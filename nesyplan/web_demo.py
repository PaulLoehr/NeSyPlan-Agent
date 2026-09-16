#!/usr/bin/env python3
"""Web demonstrator: a browser chat UI over the tracked executor.

Graphical counterpart to nesyplan/demo.py (the terminal chat). It runs the SAME engine --
the AgenticSession / _oneshot_core loop, the same selectable modes and models, the same
deterministic /cleanup -- but instead of printing to a TTY it serves a single-page chat
UI and STREAMS the agent's reasoning, tool calls and feedback to the browser over
Server-Sent Events (SSE). Stdlib only (http.server + SSE): no web framework, no build step,
no new dependencies, matching the rest of nesyplan/.

Architecture:
  - WebSession   -- the live session (robot + llm + log + mode/config + AgenticSession),
                    the exact analog of demo.Demo but publishing structured EVENTS instead
                    of printing. One per server process (single-user demonstrator).
  - WebLLM       -- wraps nesyplan.llm.LLMClient like demo.SpinnerLLM, but emits
                    reasoning/content/context events rather than printing them.
  - EventBus     -- thread-safe fan-out: an append-only history (replayed to a fresh
                    browser so a reload rebuilds the transcript) plus per-client queues.
  - _Handler     -- routes GET / (+ static), GET /events (SSE), GET /state, GET /config,
                    GET /context (the two context layers, message by message),
                    POST /task, POST /command (mode / model / cleanup / stop / new_session).

The orchestrator emits its progress via the optional `emit` sink added to AgenticSession
and _oneshot_core (nesyplan/orchestrator.py); WebLLM adds the reasoning/content/context
events; the cache-write hook adds what the reasoning cache persists. Everything the
terminal demo prints has a matching event here.

Run:
    ./scripts/run_web_demo.sh                 # wrapper: picks a port and opens the browser
    python3 -m nesyplan.web_demo --port 8600

There is nothing to start first. By default nesyplan.fake_robot serves the world model
in-process -- no containers, no simulator, no arm motion, instant resets. Everything above
the robot (chat, streaming, modes, models, meters, inspector) is the real thing; only the
motion is a no-op.

To drive an EXTERNAL executor instead (a simulator or robot you run yourself; not part of
this repository -- see docs/ARCHITECTURE.md):
    python3 -m nesyplan.web_demo --backend sim --url http://localhost:8100
"""

import argparse
import itertools
import json
import os
import queue
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import SimpleNamespace

from nesyplan.config import CacheMode, Policy
from nesyplan.environment import cube_ids, format_state, grid_units, is_tracked
from nesyplan.fake_robot import FakeRobot
from nesyplan.llm import LLMClient
from nesyplan.metrics import Metrics
from nesyplan.model_aliases import DEFAULT_MODEL, MODEL_ALIASES, resolve_model
from nesyplan.orchestrator import AgenticSession, _oneshot_core
from nesyplan.prompts import build_system_prompt
from nesyplan.replay import TranscriptReplay, find_transcript
from nesyplan.robot_client import DEFAULT_URL, RobotClient
from nesyplan.run import load_dotenv
from nesyplan.session_log import SessionLog
# Reuse the terminal demo's mode/model/meter helpers verbatim -- one source of truth, so the
# web and terminal demonstrators always offer exactly the same modes and numbers.
from nesyplan.demo import (
    CHARS_PER_TOKEN, DEFAULT_MODE, DEMO_MODES, MODE_DESC, MODE_ORDER, _config_for,
    default_context_window, estimate_tokens, modes_for, reasoning_optional)

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STATIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'webui')


# --- event bus ----------------------------------------------------------------

class _AbortTask(Exception):
    """Raised through a task's emit sink to unwind an in-flight agent loop -- either because
    the user pressed Stop (WebSession.request_stop) or because a new session took over
    mid-task, so the old loop stops instead of racing the new one."""


class EventBus:
    """Thread-safe fan-out of JSON events to any number of SSE subscribers.

    Keeps a capped, append-only history so a browser that connects (or reloads) mid-session
    can replay the transcript and rebuild the view, then follow live. Each subscriber gets
    its own bounded queue; a slow/dead client cannot block the producer (its queue just
    fills and it is dropped on the next publish).
    """

    def __init__(self, history_cap=6000):
        self._lock = threading.Lock()
        self._subs = set()
        self._history = []
        self._cap = history_cap
        self._seq = 0

    def publish(self, event):
        with self._lock:
            self._seq += 1
            event = dict(event, seq=self._seq)
            self._history.append(event)
            if len(self._history) > self._cap:
                self._history = self._history[-self._cap:]
            dead = []
            for q in self._subs:
                try:
                    q.put_nowait(event)
                except queue.Full:
                    dead.append(q)
            for q in dead:
                self._subs.discard(q)
        return event

    def subscribe(self):
        """Register a new subscriber. Returns (queue, history_snapshot) to replay first."""
        q = queue.Queue(maxsize=10000)
        with self._lock:
            history = list(self._history)
            self._subs.add(q)
        return q, history

    def unsubscribe(self, q):
        with self._lock:
            self._subs.discard(q)

    def reset(self):
        """Drop the replay history so a fresh session starts a clean transcript. The seq
        counter stays monotonic, so connected clients' de-dupe (last_seq) keeps working."""
        with self._lock:
            self._history = []


# --- LLM wrapper that emits instead of prints ---------------------------------

class WebLLM:
    """Wrap an LLMClient so each completion emits live events (llm_start -> context ->
    reasoning -> content -> llm_end) to the event bus, mirroring demo.SpinnerLLM's TTY view.
    Everything except chat()/with_model() is forwarded to the inner client, so the
    orchestrator, AgenticSession and _oneshot_core stay untouched.

    with_model() re-wraps for the cache SUMMARY side-call: it stays silent (no content /
    reasoning / context events) because that side-call is not the main-loop context -- the
    distilled note is surfaced once via the cache-write hook instead.
    """

    def __init__(self, inner, publish, context_window=0, show_thoughts=True,
                 show_content=True, show_context=True, split_fn=None):
        self._inner = inner
        self._publish = publish
        self.context_window = context_window
        self._show_thoughts = show_thoughts
        self._show_content = show_content
        self._show_context = show_context
        # Optional callable -> {'base': tok, 'loop': tok, 'loop_msgs': n}: how the request the
        # model is about to see splits into its two context layers (see
        # WebSession._context_split). Rides along on the `context` event.
        self._split_fn = split_fn
        self.last_prompt_tokens = 0

    def __getattr__(self, name):
        return getattr(self._inner, name)   # model / base_url / temperature / on_effort / ...

    def with_model(self, model):
        return WebLLM(self._inner.with_model(model), self._publish,
                      context_window=self.context_window, show_thoughts=False,
                      show_content=False, show_context=False)

    def chat(self, messages, tools=None, tool_choice=None, reasoning=None):
        if self._show_context:
            self._publish({'type': 'llm_start', 'reasoning': bool(reasoning)})
        started = time.monotonic()
        try:
            resp = self._inner.chat(messages, tools=tools, tool_choice=tool_choice,
                                    reasoning=reasoning)
        except Exception as exc:
            if self._show_context:
                self._publish({'type': 'llm_end'})
            raise
        # How long the thinking took: the response's own figure when it has one (replay carries
        # the recorded duration), else this call's wall time.
        elapsed_ms = resp.latency_ms if getattr(resp, 'latency_ms', None) is not None \
            else int((time.monotonic() - started) * 1000)
        self.last_prompt_tokens = (resp.usage or {}).get('prompt_tokens') or 0
        # The `context` event carries the two ESTIMATED conversation layers (see
        # demo.estimate_tokens) -- not this call's real prompt_tokens, which reach the UI as the
        # context peak in `meters`. The layers deliberately fall short of the real number: the
        # tool schemas (~780 tokens) go out with every turn but are not messages.
        if self._show_context:
            split = self._split_fn() if self._split_fn else None
            if split:
                self._publish({'type': 'context', **split})
        if self._show_thoughts and reasoning and resp.trace:
            # Ship the trace WITH its cost: how long it took and how much of it there is. The
            # token count is the API's own reasoning_tokens where the provider reports it
            # (phoenix reports 0 despite a real trace), else estimated from the text -- flagged
            # as such so the UI never presents a guess as a measurement.
            details = (resp.usage or {}).get('completion_tokens_details') or {}
            measured = details.get('reasoning_tokens') or 0
            self._publish({'type': 'reasoning', 'text': resp.trace, 'ms': elapsed_ms,
                           'chars': len(resp.trace),
                           'tokens': measured or int(len(resp.trace) / CHARS_PER_TOKEN),
                           'tokens_estimated': not measured})
        content = (resp.content or '').strip()
        if self._show_content and content:
            self._publish({'type': 'content', 'text': content})
        if self._show_context:
            self._publish({'type': 'llm_end'})
        return resp


# --- session ------------------------------------------------------------------

def _message_view(msg):
    """JSON-safe view of one conversation message for the context inspector: role + content,
    plus tool calls flattened to name/arguments and the id a tool result answers."""
    view = {'role': msg.get('role'), 'content': msg.get('content') or ''}
    calls = msg.get('tool_calls') or []
    if calls:
        view['tool_calls'] = [{'name': (c.get('function') or {}).get('name') or '',
                               'arguments': (c.get('function') or {}).get('arguments') or ''}
                              for c in calls]
    if msg.get('role') == 'tool':
        view['tool_call_id'] = msg.get('tool_call_id') or ''
    return view

class WebSession:
    """The live demonstrator session, driven by HTTP requests instead of a REPL.

    Mirrors demo.Demo: one persistent agentic conversation across tasks (mode switches keep
    it), one-shot re-plans each task, one SessionLog for the whole session. Publishes an
    event for everything the terminal demo would print. A single task runs at a time
    (guarded by `_busy`); mode/model/cleanup are refused while a task is in flight.
    """

    def __init__(self, args, robot, bus, log, replay=None):
        self.args = args
        self.robot = robot
        self.bus = bus
        self.log = log
        self.metrics = Metrics()
        # 'sim' -> the command server; 'fake' -> in-process world model; 'replay' -> that plus
        # canned completions from a recorded transcript (nesyplan/replay.py).
        self.backend = args.backend
        self.replay = replay
        self.max_steps = args.max_steps
        self._busy = False
        self._busy_token = 0      # bumped each begin/force-idle; end() only clears its own
        self._task_seq = 0        # increments per task; identifies the currently-live task
        self._current_task = None # a running loop whose token != this must abort
        self._stop_requested = False  # the abort came from the user's Stop, not a new session
        self._lock = threading.Lock()

        self.model_alias = args.model
        inner = self._new_inner_llm(args.model)
        self.context_window = args.context_window or default_context_window(inner.model)
        self.llm = self._wrap_llm(inner)
        self.available_modes = modes_for(self.llm.model)

        self.mode = None
        self.config = None
        self.session = None       # AgenticSession for agentic modes; None for one-shot
        self.messages = None      # persistent agentic conversation (survives mode switches)
        self.oneshot_turn = 0
        # Where the CURRENT task's turns start in self.messages -- everything before it is the
        # cross-task context that survives compaction, everything from it on is this task's
        # loop. Set per task in run_task(); drives the two-layer context view.
        self._task_base = None
        self._last_loop_tokens = 0   # loop size at the last model call (for the compaction note)
        # The finished task's trail as it was BEFORE compaction. Kept so the inspector can still
        # show the whole last loop; cleared when the next task starts.
        self._loop_snapshot = None
        self._loop_pair = None    # what that trail was compacted TO (now part of the Verlauf)
        self._task_text = None    # the running task's text (to compact an unfinished one)
        state = self.robot.get_state()
        self.grid = grid_units(state)

    def _wrap_llm(self, inner):
        return WebLLM(inner, self.bus.publish, context_window=self.context_window,
                      split_fn=self._context_split)

    # -- the two context layers --

    def _context_split(self):
        """Split the request the model is about to see into its two layers, in estimated tokens:

          base -- the CROSS-TASK context: system prompt + every earlier task collapsed to one
                  user->assistant pair. This is what survives when the task ends.
          loop -- THIS task's live trail: the task seed (with its world-state snapshot) plus
                  every turn's assistant/tool message. Compacted away at the end of the task.

        Returns None for one-shot (a single call, no persistent history to split) so the view
        falls back to a plain fullness bar. The demo runs with context_k=None, so the assembled
        request is exactly self.messages and the split lines up with what was actually sent."""
        msgs = self.messages
        if not msgs or self.config is None or self.config.policy == Policy.ONESHOT:
            return None
        base = len(msgs) if self._task_base is None else max(0, min(self._task_base, len(msgs)))
        # Between tasks the loop layer keeps reporting the FINISHED loop -- that is exactly what
        # the inspector shows -- while `base` already carries its compacted pair.
        loop_msgs = self._loop_snapshot if self._loop_snapshot is not None else msgs[base:]
        loop = estimate_tokens(loop_msgs)
        self._last_loop_tokens = loop
        return {'base': estimate_tokens(msgs[:base]), 'loop': loop, 'loop_msgs': len(loop_msgs)}

    def _publish_context(self):
        """Push the split as it stands NOW, independent of a model call, so the sidebar reflects
        a task's compaction the moment it happens."""
        split = self._context_split()
        if split:
            self.publish({'type': 'context', **split})

    def _seal_task(self):
        """A task is over: fold its compacted pair into the CROSS-TASK layer right away, so the
        Verlauf shows the finished task immediately instead of only from the next task on. The
        pair is remembered separately (`_loop_pair`) because the loop layer keeps showing the
        pre-compaction trail and has to display what that trail collapsed to."""
        if self._task_base is None or not self.messages:
            return
        base = max(0, min(self._task_base, len(self.messages)))
        self._loop_pair = [dict(m) for m in self.messages[base:]]
        self._task_base = len(self.messages)
        self._publish_context()

    def _keep_loop_snapshot(self, trail):
        """AgenticSession hands over the task's turn trail right before compaction replaces it
        with one user->assistant pair. Hold on to it so the inspector keeps showing the WHOLE
        last loop (reasoning-in-content, every tool call and result) until the next task
        starts -- otherwise the loop layer collapses to two messages the moment done() lands."""
        self._loop_snapshot = trail

    def _publish_compaction(self):
        """Say out loud what the loop context collapsed to. The turn-by-turn trail that drove
        the loop is replaced by ONE user->assistant pair, and that pair is all the next task
        carries -- the whole point of the two-layer context, so the demonstrator shows it."""
        if self._task_base is None or not self.messages:
            return
        before, after = self._last_loop_tokens or 0, estimate_tokens(self.messages[self._task_base:])
        if before > after:
            self.publish({'type': 'ctx_compact', 'before': before, 'after': after,
                          'carried': estimate_tokens(self.messages)})

    def publish(self, event):
        self.bus.publish(event)
        # After anything that can change the world, push a fresh snapshot so the sidebar
        # state view updates live (no polling). bus.publish (not self.publish) -> no recursion.
        if isinstance(event, dict) and event.get('type') in ('feedback', 'done'):
            try:
                self.bus.publish({'type': 'state', 'state': self.robot.get_state()})
            except Exception:
                pass

    def _begin_task(self):
        """Claim the latest task token; any older running loop aborts at its next emit."""
        self._task_seq += 1
        self._current_task = self._task_seq
        return self._current_task

    def _emit_for(self, my_task):
        """An emit sink bound to one task: publishes normally, but raises _AbortTask once a
        newer task (or new_session / Stop) has superseded this one, unwinding the stale loop."""
        def emit(event):
            if self._current_task != my_task:
                raise _AbortTask()
            self.publish(event)
        return emit

    # -- stopping a running task --

    def request_stop(self):
        """User-requested abort of the in-flight task/cleanup. Returns False if nothing runs.

        The loop is NOT killed mid-move: clearing the task token makes the loop's NEXT emit
        raise _AbortTask, and in an agent turn that is the `tool_call` event -- fired before
        the robot command goes out -- so the arm never starts a move it was told to abandon.
        A turn already waiting on the model finishes that one LLM call first.

        The busy flag deliberately STAYS set until the worker has actually unwound: the
        agentic loop appends to the same self.messages the next task would use, so a new task
        must not start alongside it. The UI shows a "wird beendet…" state until task_end."""
        with self._lock:
            if not self._busy:
                return False
            self._stop_requested = True
        self._current_task = None   # -> the loop aborts at its next emit
        self.bus.publish({'type': 'stopping'})
        return True

    def _finish_aborted(self, my_task, repair_history=True):
        """Close out a task/cleanup that was superseded. On a user Stop: repair the
        conversation, then end the task visibly so the composer unlocks. On new_session: stay
        silent -- that path has already reset the view, the counters and the busy flag."""
        if not self._stop_requested:
            return
        self._stop_requested = False
        self.publish({'type': 'note', 'text': 'Stopped — the task did not finish.'})
        if repair_history:
            # Collapse the stopped task like a finished one; only if there is no trail to
            # collapse do we fall back to just closing the open tool call, which is the minimum
            # the next request needs to be valid.
            if not self._compact_unfinished('(Stopped by the user before this task was finished.)'):
                self._answer_dangling_tool_calls('aborted: the user stopped the task')
        try:
            self.bus.publish({'type': 'state', 'state': self.robot.get_state()})
        except Exception:
            pass
        self._publish_meters()   # the partial run still cost tokens -- show them
        self.publish({'type': 'task_end', 'status': 'aborted'})

    def _compact_unfinished(self, marker):
        """Collapse a task that the orchestrator never got to compact. Returns True if it did.

        done()/max_steps/chat all end inside AgenticSession.run_task, which compacts the task's
        turns down to one user->assistant pair. A user Stop or a mid-task error instead unwinds
        through an exception, so that never runs -- and the raw trail (every tool call and its
        result) then stays in the conversation for good: from the next task on it sits in the
        CROSS-TASK layer of every request, precisely what compaction exists to prevent. So do
        the same collapse here, and hand the trail to the inspector as the loop snapshot."""
        if self.config is None or self.config.policy == Policy.ONESHOT:
            return False   # one-shot prompts are throwaway; self.messages is not their history
        if not self.messages or self._task_base is None:
            return False
        base = max(0, min(self._task_base, len(self.messages)))
        if base >= len(self.messages):
            return False   # the task added nothing -- nothing to collapse
        self._loop_snapshot = [dict(m) for m in self.messages[base:]]
        self.messages[base:] = [
            {'role': 'user', 'content': self._task_text or ''},
            {'role': 'assistant', 'content': marker},
        ]
        self.log.add({'role': 'system', 'content': f'[compacted unfinished task: {marker}]'})
        self._publish_compaction()
        self._seal_task()
        return True

    def _answer_dangling_tool_calls(self, error):
        """Keep the persistent conversation valid when an agentic loop was cut short mid-turn.

        A stopped loop unwinds at its `tool_call` event, and a lost command server raises at
        the robot call -- both AFTER the assistant's tool-call message was appended to the
        history but BEFORE its result. The OpenAI format requires every tool_call to be
        answered, so the next task's request would be rejected. Answer them with the truth.
        No-op for one-shot (no persistent history) and whenever the turn did complete."""
        if not self.messages:
            return
        last = self.messages[-1]
        if not (last.get('role') == 'assistant' and last.get('tool_calls')):
            return
        for call in last['tool_calls']:
            msg = {'role': 'tool', 'tool_call_id': call.get('id', ''),
                   'content': json.dumps({'ok': False, 'error': error})}
            self.messages.append(msg)
            self.log.add(msg)

    # -- config / state snapshots for the UI --

    def config_payload(self):
        state = self.robot.get_state()
        cubes = state.get('cubes') or {}
        return {
            'backend': self.backend,
            # Which recording, which task of it, how fast -- None off replay.
            'replay': self.replay.status() if self.replay is not None else None,
            'model': self.llm.model,
            'model_alias': self.model_alias,
            'models': list(MODEL_ALIASES),
            # alias -> full id, so the dropdown can show which model (and provider) an
            # alias stands for without the browser knowing the alias table.
            'model_ids': dict(MODEL_ALIASES),
            'reasoning_optional': reasoning_optional(self.llm.model),
            'mode': self.mode,
            'modes': [{'label': m, 'desc': MODE_DESC[m], 'available': m in self.available_modes}
                      for m in MODE_ORDER],
            'available_modes': self.available_modes,
            'window': self.context_window,
            'grid': self.grid,
            'busy': self._busy,
            'cubes': [{'id': c, 'color': (cubes[c] or {}).get('color'),
                       'number': (cubes[c] or {}).get('number')}
                      for c in cube_ids(state)],
        }

    def state_payload(self):
        return self.robot.get_state()

    def context_payload(self):
        """The two context layers message by message -- what the inspector (GET /context) shows.

        This is the conversation the model is actually given, verbatim, split at _task_base:
        `base` is what survives the task, `loop` is the current task's trail. The demo runs with
        context_k=None, so _assemble() passes self.messages through unchanged and this IS the
        request (a reflection instruction, appended for one PERIODIC turn, is the only addition).
        One-shot builds a fresh prompt per task and keeps no history, hence the empty layers.

        Between tasks the loop layer shows the FINISHED task's pre-compaction trail (kept by
        _keep_loop_snapshot) with `compacted: true` plus the pair it collapsed to, because the
        live list holds only that pair from the moment done() lands."""
        msgs = self.messages or []
        base = len(msgs) if self._task_base is None else max(0, min(self._task_base, len(msgs)))
        oneshot = self.config is not None and self.config.policy == Policy.ONESHOT
        snapshot = self._loop_snapshot
        loop_msgs = snapshot if snapshot is not None else msgs[base:]
        loop = {'tokens': estimate_tokens(loop_msgs),
                'messages': [_message_view(m) for m in loop_msgs],
                'compacted': snapshot is not None}
        if snapshot is not None:
            pair = self._loop_pair or []   # already folded into `base` by _seal_task
            loop['compacted_to'] = [_message_view(m) for m in pair]
            loop['compacted_tokens'] = estimate_tokens(pair)
        return {
            'mode': self.mode,
            'oneshot': oneshot,
            'window': self.context_window,
            'busy': self._busy,
            'base': {'tokens': estimate_tokens(msgs[:base]),
                     'messages': [_message_view(m) for m in msgs[:base]]},
            'loop': loop,
        }

    # -- mode / model switching (parallels demo.Demo.set_mode) --

    def set_mode(self, label, announce=True):
        if label not in self.available_modes:
            raise ValueError(f'mode {label!r} not available for this model; '
                             f'choices: {", ".join(self.available_modes)}')
        self.mode = label
        self.config = _config_for(label, SimpleNamespace(max_steps=self.max_steps))
        state = self.robot.get_state()
        if self.config.policy == Policy.ONESHOT:
            self.session = None
        else:
            if self.messages is None:
                system = {'role': 'system', 'content': build_system_prompt(state, self.config)}
                self.messages = [system]
                self.log.add(system)
                self.log.track_conversation(self.messages)
            else:
                self.log.add({'role': 'system', 'content': f'[mode -> {label}]'})
            self.session = AgenticSession(
                self.config, self.robot, self.llm, self.log, self.metrics,
                cube_ids(state), grid_units(state), self.messages,
                compact_history=True, on_cache_write=self._on_cache_write, emit=self.publish,
                on_compact=self._keep_loop_snapshot)
        if announce:
            self.publish({'type': 'mode_changed', 'mode': label, 'desc': MODE_DESC[label]})

    def _new_inner_llm(self, alias):
        """The raw LLM client for `alias` -- a real one, or the recording in replay mode.

        The single place a client is built, so `--backend replay` cannot leak a live request:
        switching models in the UI stays canned (and keeps the recording's position, which
        lives in the TranscriptReplay, not in the client)."""
        if self.replay is not None:
            return self.replay.client(resolve_model(alias))
        return LLMClient(model=alias)

    def switch_model(self, alias):
        """Swap the LLM for a new model, KEEPING the running conversation: the message history
        is model-agnostic (OpenAI-format), so the new model just continues where the old one
        left off. Only the mode changes if the new model does not offer the current one. Use
        `new_session()` to explicitly start fresh."""
        self.model_alias = alias
        inner = self._new_inner_llm(alias)
        self.context_window = self.args.context_window or default_context_window(inner.model)
        self.llm = self._wrap_llm(inner)
        self.available_modes = modes_for(self.llm.model)
        # The transcript's `model` was set once at startup, so a session that switches models
        # would be filed under the wrong one. Update it and drop a marker in the message log, so
        # a mixed session stays attributable turn by turn.
        self.log.set_meta(model=self.llm.model, model_alias=alias)
        self.log.add({'role': 'system', 'content': f'[model -> {self.llm.model}]'})
        # Keep the current mode if the new model still offers it, else fall back to the default.
        target_mode = self.mode if self.mode in self.available_modes else DEFAULT_MODE
        self.set_mode(target_mode, announce=False)   # rebuilds the session on the SAME messages
        self.publish({'type': 'model_changed', 'model': self.llm.model, 'alias': alias,
                      'window': self.context_window, 'mode': self.mode,
                      'reasoning_optional': reasoning_optional(self.llm.model),
                      'available_modes': self.available_modes})

    def new_session(self):
        """Start a fresh session NOW, whatever is running: signal any in-flight task to abort
        (its next emit raises _AbortTask), force the busy flag idle so the new session can act
        immediately, forget the conversation context + reset the counters, re-seed a clean
        conversation, and clear the event history. The physical world is left untouched (use
        /cleanup to also tidy the cubes)."""
        self._current_task = None   # supersede any running task/cleanup -> it stops at its next step
        self._stop_requested = False  # a pending Stop is moot: this reset owns the view now
        self.force_idle()           # clear busy (token bump: a stale worker's end() becomes a no-op)
        self.messages = None
        self.oneshot_turn = 0
        self._task_base = None
        self._last_loop_tokens = 0
        self._loop_snapshot = self._loop_pair = None
        self.metrics = Metrics()
        self.set_mode(self.mode, announce=False)   # re-seed a fresh conversation on empty history
        self.bus.reset()
        self.bus.publish({'type': 'session_reset'})   # direct: not gated by a task emit

    def _on_cache_write(self, mode, prev_content, new_content):
        """Announce what this turn persisted into the assistant content.

        BOTH cache modes send the text, because in both it now IS the assistant's content --
        the verbatim trace under `raw`, the distilled card under `summary` -- and therefore
        exactly what the model re-reads on the next turn. Showing it is the whole point of the
        demonstrator: the cache dimension is otherwise invisible. `overwrite` records that the
        model had written content of its own and this replaced it.
        """
        self.publish({'type': 'cache_write',
                      'mode': 'raw' if mode == CacheMode.RAW else 'summary',
                      'overwrite': bool((prev_content or '').strip()),
                      'text': new_content})

    # -- running a task --

    def run_task(self, task):
        my_task = self._begin_task()
        emit = self._emit_for(my_task)
        self._task_text = task
        self.publish({'type': 'user_msg', 'text': task})
        self.publish({'type': 'task_start', 'task': task, 'mode': self.mode})
        try:
            if self.config.policy == Policy.ONESHOT:
                self._task_base = None   # one-shot keeps no history: nothing to split or compact
                state = self.robot.get_state()
                messages = [
                    {'role': 'system', 'content': build_system_prompt(state, self.config)},
                    {'role': 'user', 'content': task},
                ]
                for m in messages:
                    self.log.add(m)
                result = _oneshot_core(self.config, self.robot, self.llm, self.log,
                                       self.metrics, messages, base_turn=self.oneshot_turn,
                                       emit=emit)
                self.oneshot_turn += result['steps'] + 1
            else:
                self.session.emit = emit   # bind the loop's events to THIS task (abortable)
                self._task_base = len(self.messages)   # from here on it is THIS task's loop
                # the previous loop stops being what the view shows
                self._loop_snapshot = self._loop_pair = None
                result = self.session.run_task(task)
                self._publish_compaction()
                self._seal_task()   # the new pair belongs to the Verlauf from now, not the loop
            status = result.get('status')
        except _AbortTask:
            # Stop pressed, or a new session took over mid-task.
            return self._finish_aborted(my_task)
        except ConnectionError as exc:
            self.publish({'type': 'error', 'text': f'lost the command server: {exc}'})
            if not self._compact_unfinished('(Interrupted: lost the command server.)'):
                self._answer_dangling_tool_calls(f'interrupted: {exc}')
            status = 'error'
        except Exception as exc:   # a bad task must not kill the server
            self.publish({'type': 'error', 'text': f'{type(exc).__name__}: {exc}'})
            if not self._compact_unfinished(f'(Interrupted: {type(exc).__name__}.)'):
                self._answer_dangling_tool_calls(f'interrupted: {type(exc).__name__}: {exc}')
            status = 'error'
        if self._current_task != my_task:
            # Superseded while finalising: the loop itself already returned cleanly, so there
            # is nothing to repair -- just close the task out for the view.
            return self._finish_aborted(my_task, repair_history=False)
        self._publish_meters()
        self.publish({'type': 'task_end', 'status': status})

    def cleanup(self):
        """Deterministically return every placed/held cube to storage, emitting progress
        (mirrors demo.deterministic_cleanup but as events instead of prints)."""
        my_task = self._begin_task()
        self.publish({'type': 'task_start', 'task': '/cleanup', 'mode': self.mode})
        moved = 0
        try:
            for _ in range(len(cube_ids(self.robot.get_state())) + 1):
                if self._current_task != my_task:
                    # Stop pressed, or a new session superseded the cleanup. Checked between
                    # stores, so the arm always finishes the move it started.
                    return self._finish_aborted(my_task, repair_history=False)
                state = self.robot.get_state()
                cubes = state.get('cubes') or {}
                held = state.get('held')
                movable = [c for c, info in cubes.items()
                           if (info or {}).get('location') in ('area', 'held')]
                if not movable:
                    break
                target = held if held else max(movable, key=lambda c: (cubes[c].get('level') or 0))
                self.publish({'type': 'tool_call', 'name': 'store', 'args': {'cube_id': target},
                              'reasoning': False})
                fb = self.robot.send_command('store', cube_id=target)
                ok = bool(fb.get('ok'))
                self.publish({'type': 'feedback', 'ok': ok, 'message': fb.get('message'),
                              'error': fb.get('error'),
                              'held': (fb.get('observation') or {}).get('held')})
                if not ok:
                    self.publish({'type': 'note', 'text': 'cleanup stopped (store failed)'})
                    break
                moved += 1
        except ConnectionError as exc:
            self.publish({'type': 'error', 'text': f'lost the command server: {exc}'})
        if self._current_task != my_task:
            return self._finish_aborted(my_task, repair_history=False)   # superseded
        self.publish({'type': 'done',
                      'reason': (f'cleanup: {moved} cube(s) returned to storage' if moved
                                 else 'cleanup: nothing to do (all cubes already in storage)')})
        self.publish({'type': 'task_end', 'status': 'cleanup'})

    def _publish_meters(self):
        peak = getattr(self.llm, 'last_prompt_tokens', 0) or 0
        carried = estimate_tokens(self.messages) if self.messages else 0
        t = self.metrics.tokens
        self.publish({'type': 'meters', 'context_peak': peak, 'window': self.context_window,
                      'carried': carried, 'tokens': dict(t)})

    # -- busy guard: only one task/cleanup at a time --

    def try_begin(self):
        """Claim the busy flag for a new task/cleanup. Returns a monotonic token to hand back
        to end(); False if a task is already running. The token lets end() no-op when a later
        force_idle() has superseded this claim (see _busy_token)."""
        with self._lock:
            if self._busy:
                return False
            self._busy = True
            self._busy_token += 1
            return self._busy_token

    def end(self, token):
        """Release the busy flag, but only if this claim is still the current one -- a worker
        that force_idle() has already superseded (token bumped past it) becomes a no-op instead
        of clearing the busy flag out from under a freshly-started task."""
        with self._lock:
            if token == self._busy_token:
                self._busy = False

    def force_idle(self):
        """Force the busy flag idle NOW and invalidate the in-flight claim, so a stale worker's
        end() cannot later clear a fresh task. Lets new_session() take over immediately."""
        with self._lock:
            self._busy = False
            self._busy_token += 1

    @property
    def busy(self):
        return self._busy


# --- HTTP handler -------------------------------------------------------------

_CONTENT_TYPES = {'.html': 'text/html; charset=utf-8', '.js': 'text/javascript; charset=utf-8',
                  '.css': 'text/css; charset=utf-8', '.svg': 'image/svg+xml',
                  '.png': 'image/png', '.ico': 'image/x-icon'}


def _run_task_thread(session, fn, token):
    """Run `fn` (a task or cleanup) in a daemon thread, clearing the busy flag when done.
    `token` is the try_begin() claim: end() ignores it if force_idle() has since superseded it."""
    def worker():
        try:
            fn()
        finally:
            session.end(token)
    threading.Thread(target=worker, daemon=True).start()


class _QuietServer(ThreadingHTTPServer):
    """ThreadingHTTPServer that does not print a traceback when a client just goes away.

    A closed browser tab (or any dropped SSE stream) resets its keep-alive connection, and
    the reset surfaces in socketserver's own request loop -- outside the handler, so the
    handler's own except clause cannot see it. The default handle_error then dumps a
    ConnectionResetError traceback that reads like a server crash. Everything else is still
    reported."""

    def handle_error(self, request, client_address):
        if isinstance(sys.exc_info()[1], (ConnectionResetError, BrokenPipeError)):
            return
        super().handle_error(request, client_address)


class _Handler(BaseHTTPRequestHandler):
    server_version = 'NeSyPlanWebDemo/1.0'
    session = None   # set on the server instance below

    def log_message(self, *args):
        pass   # keep the console clean; the session log + events are the record

    # -- helpers --

    def _send_json(self, obj, status=200):
        body = json.dumps(obj).encode('utf-8')
        self.send_response(status)
        self.send_header('Content-Type', 'application/json; charset=utf-8')
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _read_json(self):
        length = int(self.headers.get('Content-Length') or 0)
        if not length:
            return {}
        try:
            return json.loads(self.rfile.read(length) or b'{}')
        except ValueError:
            return {}

    def _serve_static(self, name):
        safe = os.path.normpath(name).lstrip('/')
        path = os.path.join(STATIC_DIR, safe)
        if not path.startswith(STATIC_DIR) or not os.path.isfile(path):
            self.send_error(404)
            return
        with open(path, 'rb') as fh:
            body = fh.read()
        ext = os.path.splitext(path)[1]
        self.send_response(200)
        self.send_header('Content-Type', _CONTENT_TYPES.get(ext, 'application/octet-stream'))
        # Never cache the UI: a browser holding an old app.js against a restarted server would
        # silently talk an older event/command contract. Re-read per request is free here.
        self.send_header('Cache-Control', 'no-store')
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    # -- routes --

    def do_GET(self):
        path = self.path.split('?', 1)[0]
        if path == '/':
            self._serve_static('index.html')
        elif path == '/config':
            self._send_json(self.session.config_payload())
        elif path == '/state':
            try:
                self._send_json(self.session.state_payload())
            except ConnectionError as exc:
                self._send_json({'error': str(exc)}, status=502)
        elif path == '/context':
            self._send_json(self.session.context_payload())
        elif path == '/events':
            self._serve_events()
        else:
            # Any other path is a static asset (app.js, style.css, vendor/vue...);
            # _serve_static guards against traversal and 404s on a miss.
            self._serve_static(path.lstrip('/'))

    def do_POST(self):
        path = self.path.split('?', 1)[0]
        if path == '/task':
            self._post_task()
        elif path == '/command':
            self._post_command()
        else:
            self.send_error(404)

    def _post_task(self):
        body = self._read_json()
        task = (body.get('task') or '').strip()
        if not task:
            self._send_json({'ok': False, 'error': 'empty task'}, status=400)
            return
        token = self.session.try_begin()
        if not token:
            self._send_json({'ok': False, 'error': 'busy'}, status=409)
            return
        if self.session.replay is not None:
            # In replay the composer is a "play the next recorded task" button: whatever was
            # typed is dropped in favour of the recorded task text, so the request and the
            # reasoning that answers it stay consistent. Wraps around at the end (see
            # TranscriptReplay.next_task), so the stream can be re-triggered indefinitely.
            task = self.session.replay.next_task()
        _run_task_thread(self.session, lambda: self.session.run_task(task), token)
        self._send_json({'ok': True})

    def _post_command(self):
        body = self._read_json()
        cmd = (body.get('cmd') or '').strip().lower()
        if cmd == 'state':
            self._send_json({'ok': True, 'state': self.session.state_payload()})
            return
        if cmd == 'stop':
            # Idempotent: clicking again while the loop unwinds just re-announces `stopping`;
            # once it is idle the answer is 409 and the UI is already back to the send button.
            if not self.session.request_stop():
                self._send_json({'ok': False, 'error': 'idle'}, status=409)
                return
            self._send_json({'ok': True})
            return
        if cmd == 'cleanup':
            token = self.session.try_begin()
            if not token:
                self._send_json({'ok': False, 'error': 'busy'}, status=409)
                return
            _run_task_thread(self.session, self.session.cleanup, token)
            self._send_json({'ok': True})
            return
        if cmd == 'new_session':
            if self.session.busy:
                self._send_json({'ok': False, 'error': 'busy'}, status=409)
                return
            self.session.new_session()
            self._send_json({'ok': True, 'config': self.session.config_payload()})
            return
        if cmd in ('mode', 'model'):
            if self.session.busy:
                self._send_json({'ok': False, 'error': 'busy'}, status=409)
                return
            try:
                if cmd == 'mode':
                    self.session.set_mode(body.get('mode'))
                else:
                    alias = body.get('model')
                    if alias not in MODEL_ALIASES:
                        raise ValueError(f'unknown model {alias!r}; choices: '
                                         f'{", ".join(MODEL_ALIASES)}')
                    self.session.switch_model(alias)
                self._send_json({'ok': True, 'config': self.session.config_payload()})
            except Exception as exc:
                self._send_json({'ok': False, 'error': str(exc)}, status=400)
            return
        self._send_json({'ok': False, 'error': f'unknown command {cmd!r}'}, status=400)

    def _serve_events(self):
        q, history = self.session.bus.subscribe()
        self.send_response(200)
        self.send_header('Content-Type', 'text/event-stream')
        self.send_header('Cache-Control', 'no-cache')
        self.send_header('Connection', 'keep-alive')
        self.send_header('X-Accel-Buffering', 'no')
        self.end_headers()
        try:
            for event in history:
                self._sse(event)
            while True:
                try:
                    event = q.get(timeout=15)
                    self._sse(event)
                except queue.Empty:
                    self.wfile.write(b': ping\n\n')   # keep-alive comment
                    self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError, OSError):
            pass   # client went away
        finally:
            self.session.bus.unsubscribe(q)

    def _sse(self, event):
        self.wfile.write(f'data: {json.dumps(event)}\n\n'.encode('utf-8'))
        self.wfile.flush()


# --- entry point --------------------------------------------------------------

def parse_args(argv):
    p = argparse.ArgumentParser(description='Web demonstrator over the tracked executor.')
    p.add_argument('--backend', default=os.environ.get('EVAL_BACKEND', 'fake'),
                   choices=['fake', 'sim', 'replay'],
                   help='Where the world lives. "fake" (default) runs the world model '
                        'in-process -- no containers, no simulator, no arm motion, instant. '
                        '"sim" talks to an external command server on --url, where a simulated '
                        'or real arm actually moves (that executor is not part of this '
                        'repository; see docs/ARCHITECTURE.md). "replay" adds canned completions '
                        'from a recorded transcript, so no LLM is called either: the whole UI '
                        'runs off a past session. (default: %(default)s)')
    p.add_argument('--transcript', default=None,
                   help='Transcript to replay (--backend replay; default: the newest usable one '
                        'under --log-dir).')
    p.add_argument('--replay-speed', type=float, default=None,
                   help='Replay playback speed as a multiple of the recorded per-turn latency '
                        '(default: 4 = four times faster than the original run).')
    p.add_argument('--url', default=os.environ.get('AGENT_ROBOT_URL') or DEFAULT_URL,
                   help='Command server URL (--backend sim only; default: %(default)s).')
    p.add_argument('--model', default=(os.environ.get('AGENT_MODEL') or DEFAULT_MODEL),
                   help='LLM model: alias (qwen3-32b|qwen3-30b-a3b|qwen3-14b, or phoenix|kimi|command|merlin) or full id '
                        '(default: %(default)s).')
    p.add_argument('--mode', default=None, choices=list(DEMO_MODES),
                   help='Start in this mode (default: react, or a model-available fallback).')
    p.add_argument('--port', type=int, default=int(os.environ.get('AGENT_WEB_PORT') or 8600),
                   help='Port for the web UI (default: %(default)s).')
    p.add_argument('--host', default='127.0.0.1', help='Bind address (default: %(default)s).')
    p.add_argument('--max-steps', type=int, default=25, help='Safety cap on agent turns per task.')
    p.add_argument('--context-window', type=int,
                   default=int(os.environ.get('AGENT_CONTEXT_WINDOW') or 0) or None,
                   help='Token window for the context-fullness meter (default: per-model).')
    p.add_argument('--log-dir',
                   default=os.environ.get('LLM_LOG_DIR') or os.path.join(REPO_ROOT, 'llm_logs'),
                   help='Where to write the demo session transcript (default: %(default)s).')
    return p.parse_args(argv)


def main(argv=None):
    load_dotenv(os.path.join(REPO_ROOT, '.env'))
    args = parse_args(argv if argv is not None else sys.argv[1:])

    replay = None
    if args.backend in ('fake', 'replay'):
        # No container, no simulator: nesyplan.fake_robot runs the tracked executor's world
        # model (and its exact validation/feedback) in-process, so the whole UI works -- only
        # the arm motion is a no-op. Starts in the initial all-in-storage layout.
        robot = FakeRobot()
        print('Backend: fake (in-process tracked world model -- no simulator, no arm motion)')
        if args.backend == 'replay':
            try:
                replay = TranscriptReplay(find_transcript(args.log_dir, args.transcript),
                                          speed=args.replay_speed)
            except (FileNotFoundError, ValueError) as exc:
                print(f'error: {exc}', file=sys.stderr)
                return 2
            st = replay.status()
            print(f'Replay:  {st["file"]} -- {st["tasks"]} recorded task(s) of '
                  f'{st["model"] or "?"}, {st["speed"]:g}x speed. No LLM is called; the '
                  f'composer plays the next recorded task.')
            # Start on the model the recording used (unless --model said otherwise), so the
            # offered modes match what actually produced these turns.
            if args.model == (os.environ.get('AGENT_MODEL') or DEFAULT_MODEL):
                by_id = {full: alias for alias, full in MODEL_ALIASES.items()}
                args.model = by_id.get(replay.recorded_model, args.model)
    else:
        robot = RobotClient(args.url)
        print(f'Connecting to command server at {args.url} ...')
    try:
        state = robot.get_state()
    except ConnectionError as exc:
        print(f'error: {exc}', file=sys.stderr)
        return 2
    if not is_tracked(state):
        print('error: the demo needs a TRACKED executor (mode="tracked"); the one on '
              f'{args.url} reports otherwise.\n'
              '  Run without an external executor instead: --backend fake', file=sys.stderr)
        return 2

    bus = EventBus()
    # In replay the session is nominally the RECORDED model -- the transcript this run writes
    # must not claim a live model that was never asked anything.
    log_model = replay.recorded_model if replay is not None else LLMClient(model=args.model).model
    log = SessionLog(args.log_dir, 'demo', task='(interactive web demo session)',
                     model=log_model, base_url=None)
    log.set_meta(backend=args.backend, kind='web_demo',
                 **({'replay_of': replay.name} if replay is not None else {}))
    session = WebSession(args, robot, bus, log, replay=replay)
    start_mode = args.mode if (args.mode and args.mode in session.available_modes) else DEFAULT_MODE
    session.set_mode(start_mode, announce=False)

    n = len(state.get('cubes') or {})
    where = 'Connected' if args.backend == 'sim' else 'Ready'
    print(f'{where}: {n} cubes, grid {session.grid}x{session.grid}, mode=tracked.')
    print(f'LLM: model={session.llm.model}  window={session.context_window}  mode={session.mode}')
    if log.path:
        print(f'Session log: {log.path}')

    _Handler.session = session
    httpd = _QuietServer((args.host, args.port), _Handler)
    url = f'http://{args.host}:{args.port}/'
    print(f'\n==> Web demonstrator ready:  {url}\n    (Ctrl-C to stop)\n')
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print('\nStopping ...')
    finally:
        httpd.shutdown()
        log.finish(status='demo_end')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
