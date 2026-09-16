"""Replay a recorded session transcript through the LIVE pipeline (stdlib only).

Working on the UI needs a realistic stream: reasoning blocks of real length, real tool
calls, real feedback, a world that evolves, meters that move. A live run needs a reachable
LLM endpoint (and, off `--backend fake`, containers) -- and iterating on CSS by burning
tokens on a real model is slow, costly and never twice the same.

So this fakes exactly ONE thing: the network call. `ReplayLLM` duck-types
nesyplan.llm.LLMClient and serves the assistant messages a past session recorded -- traces,
tool calls, token usage and per-turn latency included. Everything else runs for real:
AgenticSession drives the loop, the tracked world model (nesyplan.fake_robot) executes the
recorded tool calls and produces its own feedback, the reasoning cache writes its notes, the
metrics accumulate, and web_demo's WebLLM emits the same events it emits live. The stream is
therefore identical to a live run BY CONSTRUCTION rather than by imitation -- there is no
second code path that could drift from the real one.

Two consequences worth knowing:

  - The world is LIVE, the actions are canned. Replayed into a world that diverged from the
    recording (you switched modes, cleaned up mid-way, or the recording started elsewhere),
    a recorded place() can legitimately be REJECTED by the world model. That is not a bug in
    the replay: it is the real validation talking, and the loop reacts as it would live.
  - When a task's recorded completions run out before it called done(), a synthetic done()
    closes the task instead of letting the loop spin.

The cache SUMMARY side-call is served from the recording too (via with_model, exactly like
the live summarizer client), so a replay never reaches out to the network -- not even for a
summary. Where the recording holds no summarizer turns (it ran a different cache mode), a
one-line note is distilled from the trace locally.

Usage:
    ./scripts/run_web_demo.sh replay                      # newest transcript, no containers
    python3 -m nesyplan.web_demo --backend replay --transcript llm_logs/demo/<file>.json
"""

import glob
import json
import os
import time

from nesyplan.llm import ChatResponse

# The marker nesyplan.cache's summarizer call writes into the log before its own messages;
# the user+assistant pair that follows belongs to the summarizer, not to the driver loop.
SUMMARIZER_MARKER = '[cache summarizer call]'

# A recorded task turn is "<task text>\n\nCurrent world state:\n<state>" (see
# AgenticSession.run_task). Only the text is the task: the live loop appends the CURRENT
# state itself, so replaying the recorded suffix would duplicate it -- in the request and,
# worse, in the user bubble the UI shows.
TASK_STATE_MARKER = '\n\nCurrent world state:'

# Playback speed as a multiple of the recorded latency (4 = four times faster than the
# original run). Each wait is clamped so the UI neither flickers nor stalls: a 20 s runaway
# reasoning turn must not freeze a CSS iteration.
DEFAULT_SPEED = 4.0
MIN_WAIT = 0.2
MAX_WAIT = 3.0
FALLBACK_LATENCY_MS = 1200   # recordings predating the per-turn latency field

# Fields the API returns empty on every message; dropped so the replayed conversation (and
# the context inspector showing it) stays as clean as a hand-built one.
_NOISE_KEYS = ('refusal', 'annotations', 'audio', 'function_call')


class _Completion:
    """One recorded assistant turn: the message plus what it cost and how long it took."""

    __slots__ = ('message', 'usage', 'latency_ms')

    def __init__(self, message, usage=None, latency_ms=None):
        self.message = message
        self.usage = usage
        self.latency_ms = latency_ms


class _Block:
    """One recorded task: the user's text and the completions that answered it."""

    def __init__(self, task):
        self.task = task
        self.turns = []       # driver completions, in order
        self.summaries = []   # cache-summarizer completions, in order


def _clean(message):
    """The recorded assistant message with the always-empty API fields dropped."""
    return {k: v for k, v in message.items() if not (k in _NOISE_KEYS and not v)}


def _api_usage(recorded):
    """turns[].usage (already flattened by nesyplan.metrics) -> the OpenAI-shaped object.

    metrics.normalize_usage and WebLLM both read the API spelling (prompt_tokens, ...), so
    translate back rather than teaching them a second shape.
    """
    if not recorded:
        return None
    prompt = recorded.get('prompt') or 0
    completion = recorded.get('completion') or 0
    return {
        'prompt_tokens': prompt,
        'completion_tokens': completion,
        'total_tokens': recorded.get('total') or (prompt + completion),
        'completion_tokens_details': {'reasoning_tokens': recorded.get('reasoning') or 0},
    }


def load_blocks(path):
    """Split a transcript into (meta, [_Block]) -- one block per recorded task.

    The transcript is the append-only message log, so a task is "a user message and every
    driver assistant message until the next one". Two things must be filtered out: the
    system markers the demo writes ([mode -> x], [compacted ...]) and the cache
    summarizer's own user+assistant pair, which would otherwise be replayed as driver turns.
    """
    with open(path, encoding='utf-8') as fh:
        data = json.load(fh)
    messages = data.get('messages') or []
    turn_records = data.get('turns') or []

    blocks, current, next_turn, in_summarizer = [], None, 0, False
    for msg in messages:
        role, content = msg.get('role'), msg.get('content')
        if role == 'system':
            if isinstance(content, str) and SUMMARIZER_MARKER in content:
                in_summarizer = True
            continue
        if role == 'tool':
            continue          # the live world model produces its own feedback
        if role == 'user':
            if in_summarizer:
                continue      # the summarizer's prompt (the trace), not a task
            task = (content or '').split(TASK_STATE_MARKER, 1)[0].strip()
            current = _Block(task)
            blocks.append(current)
            continue
        if role != 'assistant' or current is None:
            continue
        if in_summarizer:
            current.summaries.append(_Completion(_clean(msg)))
            in_summarizer = False
            continue
        record = turn_records[next_turn] if next_turn < len(turn_records) else {}
        next_turn += 1
        current.turns.append(_Completion(_clean(msg), _api_usage(record.get('usage')),
                                         record.get('latency_ms')))
    return data, [b for b in blocks if b.turns]


# A recording worth defaulting to shows more than a single short exchange -- below this many
# driver turns the stream is over before a UI change can be judged (an explicit --transcript
# is of course replayed whatever its size).
SUBSTANTIAL_TURNS = 6


def find_transcript(log_dir, explicit=None):
    """Resolve which transcript to replay: an explicit path, else the best default.

    "Usable" = at least one task block with a tool call, so the replay shows the agent doing
    something. Preference order: web_demo recordings (real demo sessions -- several tasks over
    one evolving world) over research runs, substantial ones over thin ones, then newest. So
    the default is the last session you recorded, unless that was a two-turn stub, in which
    case a fuller one is the more useful thing to iterate against.
    """
    if explicit:
        if not os.path.isfile(explicit):
            raise FileNotFoundError(f'transcript not found: {explicit}')
        return explicit

    candidates = []
    for path in glob.glob(os.path.join(log_dir, '**', '*.json'), recursive=True):
        try:
            data, blocks = load_blocks(path)
        except Exception:   # noqa: BLE001 -- a half-written log must not break the picker
            continue
        turns = sum(len(b.turns) for b in blocks)
        if not any(t.message.get('tool_calls') for b in blocks for t in b.turns):
            continue
        if data.get('replay_of'):
            continue          # a replay session's own transcript: replaying it would just
                              # replay a replay, and its turns are a copy of the original's
                              # anyway. Still selectable explicitly via --transcript.
        candidates.append((1 if data.get('kind') == 'web_demo' else 0,
                           1 if turns >= SUBSTANTIAL_TURNS else 0,
                           os.path.getmtime(path), path))
    if not candidates:
        raise FileNotFoundError(
            f'no replayable transcript under {log_dir} -- record one first '
            f'(./scripts/run_web_demo.sh fake) or pass --transcript')
    return max(candidates)[3]


class TranscriptReplay:
    """The cursor over a recording, shared by every ReplayLLM it hands out.

    One per process. `next_task()` arms the next recorded task; the clients then pop that
    task's completions. The cursor lives here, not in the client, so switching models in the
    UI (which builds a NEW client) continues where the old one left off.
    """

    def __init__(self, path, speed=DEFAULT_SPEED, sleep=time.sleep):
        self.path = path
        self.meta, self.blocks = load_blocks(path)
        if not self.blocks:
            raise ValueError(f'{path} holds no replayable task (no assistant turns)')
        self.speed = max(0.1, float(speed or DEFAULT_SPEED))
        self._sleep = sleep
        self._played = 0      # how many tasks have been armed (wraps over the recording)
        self._queue = []      # the armed task's remaining driver completions
        self._summaries = []

    # -- what the recording was --

    @property
    def recorded_model(self):
        return self.meta.get('model') or ''

    @property
    def name(self):
        return os.path.basename(self.path)

    def status(self):
        """Where we are in the recording -- surfaced in /config for the UI."""
        return {'file': self.name, 'model': self.recorded_model,
                'task': (self._played - 1) % len(self.blocks) + 1 if self._played else 0,
                'tasks': len(self.blocks), 'speed': self.speed}

    # -- driving --

    def next_task(self):
        """Arm the next recorded task and return its text, wrapping around at the end.

        Wrapping is deliberate: it makes the recording an endlessly re-triggerable stream, so
        a UI change can be looked at as often as needed without restarting anything.
        """
        block = self.blocks[self._played % len(self.blocks)]
        self._played += 1
        self._queue = list(block.turns)
        self._summaries = list(block.summaries)
        return block.task

    def next_completion(self):
        if not self._queue:
            return self._synthetic_done()
        completion = self._queue.pop(0)
        self._wait(completion.latency_ms)
        # The RECORDED duration travels with the response, so the UI reports how long the
        # model actually thought -- not how long this (sped-up) replay waited.
        return ChatResponse(completion.message, completion.usage, completion.latency_ms)

    def next_summary(self, messages):
        """A cache-summarizer completion: recorded if the recording has one, else distilled
        locally from the trace it was handed -- so a replay never calls out for a summary."""
        if self._summaries:
            completion = self._summaries.pop(0)
            self._wait(completion.latency_ms)
            return ChatResponse(completion.message, completion.usage)
        return ChatResponse({'role': 'assistant', 'content': _distill(messages)},
                            {'prompt_tokens': 0, 'completion_tokens': 0, 'total_tokens': 0})

    def client(self, model=None):
        """A ReplayLLM on this recording, nominally running `model` (default: the recorded one)."""
        return ReplayLLM(self, model or self.recorded_model)

    # -- internals --

    def _wait(self, latency_ms):
        secs = (latency_ms or FALLBACK_LATENCY_MS) / 1000.0 / self.speed
        self._sleep(max(MIN_WAIT, min(MAX_WAIT, secs)))

    def _synthetic_done(self):
        """Close a task whose recording ran out before it called done()."""
        self._wait(MIN_WAIT * 1000 * self.speed)
        return ChatResponse({
            'role': 'assistant', 'content': '',
            'tool_calls': [{'id': 'replay_done', 'type': 'function', 'function': {
                'name': 'done',
                'arguments': json.dumps({'reason': 'Aufzeichnung dieses Tasks zu Ende (Replay).'})}}],
        }, None)


def _distill(messages):
    """The first meaningful line of the trace the summarizer was asked to compress."""
    trace = ''
    for msg in reversed(messages or []):
        content = msg.get('content')
        if msg.get('role') == 'user' and isinstance(content, str):
            trace = content.split('REASONING TRACE:', 1)[-1]
            break
    for line in trace.splitlines():
        line = line.strip()
        if len(line) > 20:
            return line[:220]
    return '-'


class ReplayLLM:
    """Serves RECORDED completions with nesyplan.llm.LLMClient's surface.

    The pipeline touches only .chat(), .model, .base_url and .with_model(), so those are what
    this provides; `base_url` reads `replay://<file>` so every log line and the session
    transcript say plainly that no endpoint was involved.
    """

    def __init__(self, replay, model, summarizer=False):
        self._replay = replay
        self._summarizer = summarizer
        self.model = model
        self.provider = 'replay'
        self.base_url = f'replay://{replay.name}'
        self.temperature = (replay.meta.get('config') or {}).get('temperature')
        self.top_p = (replay.meta.get('config') or {}).get('top_p')
        self.on_effort = None

    def with_model(self, model):
        """The cache summarizer's sibling client -- also canned (see next_summary)."""
        return ReplayLLM(self._replay, model, summarizer=True)

    def chat(self, messages, tools=None, tool_choice=None, reasoning=None):
        if self._summarizer:
            return self._replay.next_summary(messages)
        return self._replay.next_completion()
