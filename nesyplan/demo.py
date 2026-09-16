#!/usr/bin/env python3
"""Interactive demonstrator: a terminal chat that drives the executor.

A terminal chat for showing the agent off. Unlike nesyplan/run.py (one task, one reset)
and nesyplan/eval.py (a matrix with a reset per cell), this is a CONTINUOUS session over
ONE evolving world:

  - Starting point: all cubes in storage (this driver checks on connect and offers
    /cleanup if the world is not in that state).
  - The user types a task in natural language ("stelle den roten Wuerfel in die Mitte");
    the agent carries it out on the arm.
  - When it is done, the user types the NEXT task -- NO env reset, it continues from the
    current state (the structure stays as built).
  - Cubes can be returned to storage: the agent has a store() tool (so "raeume alle
    Wuerfel auf" works as a task) AND there is a deterministic /cleanup command that tidies
    up without the LLM (a reliable re-demo reset to the all-in-storage layout).
  - Before the first task the user picks the MODEL (default qwen3-32b; see
    nesyplan/model_aliases.py -- the endpoint follows the model, so the list spans
    providers) and then the MODE; /mode switches the mode live. The reasoning-OFF modes
    (oneshot_nocot, reason_first_noreason, on_error*) are offered ONLY for models whose
    chain-of-thought an effort of "none" actually disables -- the qwen3 aliases and phoenix;
    on the others they would silently collapse into their always-on cousins. The always-on
    modes (oneshot, react, react_raw, react_summary) are available on every model.

The chat is a genuine continuous conversation for the agentic modes: one AgenticSession
(nesyplan/orchestrator.py) persists the message history + reasoning cache across tasks, so
follow-ups like "now move it a bit left" have context. But the carried context is COMPACTED
per task (AgenticSession compact_history): the mode's turn-by-turn reasoning/tool trail
governs the live loop, yet once a task finishes only a clean chat pair survives it -- the
task text plus the done() reason as assistant content. So between tasks (the "normal chat")
the reasoning is gone; the next task is re-seeded with a fresh world-state snapshot anyway.
One-shot has no conversation by nature -- each one-shot task re-plans from the current
world state.

Run it with no setup at all -- the world model runs in-process:
    python3 -m nesyplan.demo

To drive an EXTERNAL executor instead (a simulator or robot you run yourself; not part of
this repository -- see docs/ARCHITECTURE.md):
    python3 -m nesyplan.demo --backend sim --url http://localhost:8100
"""

import argparse
import itertools
import os
import sys
import threading
import time

from nesyplan.config import CacheMode, EpisodeConfig, Policy
from nesyplan.environment import cube_ids, format_state, grid_units, is_tracked
from nesyplan.fake_robot import FakeRobot
from nesyplan.llm import LLMClient
from nesyplan.metrics import Metrics
from nesyplan.model_aliases import DEFAULT_ALIAS, MODEL_ALIASES
from nesyplan.orchestrator import AgenticSession, _oneshot_core
from nesyplan.prompts import build_system_prompt
from nesyplan.robot_client import DEFAULT_URL, RobotClient
from nesyplan.run import load_dotenv
from nesyplan.session_log import SessionLog

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# --- Claude-Code-style terminal view ------------------------------------------
# Emit cursor/colour control only on a real TTY; degrade to plain lines otherwise.
_TTY = sys.stdout.isatty()
_DIM = '\033[2m' if _TTY else ''
_CYAN = '\033[36m' if _TTY else ''
_RESET = '\033[0m' if _TTY else ''
_SPIN_FRAMES = '⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏'
MAX_TRACE_LINES = 40   # cap the printed reasoning trace (the full text is in the session log)


class _Spinner:
    """A tiny live spinner with an elapsed-seconds counter, redrawn in place on one line.

    Runs in a daemon thread so it keeps ticking while the blocking LLM call is in flight;
    the whole lifetime is inside a `with` block, so it is always stopped and its line wiped
    before anything else prints.
    """

    def __init__(self, label):
        self.label = label
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _run(self):
        t0 = time.monotonic()
        for frame in itertools.cycle(_SPIN_FRAMES):
            if self._stop.is_set():
                break
            elapsed = time.monotonic() - t0
            sys.stdout.write(f'\r{_CYAN}{frame}{_RESET} {self.label}… {_DIM}{elapsed:5.1f}s{_RESET}   ')
            sys.stdout.flush()
            self._stop.wait(0.1)

    def __enter__(self):
        if _TTY:
            self._thread.start()
        else:
            sys.stdout.write(f'  ({self.label}…)\n')
            sys.stdout.flush()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        if _TTY:
            self._thread.join(timeout=0.5)
            sys.stdout.write('\r' + ' ' * 48 + '\r')   # wipe the spinner line
            sys.stdout.flush()


def _print_block(icon, kind, text, dim=False):
    """Print a titled, indented text block (reasoning / content), optionally dimmed."""
    lines = text.strip().splitlines()
    hidden = max(0, len(lines) - MAX_TRACE_LINES)
    lines = lines[:MAX_TRACE_LINES]
    print(f'{_DIM}{icon} {kind}{_RESET}')
    body = _DIM if dim else ''
    for line in lines:
        print(f'{body}  {line}{_RESET}' if body else f'  {line}')
    if hidden:
        print(f'{_DIM}  … ({hidden} more lines; full trace in the session log){_RESET}')


# --- Context-fullness meter ---------------------------------------------------
# The honest "how full is the context" number is the last driver call's prompt_tokens
# (measured by the API). The carried (post-compaction) context between tasks has not
# been sent yet, so it is ESTIMATED (~chars/4, labelled ~). Window sizes below are from
# the model cards (phoenix 65K, command 128K, kimi 256K); merlin is unverified -> assumed
# like phoenix. qwen3-32b is the pessimistic case: OpenRouter fans one id out over upstreams
# with DIFFERENT windows (40K on DeepInfra/Nebius, 128K on Alibaba/SiliconFlow) and picks one
# per request, so the meter takes the smallest -- it may under-report headroom, never
# over-report it. Override any of them with --context-window / AGENT_CONTEXT_WINDOW.
_CONTEXT_WINDOWS = (('kimi', 262144), ('command', 131072), ('phoenix', 65536),
                    ('merlin', 65536), ('qwen3-32b', 40960))
DEFAULT_CONTEXT_WINDOW = 32768   # only for an id matching none of the above


def default_context_window(model):
    """A best-effort token window for the meter's percentage, matched on the model id."""
    m = (model or '').lower()
    for key, win in _CONTEXT_WINDOWS:
        if key in m:
            return win
    return DEFAULT_CONTEXT_WINDOW


# Calibrated against the endpoints themselves (same request, one text appended once vs twice
# -> the prompt_tokens delta is that text's real price, per-message overhead cancels out):
#
#   chars/token   phoenix   kimi   command        per text kind (phoenix)
#   ------------------------------------          German prose      3.6
#   weighted        3.64     3.37     3.72        German user task  3.2
#                                                 English prompt    4.2
#   per message: ~5 tokens of role/delimiter      world-state block 3.6
#   overhead (measured on phoenix)                tool feedback JSON 3.1
#
# 3.6 sits in the middle of the three models; the old /4 undercounted by ~10%. NOT included
# (they are not messages): the tool schemas, ~780 tokens re-sent on every single turn.
CHARS_PER_TOKEN = 3.6
TOKENS_PER_MESSAGE = 5


def estimate_tokens(messages):
    """Token estimate for a conversation -- no API call, no tokenizer.

    Counts the text that gets re-sent (content + each tool call's name and arguments) at
    CHARS_PER_TOKEN, plus TOKENS_PER_MESSAGE of chat-template overhead per message. The
    overhead matters: a loop of 20 short tool messages costs ~100 tokens in delimiters alone.
    """
    chars = 0
    count = 0
    for m in messages or []:
        count += 1
        chars += len(str(m.get('content') or ''))
        for tc in m.get('tool_calls') or []:
            fn = tc.get('function') or {}
            chars += len(str(fn.get('name') or '')) + len(str(fn.get('arguments') or ''))
    return int(chars / CHARS_PER_TOKEN) + count * TOKENS_PER_MESSAGE


def _fmt_tok(n):
    return f'{n / 1000:.1f}k' if n >= 1000 else str(int(n))


def _ctx_bar(frac, width=10):
    frac = max(0.0, min(1.0, frac))
    filled = int(round(frac * width))
    return '█' * filled + '░' * (width - filled)


def context_gauge(tokens, window):
    """'5.4k/65.5k [█░░░░░░░░░] 8%' when the window is known, else '5.4k tok'."""
    if window:
        frac = tokens / window
        return f'{_fmt_tok(tokens)}/{_fmt_tok(window)} [{_ctx_bar(frac)}] {frac * 100:.0f}%'
    return f'{_fmt_tok(tokens)} tok'


class SpinnerLLM:
    """Wrap an LLMClient so the demo SHOWS reasoning happening (live spinner + elapsed
    seconds), prints the reasoning trace + the model's content after each call, and prints a
    per-turn CONTEXT METER (the call's real input tokens vs. the window) so you can watch the
    context grow INSIDE the loop -- a Claude-Code-style view. --hide-thoughts hides only the
    reasoning TRACE (💭); the model's CONTENT (📝) is always shown, so conversational replies
    stay visible. Everything except chat() is forwarded to the inner client (model, base_url,
    with_model, ...), so the orchestrator, run_episode and eval stay untouched.

    with_model() re-wraps for the cache SUMMARY side-call (a sibling client, pinned to the
    summarizer model): it keeps its spinner but SUPPRESSES its own content/reasoning prints
    and the context meter -- that side-call is not the main-loop context, and the demo shows
    the distilled note once via the cache-write hook instead.
    """

    def __init__(self, inner, show_thoughts=True, context_window=0, show_context=True,
                 show_content=True):
        self._inner = inner
        self._show_thoughts = show_thoughts   # the reasoning TRACE (💭); hidden by --hide-thoughts
        self._show_content = show_content     # the model's CONTENT (📝); always on for the driver so
                                              # replies stay visible even with --hide-thoughts
        self._show_context = show_context     # driver prints the per-turn meter; summarizer sibling does not
        self.context_window = context_window  # token window for the meter percentage (0 = unknown)
        self.last_prompt_tokens = 0           # input tokens of the last call -- the live context meter

    def __getattr__(self, name):
        return getattr(self._inner, name)   # model / base_url / temperature / on_effort / ...

    def with_model(self, model):
        # The cache SUMMARY side-call: keep the spinner, but do NOT auto-print its reasoning,
        # its content, or a context meter -- the demo shows the distilled result via the
        # cache-write hook instead, clearly labelled (so it is not printed twice).
        return SpinnerLLM(self._inner.with_model(model), show_thoughts=False,
                          context_window=self.context_window, show_context=False,
                          show_content=False)

    def chat(self, messages, tools=None, tool_choice=None, reasoning=None):
        with _Spinner('reasoning' if reasoning else 'thinking'):
            resp = self._inner.chat(messages, tools=tools, tool_choice=tool_choice,
                                    reasoning=reasoning)
        self.last_prompt_tokens = (resp.usage or {}).get('prompt_tokens') or 0
        if self._show_context and self.last_prompt_tokens:
            print(f'{_DIM}📊 context {context_gauge(self.last_prompt_tokens, self.context_window)}{_RESET}')
        if self._show_thoughts and reasoning and resp.trace:
            _print_block('💭', 'reasoning', resp.trace, dim=True)
        content = (resp.content or '').strip()
        if self._show_content and content:
            _print_block('📝', 'content', content)
        return resp

# The demonstrator's selectable modes: label -> EpisodeConfig kwargs. A curated subset of
# nesyplan/eval.py's CONFIGS_BY_LABEL, kept local so the demo stays import-light (no
# eval/judge/render deps). allow_store is turned on for ALL of them (see _config_for).
#
# `no_reason` (no_initial_reasoning) and the on_error policy give modes with reasoning-OFF
# turns. Those differ from their always-on cousins ONLY where reasoning_effort:"none" truly
# disables CoT -- phoenix (see the phoenix-reasoning-toggle finding); kimi always thinks and
# command ignores the param, so there on_error==react and *_nocot==with-cot. Such modes are
# flagged `reason_off` and offered ONLY for reasoning-optional models (see modes_for).
DEMO_MODES = {
    'oneshot':               dict(policy='oneshot',      cache='none'),
    'react':                 dict(policy='always',       cache='none'),
    'react_raw':             dict(policy='always',       cache='raw'),
    'react_summary':         dict(policy='always',       cache='summary'),
    # --- reasoning-OFF modes: offered only where reasoning can be disabled (qwen3-*, phoenix) ---
    'oneshot_nocot':         dict(policy='oneshot',      cache='none',    no_reason=True, reason_off=True),
    'reason_first_noreason': dict(policy='reason_first', cache='none',    no_reason=True, reason_off=True),
    'on_error':              dict(policy='on_error',     cache='none',    reason_off=True),
    'on_error_raw':          dict(policy='on_error',     cache='raw',     reason_off=True),
    'on_error_summary':      dict(policy='on_error',     cache='summary', reason_off=True),
}
MODE_ORDER = list(DEMO_MODES)   # stable display order for the picker
MODE_DESC = {
    'oneshot':               'plan everything up front, then execute blind (with reasoning)',
    'react':                 'reason every turn, one action at a time',
    'react_raw':             'ReAct + keep the full reasoning trace as memory',
    'react_summary':         'ReAct + keep a distilled summary as memory',
    'oneshot_nocot':         'plan up front and execute blind, NO reasoning at all',
    'reason_first_noreason': 'agentic loop, NO reasoning -- act each turn from feedback only',
    'on_error':              'reason first, then re-think only after an error',
    'on_error_raw':          'on-error + full reasoning trace as memory',
    'on_error_summary':      'on-error + distilled summary as memory',
}
DEFAULT_MODE = 'react_summary'   # universal (not reason_off, so available on every model)


# Model ids (substring match) whose chain-of-thought can genuinely be turned OFF, i.e. an
# effort of "none" leaves the reasoning trace empty -- VERIFIED per model with
# `python3 -m nesyplan.probe_reasoning`, never assumed: whether "none" is honoured is a
# property of the serving backend, not of the API.
#
# This gate is not cosmetic. If a model silently IGNORES "none", every reasoning-OFF mode
# becomes a duplicate of its reasoning-ON cousin, and a comparison between them measures
# nothing while looking like it measures something. So a model is listed here only after a
# probe showed a 0-char trace.
#   qwen3-*  -- verified 2026-08-31: 14b / 30b-a3b / 32b all drop to a 0-char trace at
#               effort "none" while still calling tools (default effort: ~400-470 chars).
#   phoenix  -- honours it.
#   merlin   -- honours it too, but excluded per request; add 'merlin' to include it.
#   kimi     -- always thinks; command -- ignores the parameter. Both stay out.
_REASONING_OPTIONAL = ('qwen3-', 'phoenix')


def reasoning_optional(model):
    """True if reasoning can be turned OFF on this model (an effort of "none" is honored).

    The reasoning-OFF demo modes are offered only for such models (see modes_for), because
    elsewhere they would silently behave like their always-on cousins.
    """
    m = (model or '').lower()
    return any(key in m for key in _REASONING_OPTIONAL)


def modes_for(model):
    """Mode labels offered for `model`: the reasoning-OFF modes appear only where reasoning
    can actually be disabled (reasoning_optional); everyone gets the always-on modes."""
    opt = reasoning_optional(model)
    return [label for label in MODE_ORDER if opt or not DEMO_MODES[label].get('reason_off')]


def _config_for(label, args):
    """Build a validated EpisodeConfig for a demo mode label (store enabled)."""
    cfg = DEMO_MODES[label]
    return EpisodeConfig(
        policy=Policy(cfg['policy']),
        cache_mode=CacheMode(cfg.get('cache', 'none')),
        reason_on_first=not cfg.get('no_reason', False),   # no_reason -> plan/act with CoT off
        allow_store=True,          # the demo always exposes store() (return a cube to storage)
        allow_chat=True,           # NeSyPlan may reply in prose (greet / answer / ask for a task)
                                   # without a tool call -- no nudge, no done() needed to leave the loop
        max_no_tool_calls=1,       # but MID-BUILD prose ends the task instead of nudging in circles;
                                   # raise to 2 to allow one nudge before giving up
        max_steps=args.max_steps,
    ).validate()


def pick_model(args):
    """Resolve the driver model: --model if given, else an interactive picker over the known
    aliases. Returns an alias (or the explicit --model string); LLMClient resolves it to the
    full id. Default = AGENT_MODEL if it names a known alias, else qwen3-32b."""
    if args.model:
        return args.model
    aliases = list(MODEL_ALIASES)
    env_alias = (os.environ.get('AGENT_MODEL') or '').strip().lower()
    default = env_alias if env_alias in MODEL_ALIASES else DEFAULT_ALIAS
    print('Select a model:')
    for i, alias in enumerate(aliases, 1):
        mark = ' (default)' if alias == default else ''
        print(f'  {i}. {alias:10s} {MODEL_ALIASES[alias]}{mark}')
    while True:
        try:
            choice = input(f'Model [1-{len(aliases)} or name, Enter = {default}]: ').strip()
        except (EOFError, KeyboardInterrupt):
            print()
            sys.exit(0)
        if not choice:
            return default
        if choice.lower() in MODEL_ALIASES:
            return choice.lower()
        if choice.isdigit() and 1 <= int(choice) <= len(aliases):
            return aliases[int(choice) - 1]
        print(f'  invalid; enter 1-{len(aliases)} or an alias name.')


def pick_mode(args, available):
    """Resolve the mode: --mode if given (must be available for the chosen model), else an
    interactive numbered picker over `available` -- the modes offered for this model."""
    if args.mode:
        if args.mode not in DEMO_MODES:
            print(f'error: unknown --mode {args.mode!r}; choices: {", ".join(MODE_ORDER)}',
                  file=sys.stderr)
            sys.exit(2)
        if args.mode not in available:
            print(f'error: --mode {args.mode!r} needs a model whose reasoning can be disabled '
                  f'({DEFAULT_ALIAS} and the other qwen3 aliases do); this model offers: '
                  f'{", ".join(available)}', file=sys.stderr)
            sys.exit(2)
        return args.mode
    print('Select a mode:')
    for i, label in enumerate(available, 1):
        mark = ' (default)' if label == DEFAULT_MODE else ''
        print(f'  {i}. {label:22s} {MODE_DESC[label]}{mark}')
    while True:
        try:
            choice = input(f'Mode [1-{len(available)} or name, Enter = {DEFAULT_MODE}]: ').strip()
        except (EOFError, KeyboardInterrupt):
            print()
            sys.exit(0)
        if not choice:
            return DEFAULT_MODE
        if choice in available:
            return choice
        if choice.isdigit() and 1 <= int(choice) <= len(available):
            return available[int(choice) - 1]
        print(f'  invalid; enter 1-{len(available)} or one of: {", ".join(available)}')


def deterministic_cleanup(robot):
    """Return every placed/held cube to storage WITHOUT the LLM.

    Stores the currently-held cube first, then repeatedly the highest-level cube in the
    building area -- the top of the tallest stack is never buried, so store() always
    succeeds and we converge to the all-in-storage layout. Bounded by the cube count so a
    persistent failure can never loop forever.
    """
    moved = 0
    for _ in range(len(cube_ids(robot.get_state())) + 1):
        state = robot.get_state()
        cubes = state.get('cubes') or {}
        held = state.get('held')
        movable = [c for c, info in cubes.items() if (info or {}).get('location') in ('area', 'held')]
        if not movable:
            break
        target = held if held else max(movable, key=lambda c: (cubes[c].get('level') or 0))
        fb = robot.send_command('store', cube_id=target)
        ok = bool(fb.get('ok'))
        print(f'  store {target} -> {"ok" if ok else "FAIL: " + str(fb.get("error"))}')
        if not ok:
            print('  cleanup: stopping (store failed).')
            return moved
        moved += 1
    print(f'Cleanup done: {moved} cube(s) returned to storage.'
          if moved else 'Cleanup: nothing to do (all cubes already in storage).')
    return moved


HELP = """Commands:
  <task>            give the agent a natural-language task (e.g. "stack red on blue in the center")
  /cleanup          return ALL cubes to storage (deterministic, no LLM) -- tidy up for the next demo
  /mode [name]      switch mode (no name = show the picker); keeps the world AND the chat (only the mode's mechanics reset)
  /state            print the current world state
  /help             show this help
  /quit             exit the demonstrator
The agent can also tidy up on request: try "put the red cube back" or "clear everything"."""


class Demo:
    """Holds the live session: robot, llm, log, current mode/config, and (for agentic
    modes) the persistent AgenticSession. One SessionLog spans the whole demo."""

    def __init__(self, args, robot, llm, log):
        self.args = args
        self.robot = robot
        self.llm = llm
        self.log = log
        self.metrics = Metrics()
        self.mode = None
        self.config = None
        self.session = None       # AgenticSession for agentic modes; None for oneshot
        self.messages = None      # the PERSISTENT agentic conversation -- survives mode switches
        self.oneshot_turn = 0     # running turn offset so one-shot log turns keep increasing
        # Single source: the SpinnerLLM already resolved the window (per-turn meter uses it too).
        self.context_window = getattr(llm, 'context_window', 0) or default_context_window(llm.model)
        self.available_modes = modes_for(llm.model)   # reasoning-off modes only on phoenix

    def set_mode(self, label):
        """Switch to mode `label`, KEEPING the world AND the running chat. Only the mode's
        mechanics reset (policy, cache, think budget); the conversation carries over, so after
        a switch you can still refer to earlier tasks. The persistent chat is agentic; one-shot
        has no conversation (it re-plans each task) so it neither uses nor discards it -- the
        chat is preserved for when you switch back to an agentic mode."""
        self.mode = label
        self.config = _config_for(label, self.args)
        state = self.robot.get_state()
        if self.config.policy == Policy.ONESHOT:
            self.session = None
        else:
            if self.messages is None:   # seed the persistent agentic chat once (first agentic mode)
                # All agentic demo modes share this system prompt (content_plan is off), so it
                # stays valid across switches; the world layout is delivered per task, not here.
                system = {'role': 'system', 'content': build_system_prompt(state, self.config)}
                self.messages = [system]
                self.log.add(system)
                self.log.track_conversation(self.messages)
            else:
                self.log.add({'role': 'system', 'content': f'[mode -> {label}]'})
            self.session = AgenticSession(self.config, self.robot, self.llm, self.log,
                                          self.metrics, cube_ids(state), grid_units(state),
                                          self.messages, compact_history=True,
                                          on_cache_write=self._show_cache_write)
        print(f'Mode: {self.mode}  ({MODE_DESC[self.mode]})')

    def _show_cache_write(self, mode, prev_content, new_content):
        """Show what the reasoning cache PERSISTS into the assistant content this turn -- the
        text that is actually carried forward, which the model's own content (printed above by
        SpinnerLLM) does not reveal: phoenix/command usually write nothing, and kimi's own note
        gets overwritten. RAW persists the reasoning trace verbatim (already shown above as the
        reasoning block, so just noted); SUMMARY persists a distilled note (shown in full)."""
        if self.args.hide_thoughts:
            return
        overwrite = " (overwrites the model's own content above)" if (prev_content or '').strip() else ''
        if mode == CacheMode.RAW:
            print(f'{_DIM}💾 persisted to content: the full reasoning trace above, verbatim{overwrite}{_RESET}')
        else:
            _print_block('💾', f'persisted to content — distilled summary{overwrite}', new_content)

    def run_task(self, task):
        print(f'\n--- {self.mode}: {task!r} ---')
        if self.config.policy == Policy.ONESHOT:
            state = self.robot.get_state()
            # The one-shot bakes the current layout into its system prompt (it plans blind,
            # so nothing goes stale), hence the user message is just the task -- no duplicate
            # "Current world state" block. build_system_prompt(state, ...) is per task, so a
            # continue-from-current run replans against the real layout.
            messages = [
                {'role': 'system', 'content': build_system_prompt(state, self.config)},
                {'role': 'user', 'content': task},
            ]
            for m in messages:
                self.log.add(m)
            result = _oneshot_core(self.config, self.robot, self.llm, self.log, self.metrics,
                                   messages, base_turn=self.oneshot_turn)
            self.oneshot_turn += result['steps'] + 1
        else:
            self.session.run_task(task)
        self.print_context_meter()
        self.print_token_counter()

    def context_chip(self):
        """Compact 'how full is the context right now' chip for the prompt line: the carried
        (post-compaction) context the NEXT task builds on, plus the running token total."""
        carried = estimate_tokens(self.messages)
        if self.context_window:
            ctx = f'ctx ~{_fmt_tok(carried)}/{_fmt_tok(self.context_window)} {carried / self.context_window * 100:.0f}%'
        else:
            ctx = f'ctx ~{_fmt_tok(carried)}'
        return f'{ctx} · {_fmt_tok(self.metrics.tokens["total"])} tok'

    def print_context_meter(self):
        """After a task: the REAL peak the model just hit (last call's input tokens) and the
        (estimated) context carried forward -- showing the compaction keeps the base small."""
        peak = getattr(self.llm, 'last_prompt_tokens', 0) or 0
        carried = estimate_tokens(self.messages)
        print(f'{_DIM}  context: peaked {context_gauge(peak, self.context_window)} '
              f'· carrying ~{_fmt_tok(carried)} forward{_RESET}')

    def print_token_counter(self):
        """Cumulative tokens for the WHOLE demo session -- driver calls AND the cache summary
        side-calls (the orchestrator folds those into the shared metrics). The endpoint counts
        reasoning INSIDE completion_tokens, so these API totals are exact and complete (no
        estimation): `total` = in + out, `out` already includes any reasoning tokens."""
        t = self.metrics.tokens
        parts = [f'{_fmt_tok(t["total"])} total', f'{_fmt_tok(t["prompt"])} in',
                 f'{_fmt_tok(t["completion"])} out']
        if t.get('cache_injection'):
            parts.append(f'incl. {_fmt_tok(t["cache_injection"])} summary')
        print(f'{_DIM}  tokens: ' + ' · '.join(parts) + _RESET)


def parse_args(argv):
    p = argparse.ArgumentParser(description='Interactive demonstrator over the tracked executor.')
    p.add_argument('--mode', default=None, choices=list(DEMO_MODES),
                   help='Skip the picker and start in this mode.')
    p.add_argument('--model', default=None,
                   help='LLM model: alias (qwen3-32b|qwen3-30b-a3b|qwen3-14b, or phoenix|kimi|command|merlin) or full id '
                        '(default: AGENT_MODEL env, else qwen3-32b).')
    p.add_argument('--backend', default=os.environ.get('DEMO_BACKEND', 'fake'),
                   choices=['fake', 'sim'],
                   help='Where the world lives. "fake" (default) runs it in-process -- no '
                        'simulator, no containers. "sim" drives an external command server on '
                        '--url (not part of this repository; see docs/ARCHITECTURE.md).')
    p.add_argument('--url', default=os.environ.get('AGENT_ROBOT_URL') or DEFAULT_URL,
                   help='Command server URL for --backend sim (default: %(default)s).')
    p.add_argument('--max-steps', type=int, default=25, help='Safety cap on agent turns per task.')
    p.add_argument('--context-window', type=int,
                   default=int(os.environ.get('AGENT_CONTEXT_WINDOW') or 0) or None,
                   help='Token window for the context-fullness meter percentage. '
                        'Omitted -> a per-model default (see default_context_window).')
    p.add_argument('--hide-thoughts', action='store_true',
                   help='Hide the reasoning TRACE blocks (💭); the model content (📝, incl. '
                        'conversational replies) and the context meter stay visible.')
    p.add_argument('--log-dir', default=os.environ.get('LLM_LOG_DIR') or os.path.join(REPO_ROOT, 'llm_logs'),
                   help='Where to write the demo session transcript (default: %(default)s).')
    return p.parse_args(argv)


def main(argv=None):
    load_dotenv(os.path.join(REPO_ROOT, '.env'))
    args = parse_args(argv if argv is not None else sys.argv[1:])

    if args.backend == 'fake':
        robot = FakeRobot()
        print('Backend: fake (world model in-process -- no simulator, no arm motion).')
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

    n = len(state.get('cubes') or {})
    print(f'Connected: {n} cubes, grid {grid_units(state)}x{grid_units(state)}, mode=tracked.')
    not_stored = [c for c, info in (state.get('cubes') or {}).items()
                  if (info or {}).get('location') in ('area', 'held')]
    if not_stored:
        print(f'Note: {len(not_stored)} cube(s) not in storage ({", ".join(sorted(not_stored))}). '
              'The demo starts best all-in-storage -- type /cleanup to tidy up.')

    inner = LLMClient(model=pick_model(args))
    window = args.context_window or default_context_window(inner.model)
    llm = SpinnerLLM(inner, show_thoughts=not args.hide_thoughts, context_window=window)
    print(f'LLM: model={llm.model}  base_url={llm.base_url}  context_window={window}')
    if not reasoning_optional(llm.model):
        print(f'Note: reasoning cannot be disabled on this model, so the reasoning-off modes '
              f'(oneshot_nocot, reason_first_noreason, on_error*) are not offered -- pick phoenix for those.')

    log = SessionLog(args.log_dir, 'demo', task='(interactive demo session)',
                     model=llm.model, base_url=llm.base_url)
    log.set_meta(backend=args.backend, kind='demo')
    if log.path:
        print(f'Session log: {log.path}')

    demo = Demo(args, robot, llm, log)
    demo.set_mode(pick_mode(args, demo.available_modes))
    print('\nType a task, or /help for commands.\n')

    while True:
        try:
            line = input(f'[{demo.mode} · {demo.context_chip()}] task> ').strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not line:
            continue

        if line.startswith('/'):
            parts = line[1:].split(None, 1)
            cmd = parts[0].lower() if parts else ''
            rest = parts[1].strip() if len(parts) > 1 else ''
            if cmd in ('quit', 'exit', 'q'):
                break
            elif cmd in ('help', 'h', '?'):
                print(HELP)
            elif cmd == 'state':
                print(format_state(robot.get_state()))
            elif cmd == 'cleanup':
                try:
                    deterministic_cleanup(robot)
                except ConnectionError as exc:
                    print(f'error: lost the command server: {exc}', file=sys.stderr)
                    break
            elif cmd == 'mode':
                if rest and rest not in demo.available_modes:
                    reason = ('unknown mode' if rest not in DEMO_MODES
                              else 'not available for this model (reasoning-off needs phoenix)')
                    print(f'/mode {rest!r}: {reason}; choices: {", ".join(demo.available_modes)}')
                else:
                    demo.set_mode(rest if rest in demo.available_modes
                                  else pick_mode(argparse.Namespace(mode=None), demo.available_modes))
            else:
                print(f'unknown command /{cmd}; try /help')
            continue

        # A natural-language task.
        try:
            demo.run_task(line)
        except KeyboardInterrupt:
            print('\n(task interrupted; back to prompt)')
        except ConnectionError as exc:
            print(f'error: lost the command server: {exc}', file=sys.stderr)
            break
        except Exception as exc:   # a bad task must not kill the whole demo
            print(f'error running task: {type(exc).__name__}: {exc}', file=sys.stderr)

    log.finish(status='demo_end')
    print('Demo ended.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
