#!/usr/bin/env python3
"""Run an eval CAMPAIGN: a matrix of {tasks} x {configs} against the world model.

Drives nesyplan.orchestrator.run_episode once per matrix cell, resets the world to the
identical initial state between cells (so runs are comparable), scores the final /state,
renders a PNG of the result, and streams one flat row per run into a campaign folder:

    results/<campaign>/
      manifest.json     the resolved matrix + model + git sha + timestamps
      runs/<config>/<session_id>.json   full transcripts (nesyplan.session_log)
      runs/<config>/<session_id>.png    picture of the final state (human/vision review)
      results.jsonl     one row per run: task, config, success, tokens, latency, ... , log_path
      summary.md        aggregated success matrix + per-config roll-up (nesyplan.aggregate)

Scoring is two-tier. A task that ships a CHECKER (nesyplan/checkers.py) is scored
symbolically against the final /state -- deterministic, and the primary number. Tasks
without one fall back to the LLM judge (nesyplan/judge.py), which is why an ad-hoc
`--task-prompt "<anything>"` still works with zero code. The judge runs on every task
regardless and its agreement with the checker is recorded, so the judge's reliability is
itself measurable. See docs/EXPERIMENT.md for why the checker carries the headline result.

Backends (--backend):
  - fake (default): run the world model in-process (nesyplan/fake_robot.py) -- no
    simulator, no containers, instant per-cell reset. This is the reproducible path and
    the one the shipped experiment uses.
  - sim: drive an external command server over --url. That executor is NOT part of this
    repository (see docs/ARCHITECTURE.md); the flag exists so this harness can be pointed
    at a real simulator or robot if you have one. It exposes no reset action, so the
    per-cell reset is performed with store() commands instead of a scene reload.

Usage (see scripts/run_experiment.sh for the friendly wrapper):
    python3 -m nesyplan.eval --campaign smoke
    python3 -m nesyplan.eval --campaign harness      # the shipped experiment
    python3 -m nesyplan.eval --tasks german_flag,pyramid --configs react,reason_first_summary
    python3 -m nesyplan.eval --task-prompt "build a smiley face"   # ad-hoc, judge-scored
    python3 -m nesyplan.eval --list          # print the matrix and exit
    python3 -m nesyplan.eval --dry-run       # show cells (incl. resume-skips), run nothing

Resumable: a cell already present in results.jsonl is skipped, so an interrupted
campaign continues where it stopped (pass --fresh to ignore prior results).
"""

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime

from nesyplan.aggregate import write_summary
from nesyplan.checkers import run_check
from nesyplan.config import CacheMode, EpisodeConfig, Policy
from nesyplan.demo import deterministic_cleanup
from nesyplan.environment import FEEDBACK_LEVELS
from nesyplan.fake_robot import FakeRobot
from nesyplan.judge import judge_episode
from nesyplan.llm import LLMClient, resolve_sampling
from nesyplan.model_aliases import resolve_model
from nesyplan.orchestrator import run_episode
from nesyplan.render import available as render_available
from nesyplan.render import render_state
from nesyplan.robot_client import DEFAULT_URL, RobotClient
from nesyplan.run import load_dotenv
from nesyplan.scenarios import get_scenario, materialize
from nesyplan.session_log import SessionLog
from nesyplan.tasks import TASK_GROUPS, Task, _area, _layout, get_tasks
from nesyplan.viewer import write_report

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# --- The lean pilot matrix (the "schlank" option). Each entry is one config; the
#     label is stable and used as the transcript subfolder + the results.jsonl key.
#     Spread is chosen to isolate the proposal's hypotheses on a small budget:
#       H1 (feedback):  oneshot_nocot vs reason_first_noreason vs reason_first
#       H3 (cache):     reason_first vs reason_first_summary; self_triggered {none,summary}
#       H2 (invocation): self_triggered vs react (ReAct upper bound)
DEFAULT_CONFIGS = [
    {'label': 'oneshot',               'policy': 'oneshot',        'cache': 'none'},
    {'label': 'oneshot_nocot',         'policy': 'oneshot',        'cache': 'none', 'no_initial_reasoning': True},
    {'label': 'reason_first_noreason', 'policy': 'reason_first',   'cache': 'none', 'no_initial_reasoning': True},
    {'label': 'react',                 'policy': 'always',         'cache': 'none'},
    {'label': 'reason_first',          'policy': 'reason_first',   'cache': 'none'},
    {'label': 'reason_first_summary',  'policy': 'reason_first',   'cache': 'summary'},
    {'label': 'self_triggered',        'policy': 'self_triggered', 'cache': 'none'},
    {'label': 'self_triggered_summary','policy': 'self_triggered', 'cache': 'summary'},
    {'label': 'on_error_summary',      'policy': 'on_error',       'cache': 'summary'},
]

# H6 (content-as-memory) -- selectable via --configs, NOT in the default run (keeps the
# pilot lean). content_plan = prompt the model to keep a GOAL/DONE/NEXT plan in content;
# strip_content = ablate content from re-sent history (mainly meaningful for models that
# write content natively, e.g. kimi). See docs/EXPERIMENT.md and docs/FINDINGS.md Finding D.
EXTRA_CONFIGS = [
    {'label': 'react_content',        'policy': 'always',       'cache': 'none', 'content_plan': True},
    {'label': 'react_strip',          'policy': 'always',       'cache': 'none', 'strip_content': True},
    {'label': 'reason_first_content', 'policy': 'reason_first', 'cache': 'none', 'content_plan': True},
    # ReAct with the reasoning-cache lever (persist the trace into content each turn).
    {'label': 'react_summary',        'policy': 'always',       'cache': 'summary'},
    {'label': 'react_raw',            'policy': 'always',       'cache': 'raw'},
    # on_error (reason after any failed tool result) with the cache lever
    # (on_error_summary is already in DEFAULT_CONFIGS).
    {'label': 'on_error',             'policy': 'on_error',     'cache': 'none'},
    {'label': 'on_error_raw',         'policy': 'on_error',     'cache': 'raw'},

    # --- the deliberation floor -----------------------------------------------------
    # ACT: agentic (reads feedback, one action per turn) but NEVER reasons -- reason_first
    # with the initial reasoning turned off, so should_reason() returns False on every turn.
    # Byte-identical to the older label `reason_first_noreason`, which stays for continuity
    # with the campaigns in results/; prefer `act` in new campaigns, it says what it is.
    {'label': 'act',                  'policy': 'reason_first', 'cache': 'none', 'no_initial_reasoning': True},

    # --- feedback informativeness (dimension C) -------------------------------------
    # Same policy, same cache, same world: the ONLY difference is how much of a rejection
    # the model is allowed to read (nesyplan/environment.py:FEEDBACK_LEVELS). The 'fix'
    # arms are just `act` / `react` / `on_error` above, so the design is
    # {act, react, on_error} x {terse, why, fix}.
    {'label': 'act_fb_terse',         'policy': 'reason_first', 'cache': 'none',
     'no_initial_reasoning': True, 'feedback': 'terse'},
    {'label': 'act_fb_why',           'policy': 'reason_first', 'cache': 'none',
     'no_initial_reasoning': True, 'feedback': 'why'},
    {'label': 'react_fb_terse',       'policy': 'always',       'cache': 'none', 'feedback': 'terse'},
    {'label': 'react_fb_why',         'policy': 'always',       'cache': 'none', 'feedback': 'why'},
    # react_summary completes the reasoning ladder for the feedback dimension: no reasoning
    # (act) -> reasoning (react) -> reasoning + a persisted plan (react_summary). Crossing
    # all three with terse|why|fix turns "informative feedback helps" (a main effect nobody
    # doubts) into "how much explanation a model needs depends on how much it may think"
    # (an interaction). The predicted terse->fix gradient is steepest for act, flattest here.
    {'label': 'react_summary_fb_terse', 'policy': 'always',     'cache': 'summary', 'feedback': 'terse'},
    {'label': 'react_summary_fb_why',   'policy': 'always',     'cache': 'summary', 'feedback': 'why'},
    # on_error x feedback is deliberately NOT in the default cross: its trigger fires on the
    # ok=False flag (identical in all three arms) while the value comes from the text, so the
    # two are entangled in a way that needs its own experiment.
    {'label': 'on_error_fb_terse',    'policy': 'on_error',     'cache': 'none', 'feedback': 'terse'},
    {'label': 'on_error_fb_why',      'policy': 'on_error',     'cache': 'none', 'feedback': 'why'},
]
CONFIGS_BY_LABEL = {c['label']: c for c in DEFAULT_CONFIGS + EXTRA_CONFIGS}

# Named config sets -- one flag instead of a list to retype. Each is a full arm of the
# design, so a campaign is `--configs @modes` or `--configs @feedback`.
#   @modes     WHEN to deliberate + HOW the plan persists: floor (act) -> blind plan with
#              and without deliberation (oneshot / oneshot_nocot) -> ReAct -> ReAct with the
#              plan written back into content (summary/raw) -> reason only when the world
#              objects (on_error). oneshot_nocot is the absolute floor of the whole design:
#              one call, no thinking, no feedback ever.
#   @feedback  WHAT the symbolic layer says back, crossed with the reasoning ladder:
#              {act, react, react_summary} x {terse, why, fix}. The fix column IS
#              act/react/react_summary from @modes, so combining the two groups costs only
#              six extra cells (resolve_configs dedupes the overlap).
#   @lean      five cells for a first pass or a new model: floor, blind, ceiling, cheap-good.
CONFIG_GROUPS = {
    # @harness -- the shipped experiment (docs/EXPERIMENT.md). A 2x2 over the two things
    # that actually distinguish an agent from a planner, plus two repairs for the small-model
    # failure mode:
    #
    #                      no thinking        thinking
    #     plan once        oneshot_nocot      oneshot         <- never reads feedback
    #     act turn by turn act                react           <- reads feedback every turn
    #
    #     react_raw        + the VERBATIM trace carried in `content` between turns
    #     react_summary    + a distilled note instead of the verbatim trace
    #     on_error_summary + that note, but thinking ONLY after a rejection (the cheap arm)
    #
    # The last three exist because a reasoning trace is not returned to the next request:
    # unless the model writes its conclusion into `content` itself -- which ~30B models
    # routinely do not (measured at 0% for qwen3-32b, see docs/EXPERIMENT.md) -- it is lost.
    # RAW and SUMMARY supply that memory externally, and differ only in whether the trace is
    # compressed first. Keeping BOTH in the shipped set is the point: it is what separates
    # "the memory mechanism does not work" from "the compression loses something".
    '@harness': ['oneshot_nocot', 'oneshot', 'act', 'react', 'react_raw', 'react_summary',
                 'on_error_summary'],
    '@modes': ['act', 'oneshot', 'oneshot_nocot', 'react', 'react_summary', 'react_raw',
               'on_error', 'on_error_summary'],
    '@feedback': ['act_fb_terse', 'act_fb_why', 'act',
                  'react_fb_terse', 'react_fb_why', 'react',
                  'react_summary_fb_terse', 'react_summary_fb_why', 'react_summary'],
    '@lean': ['act', 'oneshot', 'react', 'react_summary', 'on_error_summary'],
}

# The judge is ALWAYS the same model, independent of the driver, so a judged verdict is
# scored consistently across every run. It defaults to a PUBLICLY reachable model so the
# repository works with nothing but an OpenRouter key.
#
# Caveat, stated plainly: this judge is no stronger than the models it scores, and an LLM
# judge on this domain is documented to be unreliable (docs/FINDINGS.md). That is exactly
# why the shipped experiment runs on @rebuild, where every task has a symbolic CHECKER and
# the judge is only recorded alongside it for agreement. Do not build a claim on a
# judge-only number. Override with --judge-model.
DEFAULT_JUDGE_MODEL = 'qwen3-32b'


def make_config(cfg, args):
    """Turn a matrix-cell dict + global run args into a validated EpisodeConfig."""
    return EpisodeConfig(
        policy=Policy(cfg['policy']),
        cache_mode=CacheMode(cfg.get('cache', 'none')),
        periodic_n=cfg.get('n'),
        context_k=cfg.get('context_k'),
        on_failure_trigger=cfg.get('on_failure', False),
        pre_done_check=cfg.get('pre_done', False),
        escalation_budget=cfg.get('escalation_budget'),
        strip_content=cfg.get('strip_content', False),
        content_plan=cfg.get('content_plan', False),
        # Per-cell 'feedback' wins; otherwise the campaign-wide --feedback-level applies,
        # so a whole campaign can be re-run at a different informativeness in one flag.
        feedback_level=cfg.get('feedback') or args.feedback_level,
        reason_on_first=not cfg.get('no_initial_reasoning', False),
        reasoning_on_effort=cfg.get('on_effort'),   # None -> omit -> model's own default level

        max_steps=args.max_steps,
        temperature=args.temperature,
        top_p=args.top_p,
        seed=args.seed,
    ).validate()


def git_sha():
    try:
        return subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=REPO_ROOT,
                                       stderr=subprocess.DEVNULL).decode().strip()
    except (subprocess.SubprocessError, OSError):
        return None


def wait_health(url, timeout=180, interval=3):
    robot = RobotClient(url)
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            h = robot.health()
            if h.get('ok') or h.get('status') in ('ok', 'ready', 'healthy'):
                return True
        except Exception:
            pass
        time.sleep(interval)
    raise RuntimeError(f'command server never became healthy on {url} within {timeout}s')


def make_robot(args):
    """Build the robot backend: a FakeRobot (in-process, no sim) or a RobotClient (:8100)."""
    if args.backend == 'fake':
        return FakeRobot()
    return RobotClient(args.url)


def reset_env(args, robot, scenario=None):
    """Reset the world to this cell's initial state, then build the inherited layout.

    `scenario` (nesyplan/scenarios.py) is the inherited starting layout a rebuild-tier task
    needs. The fake backend takes it directly (a pure-Python state assignment); a command
    server owns its own world and has no reset action, so there the layout is produced with
    real commands: store() every cube back to storage, then pick/place the scenario.
    """
    if args.no_reset:
        return
    if args.backend == 'fake':
        # pure-Python state reset -- instant, nothing to boot
        robot.reset(scenario if scenario and scenario.placed else None)
        return
    # --backend sim: the executor exposes pick/place/store/done and nothing else, so the
    # only way back to the all-in-storage start state is to store the cubes one by one.
    # deterministic_cleanup() does exactly that (no LLM, bounded by the cube count).
    print('  [reset] returning all cubes to storage via store() ...')
    deterministic_cleanup(robot)
    if scenario and scenario.placed:
        print(f'  [reset] building start layout {scenario.id!r} with the arm '
              f'({len(scenario.placed)} cubes) ...')
        materialize(robot, scenario)


def load_done(results_path):
    """Set of (model, task_id, config_label, rep) cells already recorded (for --resume)."""
    done = set()
    if not os.path.isfile(results_path):
        return done
    with open(results_path, encoding='utf-8') as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except ValueError:
                continue
            done.add((r.get('model'), r.get('task'), r.get('config'), r.get('rep', 0)))
    return done


def run_cell(model, task, cfg, rep, args, campaign, campaign_dir, robot):
    """Reset, run one episode, check the final state, return the flat results row."""
    scenario = get_scenario(args.scenario if args.scenario is not None else task.scenario)
    reset_env(args, robot, scenario)

    config = make_config(cfg, args)
    robot.get_state()  # sanity: raises ConnectionError early if the server vanished

    llm = LLMClient(model=model, temperature=args.temperature, top_p=args.top_p,
                    on_effort=cfg.get('on_effort'))   # None -> model's own default effort
    # Record the sampling actually used (per-model recommended, unless overridden).
    config.temperature, config.top_p = llm.temperature, llm.top_p
    log = SessionLog(os.path.join(campaign_dir, 'runs'), cfg['label'], task=task.prompt,
                     model=llm.model, base_url=llm.base_url, config=config.to_dict(),
                     seed=config.seed)
    log.set_meta(campaign=campaign, task_id=task.id, difficulty=task.difficulty,
                 config_label=cfg['label'], rep=rep)

    exit_code, err = None, None
    try:
        exit_code = run_episode(task.prompt, config, robot, llm, log)
    except BaseException as exc:   # one bad cell must not abort the campaign
        err = repr(exc)
        print(f'  [cell error] {err}')

    final = robot.get_state()

    # Artifact: a PNG of the final state next to the transcript, for human/vision review.
    image_path = None
    if log.path and not args.no_image:
        image_path = render_state(final, os.path.splitext(log.path)[0] + '.png', task=task.prompt)

    # Scoring: the LLM judge, always -- one scorer for every task (nesyplan/judge.py).
    # Kept deterministic (temp=0) for stable, reproducible verdicts. NOTE: on a Qwen3
    # thinking model (phoenix) this means judge-reasoning-ON runs at greedy decoding --
    # slow and prone to runaway traces (Finding C2); prefer --no-judge-reasoning there
    # (reasoning-off + temp=0 is clean and deterministic).
    judge_llm = LLMClient(model=args.judge_model or DEFAULT_JUDGE_MODEL, temperature=0.0,
                          on_effort=args.judge_effort)
    judge = judge_episode(task.prompt, final, judge_llm,
                          reasoning=not args.no_judge_reasoning, log=log)

    # PRIMARY scorer: the task's symbolic checker when it has one (nesyplan/checkers.py) --
    # deterministic, over exact ground-truth state. The judge still runs on every task and
    # its verdict is recorded either way, so (a) judge-only tasks stay scoreable and (b) the
    # two can be compared: `scorer_agree` is the per-run input to "how much can this judge
    # be trusted", which is a reportable result rather than an assumption.
    check = run_check(task, final)
    if check and check['success'] is not None:
        success, detail, scorer = check['success'], check['detail'], 'checker'
    else:
        success, detail, scorer = judge['success'], judge.get('reasoning'), 'judge'
    agree = (None if not check or check['success'] is None or judge['success'] is None
             else check['success'] == judge['success'])

    conf = f' (conf {judge["confidence"]})' if judge.get('confidence') is not None else ''
    verdict = 'PASS' if success else 'fail' if success is False else '?'
    print(f'  -> [{scorer}] {verdict}: {detail}')
    if agree is False:
        print(f'  -> !! judge DISAGREES{conf}: {judge.get("reasoning")}')

    metrics = dict(log.data.get('metrics') or {})
    metrics['task_success'] = success
    log.set_metrics(metrics)
    log.set_meta(evaluation={
        'success': success, 'detail': detail, 'scorer': scorer,
        'check': check, 'scorer_agree': agree,
        'judge': {k: judge[k] for k in ('success', 'confidence', 'reasoning')},
        'final_layout': _layout(_area(final)), 'final_cubes': final.get('cubes'),
        'image': os.path.relpath(image_path, campaign_dir) if image_path else None,
    })

    tokens = metrics.get('tokens') or {}
    judge_tokens = judge.get('tokens') or {}
    return {
        'campaign': campaign, 'task': task.id, 'difficulty': task.difficulty,
        'scenario': scenario.id,
        'config': cfg['label'], 'policy': cfg['policy'], 'cache': cfg.get('cache', 'none'),
        'feedback_level': config.feedback_level,
        'no_initial_reasoning': cfg.get('no_initial_reasoning', False),
        'rep': rep, 'seed': args.seed, 'temperature': llm.temperature, 'top_p': llm.top_p,
        'model': llm.model,
        'success': success, 'check_detail': detail,
        'scorer': scorer, 'scorer_agree': agree,
        'check_success': (check or {}).get('success'),
        'check_spec': (check or {}).get('spec'),
        'judge_success': judge['success'],
        'judge_confidence': judge['confidence'],
        'judge_reasoning': judge['reasoning'],
        'judge_tokens_total': judge_tokens.get('total'),
        'image_path': os.path.relpath(image_path, campaign_dir) if image_path else None,
        'outcome': metrics.get('outcome'), 'steps': metrics.get('steps'),
        'reasoning_turns_total': metrics.get('reasoning_turns_total'),
        'reasoning_turns': metrics.get('reasoning_turns'),
        'tool_failures': metrics.get('tool_failures'),
        'think_calls': metrics.get('think_calls'),
        'get_state_calls': metrics.get('get_state_calls'),
        'tokens_total': tokens.get('total'), 'tokens_reasoning': tokens.get('reasoning'),
        'tokens_reasoning_est': tokens.get('reasoning_est'),
        'reasoning_chars': metrics.get('reasoning_chars'),
        'reasoning_traces': metrics.get('reasoning_traces'),
        'tokens_completion': tokens.get('completion'), 'tokens_prompt': tokens.get('prompt'),
        'tokens_cache_injection': tokens.get('cache_injection'),
        'latency_ms_total': metrics.get('latency_ms_total'),
        'exit_code': exit_code, 'error': err, 'session_id': log.session_id,
        'log_path': os.path.relpath(log.path, campaign_dir) if log.path else None,
        'started_at': log.data.get('started_at'), 'finished_at': log.data.get('finished_at'),
    }


def parse_args(argv):
    p = argparse.ArgumentParser(description='Run an eval campaign (tasks x configs matrix).')
    p.add_argument('--campaign', default=None,
                   help='Campaign name -> results/<campaign>/ (default: <timestamp>_pilot).')
    p.add_argument('--tasks', default=None,
                   help='Comma-separated task ids or @groups (default: all, in ladder order). '
                        'Groups: ' + ', '.join(TASK_GROUPS) + '. The rebuild tier '
                        '(@rebuild: flag_excavate,unbox_red,flag_repair,pyramid_rebuild) starts '
                        'from an inherited layout and is scored by code, not the judge.')
    p.add_argument('--configs', default=None,
                   help='Comma-separated config labels or @groups (default: the lean pilot set). '
                        'Groups: ' + ', '.join(CONFIG_GROUPS) + '.')
    p.add_argument('--feedback-level', default='fix', choices=list(FEEDBACK_LEVELS),
                   dest='feedback_level',
                   help='How much of a rejected action the model gets to read: "terse" (only '
                        'that it failed), "why" (+ the violated precondition), "fix" (+ what to '
                        'do about it = full message, default). Campaign-wide default; the '
                        '*_fb_* configs pin their own level per cell.')
    p.add_argument('--scenario', default=None,
                   help='Force an initial layout for EVERY cell (nesyplan/scenarios.py), '
                        'overriding each task\'s own. Use "empty" for the legacy all-in-storage '
                        'start. Omit to let each task bring its own.')
    p.add_argument('--task-prompt', default=None,
                   help='Run a single AD-HOC task with this natural-language prompt. '
                        'Overrides --tasks.')
    p.add_argument('--judge-model', default=None,
                   help=f'Model for the judge -- alias or full id. Default: {DEFAULT_JUDGE_MODEL} '
                        '(publicly reachable over OpenRouter), pinned independently of --model so '
                        'scoring stays consistent when driver models are compared.')
    p.add_argument('--judge-effort', default=None,
                   help='reasoning_effort for the judge. Omit (default) to let the judge model use its '
                        'own default level; pass low/medium/high to pin one. (Also see --no-judge-reasoning.)')
    p.add_argument('--no-judge-reasoning', action='store_true', help='Run the judge reasoning-off.')
    p.add_argument('--no-image', action='store_true', help='Do not render the per-run PNG artifact.')
    p.add_argument('--reps', type=int, default=1,
                   help='Repetitions per cell, averaged for variance. The default sampling is '
                        'non-zero (temp 0.6), so reps>1 is meaningful and recommended for a real eval.')
    p.add_argument('--backend', default=os.environ.get('EVAL_BACKEND', 'fake'),
                   choices=['fake', 'sim'],
                   help='Robot backend. "fake" (default) runs the world model in-process -- no '
                        'simulator, no containers, instant reset. "sim" talks to an external '
                        'command server on --url (see docs/ARCHITECTURE.md); that server is not '
                        'part of this repository.')
    p.add_argument('--url', default=os.environ.get('AGENT_ROBOT_URL') or DEFAULT_URL,
                   help='Command server base URL for --backend sim (default: %(default)s).')
    p.add_argument('--model', default=None,
                   help='LLM model: a short alias (qwen3-32b|qwen3-30b-a3b|qwen3-14b, or phoenix|kimi|command|merlin) or a full '
                        'model id (default: AGENT_MODEL env, else qwen3-32b). Comma-separate '
                        'several (e.g. phoenix,command,kimi) to span models in ONE campaign, '
                        'run sequentially. See model_aliases.py.')
    p.add_argument('--max-steps', type=int, default=25)
    p.add_argument('--temperature', type=float, default=None,
                   help='Sampling temperature. Omit to use the shared default (0.0 = greedy) applied '
                        'to EVERY model for comparability. See nesyplan/llm.py:DEFAULT_TEMPERATURE.')
    p.add_argument('--top-p', type=float, default=None, dest='top_p',
                   help='Nucleus sampling top_p. Omit to use the shared default (none) for every model.')
    p.add_argument('--seed', type=int, default=None)
    p.add_argument('--no-reset', action='store_true',
                   help='Do NOT reset between cells (fast pipeline test; runs are NOT comparable).')
    p.add_argument('--fresh', action='store_true',
                   help='Ignore any existing results.jsonl and re-run every cell.')
    p.add_argument('--list', action='store_true', help='Print the resolved matrix and exit.')
    p.add_argument('--dry-run', action='store_true',
                   help='Show which cells would run (and which are resume-skipped); run nothing.')
    return p.parse_args(argv)


def resolve_configs(spec):
    """Resolve a --configs spec: labels, or '@group' names (CONFIG_GROUPS), or a mix."""
    if not spec:
        return list(DEFAULT_CONFIGS)
    labels = []
    for s in (s.strip() for s in spec.split(',')):
        if not s:
            continue
        for label in (CONFIG_GROUPS[s] if s in CONFIG_GROUPS else [s]):
            if label not in labels:   # groups overlap (act/react are in both) -> dedupe,
                labels.append(label)  # or the same cell would be run twice in one campaign
    missing = [l for l in labels if l not in CONFIGS_BY_LABEL]
    if missing:
        raise KeyError(f'unknown config label(s): {missing}. Known: {list(CONFIGS_BY_LABEL)} '
                       f'or a group: {list(CONFIG_GROUPS)}')
    return [CONFIGS_BY_LABEL[l] for l in labels]


def main(argv=None):
    load_dotenv(os.path.join(REPO_ROOT, '.env'))
    args = parse_args(argv if argv is not None else sys.argv[1:])

    try:
        if args.task_prompt:
            tasks = [Task('adhoc', '--', args.task_prompt)]
        else:
            tasks = get_tasks([s.strip() for s in args.tasks.split(',')] if args.tasks else None)
        configs = resolve_configs(args.configs)
    except KeyError as exc:
        print(f'error: {exc}', file=sys.stderr)
        return 2

    # --model accepts a comma-separated list -> one campaign spanning several models
    # (run sequentially, models outermost). Each row records its own model.
    models = [m.strip() for m in args.model.split(',')] if args.model else [None]
    cells = [(m, t, c, rep) for m in models for t in tasks for c in configs
             for rep in range(args.reps)]

    if args.list:
        print(f'Models  ({len(models)}): ' + ', '.join(resolve_model(m) for m in models))
        print(f'Tasks   ({len(tasks)}): ' + ', '.join(
            f'{t.id}[{t.difficulty}]'
            + (f'@{args.scenario if args.scenario is not None else t.scenario}'
               if (args.scenario or t.scenario) else '')
            + ('*' if callable(t.check) else '')
            for t in tasks) + '   (@ = inherited layout, * = code-scored)')
        print(f'Configs ({len(configs)}): ' + ', '.join(
            c['label'] + (f'[fb:{c["feedback"]}]' if c.get('feedback') else '') for c in configs))
        print(f'Feedback level (default for cells without their own): {args.feedback_level}')
        print(f'Cells: {len(cells)}  (= {len(models)} models x {len(tasks)} tasks '
              f'x {len(configs)} configs x {args.reps} reps)')
        # Only the ones NOT already in the listed matrix -- otherwise a config that the
        # selected set does run (react_raw under @harness, say) is printed as if it did not.
        selected = {c['label'] for c in configs}
        rest = [c['label'] for c in EXTRA_CONFIGS if c['label'] not in selected]
        if rest:
            print('Also selectable via --configs (not in the set above): ' + ', '.join(rest))
        return 0

    campaign = args.campaign or (datetime.now().strftime('%Y-%m-%d_%H-%M') + '_pilot')
    campaign_dir = os.path.join(REPO_ROOT, 'results', campaign)
    results_path = os.path.join(campaign_dir, 'results.jsonl')

    done = set() if args.fresh else load_done(results_path)
    pending = [(m, t, c, r) for (m, t, c, r) in cells
               if (resolve_model(m), t.id, c['label'], r) not in done]

    print(f'Campaign: {campaign}  ({campaign_dir})')
    print(f'Backend: {args.backend}' + ('  (in-process, no simulator)' if args.backend == 'fake'
                                         else f'  (tracked command server @ {args.url})'))
    print(f'Matrix: {len(models)} models x {len(tasks)} tasks x {len(configs)} configs '
          f'x {args.reps} reps = {len(cells)} cells')
    coded = sum(1 for t in tasks if callable(t.check))
    print(f'Scorer: symbolic checker for {coded}/{len(tasks)} tasks, LLM judge for the rest '
          f'(judge runs on all, agreement recorded)'
          + ('  (+PNG per run)' if not args.no_image else '  (no PNG)'))
    # The 'why' arm needs the executor to split diagnosis from remedy, which the in-process
    # world model does; an external executor may send one flat sentence, so 'why' would
    # silently be identical to 'fix' there. Say so rather than publish an arm that is not one.
    levels = {c.get('feedback') or args.feedback_level for c in configs}
    if args.backend != 'fake' and 'why' in levels:
        print('WARNING: feedback level "why" is only guaranteed on --backend fake; an external '
              'executor that sends one un-split message degrades "why" to "fix".')
    if done:
        print(f'Resume: {len(cells) - len(pending)} already done, {len(pending)} pending.')
    if args.dry_run:   # inspect the plan without writing anything to disk
        for m, t, c, r in pending:
            print(f'  would run: {resolve_model(m):40s} {t.id:14s} x {c["label"]}'
                  + (f' rep{r}' if args.reps > 1 else ''))
        print(f'({len(pending)} cells would run)')
        return 0

    os.makedirs(os.path.join(campaign_dir, 'runs'), exist_ok=True)
    if not pending:
        print('Nothing to do (all cells present). Use --fresh to re-run.')
        write_summary(campaign_dir)
        return 0

    if not args.no_image and not render_available():
        print('note: matplotlib not importable -- per-run PNGs will be skipped. '
              'Install it (pip install --user matplotlib) or pass --no-image to silence this.')

    # Manifest: make the campaign self-describing before any run happens.
    eff_temp, eff_top_p = resolve_sampling(args.temperature, args.top_p)
    manifest = {
        'campaign': campaign, 'created_at': datetime.now().isoformat(timespec='seconds'),
        'git_sha': git_sha(), 'model': args.model,
        'models': [resolve_model(m) for m in models], 'backend': args.backend,
        'url': args.url,
        'scorer': 'judge', 'judge_model': args.judge_model or DEFAULT_JUDGE_MODEL,
        'judge_effort': args.judge_effort, 'judge_reasoning': not args.no_judge_reasoning,
        'render_png': not args.no_image,
        'temperature': eff_temp, 'top_p': eff_top_p, 'max_steps': args.max_steps, 'seed': args.seed,
        'reps': args.reps, 'no_reset': args.no_reset,
        'feedback_level': args.feedback_level, 'scenario_override': args.scenario,
        'tasks': [{'id': t.id, 'difficulty': t.difficulty, 'prompt': t.prompt,
                   'scenario': (args.scenario if args.scenario is not None else t.scenario) or 'empty',
                   'check': getattr(t.check, 'spec', None)} for t in tasks],
        'configs': configs,
    }
    with open(os.path.join(campaign_dir, 'manifest.json'), 'w', encoding='utf-8') as fh:
        json.dump(manifest, fh, indent=2, ensure_ascii=False)

    robot = make_robot(args)

    if args.no_reset:
        print('WARNING: --no-reset is set; cells are NOT comparable (shared drifting state).')
    elif args.backend == 'sim':
        # Fail fast with a clear message if the executor is not up yet (fake needs nothing).
        try:
            wait_health(args.url, timeout=15)
        except RuntimeError:
            print(f'error: no command server reachable on {args.url}.\n'
                  f'  --backend sim expects an external executor (see docs/ARCHITECTURE.md);\n'
                  f'  to run without one, drop the flag: --backend fake', file=sys.stderr)
            return 2

    for i, (model, task, cfg, rep) in enumerate(pending, start=1):
        tag = f'{resolve_model(model)} · {task.id} x {cfg["label"]}' + (f' rep{rep}' if args.reps > 1 else '')
        print(f'\n=== [{i}/{len(pending)}] {tag}  (task {task.difficulty}) ===')
        try:
            row = run_cell(model, task, cfg, rep, args, campaign, campaign_dir, robot)
        except ConnectionError as exc:
            print(f'error: lost the command server: {exc}', file=sys.stderr)
            return 2
        with open(results_path, 'a', encoding='utf-8') as fh:
            fh.write(json.dumps(row, ensure_ascii=False, default=str) + '\n')
        write_summary(campaign_dir)   # keep summary.md live as the campaign progresses

    print(f'\nCampaign complete. Summary: {os.path.join(campaign_dir, "summary.md")}')
    print(f'Index: {results_path}')
    try:   # best-effort: a browsable report; never let it fail the campaign
        report = write_report(campaign_dir)
        print(f'Report: {report}  (open in a browser)')
    except Exception as exc:  # noqa: BLE001
        print(f'note: could not write report.html ({exc!r}); '
              f'run `python3 -m nesyplan.viewer {campaign_dir}` to retry.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
