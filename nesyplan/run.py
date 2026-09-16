#!/usr/bin/env python3
"""Run ONE orchestration episode: python3 -m nesyplan.run "<task>" [flags].

Wires the harness together -- loads the repo-root .env, builds the RobotClient (to the
tracked executor on :8100), the LLMClient (Phoenix), and the unified SessionLog, then
calls run_episode. Every proposal mode is expressed via flags:

  --policy   oneshot | always | reason_first | periodic | self_triggered
  --cache    none | raw | summary
  --n N                  reasoning period for --policy periodic
  --context-k K          keep only the last K assistant turns in context (omit = full)
  --on-failure           hard trigger: force reasoning after any failed tool result
  --pre-done             reject the first reasoning-off done(); the retry reasons
  --escalation-budget B  cap on granted think() calls (self_triggered)
  --on-effort E          effort on reasoning-ON turns; omit (default) = model's own level

Examples:
  python3 -m nesyplan.run "build the german flag as a stack" --policy always --cache none
  python3 -m nesyplan.run "stack red, green, blue in the center" \
      --policy self_triggered --cache summary --on-failure --pre-done
"""

import argparse
import os
import sys

from nesyplan.config import CacheMode, EpisodeConfig, Policy
from nesyplan.envfile import load_dotenv
from nesyplan.fake_robot import FakeRobot
from nesyplan.llm import LLMClient
from nesyplan.orchestrator import run_episode
from nesyplan.robot_client import DEFAULT_URL, RobotClient
from nesyplan.session_log import SessionLog


# Re-exported so the existing `from nesyplan.run import load_dotenv` keeps working; the
# implementation lives in the leaf module so probe_reasoning can use it without importing
# the agent stack.
__all__ = ['load_dotenv', 'main']


def build_config(args):
    return EpisodeConfig(
        policy=Policy(args.policy),
        cache_mode=CacheMode(args.cache),
        periodic_n=args.n,
        context_k=args.context_k,
        on_failure_trigger=args.on_failure,
        pre_done_check=args.pre_done,
        escalation_budget=args.escalation_budget,
        strip_content=args.strip_content,
        content_plan=args.content_plan,
        reason_on_first=not args.no_initial_reasoning,
        reasoning_on_effort=args.on_effort,
        max_steps=args.max_steps,
        temperature=args.temperature,
        top_p=args.top_p,
        seed=args.seed,
    ).validate()


def main(argv=None):
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    load_dotenv(os.path.join(repo_root, '.env'))

    p = argparse.ArgumentParser(description='Run one reasoning-orchestration episode.')
    p.add_argument('task', help='Natural-language task for the robot.')
    p.add_argument('--policy', default='always',
                   choices=[m.value for m in Policy], help='Invocation policy (default: %(default)s).')
    p.add_argument('--cache', default='none',
                   choices=[c.value for c in CacheMode], help='Reasoning cache mode (default: %(default)s).')
    p.add_argument('--n', type=int, default=None, help='Reasoning period for --policy periodic.')
    p.add_argument('--context-k', type=int, default=None,
                   help='Keep only the last K assistant turns in context (omit = full history).')
    p.add_argument('--strip-content', action='store_true',
                   help='Drop assistant content from the re-sent history (content-as-memory ablation: '
                        'removes e.g. kimi\'s native plan-in-content).')
    p.add_argument('--content-plan', action='store_true',
                   help='Prompt the model to keep a running GOAL/DONE/NEXT plan in message content '
                        'each turn (persistence via content).')
    p.add_argument('--on-failure', action='store_true',
                   help='Hard trigger: force reasoning after any failed tool result.')
    p.add_argument('--pre-done', action='store_true',
                   help='Reject the first reasoning-off done(); the retry runs reasoning-on.')
    p.add_argument('--escalation-budget', type=int, default=None,
                   help='Cap on granted think() calls (self_triggered).')
    p.add_argument('--no-initial-reasoning', action='store_true',
                   help='Do not force reasoning on turn 1 (or the single oneshot call). Yields the '
                        'no-reasoning baselines: oneshot=plan without CoT, reason_first=never reasons, '
                        'self_triggered=think available from turn 1, model decides.')
    p.add_argument('--on-effort', default=None,
                   help='reasoning_effort on reasoning-ON turns. Omit (default) to send NO effort '
                        'so the model thinks at its own default level; pass low/medium/high to pin one.')
    p.add_argument('--backend', default=os.environ.get('EVAL_BACKEND', 'fake'),
                   choices=['fake', 'sim'],
                   help='Robot backend. "fake" (default) runs the world model in-process -- no '
                        'simulator, no containers. "sim" drives an external command server on '
                        '--url (not part of this repository; see docs/ARCHITECTURE.md). '
                        '(default: %(default)s)')
    p.add_argument('--url', default=os.environ.get('AGENT_ROBOT_URL') or DEFAULT_URL,
                   help='Command server URL for --backend sim (default: %(default)s).')
    p.add_argument('--model', default=None,
                   help='LLM model: a short alias (qwen3-32b|qwen3-30b-a3b|qwen3-14b, or phoenix|kimi|command|merlin) or a full '
                        'model id (default: AGENT_MODEL env, else qwen3-32b). See model_aliases.py.')
    p.add_argument('--max-steps', type=int, default=25, help='Safety cap on agent turns (default: %(default)s).')
    p.add_argument('--temperature', type=float, default=None,
                   help='Sampling temperature. Omit to use the shared default (0.6) applied to '
                        'EVERY model for comparability. See nesyplan/llm.py:DEFAULT_TEMPERATURE.')
    p.add_argument('--top-p', type=float, default=None, dest='top_p',
                   help='Nucleus sampling top_p. Omit to use the shared default (0.95) for every model.')
    p.add_argument('--seed', type=int, default=None, help='Recorded in the log for reproducibility bookkeeping.')
    p.add_argument('--log-dir', default=os.environ.get('LLM_LOG_DIR') or os.path.join(repo_root, 'llm_logs'),
                   help='Where to write the per-run JSON session transcript (default: %(default)s).')
    args = p.parse_args(argv)

    try:
        config = build_config(args)
    except ValueError as exc:
        print(f'error: {exc}', file=sys.stderr)
        return 2

    if args.backend == 'fake':
        robot = FakeRobot()
        print('Backend: fake (in-process world model, no simulator)')
    else:
        robot = RobotClient(args.url)
        print(f'Connecting to command server at {args.url} ...')
    try:
        state = robot.get_state()
    except ConnectionError as exc:
        print(f'error: {exc}', file=sys.stderr)
        return 2
    n_cubes = len(state.get('cubes') or {})
    print(f'Initial state: {n_cubes} cubes, grid {state.get("grid_units")}x{state.get("grid_units")}, '
          f'mode={state.get("mode", "basic")}')

    llm = LLMClient(model=args.model, temperature=args.temperature, top_p=args.top_p,
                    on_effort=args.on_effort)
    # Record the sampling actually used (per-model recommended, unless overridden) in the log.
    config.temperature, config.top_p = llm.temperature, llm.top_p
    print(f'LLM: model={llm.model} base_url={llm.base_url} '
          f'on_effort={llm.on_effort or "model-default"} '
          f'temperature={llm.temperature} top_p={llm.top_p}')
    print(f'Policy: {config.policy.value}  Cache: {config.cache_mode.value}  Task: {args.task!r}\n')

    log = SessionLog(args.log_dir, config.policy.value, task=args.task, model=llm.model,
                     base_url=llm.base_url, config=config.to_dict(), seed=config.seed)
    log.set_meta(backend=args.backend)
    if log.path:
        print(f'Session log: {log.path}')

    try:
        return run_episode(args.task, config, robot, llm, log)
    except (ConnectionError, RuntimeError) as exc:
        print(f'error: {exc}', file=sys.stderr)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
