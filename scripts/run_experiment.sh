#!/bin/bash
#
# Run the shipped experiment: does an agentic harness beat one-shot planning, and what
# does thinking buy on top? See docs/EXPERIMENT.md for the design and the results.
#
# The matrix is 4 tasks x 7 configs x 2 reps = 56 episodes, all on the in-process world
# model -- no simulator, no containers. It costs LLM tokens, not setup.
#
#   tasks   @rebuild  -- the four inherited-layout tasks (a cube is buried / boxed in /
#                        the goal cells are occupied). Every one is scored by a symbolic
#                        CHECKER, not by an LLM judge.
#   configs @harness  -- oneshot_nocot, oneshot, act, react, react_summary, on_error_summary
#
# Usage:
#   ./scripts/run_experiment.sh                        # the full 56-episode campaign
#   ./scripts/run_experiment.sh --list                 # print the matrix, run nothing
#   ./scripts/run_experiment.sh --reps 1               # half the cost, noisier
#   ./scripts/run_experiment.sh --model qwen3-14b      # a different model
#   ./scripts/run_experiment.sh --campaign my_run      # results/my_run/
#
# Results land in results/<campaign>/ (manifest.json, runs/, results.jsonl, summary.md).
# Resumable: re-run the same --campaign and finished cells are skipped.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"
cd "$REPO_DIR"

PYTHON="python3"; command -v python3 >/dev/null 2>&1 || PYTHON="python"

if [ ! -f .env ]; then
  echo "error: no .env found. Copy .env.example to .env and put your API key in it." >&2
  exit 1
fi

# Sampling: 0.6 / 0.95, NOT the greedy temp-0 default. Two reasons, both in
# nesyplan/llm.py: greedy is the degenerate regime for Qwen3-family thinking models (it
# inflates reasoning and biases effect magnitudes), and the endpoint is not deterministic
# at temp 0 anyway -- so greedy would hide variance rather than remove it, making --reps
# meaningless. 0.6/0.95 is also Qwen3's own recommended thinking-mode sampling.
#
# Every flag is forwarded, and a later one wins (argparse), so the values here are only
# defaults -- `--reps 1` or `--tasks flag_excavate` override them.
exec "$PYTHON" -m nesyplan.eval \
  --campaign harness \
  --tasks @rebuild \
  --configs @harness \
  --reps 2 \
  --temperature 0.6 \
  --top-p 0.95 \
  "$@"
