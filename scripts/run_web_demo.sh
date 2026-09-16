#!/bin/bash
#
# Web demonstrator: a browser chat UI over the NeSyPlan agent.
#
# There is nothing to install and nothing to start first -- the world model runs inside
# this process (nesyplan/fake_robot.py). All you need is an API key in .env.
#
# Usage:
#   ./scripts/run_web_demo.sh                       # the demonstrator
#   ./scripts/run_web_demo.sh --model qwen3-14b     # any flag is forwarded to web_demo.py
#   ./scripts/run_web_demo.sh replay                # no LLM either: play back a transcript
#
# Env:
#   PORT=8600     web UI port (also settable via --port)
#   NO_OPEN=1     do not open the browser
#
# To drive an EXTERNAL executor (a simulator or robot you run yourself; not part of this
# repository -- see docs/ARCHITECTURE.md):
#   ./scripts/run_web_demo.sh --backend sim --url http://localhost:8100
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"
cd "$REPO_DIR"

# A leading `replay` is shorthand for --backend replay (canned turns from a recording).
# Kept as a plain string, not an array: macOS still ships bash 3.2, where expanding an
# EMPTY array under `set -u` is itself an "unbound variable" error.
BACKEND=""
case "${1:-}" in
  replay) BACKEND="replay"; shift ;;
  fake)   BACKEND="fake";   shift ;;   # the default; accepted for symmetry
esac

WEB_PORT="${PORT:-8600}"
URL="http://127.0.0.1:${WEB_PORT}/"

PYTHON="python3"; command -v python3 >/dev/null 2>&1 || PYTHON="python"

if [ ! -f .env ]; then
  echo "Note: no .env found. Copy .env.example to .env and put your API key in it,"
  echo "      otherwise the first task will fail with a missing-key error."
  echo
fi

# Open the browser shortly after the server binds, unless NO_OPEN=1.
if [ "${NO_OPEN:-0}" != "1" ] && command -v open >/dev/null 2>&1; then
  ( sleep 1.5; open "$URL" >/dev/null 2>&1 || true ) &
elif [ "${NO_OPEN:-0}" != "1" ] && command -v xdg-open >/dev/null 2>&1; then
  ( sleep 1.5; xdg-open "$URL" >/dev/null 2>&1 || true ) &
fi

# A --port / --backend in "$@" still wins (argparse: last one set).
if [ -n "$BACKEND" ]; then
  exec "$PYTHON" -m nesyplan.web_demo --port "$WEB_PORT" --backend "$BACKEND" "$@"
fi
exec "$PYTHON" -m nesyplan.web_demo --port "$WEB_PORT" "$@"
