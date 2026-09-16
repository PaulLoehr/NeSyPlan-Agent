"""Minimal .env loader, shared by every entry point.

A leaf module on purpose: it imports nothing from the package, so the cheap tools
(`probe_reasoning`) can read the key file without dragging in the agent stack that
`run.py` pulls behind it.

Deliberately not python-dotenv. The format used here is the intersection everyone
agrees on -- `KEY=VALUE`, `#` comments, an optional `export ` prefix, optional
surrounding quotes -- and supporting it costs fifteen lines instead of a dependency,
which is what lets the README say "nothing to install".
"""
import os


def load_dotenv(path):
    """Load KEY=VALUE lines from `path` into os.environ. Missing file: do nothing.

    A variable already present in the environment WINS: an explicit
    `OPENROUTER_API_KEY=... python3 -m nesyplan.eval` must not be silently overridden
    by whatever happens to sit in the repo's .env.
    """
    if not os.path.isfile(path):
        return
    with open(path, encoding='utf-8') as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            if line.startswith('export '):
                line = line[len('export '):]
            if '=' not in line:
                continue
            key, _, value = line.partition('=')
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = value
