#!/usr/bin/env python3
"""Probe whether an effort of "none" actually turns chain-of-thought OFF, per model.

The effort level is an API *hint* -- whether "none" suppresses the assistant message's
`reasoning` field depends on the model's serving backend, not on the API. This sends the
SAME prompt to each aliased model twice (effort "none" vs "high") and prints, for each,
the returned reasoning-trace length + a content snippet. If the "none" trace is not ~0,
that model cannot have its CoT disabled here -- which makes the harness's no-reasoning
legs (oneshot_nocot, reason_first_noreason, every reasoning-OFF turn) invalid for it, and
it must stay out of demo._REASONING_OPTIONAL. The failure is silent and expensive: a model
that ignores "none" makes every reasoning-OFF arm a duplicate of its reasoning-ON cousin,
so the comparison measures nothing while producing a table that looks like it does.

Each model is probed at ITS OWN provider, with that provider's key, headers and reasoning
dialect (nesyplan/providers.py) -- so Phoenix and OpenRouter models can be compared in one
run. The request is still hand-rolled rather than going through nesyplan/llm.py: this is
the tool that verifies what the client's toggle is worth, so it must not depend on it.

Run from the repo root (reads the git-ignored .env for BEARER_TOKEN / OPENROUTER_API_KEY):
    python3 -m nesyplan.probe_reasoning
    python3 -m nesyplan.probe_reasoning --models kimi,phoenix    # subset
    python3 -m nesyplan.probe_reasoning --models qwen3-32b       # a new model, before trusting it
"""
import argparse
import json
import os
import time
import urllib.error
import urllib.request

from nesyplan.envfile import load_dotenv
from nesyplan.model_aliases import MODEL_ALIASES, resolve_model
from nesyplan.providers import endpoint_for, reasoning_payload, routing_payload

# A prompt a thinking model would normally reason about (small, to keep it cheap).
PROMPT = ("You have cubes numbered 1..6. Pick three whose numbers sum to 10 and "
          "give just the three numbers. One short line.")


def _call(endpoint, model, effort, timeout=150):
    payload = {'model': model, 'temperature': 0.0,
               'messages': [{'role': 'user', 'content': PROMPT}]}
    payload.update(reasoning_payload(endpoint.provider, effort))
    payload.update(routing_payload(endpoint.provider, True))
    if not endpoint.api_key:
        return {'err': f'no key: set {endpoint.key_env} in .env'}
    headers = {'Content-Type': 'application/json',
               'Authorization': f'Bearer {endpoint.api_key}'}
    headers.update(endpoint.headers)
    req = urllib.request.Request(endpoint.base_url.rstrip('/') + '/chat/completions',
                                 data=json.dumps(payload).encode(), headers=headers, method='POST')
    t0 = time.monotonic()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            body = json.load(r)
    except urllib.error.HTTPError as e:
        return {'err': f'HTTP {e.code} {e.reason}: {e.read().decode("utf-8", "replace")[:160]}'}
    except Exception as e:  # noqa: BLE001 -- probe: report any failure, keep going
        return {'err': f'{type(e).__name__}: {e}'}
    msg = body['choices'][0]['message']
    trace = msg.get('reasoning') or msg.get('reasoning_content') or ''
    usage = body.get('usage') or {}
    return {'ms': int((time.monotonic() - t0) * 1000),
            'trace_chars': len(trace),
            'completion_tokens': usage.get('completion_tokens'),
            'content': (msg.get('content') or '').strip().replace('\n', ' ')[:80]}


def main(argv=None):
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    load_dotenv(os.path.join(repo_root, '.env'))

    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--models', default=None,
                   help='Comma-separated aliases or full ids (default: all aliases in model_aliases.py).')
    args = p.parse_args(argv)

    if args.models:
        targets = [(m.strip(), resolve_model(m.strip())) for m in args.models.split(',') if m.strip()]
    else:
        targets = list(MODEL_ALIASES.items())

    endpoints = {alias: endpoint_for(mid) for alias, mid in targets}
    for alias, ep in endpoints.items():
        print(f'{alias:10s} -> {ep.provider:10s} {ep.base_url}  '
              f'({ep.key_env}: {"set" if ep.api_key else "MISSING"})')
    print()
    print(f'{"model":10s} {"effort":6s} | {"ms":>6s} {"trace_ch":>8s} {"comp_tok":>8s} | content')
    print('-' * 92)
    for alias, mid in targets:
        for effort in ('none', 'high'):
            r = _call(endpoints[alias], mid, effort)
            if 'err' in r:
                print(f'{alias:10s} {effort:6s} | ERROR: {r["err"]}')
                continue
            print(f'{alias:10s} {effort:6s} | {r["ms"]:>6d} {r["trace_chars"]:>8d} '
                  f'{str(r["completion_tokens"]):>8s} | {r["content"]}')
        print()
    print('Read: a "none" trace of ~0 == CoT can genuinely be disabled (verified for the '
          'qwen3 aliases) -> the model may join demo._REASONING_OPTIONAL. A large or '
          'identical-to-"high" trace == the toggle is a no-op for that model -> keep it out.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
