"""LLM-as-judge: score a finished episode from its final /state -- no per-task checker.

This is the GENERAL scorer. Give it the task text and the final world state and it
returns a semantic pass/fail, so you can pose ARBITRARY tasks without writing a checker
first: the model that can build the flag can also judge whether a flag was built. It
runs over the same Phoenix endpoint as the actor (nesyplan/llm.py), one call per episode.

This is the SOLE scorer -- there are no scripted checkers. It is strong on
spatial/structural goals; on precise arithmetic or off-by-one logic it is less reliable,
so spot-check those verdicts against the per-run PNG (nesyplan/render.py).

Honest trade-off: the tracked executor's /state IS ground truth in sim (the arm places
cubes at exactly those coordinates -- no perception gap), so this judge reasons over
exact numbers, not pixels. On real hardware, where the world model can drift from
reality, a judge over a real photo would add signal this one cannot.
"""

import json
import re

from nesyplan.environment import catalog_block, format_state, grid_units
from nesyplan.metrics import normalize_usage

JUDGE_SYSTEM = """You are a strict evaluator for a cube-stacking robot. Given a TASK and \
the FINAL world state the robot produced, decide whether the task was accomplished.

Judge only the final configuration against the task's intent -- not how the robot got \
there, not efficiency. Be strict about explicit constraints (exact colors, exact order, \
counts, "all cubes", specific positions) and reasonable about vague ones ("near the \
center", "roughly in a circle"). If the task is genuinely ambiguous, accept any sensible \
interpretation. A cube still in "storage" was never placed.

Respond with ONLY a JSON object -- no prose, no code fence:
{"success": true|false, "confidence": 0.0-1.0, "reasoning": "one or two sentences"}"""


def build_judge_prompt(task, state):
    """The user message for the judge: task + catalog + coordinate system + final state."""
    grid = grid_units(state)
    return f"""TASK:
{task}

Cube catalog (id, color, number):
{catalog_block(state)}

Coordinate system: {grid}x{grid} grid; ({grid / 2:g},{grid / 2:g}) is the center; z is \
the stack level (0 = on the table).

FINAL world state:
{format_state(state)}

Did the robot accomplish the TASK? Reply with the JSON verdict only."""


def parse_verdict(text):
    """Pull the {success, confidence, reasoning} object out of the judge's reply, or None."""
    text = (text or '').strip()
    fence = re.search(r'```(?:json)?\s*\n(.*?)```', text, re.DOTALL)
    if fence:
        text = fence.group(1).strip()
    match = re.search(r'\{.*\}', text, re.DOTALL)
    if not match:
        return None
    try:
        data = json.loads(match.group(0))
    except ValueError:
        return None
    if not isinstance(data, dict) or 'success' not in data:
        return None
    conf = data.get('confidence')
    try:
        conf = float(conf) if conf is not None else None
    except (TypeError, ValueError):
        conf = None
    return {'success': bool(data['success']), 'confidence': conf,
            'reasoning': str(data.get('reasoning', ''))}


def judge_episode(task, state, llm, *, reasoning=True, log=None):
    """Score one finished episode. Returns {success, confidence, reasoning, tokens}.

    success is True/False on a parsed verdict, or None if the call failed or the reply
    was unparseable (recorded in `reasoning` so the run stays diagnosable). If a
    SessionLog is passed, the judge exchange is appended to it for transparency (clearly
    marked, mirroring how the C2 cache summarizer's side-call is logged).
    """
    prompt = build_judge_prompt(task, state)
    messages = [{'role': 'system', 'content': JUDGE_SYSTEM},
                {'role': 'user', 'content': prompt}]
    try:
        resp = llm.chat(messages, reasoning=reasoning)
    except Exception as exc:
        return {'success': None, 'confidence': None,
                'reasoning': f'judge call failed: {exc!r}', 'tokens': normalize_usage(None)}

    verdict = parse_verdict(resp.content)
    if verdict is None:
        verdict = {'success': None, 'confidence': None,
                   'reasoning': f'unparseable judge output: {(resp.content or "")[:200]!r}'}
    verdict['tokens'] = normalize_usage(resp.usage)

    if log is not None:
        log.add({'role': 'system', 'content': '[judge call -- not shown to the agent]'})
        log.add({'role': 'user', 'content': prompt})
        log.add(resp.message)   # raw -> keeps the judge's own reasoning trace
    return verdict
