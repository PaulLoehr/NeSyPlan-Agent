"""Single source of truth for the cube-world domain, agent-side.

Everything the orchestrator and prompts need to know about the world is derived here from
the executor's /state observation -- the grid size, the cube catalog (ids + colors +
numbers), and the compact volatile-state rendering.

The EXECUTOR is authoritative for the world, not this module: cube ids and the grid size
are never hardcoded here, they come from the live observation. That is what lets the same
agent code drive the in-process world model (fake_robot.py) and an external executor
without knowing which one it is talking to.
"""

DEFAULT_GRID_UNITS = 6
PLACE_MARGIN = 0.5  # cube centers must stay this far from every edge (matches the tracked executor)


def grid_units(state):
    return state.get('grid_units', DEFAULT_GRID_UNITS)


def cube_ids(state):
    """Sorted cube ids from the observation (stable order for tool enums + prompts)."""
    return sorted((state.get('cubes') or {}).keys())


def is_tracked(state):
    """The tracked executor advertises mode='tracked'; basic omits it."""
    return state.get('mode') == 'tracked'


def catalog_block(state):
    """The cube catalog block for the system prompt: one line per cube with color/number.

    The colors/numbers arrive with every /state (the executor merges the catalog
    into each cube), so the model reasons over them instead of guessing from the id.
    """
    cubes = state.get('cubes') or {}
    lines = []
    for cid in cube_ids(state):
        info = cubes.get(cid) or {}
        attrs = [f'{k}: {info[k]}' for k in ('color', 'number') if info.get(k) is not None]
        lines.append(f'  - {cid} ({", ".join(attrs)})' if attrs else f'  - {cid}')
    return '\n'.join(lines)


def format_state(state):
    """Compact, human-readable rendering of the volatile world state for the model.

    Only what is held + where each cube is; colors/numbers already live in the
    system prompt's catalog, so they are not repeated here (much cheaper to re-read
    than the full-state JSON).
    """
    held = state.get('held')
    cubes = state.get('cubes') or {}
    lines = [f'Holding: {held if held else "nothing"}', 'Cubes:']
    for cid in cube_ids(state):
        info = cubes.get(cid) or {}
        loc = info.get('location')
        if loc == 'held':
            where = 'held (in the gripper)'
        elif loc == 'area':
            where = f'building area at x={info.get("x")}, y={info.get("y")}, level={info.get("level")}'
        else:
            where = 'storage (not yet placed)'
        lines.append(f'  - {cid}: {where}')
    return '\n'.join(lines)


# --- feedback informativeness (the "what does the symbolic layer say back" lever) ------
#
# Every rejection the executor computes has two halves: WHAT is wrong (diagnosis) and WHAT
# TO DO about it (remedy) -- see fake_robot._reject. These three levels decide how much of
# that reaches the model, and nothing else changes: the ok=False flag, the world state and
# the policy's error trigger are identical in all three arms. So a difference between arms
# is attributable to the TEXT alone, which is what makes this a clean manipulation.
#
#   terse -- "it failed", no reason at all. The model must infer the cause from the state.
#   why   -- diagnosis only. Names the violated precondition, suggests nothing.
#   fix   -- diagnosis + remedy (the executor's full message; the shipped behaviour).
FEEDBACK_LEVELS = ('terse', 'why', 'fix')
TERSE_ERROR = 'action failed'


def error_at_level(feedback, level):
    """The error text a model at `level` gets to read.

    Falls back to the full sentence when the backend did not split it: the in-process world
    model emits `error_parts`, but an external executor may send one flat string, so on
    --backend sim the `why` arm cannot be cut faithfully (eval.py warns about this).
    """
    if level == 'terse':
        return TERSE_ERROR
    full = feedback.get('error')
    if level == 'why':
        return (feedback.get('error_parts') or {}).get('diagnosis') or full
    return full


def concise_feedback(feedback, level='fix'):
    """Trim an executor feedback dict to what the model needs: ok + message/error.

    The executor also returns the full `observation` (and a `done` flag) with every
    reply; forwarding that verbatim every turn bloats the context. We drop it -- the
    model pulls fresh state via get_state instead.

    `level` applies the feedback-informativeness lever to the error channel (see above).
    Success messages are identical across levels on purpose -- the lever is about how
    failures are explained, so leaving the success path untouched keeps the arms
    comparable everywhere else. Default 'fix' = the shipped behaviour, so every existing
    caller is unchanged.
    """
    out = {'ok': feedback.get('ok')}
    if 'message' in feedback:
        out['message'] = feedback['message']
    if 'error' in feedback:
        out['error'] = error_at_level(feedback, level)
    return out
