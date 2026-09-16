"""Symbolic goal checkers -- a deterministic primary scorer for the structural tasks.

The LLM judge (nesyplan/judge.py) is the GENERAL scorer and stays: it is the only thing
that can score an open-ended goal ("build a structure that defends the red block") or an
ad-hoc --task-prompt. But it is also the harness's noisiest instrument -- it flips on
identical structures and has invented cube colours the scene never had (docs/FINDINGS.md,
Finding C side-note). A scorer with that much variance cannot carry a cost/robustness
front, because a 15-point config difference disappears into judge noise.

For the tasks whose goal IS formally checkable, we do not need a language model at all:
the tracked executor's /state is exact ground truth (real coordinates, no perception gap),
so the goal is a predicate over it. This module holds those predicates.

Both scorers run on every task that has a checker. The checker is the reported result;
the judge's verdict is recorded alongside, and their agreement is a reportable number
("an LLM judge over symbolic state agrees with a formal checker on X% of runs") instead
of an unexamined assumption.

A checker is a callable `check(state) -> (success: bool, detail: str)`. Tasks bind one via
Task(check=...) in nesyplan/tasks.py.
"""

FOOTPRINT = 1.0
_EPS = 1e-6
_COL_TOL = 0.5   # cubes count as "same column" when their centres are closer than this


def _placed(state):
    """[(cube_id, info)] for cubes resting in the building area, with x/y/level present."""
    out = []
    for cid, info in ((state or {}).get('cubes') or {}).items():
        info = info or {}
        if (info.get('location') == 'area' and info.get('x') is not None
                and info.get('y') is not None and info.get('level') is not None):
            out.append((cid, info))
    return out


def _by_color(state):
    """{color: (cube_id, info)} over the placed cubes (colors are unique in the catalog)."""
    return {(info.get('color') or ''): (cid, info) for cid, info in _placed(state)}


def _supported(info, others):
    """Does some cube one level below overlap this cube's footprint?"""
    return any(o.get('level') == info['level'] - 1
               and abs(o['x'] - info['x']) < FOOTPRINT - _EPS
               and abs(o['y'] - info['y']) < FOOTPRINT - _EPS
               for o in others)


def stack_check(colors, name=None):
    """The named colors form ONE vertical tower, bottom-to-top in exactly this order.

    Requires: all named cubes in the building area, all in the same column (centres within
    _COL_TOL on both axes), levels consecutive and ascending in the given order. Cubes not
    named are ignored -- the task constrains a tower, not the whole plate.
    """
    label = name or ' -> '.join(colors)

    def check(state):
        found = _by_color(state)
        missing = [c for c in colors if c not in found]
        if missing:
            where = {}
            for c in missing:
                info = ((state or {}).get('cubes') or {}).get(f'cube_{c}') or {}
                where[c] = info.get('location') or 'unknown'
            return False, (f'not a tower ({label}): {", ".join(missing)} not in the building '
                           f'area ({", ".join(f"{c}={w}" for c, w in where.items())})')

        entries = [(c,) + found[c] for c in colors]        # (color, cube_id, info)
        x0, y0 = entries[0][2]['x'], entries[0][2]['y']
        off = [c for c, _, i in entries
               if abs(i['x'] - x0) >= _COL_TOL or abs(i['y'] - y0) >= _COL_TOL]
        if off:
            layout = ', '.join(f'{c}@({i["x"]:g},{i["y"]:g},L{i["level"]})' for c, _, i in entries)
            verb = 'sits' if len(off) == 1 else 'sit'
            return False, (f'not one tower ({label}): {", ".join(off)} {verb} in another '
                           f'column -- {layout}')

        levels = [i['level'] for _, _, i in entries]
        if levels != list(range(levels[0], levels[0] + len(levels))):
            layout = ', '.join(f'{c}=L{l}' for (c, _, _), l in zip(entries, levels))
            return False, (f'wrong order/gaps ({label} bottom-to-top): got {layout}')

        return True, (f'tower {label} bottom-to-top at ({x0:g},{y0:g}), '
                      f'levels {levels[0]}..{levels[-1]}')

    check.spec = f'stack[{label}]'
    return check


def levels_check(counts, require_all=False, name=None):
    """`counts` = {level: number_of_cubes} for the building area, plus support for level > 0.

    The shape check for tasks like the 3-2-1 pyramid, where WHICH cube sits where does not
    matter but the silhouette does. require_all additionally demands that no cube is left
    in storage.
    """
    label = name or '-'.join(str(counts[k]) for k in sorted(counts))   # bottom level first

    def check(state):
        placed = _placed(state)
        infos = [i for _, i in placed]
        got = {}
        for i in infos:
            got[i['level']] = got.get(i['level'], 0) + 1
        want = {int(k): int(v) for k, v in counts.items()}
        if got != want:
            fmt = lambda d: ', '.join(f'L{k}={d[k]}' for k in sorted(d)) or 'nothing placed'
            return False, f'wrong shape ({label}): wanted {fmt(want)}, got {fmt(got)}'

        floating = [cid for cid, i in placed if i['level'] > 0 and not _supported(i, infos)]
        if floating:
            return False, f'unsupported cube(s) floating: {", ".join(sorted(floating))}'

        if require_all:
            stored = [cid for cid, info in ((state or {}).get('cubes') or {}).items()
                      if (info or {}).get('location') != 'area']
            if stored:
                return False, (f'not all blocks used ({label}): '
                               f'{", ".join(sorted(stored))} still outside the building area')

        return True, f'shape {label} built and fully supported ({len(placed)} cubes placed)'

    check.spec = f'levels[{label}]' + ('+all' if require_all else '')
    return check


def run_check(task, state):
    """Score `task` against `state` with its checker, or None when it has none.

    Returns {'success': bool, 'detail': str, 'spec': str} -- or None for judge-only tasks
    (open-ended goals, ad-hoc --task-prompt), which is the signal to fall back to the judge.
    Never raises: a checker crash is reported as an un-scored run, not a lost campaign.
    """
    check = getattr(task, 'check', None)
    if not callable(check):
        return None
    try:
        success, detail = check(state)
    except Exception as exc:   # noqa: BLE001 -- a broken predicate must not kill the cell
        return {'success': None, 'detail': f'checker error: {exc!r}',
                'spec': getattr(check, 'spec', '?')}
    return {'success': bool(success), 'detail': detail, 'spec': getattr(check, 'spec', '?')}
