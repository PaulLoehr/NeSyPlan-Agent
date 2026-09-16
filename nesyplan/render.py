"""Render a final /state observation to a PNG for human (or vision) inspection.

One picture per episode, dropped next to its transcript, so a run can be judged at a
glance instead of by reading a JSON dict. Top-down view of the building-area grid:
each placed cube is a colored square labelled with its number; stacks are drawn with a
small isometric offset per level (so height is visible from above) and spelled out
bottom->top in the footer, alongside the held/storage cubes.

matplotlib is imported LAZILY inside render_state so that importing this module (and
thus nesyplan.eval) never requires it -- `--list`/`--dry-run` and the rest of the
harness stay stdlib-only; only actually drawing a PNG needs matplotlib.
"""

import json
import os
from collections import defaultdict

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Fallback colors by name (only used if data/cubes.json can't be read); rgba there wins.
_NAME_RGB = {
    'red': (0.90, 0.10, 0.10), 'green': (0.10, 0.70, 0.10), 'blue': (0.10, 0.30, 0.90),
    'yellow': (0.95, 0.85, 0.10), 'black': (0.10, 0.10, 0.10), 'white': (0.95, 0.95, 0.95),
}


def available():
    """True if matplotlib can be imported (PNG rendering is possible)."""
    try:
        import matplotlib  # noqa: F401
        return True
    except Exception:
        return False


def _rgb_by_id():
    try:
        with open(os.path.join(REPO_ROOT, 'data', 'cubes.json'), encoding='utf-8') as fh:
            data = json.load(fh)
        return {c['id']: tuple(c['rgba'][:3]) for c in data.get('cubes', []) if c.get('rgba')}
    except (OSError, ValueError, KeyError):
        return {}


def _color_for(cid, info, rgb_by_id):
    if cid in rgb_by_id:
        return rgb_by_id[cid]
    return _NAME_RGB.get((info or {}).get('color'), (0.6, 0.6, 0.6))


def _short(cid):
    return cid.replace('cube_', '')


def render_state(state, out_path, *, task=None, title=None):
    """Draw the final world state to out_path (PNG). Returns out_path, or None on failure."""
    try:
        import matplotlib
        matplotlib.use('Agg')                       # headless: no display needed
        import matplotlib.pyplot as plt
        from matplotlib.patches import Rectangle
    except Exception as exc:
        print(f'  [render] matplotlib unavailable, skipping PNG: {exc!r}')
        return None

    grid = state.get('grid_units', 6)
    cubes = state.get('cubes') or {}
    rgb_by_id = _rgb_by_id()
    held = state.get('held')

    columns = defaultdict(list)   # (x,y) -> [(level, cid, info), ...]
    storage = []
    for cid, info in sorted(cubes.items()):
        info = info or {}
        loc = info.get('location')
        if loc == 'area' and info.get('x') is not None:
            columns[(round(info['x'], 3), round(info['y'], 3))].append((info.get('level', 0), cid, info))
        elif loc == 'held' or cid == held:
            continue                                # shown in the footer
        else:
            storage.append(cid)

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.set_xlim(0, grid)
    ax.set_ylim(0, grid)
    ax.set_aspect('equal')
    ax.set_xticks(range(grid + 1))
    ax.set_yticks(range(grid + 1))
    ax.grid(True, color='0.85', linewidth=0.8)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.plot(grid / 2, grid / 2, marker='+', color='0.55', markersize=14, markeredgewidth=1.5)

    size, off = 0.8, 0.12                           # footprint drawn; isometric offset per level
    for (x, y), members in columns.items():
        members.sort()                              # by level, bottom first
        for level, cid, info in members:
            cx, cy = x - size / 2 + off * level, y - size / 2 + off * level
            face = _color_for(cid, info, rgb_by_id)
            ax.add_patch(Rectangle((cx, cy), size, size, facecolor=face,
                                   edgecolor='0.15', linewidth=1.2, zorder=3 + level))
            num = info.get('number')
            if num is not None:
                lum = 0.299 * face[0] + 0.587 * face[1] + 0.114 * face[2]
                ax.text(cx + size / 2, cy + size / 2, str(num), ha='center', va='center',
                        color='white' if lum < 0.5 else 'black', fontsize=11,
                        fontweight='bold', zorder=3 + level + 0.5)
        if len(members) > 1:                        # stack-height badge
            n = len(members)
            ax.text(x - size / 2 + off * n + size + 0.02, y - size / 2 + off * n + size,
                    f'x{n}', ha='left', va='top', fontsize=8, color='0.3')

    ttl = title or 'Final state'
    if task:
        t = task if len(task) <= 90 else task[:87] + '...'
        ttl = f'{ttl}\n"{t}"'
    ax.set_title(ttl, fontsize=11)

    lines = []
    for (x, y), members in sorted(columns.items()):
        members.sort()
        order = ' -> '.join(_short(cid) for _, cid, _ in members)
        suffix = '  (bottom->top)' if len(members) > 1 else ''
        lines.append(f'({x:g},{y:g}): {order}{suffix}')
    footer = 'Placed: ' + ('; '.join(lines) if lines else 'none')
    if held:
        footer += f'\nHeld: {_short(held)}'
    if storage:
        footer += '\nStorage: ' + ', '.join(_short(c) for c in storage)
    fig.text(0.01, 0.005, footer, fontsize=8, va='bottom', ha='left', family='monospace')
    fig.subplots_adjust(bottom=0.16 + 0.03 * (len(lines) + bool(held) + bool(storage)))

    try:
        os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)
        fig.savefig(out_path, dpi=110)
        return out_path
    except OSError as exc:
        print(f'  [render] could not write {out_path}: {exc!r}')
        return None
    finally:
        plt.close(fig)
