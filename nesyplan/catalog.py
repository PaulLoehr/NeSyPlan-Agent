"""Runtime access to the cube catalog (data/cubes.json).

The catalog is the single source of truth for the cube set: ids, numbers, colors and the
shared cube size. Every reader loads it here, so the cube list, size, colors and numbers
are never hardcoded twice -- the world model, the prompt the model reads, and the picture
the UI draws all describe the same six cubes.

An external executor (see docs/ARCHITECTURE.md) is expected to read the same file, which
is why CUBES_CATALOG can point this loader at a shared copy. A missing or unreadable file
degrades to the built-in fallback with a warning instead of breaking.

Typical use:

    from nesyplan.catalog import load_catalog
    cat = load_catalog()
    CUBES = cat.ids                 # ['cube_red', ...] in catalog order
    CUBE_SIZE = cat.cube_size       # 0.06
    cat.describe('cube_red')        # {'id':..,'number':1,'color':'red',...}
    cat.prompt_lines()              # ['cube_red (color: red, number: 1)', ...]
"""

import json
import os

_CANDIDATE_PATHS = [
    os.environ.get('CUBES_CATALOG'),
    # nesyplan/ sits one level below the repo root -> repo-root data/cubes.json.
    os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data', 'cubes.json'),
]

# Fallback matching the committed catalog, so a run still works (with a warning)
# if the file is somehow not mounted. Keep in sync with data/cubes.json.
_FALLBACK = {
    'cube_size': 0.06,
    'mass': 0.016,
    'render_mode': 'texture',
    'cubes': [
        {'id': 'cube_red', 'number': 1, 'color': 'red', 'rgba': [0.9, 0.1, 0.1, 1.0]},
        {'id': 'cube_green', 'number': 2, 'color': 'green', 'rgba': [0.1, 0.7, 0.1, 1.0]},
        {'id': 'cube_blue', 'number': 3, 'color': 'blue', 'rgba': [0.1, 0.3, 0.9, 1.0]},
        {'id': 'cube_yellow', 'number': 4, 'color': 'yellow', 'rgba': [0.95, 0.85, 0.1, 1.0]},
        {'id': 'cube_black', 'number': 5, 'color': 'black', 'rgba': [0.05, 0.05, 0.05, 1.0]},
        {'id': 'cube_white', 'number': 6, 'color': 'white', 'rgba': [0.95, 0.95, 0.95, 1.0]},
    ],
}


class Catalog:
    """Parsed cube catalog with convenience views used by the research harness."""

    def __init__(self, data, source='<fallback>'):
        self.source = source
        self.cube_size = float(data['cube_size'])
        self.mass = float(data.get('mass', 0.016))
        self.render_mode = data.get('render_mode', 'texture')
        # Keep catalog order (stacks build bottom-to-top in this order).
        self.cubes = [dict(c) for c in data['cubes']]
        self._by_id = {c['id']: c for c in self.cubes}

    @property
    def ids(self):
        return [c['id'] for c in self.cubes]

    def describe(self, cube_id):
        return self._by_id.get(cube_id)

    def number(self, cube_id):
        c = self._by_id.get(cube_id)
        return c['number'] if c else None

    def color(self, cube_id):
        c = self._by_id.get(cube_id)
        return c['color'] if c else None

    def prompt_lines(self):
        """Human/LLM-readable one-liners, e.g. 'cube_red (color: red, number: 1)'."""
        return [f'{c["id"]} (color: {c["color"]}, number: {c["number"]})'
                for c in self.cubes]

    def info_map(self):
        """{id: {'color':.., 'number':..}} for state observations / prompts."""
        return {c['id']: {'color': c['color'], 'number': c['number']} for c in self.cubes}


def load_catalog(path=None):
    """Load the catalog, trying `path`, then $CUBES_CATALOG, then repo-root data/cubes.json.

    Falls back to a built-in copy (with a warning) if no file is found, so a missing or
    moved data file degrades gracefully instead of breaking the run.
    """
    for candidate in ([path] if path else []) + _CANDIDATE_PATHS:
        if candidate and os.path.isfile(candidate):
            with open(candidate) as fh:
                return Catalog(json.load(fh), source=candidate)
    print('[nesyplan.catalog] WARNING: data/cubes.json not found on any known path; '
          'using the built-in fallback catalog.', flush=True)
    return Catalog(_FALLBACK)
