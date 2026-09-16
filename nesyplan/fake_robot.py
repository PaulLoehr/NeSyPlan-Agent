"""The world model: an in-process, simulator-free symbolic executor.

This is the SYMBOLIC half of the neuro-symbolic loop, and the default backend. It owns
the authoritative world state and validates every command against it -- bounds, occupancy,
support, and gripper-axis reachability -- then either applies the command or rejects it
with a reason the model can act on. Nothing here is a mock of the agent's logic; the
rejections are the real mechanism the experiment measures.

It duck-types nesyplan.robot_client.RobotClient (health / get_state / send_command), so
the orchestrator cannot tell whether it is talking to this class or to an external
executor over HTTP. That symmetry is what makes --backend fake and --backend sim
interchangeable.

Provenance: this model was extracted from a containerised executor that drives a UR5e arm
with an OnRobot RG2 gripper (in simulation or on real hardware). There, the same
validation runs in pure Python and only the pick()/place() MOTION touches the simulator.
This module reproduces the state transitions and feedback strings and replaces the motion
with a no-op -- so the symbolic /state a scorer reads, and the feedback the model reasons
over, are the same, but instant and with nothing to boot. See docs/ARCHITECTURE.md.

The ONE behaviour it cannot reproduce is a physical-motion failure (IK miss, unexpected
collision, a dropped grasp): nothing moves here, so every geometrically valid command
succeeds. For the orchestration question this repository studies that is not a limitation
-- the interesting failures are symbolic. Treat a green run as "the orchestration logic is
sound"; motion is a separate concern, confirmed on real hardware elsewhere.

The cube set (ids, colors, numbers) comes from data/cubes.json via nesyplan.catalog -- the
same single source of truth an external executor is expected to read.
"""

from nesyplan.catalog import load_catalog

_CATALOG = load_catalog()
CUBES = _CATALOG.ids                 # ['cube_red', ...] in catalog order
CUBE_INFO = _CATALOG.info_map()      # {id: {'color':.., 'number':..}}

# --- geometry constants (shared with the external executor) --------------------
# A cube occupies ~one grid unit in x and y. Two cubes on the same level collide
# when their centers are less than one unit apart in BOTH axes; exactly-adjacent
# (one unit apart) is allowed, and the epsilon keeps that boundary on the allowed
# side. A cube center must stay >= PLACE_MARGIN from every edge so it doesn't
# overhang, and the building area is GRID_UNITS x GRID_UNITS cells (matches
# the executor's own grid).
FOOTPRINT = 1.0
_EPS = 1e-6
PLACE_MARGIN = 0.5
GRID_UNITS = 6


def _is_number(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool)


class FakeRobot:
    """Pure-Python tracked world model with the RobotClient method surface.

    Authoritative state, exactly like RobotEnv in the tracked executor:
      - `self._held` : the cube currently in the gripper (or None), and
      - `self.placed`: cube_id -> (x, y, level) for every cube resting in the area.
    A cube that is neither held nor in `placed` is still at its storage slot
    ("table"). No simulator, no arm -- pick/place just update this model after the
    same validation the tracked executor runs before it moves.
    """

    def __init__(self, scenario=None):
        self._held = None
        self.placed = {}   # cube_id -> (x, y, level)
        self.scenario_id = 'empty'
        if scenario is not None:
            self.reset(scenario)

    # --- RobotClient-compatible surface (what the orchestrator calls) ---------

    def health(self):
        return {'ok': True, 'status': 'ready'}

    def get_state(self):
        return self.describe_state()

    def send_command(self, action, **args):
        """Send one command; returns feedback {ok, message|error, observation}."""
        cmd = {'action': action}
        cmd.update(args)
        return self.handle_command(cmd)

    def reset(self, scenario=None):
        """Return to the initial state, optionally with cubes already in the building area.

        The instant equivalent of reloading a simulator scene between eval cells -- so
        runs stay comparable at zero startup cost.

        `scenario` is a nesyplan.scenarios.Scenario (already validated as physically
        possible). Injecting its layout directly is the fake backend's privilege; a real
        command server has to be driven there with pick/place commands instead
        (nesyplan.scenarios.materialize).
        """
        self._held = None
        self.placed = {}
        self.scenario_id = 'empty'
        if scenario is not None:
            self.placed = dict(scenario.as_dict)
            self.scenario_id = scenario.id
        return self.describe_state()

    # --- world state / observations (mirrors RobotEnv.describe_state) ---------

    def describe_state(self):
        cubes = {}
        for cube_id in CUBES:
            info = dict(CUBE_INFO.get(cube_id, {}))  # color + number from the catalog
            if cube_id == self._held:
                info['location'] = 'held'
            elif cube_id in self.placed:
                x, y, level = self.placed[cube_id]
                info.update(location='area', x=x, y=y, level=level)
            else:
                info['location'] = 'table'
            cubes[cube_id] = info
        return {
            'ok': True,
            'mode': 'tracked',   # same advertisement the tracked executor makes
            'grid_units': GRID_UNITS,
            'held': self._held,
            'cubes': cubes,
        }

    def _feedback(self, ok, msg, **extra):
        payload = {'ok': ok, ('message' if ok else 'error'): msg,
                   'observation': self.describe_state()}
        payload.update(extra)
        return payload

    def _reject(self, diagnosis, remedy=''):
        """Feedback for a REJECTED action, carrying diagnosis and remedy separately.

        The executor always computes the full explanation; how much of it the model gets to
        read is the feedback-informativeness lever, and it is applied host-side in
        nesyplan.environment.concise_feedback (terse / why / fix). Keeping the split here
        and the choice there means one executor serves all three arms, and the arms differ
        in exactly one thing: the text in the tool result. The symbolic ok=False is
        identical, so an on_error policy fires the same way in every arm.

        `error` stays the full sentence so every existing reader (console, session log,
        the sim/real executors that only ever send one string) is unchanged.
        """
        full = f'{diagnosis} {remedy}'.strip() if remedy else diagnosis
        return {'ok': False, 'error': full,
                'error_parts': {'diagnosis': diagnosis, 'remedy': remedy},
                'observation': self.describe_state()}

    # --- world-model queries (pure; copied from the tracked executor) ---------

    def _cube_at(self, x, y, level, exclude=None):
        """Id of a placed cube whose footprint overlaps (x, y) on `level`, else None."""
        for cid, (ox, oy, oz) in self.placed.items():
            if cid == exclude:
                continue
            if oz == level and abs(ox - x) < FOOTPRINT - _EPS and abs(oy - y) < FOOTPRINT - _EPS:
                return cid
        return None

    def _cube_on_top(self, cube_id):
        """Id of a cube sitting directly on top of `cube_id` (one level up), else None."""
        if cube_id not in self.placed:
            return None
        x, y, level = self.placed[cube_id]
        return self._cube_at(x, y, level + 1, exclude=cube_id)

    def _axis_blocked(self, x, y, level, axis, exclude=None):
        """True if a placed cube sits one cell away along `axis` at the same level and
        in the same perpendicular lane -- where jaws opening along that axis would
        close on it. A purely diagonal neighbour does NOT block.
        """
        for cid, (ox, oy, oz) in self.placed.items():
            if cid == exclude or oz != level:
                continue
            dx, dy = abs(ox - x), abs(oy - y)
            if axis == 'x' and dy < FOOTPRINT - _EPS and 0.5 < dx < 1.5:
                return True
            if axis == 'y' and dx < FOOTPRINT - _EPS and 0.5 < dy < 1.5:
                return True
        return False

    def _grasp_orientation(self, x, y, level, exclude=None):
        """Gripper yaw for grasping/releasing at (x, y, level). Returns (rotate, error):
        (False, None) x-axis clear; (True, None) x blocked but y clear -> rotate 90 deg;
        (None, (diagnosis, remedy)) both axes blocked -> unreachable, caller must reject.
        """
        x_blocked = self._axis_blocked(x, y, level, 'x', exclude=exclude)
        if not x_blocked:
            return False, None
        y_blocked = self._axis_blocked(x, y, level, 'y', exclude=exclude)
        if not y_blocked:
            return True, None
        return None, (
            f'cannot reach ({x}, {y}) level {level}: neighbouring cubes sit on BOTH the '
            f'x-axis (at x-1 or x+1) and the y-axis (at y-1 or y+1). The gripper opens '
            f'its jaws along one axis and needs that axis clear, so it can neither grasp '
            f'nor release here.',
            'Clear one of those neighbours (pick it up and move it), '
            'or use a cell that has at least one free axis.')

    def _validate_place(self, cube_id, x, y, z):
        """(diagnosis, remedy) if placing here is invalid, else None.

        Every rejection is split into WHAT is wrong and WHAT TO DO about it, so the
        feedback-informativeness arms (why vs. fix) can be cut from the same executor --
        see _reject().
        """
        if not _is_number(x) or not _is_number(y):
            return (f'x and y must be numbers (grid units in [0, {GRID_UNITS}]).',
                    'Re-issue place with valid coordinates.')
        lo, hi = PLACE_MARGIN, GRID_UNITS - PLACE_MARGIN
        if not (lo <= x <= hi and lo <= y <= hi):
            return (f'({x}, {y}) is off the plate: a cube center must stay within '
                    f'{lo} .. {hi} on BOTH axes (closer than {PLACE_MARGIN} to an edge '
                    f'would overhang).',
                    f'Choose x and y in {lo} .. {hi}, then place again.')
        if isinstance(z, bool) or not _is_number(z) or float(z) != int(z) or int(z) < 0:
            return (f'z must be an integer stack level >= 0 (0 = on the table, 1 = on top of '
                    f'one cube, ...); got {z!r}.',
                    'Re-issue place with a valid level.')
        level = int(z)
        occupant = self._cube_at(x, y, level, exclude=cube_id)
        if occupant is not None:
            return (f'level {level} at ({x}, {y}) is already occupied by {occupant}.',
                    f'Choose a free cell, or stack on top of it at level {level + 1}. '
                    f'Then place again.')
        if level > 0 and self._cube_at(x, y, level - 1, exclude=cube_id) is None:
            return (f'level {level} at ({x}, {y}) has no cube directly below it '
                    f'(level {level - 1} is empty), so the cube would float.',
                    'Place it at level 0, or on top of an existing cube. Then place again.')
        return None

    # --- command dispatch (mirrors RobotEnv.handle_command) -------------------

    def handle_command(self, cmd):
        """Execute one command dict and return a feedback dict. Never raises."""
        action = str(cmd.get('action') or '').strip().lower()

        if action == 'shutdown':
            return self._feedback(True, 'shutting down command server')
        if action == 'done':
            return self._feedback(True, f'task marked done: {cmd.get("reason") or ""}', done=True)

        try:
            if action == 'pick':
                return self._pick(cmd)
            if action == 'place':
                return self._place(cmd)
            if action == 'store':
                return self._store(cmd)
            return self._reject(f'unknown action {action!r}.',
                                'Use "pick", "place", "store" or "done".')
        except Exception as exc:  # noqa: BLE001 -- parity with the executor: errors are feedback
            return self._reject(f'{type(exc).__name__}: {exc}')

    def _pick(self, cmd):
        cube_id = cmd.get('cube_id')
        if cube_id not in CUBES:
            return self._reject(f'unknown cube_id {cube_id!r}.',
                                f'Valid ids: {", ".join(CUBES)}.')
        if self._held is not None:
            return self._reject(f'already holding {self._held!r}.',
                                'Place it before picking another cube.')

        if cube_id in self.placed:
            # Resting in the building area -> grasp it from there, not storage.
            blocker = self._cube_on_top(cube_id)
            if blocker is not None:
                x, y, level = self.placed[cube_id]
                return self._reject(
                    f'{cube_id} is buried: {blocker} sits on top of it (level {level + 1}).',
                    f'Pick {blocker} and place it elsewhere first, then pick {cube_id}.')
            x, y, level = self.placed[cube_id]
            rotate, error = self._grasp_orientation(x, y, level, exclude=cube_id)
            if error is not None:   # boxed in on both axes -> unreachable, do not "move"
                return self._reject(*error)
            # (tracked executor: pick_from_area moves the arm; here just update state)
            self._held = cube_id
            self.placed.pop(cube_id, None)
            #note = ' (gripper turned 90 deg to clear an x-axis neighbour)' if rotate else ''
            return self._feedback(
                True, f'picked {cube_id} from the building area (grid {x}, {y}, level {level}). Next place it.')

        # Untouched -> still at its storage slot (base pick).
        self._held = cube_id
        return self._feedback(True, f'picked {cube_id} from storage. Next place it.')

    def _place(self, cmd):
        cube_id = cmd.get('cube_id')
        if cube_id not in CUBES:
            return self._reject(f'unknown cube_id {cube_id!r}.',
                                f'Valid ids: {", ".join(CUBES)}.')
        if self._held is None:
            return self._reject('not holding any cube.', 'Pick one before placing.')
        if self._held != cube_id:
            return self._reject(f'holding {self._held!r}, not {cube_id!r}.',
                                'Only the held cube can be placed.')
        for key in ('x', 'y', 'z'):
            if key not in cmd:
                return self._reject(f'place requires x, y and z (missing {key!r}).',
                                    'Re-issue place with all three arguments.')
        x, y, z = cmd['x'], cmd['y'], cmd['z']

        # Validate BEFORE "moving"; an invalid target is feedback, not motion.
        error = self._validate_place(cube_id, x, y, z)
        if error is not None:
            return self._reject(*error)

        level = int(z)
        rotate, error = self._grasp_orientation(x, y, level, exclude=cube_id)
        if error is not None:   # boxed in on both axes -> unreachable, do not "move"
            return self._reject(*error)
        # (tracked executor: api.place moves the arm; here just update state)
        self._held = None
        self.placed[cube_id] = (float(x), float(y), level)
        #note = ' (gripper turned 90 deg to clear an x-axis neighbour)' if rotate else ''
        return self._feedback(True, f'placed {cube_id} at grid ({x}, {y}) level {level}. Next pick another cube.')

    def _store(self, cmd):
        """Return a cube to storage -- pure-Python mirror of RobotEnv._store (agent_env_tracked).

        Same validation (held / holding-another / buried / already-stored); no motion, just
        drop it out of `placed` / the gripper so its location goes back to 'table'.
        """
        cube_id = cmd.get('cube_id')
        if cube_id not in CUBES:
            return self._reject(f'unknown cube_id {cube_id!r}.',
                                f'Valid ids: {", ".join(CUBES)}.')

        if self._held == cube_id:
            self._held = None
            self.placed.pop(cube_id, None)
            return self._feedback(True, f'returned {cube_id} to storage')

        if self._held is not None:
            return self._reject(f'holding {self._held!r}.',
                                f'Put it down (place or store it) before storing {cube_id!r}.')

        if cube_id not in self.placed:
            return self._feedback(True, f'{cube_id} is already in storage; nothing to do')

        blocker = self._cube_on_top(cube_id)
        if blocker is not None:
            x, y, level = self.placed[cube_id]
            return self._reject(
                f'{cube_id} is buried: {blocker} sits on top of it (level {level + 1}).',
                f'Store or move {blocker} first, then {cube_id}.')
        x, y, level = self.placed[cube_id]
        rotate, error = self._grasp_orientation(x, y, level, exclude=cube_id)
        if error is not None:   # boxed in on both axes -> unreachable, do not "move"
            return self._reject(*error)
        self.placed.pop(cube_id, None)
        return self._feedback(
            True, f'returned {cube_id} to storage (was in the building area at grid {x}, {y}, level {level})')
