"""Named initial world layouts -- the "the world was already like this" axis.

Until now every episode started identically: all six cubes in storage, building area
empty. That start is exactly what keeps blind one-shot planning competitive -- the
planner only ever reasons about a world *it* creates, step by step, from nothing. Its
plan can be written in one pass because every precondition is trivially true at the
moment it matters.

A scenario breaks that. Cubes are already in the building area, so the task begins with
an inherited configuration the model must reason *about* rather than construct:

  - a needed cube is BURIED (something rests on top of it)      -> must be excavated
  - a needed cube is BOXED IN on both grasp axes                -> literally unreachable
    until a neighbour is moved (see fake_robot._grasp_orientation)
  - the goal cells are OCCUPIED by the wrong cubes              -> must be dismantled
  - removed cubes need PARKING SPOTS that don't block later steps

None of that is hidden information -- both prompt modes see the full layout (the one-shot
bakes it into its system prompt, the agentic loop gets it in the seed message). The
difficulty is not observation, it is that a correct plan now requires simulating
*intermediate* states: after I lift this cube, what is reachable, what did I just block,
where did I put the thing I took off? That is where a one-pass plan breaks and per-action
feedback pays for itself.

`placed` is an ordered BUILD LIST, not a dict, for two reasons: it documents a feasible
construction order (lower levels first; a cell that will be boxed in must be filled before
its neighbours), and materialize() replays exactly that order to build the layout on a
real/simulated arm.

Usage:
    from nesyplan.scenarios import get_scenario, materialize
    sc = get_scenario('boxed_red')
    robot.reset(sc)                 # FakeRobot: instant
    materialize(robot, sc)          # sim/real: actually build it with the arm
"""

from dataclasses import dataclass, field

FOOTPRINT = 1.0
_EPS = 1e-6


@dataclass
class Scenario:
    """One initial layout: cubes already resting in the building area."""
    id: str
    summary: str                       # one line, recorded in the results row + manifest
    placed: list = field(default_factory=list)   # ordered [(cube_id, x, y, level)], build order

    @property
    def as_dict(self):
        """{cube_id: (x, y, level)} -- the shape FakeRobot.placed expects."""
        return {cid: (float(x), float(y), int(z)) for cid, x, y, z in self.placed}

    def validate(self):
        """Fail loudly on a physically impossible layout (a typo, not a runtime condition).

        Checks the same three invariants the executor enforces: no two cubes overlapping on
        one level, every cube above level 0 supported from below, and every cube placeable in
        the given order (the cell it goes into must have a free grasp axis at that moment).
        """
        done = []   # [(cid, x, y, z)] already standing, in build order
        for cid, x, y, z in self.placed:
            for ocid, ox, oy, oz in done:
                if oz == z and abs(ox - x) < FOOTPRINT - _EPS and abs(oy - y) < FOOTPRINT - _EPS:
                    raise ValueError(f'scenario {self.id}: {cid} at ({x},{y},{z}) overlaps {ocid}')
            if z > 0 and not any(oz == z - 1 and abs(ox - x) < FOOTPRINT - _EPS
                                 and abs(oy - y) < FOOTPRINT - _EPS
                                 for ocid, ox, oy, oz in done):
                raise ValueError(f'scenario {self.id}: {cid} at ({x},{y},{z}) would float')
            x_blocked = any(oz == z and abs(oy - y) < FOOTPRINT - _EPS and 0.5 < abs(ox - x) < 1.5
                            for _, ox, oy, oz in done)
            y_blocked = any(oz == z and abs(ox - x) < FOOTPRINT - _EPS and 0.5 < abs(oy - y) < 1.5
                            for _, ox, oy, oz in done)
            if x_blocked and y_blocked:
                raise ValueError(
                    f'scenario {self.id}: {cid} at ({x},{y},{z}) is unplaceable in this order '
                    f'(both grasp axes blocked when its turn comes) -- move it earlier in `placed`')
            done.append((cid, x, y, z))
        return self


# --- the catalog ---------------------------------------------------------------
# Cube numbers, for reading the layouts: red 1, green 2, blue 3, yellow 4, black 5, white 6.

SCENARIOS = [
    Scenario('empty', 'building area empty, all six cubes in storage (the legacy start)'),

    # A needed cube at the BOTTOM of a 3-stack. The flag needs black; black is under two
    # cubes, so two removals (and two parking decisions) come before the first flag step.
    Scenario('buried_black',
             'black is at the bottom of a 3-stack at (2,2) under white and green',
             [('cube_black', 2, 2, 0), ('cube_white', 2, 2, 1), ('cube_green', 2, 2, 2)]),

    # A needed cube BOXED IN on both axes: a plus sign whose centre cannot be grasped at
    # all. pick(cube_red) is rejected outright, so any plan that starts with the obvious
    # target fails on its first action.
    #
    # Verified subtlety: freeing red takes a WHOLE axis, not one neighbour -- moving green
    # still leaves blue blocking x. So the recovery is "clear both x-neighbours (green and
    # blue) or both y-neighbours (white and black)", which is a real inference rather than
    # a single retry. The executor's generic remedy text ("clear one of those neighbours")
    # is a hint here, not a recipe -- deliberately, since a model that follows it literally
    # gets a second rejection and has to reason further.
    Scenario('boxed_red',
             'red sits in the centre of a plus: boxed in on BOTH grasp axes, and freeing it '
             'needs a whole axis cleared (green+blue on x, or white+black on y)',
             [('cube_red', 3, 3, 0),
              ('cube_green', 2, 3, 0), ('cube_blue', 4, 3, 0),
              ('cube_white', 3, 2, 0), ('cube_black', 3, 4, 0)]),

    # The goal itself is already built -- WRONG. Every goal cell is occupied by the very
    # cube that has to end up somewhere else, so the tower must be fully dismantled (with
    # parking) before it can be rebuilt. Minimum 8 actions.
    Scenario('flag_inverted',
             'the three flag colours are already stacked at (3,3) but upside down '
             '(black at the bottom, yellow on top)',
             [('cube_black', 3, 3, 0), ('cube_red', 3, 3, 1), ('cube_yellow', 3, 3, 2)]),

    # Clutter across the middle of the plate: nothing is unreachable, but the cubes a
    # 6-cube structure needs are scattered and the obvious footprint is taken.
    Scenario('clutter_diagonal',
             'green, white and blue lie scattered on the diagonal across the build zone',
             [('cube_green', 2, 2, 0), ('cube_white', 3, 3, 0), ('cube_blue', 4, 4, 0)]),
]

SCENARIOS_BY_ID = {s.id: s.validate() for s in SCENARIOS}
EMPTY = SCENARIOS_BY_ID['empty']


def get_scenario(scenario_id):
    """Resolve a scenario id; '' / None / 'empty' all give the empty start."""
    if not scenario_id:
        return EMPTY
    if scenario_id not in SCENARIOS_BY_ID:
        raise KeyError(f'unknown scenario {scenario_id!r}. Known: {list(SCENARIOS_BY_ID)}')
    return SCENARIOS_BY_ID[scenario_id]


def materialize(robot, scenario):
    """Build `scenario` on a backend that has no state injection (sim / real).

    FakeRobot takes the layout directly (robot.reset(scenario)); a command server behind
    :8100 owns its own world, so the only way to reach the layout is to actually pick and
    place the cubes -- which on target=real means the arm physically builds the start
    state. Replays `placed` in build order, which validate() has proven feasible.

    Raises RuntimeError on the first rejected command: a half-built start state would
    silently make the whole cell incomparable, which is worse than failing the run.
    """
    for cid, x, y, z in scenario.placed:
        for action, args in (('pick', {'cube_id': cid}),
                             ('place', {'cube_id': cid, 'x': x, 'y': y, 'z': z})):
            fb = robot.send_command(action, **args) or {}
            if not fb.get('ok'):
                raise RuntimeError(
                    f'could not materialize scenario {scenario.id!r}: '
                    f'{action}({args}) -> {fb.get("error")}')
    return robot.get_state()
