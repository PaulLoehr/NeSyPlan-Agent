"""Task registry for the eval harness -- prompts only; scoring is the LLM judge.

Every task is scored by nesyplan/judge.py (a semantic pass/fail from the task text + the
final /state). There are no scripted checkers: one scorer for everything. That is what
lets open-ended tasks with no single right answer ("build a structure that defends the
red block") be evaluated at all, and lets `eval.py --task-prompt "<anything>"` run with
zero code.

Honest note (does not change the design, just how to read results): the judge reasons
over the tracked executor's symbolic /state, which is ground truth in sim (exact
coordinates, no perception gap). It is strong on spatial/structural goals and less
reliable on precise arithmetic (largest_div4) or off-by-one logic (tower_no_adjacent) --
weigh those verdicts accordingly, e.g. by spot-checking their PNGs.

State shape (as returned by any tracked executor):
  state['cubes'][id] = {color, number, location, x, y, level}
  location in {'table'(=storage), 'area', 'held'}; x,y (floats) and level (int) present
  when location=='area'. Cubes: red(1) green(2) blue(3) yellow(4) black(5) white(6).

Feasibility (executor rules): x,y may be fractional in [0.5, 5.5]; a cube at level z is
"supported" if any cube one level down overlaps its footprint (<1.0 in both axes), so an
offset pyramid row resting across two lower cubes is valid; two cubes on one level must
be >=1.0 apart; the gripper auto-rotates 90 deg for side-by-side neighbours.
"""

from dataclasses import dataclass

from nesyplan.checkers import levels_check, stack_check


# --- state helpers (imported by nesyplan/eval.py for the evaluation record) ---

def _area(state):
    """{cube_id: info} for cubes currently placed in the building area (with x/y/level)."""
    cubes = (state or {}).get('cubes') or {}
    out = {}
    for cid, info in cubes.items():
        info = info or {}
        if info.get('location') == 'area' and info.get('x') is not None and info.get('y') is not None:
            out[cid] = info
    return out


def _layout(area):
    """Compact one-line description of where every placed cube sits (for detail strings)."""
    if not area:
        return 'building area empty'
    parts = [f'{cid}@({i["x"]:g},{i["y"]:g},L{i.get("level")})' for cid, i in sorted(area.items())]
    return ', '.join(parts)


# --- registry ----------------------------------------------------------------

@dataclass
class Task:
    id: str
    difficulty: str    # rough rung on the ladder ('--' for an ad-hoc --task-prompt)
    prompt: str        # fed to the orchestrator verbatim; scored by the LLM judge
    scenario: str = ''  # initial layout id (nesyplan/scenarios.py); '' = the empty start,
                        # i.e. all six cubes in storage. A scenario means the episode
                        # INHERITS a configuration it did not build -- see scenarios.py.
    check: object = None  # optional callable(state) -> (success, detail) from
                        # nesyplan/checkers.py. When present it is the PRIMARY scorer and
                        # the judge becomes a second opinion (agreement is recorded).


TASKS = [
    # Knowledge + ordering, NO hints about colors or order (the model must know the flag).
    Task('german_flag', 'L0',
         'Show me how the German flag looks by stacking the blocks in the correct order of colors.'),
    # Open-ended construction: a true 3-2-1 shape needs fractional x (offset middle row).
    Task('pyramid', 'L1',
         'Build a pyramid: three blocks on the bottom, two resting on top of them, and '
         'one on top of those.'),
    # Qualitative goal ("could climb").
    Task('staircase', 'L2',
         'Build a staircase with three steps that a tiny toy figure could climb from '
         'left to right.'),
    # Precise reasoning: largest permutation of the digits divisible by 4 (1..6 -> 654312).
    Task('largest_div4', 'L3',
         'Arrange all the blocks in a row so that the six-digit number they form (read '
         'left to right) is as large as possible AND divisible by 4.'),
    # Precise logic: no vertically adjacent numbers differing by exactly 1.
    Task('tower_no_adjacent', 'L4',
         'Build a tower of 5 blocks in which no two adjacent blocks have numbers that '
         'differ by exactly 1.'),
    # Genuinely open-ended -- no single correct answer.
    Task('defend_red', 'L5',
         'Build a structure that defends the red block.'),

    # --- one-shot-hard tier (F = needs feedback). Each has a valid final state, but the
    #     gripper's free-axis rule makes placement ORDER matter: a blind one-shot plan
    #     that orders wrong strands a cell that is boxed in on both axes (unreachable),
    #     while an agentic loop reads the "both axes blocked" error and reorders / re-picks
    #     a neighbour to recover. See fake_robot._grasp_orientation.
    #   plus_cross: the centre is unreachable once BOTH an x-arm and a y-arm are down, so a
    #     naive "arms first, centre last" plan fails; centre must go before it gets boxed.
    Task('plus_cross', 'F1',
         'Build a plus (+) sign flat on the table. Place the centre cube last.'),
    #   t_shape: the junction cube (where the crossbar's centre meets the stem) gains both
    #     an x-neighbour (the crossbar) and a y-neighbour (the stem), so a plan that fills
    #     the arms before the junction boxes it in on both axes; the junction must go first.
    Task('t_shape', 'F2',
         'Write the letter T with the blocks.'),

    # --- rebuild tier (R = the world was already like this) -----------------------------
    #   Every task below starts from an INHERITED layout (nesyplan/scenarios.py) instead of
    #   the empty plate, so the goal cannot be reached by construction alone -- something
    #   has to be taken apart first. All four are formally checkable (nesyplan/checkers.py),
    #   so they are scored by code and the judge only corroborates.
    #
    #   The flag prompt pins the axis convention on purpose ("read the tower top to bottom
    #   like the flag top to bottom"). The knowledge part stays (which three colours, which
    #   order); what is removed is the stripe-onto-stack AMBIGUITY that made the legacy
    #   german_flag task unscoreable -- the judge has failed correct towers over it.
    #
    #   R1: the cube the goal needs sits at the BOTTOM of a stack -> two removals and two
    #     parking decisions happen before the first goal step.
    Task('flag_excavate', 'R1',
         'Build the German flag as a vertical tower of three blocks: read the tower from '
         'top to bottom like you read the flag from top to bottom.',
         scenario='buried_black',
         check=stack_check(['yellow', 'red', 'black'], name='flag tower (yellow-red-black bottom-up)')),
    #   R2: the goal cube is BOXED IN on both grasp axes -> pick() on it is rejected
    #     outright, so any plan whose first action is the obvious one fails immediately.
    Task('unbox_red', 'R2',
         'Stack three blocks into a tower: blue at the bottom, green in the middle, red on top.',
         scenario='boxed_red',
         check=stack_check(['blue', 'green', 'red'])),
    #   R3: the goal is ALREADY BUILT, wrong. Every goal cell is occupied by a cube that has
    #     to end up elsewhere, so the tower must be fully dismantled and rebuilt (>=8 actions
    #     with three parking spots) -- the hardest thing here for a one-pass plan.
    Task('flag_repair', 'R3',
         'The three German-flag colours are already stacked in the building area, but in the '
         'wrong order. Rebuild that tower so it shows the flag correctly: read the tower from '
         'top to bottom like you read the flag from top to bottom.',
         scenario='flag_inverted',
         check=stack_check(['yellow', 'red', 'black'], name='flag tower (yellow-red-black bottom-up)')),
    #   R4: clutter, not blockage -- the cubes a six-cube structure needs are already on the
    #     plate in the wrong places, so they must be re-used or relocated.
    Task('pyramid_rebuild', 'R4',
         'Build a pyramid out of all six blocks: three on the bottom, two resting on top of '
         'them, and one on top of those.',
         scenario='clutter_diagonal',
         check=levels_check({0: 3, 1: 2, 2: 1}, require_all=True, name='3-2-1 pyramid')),
]

# Named task sets, so a campaign is one flag instead of a list to retype.
#   @rebuild  the R tier: inherited layouts, code-scored -- the discriminating set.
#   @clean    the legacy empty-plate tasks the earlier campaigns used.
TASK_GROUPS = {
    '@rebuild': ['flag_excavate', 'unbox_red', 'flag_repair', 'pyramid_rebuild'],
    '@clean': ['german_flag', 'pyramid', 'plus_cross', 't_shape'],
}

TASKS_BY_ID = {t.id: t for t in TASKS}


def get_tasks(ids=None):
    """Resolve task ids (in the given order) -- or all tasks when ids is falsy.

    An id starting with '@' expands to a named group (TASK_GROUPS), so
    `--tasks @rebuild` selects the whole rebuild tier.
    """
    if not ids:
        return list(TASKS)
    expanded = []
    for i in ids:
        expanded.extend(TASK_GROUPS[i] if i in TASK_GROUPS else [i])
    missing = [i for i in expanded if i not in TASKS_BY_ID]
    if missing:
        raise KeyError(f'unknown task id(s): {missing}. Known: {list(TASKS_BY_ID)} '
                       f'or a group: {list(TASK_GROUPS)}')
    return [TASKS_BY_ID[i] for i in expanded]
