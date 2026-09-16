"""Single source of truth for every system prompt in the harness.

Two modes are assembled from small composable blocks:

  - agentic modes (ALWAYS / REASON_FIRST / PERIODIC / SELF_TRIGGERED): "one tool
    call per turn, react to feedback" framing. (SELF_TRIGGERED's think-tool mechanics
    live in the think() tool description itself, nesyplan/tools.py, not here -- the
    API injects the tool schema into the model's context, so there is nothing to add.)
    Leaner than the one-shot mode: the coordinate system and every argument's range
    live in the tool schemas (nesyplan/tools.py) and every invalid action returns an
    explanatory error, so the prompt does not restate what the schema says or what
    feedback would teach (pick-before-place, one-held-at-a-time, in-bounds). It DOES
    keep the geometry rules -- they guide planning and (the straddle technique
    especially) are not something per-action feedback would ever suggest.
  - one-shot (ONESHOT): "emit the complete ordered plan as JSON, no feedback"
    framing. Called WITHOUT tools and executed blind, so its prompt is
    self-contained: it describes the pick/place actions and the coordinate system
    inline (no tool schema) and carries the same geometry rules (no feedback to
    recover from a mistake).

The geometry rules (RULES_BLOCK, incl. FREE_AXIS_RULE) are shared by both modes so
they never drift -- the one-shot planner and the agentic loop describe the same world.
"""

from nesyplan.config import Policy
from nesyplan.environment import catalog_block, format_state, grid_units


# The one physical constraint that is genuinely non-obvious and expensive to
# rediscover by trial, so it is stated proactively (part of RULES_BLOCK below).
FREE_AXIS_RULE = (
    'The gripper needs ONE clear axis to open its jaws: a cell that has a cube on an '
    'x-neighbour (x-1 or x+1) AND on a y-neighbour (y-1 or y+1) at the same level cannot '
    'be grasped or released there. Keep at least one axis open when placing cubes side by '
    'side, or clear a neighbour first.'
)


# The geometry the model must respect, shared verbatim by both modes so they never
# drift. Agentic keeps it because it guides planning (feedback rejects a bad placement
# but never suggests the fractional-straddle technique); one-shot needs it because it
# plans blind with no feedback at all.
RULES_BLOCK = f"""  - Support: a cube at level z>0 must rest on the level below -- some cube at level z-1 whose
    footprint overlaps it (its center within 1.0 unit in BOTH x and y). A cube with nothing
    under it would float and fails.
  - To stack straight up, reuse the same (x, y) with increasing z (0, then 1, then 2). To
    place cubes side by side on one level, keep their centers at least 1.0 unit apart. To
    straddle two lower cubes, centre the upper one between them with a fractional coordinate:
    level-0 cubes at x=2 and x=3 support a level-1 cube at x=2.5 (same y).
  - {FREE_AXIS_RULE}
  - If the task is ambiguous, choose one reasonable interpretation and proceed; do not
    deliberate at length over alternatives.
  - Only move the cubes the task requires; leave the others where they are."""


# Appended when config.content_plan is set: make the model keep a durable plan in
# `content`. This is the "persistence via content" lever -- content survives in the
# re-sent history (unlike the reasoning trace), so it acts as cross-turn memory the
# model can follow instead of re-deriving its plan every turn.
CONTENT_PLAN_INSTRUCTION = """
Keep a running plan in your MESSAGE CONTENT. On EVERY turn, in the assistant message
content (the normal text, NOT the tool call), write a brief running plan in at most
three short lines -- GOAL, DONE SO FAR, NEXT -- then make exactly one tool call. Always
fill the content, even when you call a tool: it is your durable memory across turns, so
you do not have to re-derive the whole plan each time."""

CONTENT_PLAN_INSTRUCTION_2 = """
On EVERY turn, write a brief running plan in at most
three short lines -- GOAL, DONE SO FAR, NEXT --"""



# Injected as a user message on scheduled reflection turns (PERIODIC).
REFLECTION_INSTRUCTION = (
    'Reflection checkpoint: review your actions so far and the latest feedback. '
    'Is your approach still valid and on track to satisfy the task? Adjust your plan '
    'if needed, then make your next tool call.'
)


def _agentic_prompt(state, config):
    grid = grid_units(state)

    extra = ''
    if config.content_plan:
        extra += '\n' + CONTENT_PLAN_INSTRUCTION_2

    return f"""You are NeSyPlan, an agent controlling a robot arm that picks up cubes and places them onto a \
{grid}x{grid} building-areas. Next to the building area is the storage area. Accomplish the user's task by building a structure using the provided tools. Only call one tool per turn. You can arrange cubes on the x, y so the user sees them from above, and also stack them on the z axis, so the user sees from the front.

Only build something if the user asks for it. Otherwise just have a chat with the user and try to get a task from them.

Available cubes (use these exact ids; each cube's color and number are given):
{catalog_block(state)}

{extra}"""


def _oneshot_prompt(state, config):
    grid = grid_units(state)
    center = grid / 2
    hi = grid - 0.5   # cube centers must stay this far from every edge (PLACE_MARGIN)
    # store() is only offered when the driver enables it (the demo); the eval's one-shot
    # baseline keeps the pick/place-only action space frozen. Mentioned inline because
    # one-shot has no tool schema to advertise it.
    store_line = (
        '\n  - store(cube_id): return a cube to its storage slot'
        if config.allow_store else '')
    return f"""You program a robot arm that picks up cubes and places them onto a \
{grid}x{grid} building-area grid. Next to the building area is the storage area.
You translate the user's task into a COMPLETE, \
ordered plan; it is then executed.

Available cubes (use these exact ids; each cube's color and number are given):
{catalog_block(state)}

Current world state:
{format_state(state)}

Actions and coordinates:
  - pick(cube_id): Grasp the named cube from wherever it currently is. Only one cube can be held at a time; place it before picking another.
  - place(cube_id, x, y, z): Place the currently held cube into the building area at coordinate (x, y) on stack level z.
  {store_line}

  Every place must follow a pick of the SAME cube. x and y are positions in cube widths in [0, {grid:g}] and MAY be fractional (e.g. 2.5); 
  keep centers within 0.5 .. {hi:g} so a cube stays on the plate. z is the integer stack level: 0 on the table, 1 on top of one cube, 2 on top of that, and so on.

Output the complete ordered plan as a JSON array of steps and NOTHING else -- no prose, no
comments, no markdown fences. Each step is an object with "name" ("pick" or "place") and
"arguments":
  [{{"name": "pick",  "arguments": {{"cube_id": "cube_yellow"}}}},
   {{"name": "place", "arguments": {{"cube_id": "cube_yellow", "x": 3, "y": 3, "z": 0}}}},
   {{"name": "store",  "arguments": {{"cube_id": "cube_red"}}}},
    ...]
   ]"""


def build_system_prompt(state, config):
    """The system prompt for this episode, chosen by policy.

    Agentic prompts are lean (the tool schemas + per-action feedback carry the
    constraints); the one-shot prompt is self-contained (no tools, no feedback).

    Where the mutable world layout goes differs by mode:
      - AGENTIC: kept OUT of the system prompt (it would go stale after the first
        action and sit at the top of the context forever) -- delivered once as the
        seed user message and refreshed via get_state.
      - ONE-SHOT: baked directly into the system prompt. The whole plan is emitted in
        a single shot and executed blind, so there is no "after the first action" for
        it to go stale against, and the planner must see the actual starting layout
        (cubes may already be placed on a continue-from-current run).
    """
    if config.policy == Policy.ONESHOT:
        return _oneshot_prompt(state, config)
    return _agentic_prompt(state, config)
