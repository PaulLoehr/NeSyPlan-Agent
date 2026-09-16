"""Single source of truth for the OpenAI-style tool schemas the agent may call.

The base four (get_state / pick / place / done) are grid-parameterized rather than
hardcoded to [0, 6], so the tool descriptions cannot drift from the prompt. The think()
tool is appended only when the loop exposes it (SELF_TRIGGERED, on a reasoning-off turn,
within the escalation budget).
"""


def _base_tools(cube_ids, grid):
    center = grid / 2
    hi = grid - 0.5
    return [
        {'type': 'function', 'function': {
            'name': 'get_state',
            'description': 'Look up the CURRENT world state: which cubes are in storage, which '
                           'are placed in the building area (with their grid x, y and stack level), '
                           'and which cube (if any) you are holding. Does not move the robot. Call '
                           'it whenever you are unsure what is where.',
            'parameters': {'type': 'object', 'properties': {}}}},
        {'type': 'function', 'function': {
            'name': 'pick',
            'description': 'Grasp the named cube from wherever it currently is. '
                           'Only one cube can be held at a time; place it before picking another.',
            'parameters': {'type': 'object', 'properties': {
                'cube_id': {'type': 'string', 'enum': cube_ids,
                            'description': 'Id of the cube to grasp.'}},
                'required': ['cube_id']}}},
        {'type': 'function', 'function': {
            'name': 'place',
            'description': 'Place the currently held cube into the building area at '
                           'coordinate (x, y) on stack level z.',
            'parameters': {'type': 'object', 'properties': {
                'cube_id': {'type': 'string', 'enum': cube_ids,
                            'description': 'Id of the held cube being placed.'},
                'x': {'type': 'number',
                      'description': f'Grid x in [0, {grid:g}], may be fractional (e.g. 2.5); {center:g} is '
                                     f'the center, keep it in 0.5 .. {hi:g} to stay on the plate.'},
                'y': {'type': 'number',
                      'description': f'Grid y in [0, {grid:g}], may be fractional (e.g. 3.5); {center:g} is '
                                     f'the center, keep it in 0.5 .. {hi:g} to stay on the plate.'},
                'z': {'type': 'integer',
                      'description': 'Stack level: 0 on the table, 1 on top of one cube, etc.'}},
                'required': ['cube_id', 'x', 'y', 'z']}}},
        {'type': 'function', 'function': {
            'name': 'done',
            'description': 'Call this once the task is fully accomplished. Ends the session.',
            'parameters': {'type': 'object', 'properties': {
                'reason': {'type': 'string', 'description': 'Describe what you build and how this fulfills the user\'s request.'}},
                'required': []}}},
    ]


def _store_tool(cube_ids):
    return {'type': 'function', 'function': {
        'name': 'store',
        'description': 'Return the named cube to its storage slot, removing it from the building area ',
        'parameters': {'type': 'object', 'properties': {
            'cube_id': {'type': 'string', 'enum': cube_ids,
                        'description': 'Id of the cube to send back to storage.'}},
            'required': ['cube_id']}}}


THINK_TOOL = {'type': 'function', 'function': {
    'name': 'think',
    'description': 'Request a deliberation step BEFORE acting. By default you act WITHOUT '
                   'deliberation, one tool call per turn; call this when an action failed '
                   'unexpectedly, when feedback contradicts your expectation, when you are '
                   'unsure about the next step, or before calling done(). Calling think() is '
                   'EXCLUSIVE -- on that turn you do NOT also act -- and it grants you exactly '
                   'one reasoning-enabled turn next.',
    'parameters': {'type': 'object', 'properties': {
        'reason': {'type': 'string',
                   'description': 'Why you need to deliberate now.'}},
        'required': ['reason']}}}


def build_tools(cube_ids, grid, include_think=False, include_store=False):
    """The tool list for one request.

    include_think appends think() (SELF_TRIGGERED); include_store appends store()
    (return a cube to storage -- the demo enables it, the eval leaves it off so the
    baseline action space stays frozen).
    """
    tools = _base_tools(cube_ids, grid)
    if include_store:
        tools.append(_store_tool(cube_ids))
    if include_think:
        tools.append(THINK_TOOL)
    return tools
