"""Invocation policy: decide, per turn, whether reasoning is enabled and why.

Implements the proposal's should_reason table plus the combinable hard trigger
on_failure. The pre_done check and the escalation budget are enforced in the
orchestrator loop (they intercept done()/think()), and surface here only via the
`forced_trigger` override the loop passes in (e.g. the pre_done retry).

    ALWAYS:          always
    REASON_FIRST:    turn 1 only
    PERIODIC(N):     turn 1 or turn % N == 0
    SELF_TRIGGERED:  turn 1 or the model asked (escalate)
    ON_ERROR:        turn 1 or whenever the last tool result failed (failure trigger baked in)
    (any agentic):   + on_failure hard trigger if the last tool result failed

The loop consumes a fired hard_failure (sets last_feedback=None) so it triggers
once per error occurrence; a NEW failed tool result re-arms it.
"""

from nesyplan.config import Policy


def should_reason(config, turn, escalate, last_feedback, forced_trigger=None):
    """Return (reasoning: bool, trigger: str|None) for this turn.

    trigger is one of: initial | scheduled | self | hard_failure | pre_done.
    """
    if forced_trigger:                       # e.g. the pre_done retry the loop forces
        return True, forced_trigger

    if turn == 1 and config.reason_on_first:
        return True, 'initial'

    if config.policy == Policy.ALWAYS:
        return True, 'scheduled'

    # Failure trigger: a combinable flag on any policy, and baked into ON_ERROR.
    failure_trigger = config.on_failure_trigger or config.policy == Policy.ON_ERROR
    if failure_trigger and last_feedback is not None and not last_feedback.get('ok', True):
        return True, 'hard_failure'

    if config.policy == Policy.PERIODIC:
        n = config.periodic_n or 1
        if turn % n == 0:
            return True, 'scheduled'
    elif config.policy == Policy.SELF_TRIGGERED:
        if escalate:
            return True, 'self'

    # REASON_FIRST and the off-turns of the others fall through to reasoning-off.
    return False, None
