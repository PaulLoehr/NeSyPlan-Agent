"""Configuration types for the reasoning-orchestration harness.

Every proposal mode is a point in (policy x cache) space plus a few knobs. An
EpisodeConfig fully describes one experimental run and is serialized verbatim into
the session log's `config` block so runs are self-describing and comparable.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Optional

from nesyplan.environment import FEEDBACK_LEVELS


class Policy(str, Enum):
    """Dimension 1 -- invocation policy (when is reasoning enabled?)."""
    ONESHOT = 'oneshot'              # non-agentic: one planning call, execute blindly
    ALWAYS = 'always'               # ReAct: reason every turn
    REASON_FIRST = 'reason_first'   # reason on turn 1 only, then feedback-only
    PERIODIC = 'periodic'           # reason on turn 1 and every Nth turn
    SELF_TRIGGERED = 'self_triggered'  # reason on turn 1 + turns the model requests via think()
    ON_ERROR = 'on_error'           # reason on turn 1 + whenever the env returns an error
                                    # (the on_failure hard trigger, baked in as a first-class mode)


class CacheMode(str, Enum):
    """Dimension 2 -- reasoning cache (how past reasoning persists)."""
    NONE = 'none'        # C0: traces discarded (content left as the model produced it)
    RAW = 'raw'          # C1: verbatim trace written into the assistant content
    SUMMARY = 'summary'  # C2: distilled trace essence written into the assistant content


@dataclass
class EpisodeConfig:
    policy: Policy = Policy.ALWAYS
    cache_mode: CacheMode = CacheMode.NONE
    periodic_n: Optional[int] = None     # required for PERIODIC (reason when turn % n == 0)
    context_k: Optional[int] = None      # None = full history; int = keep last K assistant turns + tails
    strip_content: bool = False          # drop assistant `content` from re-sent history (ablation: kill content-as-memory)
    content_plan: bool = False           # prompt the model to keep a running GOAL/DONE/NEXT plan in `content` each turn
    feedback_level: str = 'fix'          # how much of a rejection the model reads: 'terse' (only that it
                                         # failed) | 'why' (+ the violated precondition) | 'fix' (+ what to
                                         # do about it = the executor's full message, shipped behaviour).
                                         # Dimension C: what the symbolic layer gives back. See
                                         # nesyplan/environment.py:FEEDBACK_LEVELS.
    on_failure_trigger: bool = False     # hard trigger: force reasoning after any failed tool result
    pre_done_check: bool = False         # reject the first reasoning-off done(); retry runs reasoning-on
    escalation_budget: Optional[int] = None  # cap on granted think() calls (SELF_TRIGGERED)
    reason_on_first: bool = True         # force reasoning on turn 1 (ONESHOT: the single call).
                                         # False -> no forced initial reasoning: gives the
                                         # no-reasoning baselines and self_triggered-from-cold.
    reasoning_on_effort: Optional[str] = None  # effort on a reasoning-ON turn; None = omit (model's own default level)
    allow_store: bool = False            # expose the store() tool (return a cube to storage). Off in the
                                         # eval/ReAct baseline (keeps the action space frozen); the demo turns it on.
    allow_chat: bool = False             # accept a no-tool-call prose reply as a conversational turn instead of
                                         # nudging back into the loop (the demo turns it on: NeSyPlan may greet /
                                         # answer / ask for a task without acting, and leaves the loop without done()).
    max_no_tool_calls: Optional[int] = None  # give up after N consecutive turns where the model TALKED instead of
                                         # calling a tool (mid-build; a pre-action reply is handled by allow_chat).
                                         # None = nudge indefinitely until max_steps -- the eval baseline, kept so
                                         # past campaigns stay reproducible. The demo sets 1: a model that narrates
                                         # instead of acting ends the task rather than burning the whole step budget.
    max_steps: int = 25
    temperature: Optional[float] = None  # None = the model's recommended sampling (see nesyplan/llm.py)
    top_p: Optional[float] = None        # None = recommended/omit; set for logging the value actually used
    seed: Optional[int] = None           # recorded for reproducibility bookkeeping (not a sampler seed)

    def validate(self):
        if self.policy == Policy.PERIODIC and not (self.periodic_n and self.periodic_n >= 1):
            raise ValueError('policy=periodic requires --n >= 1')
        if self.feedback_level not in FEEDBACK_LEVELS:
            raise ValueError(f'feedback_level must be one of {FEEDBACK_LEVELS}, '
                             f'got {self.feedback_level!r}')
        if self.context_k is not None and self.context_k < 1:
            raise ValueError('--context-k must be >= 1 (omit for full history)')
        if self.escalation_budget is not None and self.escalation_budget < 0:
            raise ValueError('--escalation-budget must be >= 0')
        return self

    def to_dict(self):
        """Plain, JSON-serializable view for the session log's `config` block."""
        return {
            'policy': self.policy.value,
            'cache_mode': self.cache_mode.value,
            'periodic_n': self.periodic_n,
            'context_k': self.context_k,
            'strip_content': self.strip_content,
            'content_plan': self.content_plan,
            'feedback_level': self.feedback_level,
            'hard_triggers': {
                'on_failure': self.on_failure_trigger,
                'pre_done': self.pre_done_check,
            },
            'escalation_budget': self.escalation_budget,
            'reason_on_first': self.reason_on_first,
            'reasoning_on_effort': self.reasoning_on_effort,
            'allow_store': self.allow_store,
            'allow_chat': self.allow_chat,
            'max_no_tool_calls': self.max_no_tool_calls,
            'max_steps': self.max_steps,
            'temperature': self.temperature,
            'top_p': self.top_p,
            'seed': self.seed,
        }
