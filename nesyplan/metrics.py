"""Per-episode metric accumulation for comparable cross-mode results.

Collects exactly the quantities the proposal's Logging + Evaluation sections call
for: total / reasoning / completion / prompt / cache-injection tokens; per-turn and
per-episode latency; count and cause of reasoning turns; tool failures; get_state
calls; cache updates; plan-action mismatches; think escalations; final outcome.

task_success is intentionally left null here -- it is filled afterwards by the eval
layer's LLM judge (nesyplan/judge.py); everything else is filled by the orchestrator.
"""


def normalize_usage(usage):
    """Flatten an OpenAI-style usage object to {prompt, completion, reasoning, total}.

    The reasoning-token breakdown field is provider-specific; we read the common
    OpenAI location (completion_tokens_details.reasoning_tokens) and also accept a
    top-level reasoning_tokens, defaulting to 0 when absent.
    """
    usage = usage or {}
    prompt = usage.get('prompt_tokens') or 0
    completion = usage.get('completion_tokens') or 0
    details = usage.get('completion_tokens_details') or {}
    reasoning = details.get('reasoning_tokens')
    if reasoning is None:
        reasoning = usage.get('reasoning_tokens') or 0
    total = usage.get('total_tokens')
    if total is None:
        total = prompt + completion
    return {'prompt': prompt, 'completion': completion, 'reasoning': reasoning, 'total': total}


# The endpoint returns no usage.completion_tokens_details.reasoning_tokens (all None), so
# reasoning cost can only be estimated from the returned trace's length. ~4 chars/token is
# the usual rough ratio for English prose; it is a PROXY, comparable across configs of the
# same model, not an exact token count.
CHARS_PER_TOKEN = 4


class Metrics:
    def __init__(self):
        self.reasoning_turns = {}   # trigger cause -> count (initial|scheduled|self|hard_failure|pre_done)
        self.tool_failures = 0
        self.get_state_calls = 0
        self.cache_updates = 0
        self.think_calls = 0
        self.latency_ms_total = 0
        self.reasoning_chars = 0    # summed length of every returned reasoning trace
        self.reasoning_traces = 0   # how many turns actually came back with a trace
        self.tokens = {'total': 0, 'reasoning': 0, 'completion': 0, 'prompt': 0, 'cache_injection': 0}

    def add_trace(self, trace):
        """Fold one turn's returned reasoning trace into the cost proxy.

        Called for EVERY turn, not just reasoning-on ones: a model that cannot honour
        reasoning=False still thinks (FINDINGS Finding A), and that cost is real. This is
        the only measurement of the reasoning axis that does not depend on the provider
        reporting reasoning_tokens (it never does).
        """
        if trace:
            self.reasoning_chars += len(trace)
            self.reasoning_traces += 1

    def add_llm(self, usage, *, cache=False):
        """Fold one completion's usage into the running totals.

        cache=True marks a summarizer call (C2): its tokens are also tracked
        separately as cache-injection overhead, per the proposal.
        """
        u = normalize_usage(usage)
        self.tokens['prompt'] += u['prompt']
        self.tokens['completion'] += u['completion']
        self.tokens['reasoning'] += u['reasoning']
        self.tokens['total'] += u['total']
        if cache:
            self.tokens['cache_injection'] += u['total']

    def record_reasoning(self, trigger):
        key = trigger or 'unknown'
        self.reasoning_turns[key] = self.reasoning_turns.get(key, 0) + 1

    def finalize(self, outcome, steps, *, task_success=None):
        tokens = dict(self.tokens)
        # The reported reasoning cost for this episode. `reasoning` stays whatever the
        # provider claimed (usually 0); `reasoning_est` is the trace-length proxy that
        # actually carries signal, so the roll-up has a non-empty cost column.
        tokens['reasoning_est'] = self.reasoning_chars // CHARS_PER_TOKEN
        return {
            'outcome': outcome,
            'steps': steps,
            'task_success': task_success,
            'reasoning_turns': dict(self.reasoning_turns),
            'reasoning_turns_total': sum(self.reasoning_turns.values()),
            'reasoning_chars': self.reasoning_chars,
            'reasoning_traces': self.reasoning_traces,
            'tool_failures': self.tool_failures,
            'get_state_calls': self.get_state_calls,
            'cache_updates': self.cache_updates,
            'think_calls': self.think_calls,
            'tokens': tokens,
            'latency_ms_total': self.latency_ms_total,
        }
