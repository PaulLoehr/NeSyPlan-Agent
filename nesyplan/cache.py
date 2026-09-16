"""The reasoning cache -- proposal Dimension 2 (C0/C1/C2).

Persists the model's per-turn reasoning by writing it into the assistant message's
`content`, where it survives in the re-sent history -- the provider drops the raw
reasoning trace, so without this the model re-derives its plan from scratch every turn.
This is the SAME channel content_plan (model writes it) and strip_content (ablates it)
act on, so all three levers are consistent: they differ only in who fills `content`.

  C0 NONE     -- leave `content` as the model produced it (usually empty for phoenix).
  C1 RAW      -- content = the verbatim reasoning trace (bounded only by context_k).
  C2 SUMMARY  -- one extra reasoning-OFF call (always SUMMARIZER_MODEL, independent of
                 the driver) distills the trace into a compact essence (the
                 decisions/insight worth carrying forward). The GOAL and the
                 completed-action history are deliberately NOT restated -- they already
                 live in the system prompt and the tool-call history, so re-deriving
                 them from a single trace only makes the note wobble.

The result accumulates naturally, one short note per assistant turn, exactly the
ReAct-style trail a self-persisting model (kimi) writes on its own.
"""

import os

from nesyplan.config import CacheMode


# The SUMMARY side-call is pinned to ONE model regardless of the driver, exactly as the
# eval judge is, so summarization quality is held constant when driver models are compared.
#
# The requirement is that it honours an effort of "none": the distillation must itself be
# reasoning-off, or the cache costs more than it saves (docs/FINDINGS.md, Finding A --
# kimi and command cannot disable it and are unusable here). qwen3-32b is verified to
# honour it and is publicly reachable, which keeps the SUMMARY configs runnable with
# nothing but an OpenRouter key. Override with NESYPLAN_SUMMARIZER_MODEL.
SUMMARIZER_MODEL = os.environ.get('NESYPLAN_SUMMARIZER_MODEL') or 'qwen3-32b'


SUMMARY_INSTRUCTION = """You compress a robot cube-stacking agent's private reasoning into a short note it will
re-read on later turns. The agent already sees its task (the goal) and its full tool-call
history with results (what is already done) -- do NOT restate those. Capture ONLY the
essence of THIS reasoning trace: the key decision, the intended next step(s), and any
constraint or realization worth not re-deriving. Write 1-3 short lines, first person, no
headers and no preamble. If the trace holds nothing worth carrying forward, reply with a
single dash: -"""


class ReasoningCache:
    def __init__(self, mode, llm):
        self.mode = mode
        # Only SUMMARY makes a side-call; pin it to SUMMARIZER_MODEL regardless of the
        # driver. NONE/RAW never touch the model, so they keep the driver client as-is.
        self.llm = llm.with_model(SUMMARIZER_MODEL) if mode == CacheMode.SUMMARY else llm

    def content_for(self, trace, log=None):
        """Text to write into THIS turn's assistant `content`, plus summarizer usage.

        Returns (text_or_None, usage_or_None). NONE / an empty trace -> (None, None),
        leaving the model's own content untouched. RAW -> the verbatim trace. SUMMARY ->
        the distilled essence (None if the summarizer found nothing worth keeping).
        """
        if self.mode == CacheMode.NONE or not trace:
            return None, None
        if self.mode == CacheMode.RAW:
            return trace, None
        return self._summarize(trace, log)

    def _summarize(self, trace, log):
        messages = [
            {'role': 'system', 'content': SUMMARY_INSTRUCTION},
            {'role': 'user', 'content': f'REASONING TRACE:\n{trace}'},
        ]
        resp = self.llm.chat(messages, reasoning=False)
        if log is not None:
            log.add({'role': 'system', 'content': '[cache summarizer call]'})
            for m in messages[1:]:
                log.add(m)
            log.add(resp.message)
        essence = (resp.content or '').strip()
        if not essence or essence == '-':
            return None, resp.usage
        return essence, resp.usage
