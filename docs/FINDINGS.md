# Findings

Things measured along the way that shape how the harness is built and how its numbers
should be read. Code comments refer to these by letter, so the labels are stable.

---

## Finding A — "reasoning off" is a per-model serving capability, not an API feature

`reasoning_effort` (Phoenix) / `reasoning: {effort}` (OpenRouter) is a **hint**. Whether
`"none"` actually suppresses the chain-of-thought depends on the serving backend. Same
prompt, `none` vs. the model's default, measured as the length of the returned `reasoning`
field:

| model | trace at `none` | trace at default | off works? |
| --- | --- | --- | --- |
| `qwen3-14b` | **0 chars** | 404 | ✅ |
| `qwen3-30b-a3b` | **0 chars** | 468 | ✅ |
| `qwen3-32b` | **0 chars** | 437 | ✅ |
| `phoenix` | **0 chars** | 3096 | ✅ |
| `merlin` | **0 chars** | 2268 | ✅ |
| `kimi` | ~869 chars | 1158 | ❌ native thinking model — no off switch |
| `command` | 525 chars | 525 (byte-identical) | ❌ parameter ignored |

*(qwen rows verified 2026-08-31 with a live tool-calling request; the rest earlier on the
Pharia endpoint.)*

**Why this matters more than it looks.** If a model silently ignores `"none"`, then every
reasoning-OFF arm is a duplicate of its reasoning-ON cousin — and a comparison between
them measures nothing while producing a table that looks like it measures something. That
is silent data corruption, not a missing feature.

So the harness gates it: [`demo._REASONING_OPTIONAL`](../nesyplan/demo.py) lists only
models that have been probed, and the reasoning-OFF modes are offered only for those.
Probe a new model before trusting it:

```bash
python3 -m nesyplan.probe_reasoning --models qwen3-14b,qwen3-32b
```

Also observed: the endpoints do **not** return `usage.completion_tokens_details.
reasoning_tokens` (always `None`), so reasoning cost can only be estimated from the
trace's length. Any "reasoning tokens" column sourced from usage alone will read 0.

---

## Finding B — a per-config model comparison mostly measures config fit

Running two models in the *same* configuration and concluding one is more efficient is a
mistake this harness makes easy. The phoenix-vs-kimi gap under `react` turned out to be
phoenix's worst case against kimi's native strength: kimi writes its plan into `content`
and self-regulates; phoenix puts it in `reasoning`, which is then discarded, so it
re-derives every turn.

The scaffolding (policy + cache) is precisely the external mechanism that gives a
plan-in-`reasoning` model the efficiency a plan-in-`content` model has built in. It
therefore pays off **unevenly by model** — large for phoenix, little for kimi.

Fair comparisons: each model in *its own best* config, or name the confound out loud.

---

## Finding C — model scale and reasoning-controllability are confounded in a mixed pool

A pool spanning ~31B to ~1T that *also* differs in whether reasoning can be disabled at
all yields no interpretable cross-model number: every difference is "scale + config-fit +
controllability". This is why the shipped experiment uses **one model family at one size**
and treats model as a fixed factor rather than an axis.

**C2 — sampling.** The harness shares one sampling default across all models for
comparability. Greedy `temp=0` is deterministic and needs no repetitions, but it is the
degenerate regime for Qwen3-derived models and inflates magnitudes (a 16× gap measured at
temp 0 was ~3× at recommended sampling). It preserves config *ranking*, not magnitudes.
For any magnitude claim use `--temperature 0.6 --top-p 0.95 --reps 2` or more.

**C3 — the judge is not a trustworthy instrument here.** The LLM judge has flipped its
verdict on structurally identical final states and has described cube colours the scene
never contained. It also cannot resolve genuinely ambiguous goals: "the German flag as a
stack" leaves open whether level 0 is the flag's top or bottom, and the judge answered
that ambiguity differently across runs.

This is why [`checkers.py`](../nesyplan/checkers.py) exists and why the shipped experiment
runs only on tasks that have one. The judge still runs alongside, and its agreement with
the checker is recorded — so its reliability stays measurable instead of assumed. **Do not
build a claim on a judge-only number.**

---

## Finding D — orchestration cannot fix a capability floor

Two different failures look alike in a transcript and must not be conflated:

1. the model *re-derives* the plan each turn because nothing carried over — fixable by
   cache/policy, and the thing this harness is for;
2. the model *has* the feedback in context and simply does not act on it — a capability
   floor, where orchestration is moot.

`command-a-plus` showed the second. Crediting (or blaming) the harness for outcomes a weak
model produces under every configuration is the easiest wrong conclusion available here.

---

## Finding E — the environment has to push back, or there is nothing to measure

The original task set started from an **empty** building area: all cubes in storage, goal
freely constructible, horizon ~9–12 actions. In that setting blind one-shot planning is
nearly sufficient — and measured that way. Across three campaigns one-shot matched or beat
ReAct as often as not, at roughly a twentieth of the cost.

That is not a null result about agentic loops; it is a statement that the *environment*
never contradicted the plan. A one-pass plan is correct whenever every precondition is
trivially true at the moment it matters.

[`scenarios.py`](../nesyplan/scenarios.py) is the response: the world is already in some
inherited configuration, so the task begins with a state the model must reason *about*
rather than construct — a needed cube buried under another, boxed in on both grasp axes,
goal cells occupied by the wrong cubes. Nothing is hidden (both modes see the full
layout); what changes is that a correct plan now requires simulating *intermediate* states.
That is where a one-pass plan breaks and per-action feedback starts paying for itself.

The `@rebuild` task tier is built on this, and it is the tier the shipped experiment uses.

---

## Reproducing

```bash
# Finding A -- reasoning-off support, per model (fast, a handful of calls)
python3 -m nesyplan.probe_reasoning --models qwen3-14b,qwen3-30b-a3b,qwen3-32b

# Finding E -- the same config on an empty vs. an inherited layout
python3 -m nesyplan.eval --tasks german_flag --configs oneshot,react --reps 3
python3 -m nesyplan.eval --tasks @rebuild   --configs oneshot,react --reps 3
```
