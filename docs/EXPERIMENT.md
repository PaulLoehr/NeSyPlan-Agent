# The experiment

**The question.** An agentic loop costs roughly 10× what one planning call costs. When does
it earn that?

Everything below comes from one campaign, shipped in `results/harness/`. Re-run it with
`./scripts/run_experiment.sh`.

---

## Setup

|  |  |
| --- | --- |
| model | `qwen/qwen3-32b`, over OpenRouter — **one** model family at **one** size, on purpose (see *Why model is a fixed factor* below) |
| tasks | the four `@rebuild` tasks, each starting from an **inherited layout** |
| configs | the seven `@harness` configs below |
| matrix | 4 tasks × 7 configs × 2 reps = **56 episodes** |
| sampling | `temperature 0.6`, `top_p 0.95` — Qwen3's recommended thinking-mode sampling, deliberately **not** greedy |
| step cap | 25 actions per episode |
| feedback | `fix` — the full refusal, diagnosis and remedy |
| scoring | a **symbolic checker** per task; the LLM judge runs alongside and its agreement is recorded |
| cost | 3.34 M tokens, 4.7 h wall clock |

The tasks all start mid-build, because that is the only setting in which the question is
answerable at all. An earlier version of this task set started from an **empty** building
area — everything in storage, the goal freely constructible — and there a blind one-shot
plan is nearly sufficient: across three campaigns it matched or beat the agentic loop as
often as not, at roughly a twentieth of the cost. That is not a null result about agentic
loops; it says the *environment* never contradicted the plan. A one-pass plan is correct
whenever every precondition is trivially true at the moment it matters.

So the world starts in a configuration the agent did not build. Nothing is hidden: both
one-shot and agentic modes see the full layout. What changes is that a correct plan now has
to simulate *intermediate* states — after I lift this, what is reachable, what did I just
block, where does the thing I took off go?

The configs are two dimensions crossed, plus three memory arms:

|  | no thinking | thinking |
| --- | --- | --- |
| **plan once** | `oneshot_nocot` | `oneshot` |
| **act turn by turn** | `act` | `react` |

- `react_raw` — ReAct plus the **verbatim** reasoning trace carried in `content`
- `react_summary` — ReAct plus a **distilled note** instead of the verbatim trace
- `on_error_summary` — that note, but thinking **only after a rejection** (the cheap arm)

---

## Results

Cheapest first. `tok/reason~` is estimated from trace length: the endpoints do not return
`usage.completion_tokens_details.reasoning_tokens` (it is always `None`), so any reasoning
column sourced from usage alone would read 0. `tok/cache` is summarizer overhead. `fails`
is refused actions per run.

| config | success | rate | tok/tot | tok/reason~ | tok/cache | steps | fails | lat(s) | tok per success |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `oneshot_nocot` | 2/8 | 25% | 3 284 | 2 265 | 0 | 7.8 | 3.8 | 62 | 13 136 |
| `oneshot` | 1/8 | 12% | 3 788 | 2 778 | 0 | 8.8 | 3.0 | 75 | 30 306 |
| `react` | **6/8** | **75%** | 38 620 | 19 764 | 0 | 10.9 | 1.9 | 579 | **51 494** |
| `act` | 0/8 | 0% | 39 936 | 0 | 0 | 19.9 | 8.9 | 33 | — |
| `on_error_summary` | 3/8 | 38% | 52 466 | 11 139 | 11 192 | 14.5 | 4.2 | 348 | 139 909 |
| `react_summary` | **6/8** | **75%** | 78 811 | 15 072 | 23 430 | 15.8 | 4.2 | 486 | 105 082 |
| `react_raw` | 3/8 | 38% | **200 516** | 14 933 | 0 | 17.9 | 7.0 | 550 | **534 709** |

Which configuration solved which task (2 reps each):

| config | flag_excavate (R1) | unbox_red (R2) | flag_repair (R3) | pyramid_rebuild (R4) |
| --- | --- | --- | --- | --- |
| `oneshot_nocot` | 0/2 | 1/2 | 0/2 | 1/2 |
| `oneshot` | 1/2 | 0/2 | 0/2 | 0/2 |
| `act` | 0/2 | 0/2 | 0/2 | 0/2 |
| `react` | 0/2 | 2/2 | 2/2 | 2/2 |
| `react_raw` | 0/2 | 2/2 | 0/2 | 1/2 |
| `react_summary` | 2/2 | 2/2 | 1/2 | 1/2 |
| `on_error_summary` | 0/2 | 2/2 | 0/2 | 1/2 |

---

## What the numbers say

**1. The loop earns its cost — but only together with deliberation.**
`react` solves 6/8 where `oneshot` solves 1/8, for about 10× the tokens. That is the
headline, and it is the answer to the question this experiment asks.

**2. `act` is the floor, and it is the most informative cell in the table.**
Zero successes out of eight — while spending 39 936 tokens, *more than `react`*. It
averages 19.9 steps against a cap of 25 and 8.9 refused actions per run: an agent that
reads every refusal and thrashes against it without ever stopping to think. Feedback alone
is not the mechanism. Feedback plus deliberation is.

That `act` is also the fastest arm (33 s) is not a virtue. It is what burning a step budget
on reflex looks like when no reasoning tokens are generated.

**3. Plain ReAct — carrying no memory at all — is the best arm on this model.**
Both memory arms were supposed to beat it. Neither does. `react_summary` matches it exactly
(6/8) at twice the cost; `react_raw` is worse on both axes, 3/8 at **5.2× the tokens**. Per
unit of success, `react` costs 51 494 tokens, `react_summary` 105 082, and `react_raw`
**534 709** — an order of magnitude apart.

**4. Thinking only on error is dominated.**
`on_error_summary` reaches 3/8 while spending *more* than `react` (52 466 vs 38 620) for
half the success. Cheap deliberation is not cheap when it arrives after the mistake that
needed it.

**5. Reasoning made one-shot slightly worse, and that is noise.**
`oneshot_nocot` 2/8 against `oneshot` 1/8 is a difference of one episode out of eight. It
should not be read as a finding in either direction. Reported because leaving it out would
be selective.

---

## The memory mechanism engages — and still does not pay

Result 3 is the surprising one, and it is worth taking apart, because the obvious
explanation is wrong.

The reasoning cache exists to survive a protocol detail: a reasoning trace is **not**
carried into the next request, so whatever the model worked out is gone unless the model
wrote it into `content` itself. The first thing to check is whether this model writes
anything there on its own. Counted over every tool-calling assistant message, separating
what the *model* produced from what the *cache* injected:

| config | tool-calling turns | model-written text | avg trace per turn | avg content carried back |
| --- | --- | --- | --- | --- |
| `act` | 159 | **0 (0%)** | 1 568 ch | 0 ch |
| `react` | 87 | **0 (0%)** | 6 772 ch | 0 ch |
| `react_raw` | 142 | 1 (1%) | 3 258 ch | **2 760 ch** |
| `react_summary` | 126 | 4 (3%) | 2 496 ch | 198 ch |
| `on_error_summary` | 114 | 19 (17%) | 4 945 ch | 104 ch |

Three things follow, in order.

**The premise holds.** `qwen3-32b` writes essentially nothing into `content` when it calls a
tool — the same behaviour the earlier development runs measured on Phoenix. So `react`
really is re-deriving its plan from scratch every single turn.

**The mechanism engages.** Both memory arms do exactly what they are designed to do. Handing
something forward cuts the average reasoning trace from **6 772** characters per turn to
**3 258** (raw) and **2 496** (summary) — 2.1× and 2.7× shorter. That is the signature of a
model that stops re-deriving and starts continuing, and it is not subtle.

**It still does not convert.** Shorter thinking did not become more solved tasks. Both arms
take *more* steps than `react` (17.9 and 15.8 vs 10.9) and are refused more often (7.0 and
4.2 vs 1.9). Whatever the long re-derivation was getting right, the carried memory does not
preserve — it anchors the model to an earlier plan instead, and the extra refusals are that
plan meeting a world it no longer matches.

**Why `raw` is catastrophic rather than merely unhelpful.** The last column is the answer:
`raw` carries **2 760 characters back per turn against `summary`'s 198**, a 14× larger
payload — and it compounds, because every turn's trace joins the conversation the next turn
re-sends. One `flag_excavate` episode ran to the step cap and cost 319 525 tokens on its
own, roughly what all eight `oneshot` episodes cost together.

### The same mechanism, the opposite verdict

The earlier development runs against Aleph Alpha's internal Phoenix model found `react_raw`
the **strongest and cheapest-per-success** arm. Here it is the weakest and the most
expensive by an order of magnitude. Same code, same tasks, same cache implementation —
opposite conclusion.

The general form: **the scaffolding pays off unevenly by model**, and a mode's ranking is
not a property of the mode. The same effect showed up once before in this project's
history, between two other models under plain `react` — one wrote its plan into `content`
and self-regulated, the other put it in `reasoning`, which the protocol discards, and so
re-derived every turn. The scaffolding is precisely the external mechanism that gives the
second model what the first has natively, which is why it is worth a great deal to one and
nothing to the other.

The consequence for anyone reading a per-config comparison: running two models in the
*same* configuration and concluding one is more efficient mostly measures **config fit**.
Compare each model in *its own best* config, or name the confound out loud.

The transferable step is not "use this cache mode" but: *measure it on your model before
paying for it.* The instrument is cheap — the table above is one pass over the transcripts,
and it separates "the memory mechanism does not work" from "the memory works and the model
does not benefit."

> **A wart worth knowing about.** 463 of the tool-calling turns carry the literal string
> `ool_call>` in `content` — the tail of a `<tool_call>` template tag that the provider's
> parser cuts in the wrong place. It is junk, it is carried into the next request, and any
> analysis that counts "turns with non-empty content" will read it as the model writing
> text. It is why the table above filters on content length rather than emptiness.

---

## How to read this responsibly

- **n = 2 per cell.** Eight episodes per config. A one- or two-episode gap is noise; the
  0/8-vs-6/8 gap and the 5× cost gap are not. Raise `--reps` before quoting any single cell.
- **`react_raw`'s failures and its cost are partly entangled.** Running to the step cap both
  costs tokens and counts as a failure, so the two effects are not fully separable at this
  sample size. The 14× payload difference is the mechanism; the success gap is the weaker
  claim of the two.
- **Magnitudes are softer than rankings.** Greedy `temp=0` is deterministic and would need
  no repetitions, but it is the degenerate regime for Qwen3-derived models and inflates
  effect sizes — a 16× gap measured at temp 0 came out around 3× at recommended sampling.
  Sampling settings move the *size* of the gaps while preserving their *order*, so treat
  every magnitude here as softer than the ranking it sits in.
- **Do not build a claim on a judge-only number.** The judge agreed with the checker on
  **56/56 runs (100%)** in this campaign. That is not a licence to trust it. On this domain
  an LLM judge has flipped its verdict on structurally identical final states, has described
  cube colours the scene never contained, and cannot resolve a genuinely ambiguous goal —
  "the German flag as a stack" leaves open whether level 0 is the flag's top or bottom, and
  the judge answered that differently across runs. Every task here has a symbolic checker,
  and **the checker is what scored these runs**. The agreement number is recorded so the
  judge's reliability stays measurable rather than assumed.

### Why model is a fixed factor

A model pool spanning ~31B to ~1T that *also* differs in whether reasoning can be disabled
at all yields no interpretable cross-model number: every difference reads as "scale +
config-fit + controllability" at once. So this experiment uses **one model family at one
size** and treats model as a fixed factor rather than an axis. Nothing here generalises
across scale, and that is a deliberate restriction rather than an oversight.

There is a second reason to be careful with a cross-model reading. Two failures look alike
in a transcript and must not be conflated: a model that *re-derives* its plan each turn
because nothing carried over — fixable by cache and policy, and the thing this harness is
for — and a model that *has* the feedback in context and simply does not act on it, which
is a capability floor where orchestration is moot. Crediting or blaming the harness for
outcomes a weak model produces under every configuration is the easiest wrong conclusion
available here.

### On turning reasoning off

`reasoning_effort` (and OpenRouter's `reasoning: {effort}`) is a **hint**. Whether `"none"`
actually suppresses the chain-of-thought is a property of the serving backend, not of the
API. Measured as the length of the returned `reasoning` field, `qwen3-14b`, `qwen3-30b-a3b`
and `qwen3-32b` all drop to a genuine **0-char** trace at `none` while still calling tools;
other models ignore the parameter entirely and keep thinking.

This matters more than it looks. If a model silently ignores `"none"`, then every
reasoning-OFF arm is a duplicate of its reasoning-ON cousin, and a comparison between them
measures nothing while producing a table that looks like it measures something. That is
silent data corruption, not a missing feature. So the harness gates it:
[`demo._REASONING_OPTIONAL`](../nesyplan/demo.py) lists only models that have been probed,
and the reasoning-OFF modes (`oneshot_nocot`, `act`) are offered only for those. Probe a new
model before trusting it:

```bash
python3 -m nesyplan.probe_reasoning --models qwen3-14b,qwen3-32b
```

---

## Reproducing

```bash
./scripts/run_experiment.sh --list          # the matrix, runs nothing
./scripts/run_experiment.sh                 # the 56 episodes above
./scripts/run_experiment.sh --reps 4        # tighter, ~2x the cost
```

The campaign is resumable: re-run the same `--campaign` and finished cells are skipped.
Artifacts land in `results/harness/` — `summary.md`, `results.jsonl` and `manifest.json`
are committed; `report.html` and the per-run transcripts and PNGs are regenerated locally.
