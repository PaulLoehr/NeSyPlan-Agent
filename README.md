# NeSyPlan Agent

A neuro-symbolic planning agent for a cube-stacking robot.

An LLM proposes one high-level action at a time — *pick the red cube*, *place it at (3, 3)
on level 1*. A symbolic executor owns the world and checks that proposal against an exact
model of it: is the cube reachable, is the cell free, is anything underneath it, can the
gripper's jaws even open there? It then performs the action or **refuses** it with a
written explanation. The model never touches the world directly.

![The NeSyPlan web demonstrator mid-task](images/ui-04-done.png)

*The demonstrator after building the German flag: the reasoning behind each action, the
tool call and the executor's answer beneath it, and the world — drawn from the same state
the model reads — on the right.*

---

## What is and is not in this repository

**Here:** the agent (loop, invocation policies, reasoning cache), a faithful in-process
world model with all of its validation, a browser demonstrator, and an evaluation harness.

**Not here: the robot.** The executor that does inverse kinematics, plans motion, drives the
gripper and talks to a UR5e ships as pre-built images belonging to a third party. This
repository contains only [`robot_client.py`](nesyplan/robot_client.py), a ~50-line HTTP
client that can *speak to* such an executor if you already run one.

So, plainly: **you cannot move a physical arm with this repository alone.** Everything below
runs against the in-process world model, which reproduces the executor's symbolic behaviour
— the same state transitions, the same validation, the same refusal text — with motion as a
no-op. That is enough for every question about planning and orchestration, and it is not a
statement about motion. See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for the HTTP
contract and the limits that matter before pointing any model at real hardware.

---

## Setup

```bash
git clone <this-repo> && cd nesyplan-agent
cp .env.example .env          # then put your OpenRouter key in it
```

That is the whole installation. **Python 3.9+**, standard library only — there is no
`requirements.txt` and nothing to build. The single optional extra is **matplotlib**, used
lazily and only to draw a result PNG per run; without it, campaigns simply skip the images.

The agent, the eval judge and the cache summarizer all run on **`qwen3-32b` over
OpenRouter** by default, so one key in `.env` covers everything:

```
OPENROUTER_API_KEY=...        # https://openrouter.ai/keys
```

It is a paid model, so the account needs a little credit — without it, requests fail with
HTTP 402 before the model runs. `.env` is gitignored.

### Try it with no key at all

```bash
./scripts/run_web_demo.sh replay
```

replays a recorded session through the real UI — no key, no model, no cost. One recording
ships with the repository, so this works immediately after a clone.

---

## The browser demonstrator

```bash
./scripts/run_web_demo.sh                     # http://localhost:8600
./scripts/run_web_demo.sh --model qwen3-14b   # any flag is forwarded
PORT=9000 ./scripts/run_web_demo.sh           # different port
NO_OPEN=1 ./scripts/run_web_demo.sh           # do not open a browser
```

Type a task in plain language and watch the agent work:

> *stack the red, green and blue cube in the middle*
> *build the German flag as a stack*
> *put the yellow cube next to the black one*

The chat is **continuous**: the world is not reset between tasks, so a follow-up like *"now
shift it a bit left"* has both context and a structure to act on.

| in the UI | what it does |
| --- | --- |
| `/cleanup` | return every cube to storage, deterministically and without the model |
| `/new` | start a fresh session (clears the carried conversation) |
| `/mode <name>` | switch the orchestration mode live; the world and chat survive |
| `/model <alias>` | switch the driving model live |
| `/context` | open the inspector: exactly what the model receives, split into what persists across tasks and the current loop |

The right-hand panel draws the world from the same observation the model reads, so the
picture and the prompt can never disagree — which makes it a debugging tool as much as a
display.

### The same engine in a terminal

```bash
python3 -m nesyplan.demo
python3 -m nesyplan.demo --mode react --model qwen3-14b
python3 -m nesyplan.demo --hide-thoughts       # hide the reasoning blocks
```

Commands: `/cleanup` · `/mode [name]` · `/state` · `/help` · `/quit`.

### One scripted episode

```bash
python3 -m nesyplan.run "build a pyramid"
python3 -m nesyplan.run "build a pyramid" --policy always --cache summary
```

One task, one reset, a JSON transcript in `llm_logs/`. This is the low-level entry point:
every orchestration lever is a flag (see `--help`), which makes it the right tool for
poking at one behaviour in isolation.

---

## Running a campaign

A campaign is a matrix of **tasks × configurations × repetitions**, run against the
in-process world model, scored, and written to `results/<campaign>/`.

### The shipped experiment

```bash
./scripts/run_experiment.sh --list     # print the matrix, run nothing
./scripts/run_experiment.sh            # run it
```

That wrapper is one `nesyplan.eval` invocation with the shipped defaults pinned
(`--tasks @rebuild --configs @harness --reps 2 --temperature 0.6 --top-p 0.95`). Every flag
you add is forwarded and a later one wins, so `./scripts/run_experiment.sh --reps 1` halves
it. Results and how to read them: **[docs/EXPERIMENT.md](docs/EXPERIMENT.md)**.

### Building your own

```bash
python3 -m nesyplan.eval --campaign my_run --tasks @rebuild --configs @lean --reps 3
python3 -m nesyplan.eval --list        # the resolved matrix
python3 -m nesyplan.eval --dry-run     # which cells would run, incl. resume-skips
```

**Tasks** (`--tasks`) — twelve of them, id or `@group`:

| tier | tasks | scored by |
| --- | --- | --- |
| `@rebuild` | `flag_excavate` `unbox_red` `flag_repair` `pyramid_rebuild` | a **symbolic checker** |
| `@clean` | `german_flag` `pyramid` `plus_cross` `t_shape` | the LLM judge |
| ungrouped | `staircase` `largest_div4` `tower_no_adjacent` `defend_red` | the LLM judge |

The `@rebuild` tier starts from a layout the agent did not build and is the only tier where
every task has a code-based scorer. Prefer it for anything you intend to quote.

**Configurations** (`--configs`) — a label or a `@group`. The two dimensions are *when may
the model think* and *what survives to the next turn*:

|  | no thinking | thinking |
| --- | --- | --- |
| **plan once** | `oneshot_nocot` | `oneshot` |
| **act turn by turn** | `act` | `react` |

plus the memory arms `react_raw` (verbatim trace carried forward), `react_summary` (a
distilled note) and `on_error_summary` (that note, thinking only after a rejection).
Groups: `@harness` (the shipped seven) · `@modes` (eight) · `@feedback` (nine, crossing the
reasoning ladder with feedback verbosity) · `@lean` (five, for a first pass on a new model).
`python3 -m nesyplan.eval --list` prints everything else that is selectable.

**Other levers worth knowing:**

```bash
--reps 4                  # repetitions per cell; raise before quoting any single number
--feedback-level terse    # how much of a refusal the model may read: terse | why | fix
--scenario boxed_red      # force one starting layout for every cell
--task-prompt "build a smiley face"    # an ad-hoc task, judge-scored, no code needed
--model qwen3-14b         # comma-separate for several
--max-steps 25            # safety cap on agent turns per task
--temperature 0.6 --top-p 0.95         # sampling; see EXPERIMENT.md before using greedy
--fresh                   # ignore prior results and re-run every cell
```

**Starting layouts** (`--scenario`) — `empty` · `buried_black` (a needed cube under two
others) · `boxed_red` (unreachable on both grasp axes) · `clutter_diagonal` (the footprint
is taken) · `flag_inverted` (the goal is built, in the wrong order). Each is validated on
load against the same invariants the executor enforces, so an impossible layout fails
loudly rather than halfway through a campaign.

### What a campaign writes

```
results/<campaign>/
  manifest.json                    the resolved matrix, model, sampling, judge — self-describing
  results.jsonl                    one row per run: success, tokens, steps, refusals, latency
  summary.md                       success matrix + per-config roll-up, cheapest first
  report.html                      browsable report (open in a browser)
  runs/<config>/<id>.json          full transcript, verbatim reasoning included
  runs/<config>/<id>.png           the final world state
```

`summary.md` is rewritten after **every** cell, so a running campaign can be watched.
Campaigns are **resumable**: re-run the same `--campaign` and finished cells are skipped —
which is also how you add a configuration to an existing campaign without re-running it.

### Scoring

Two tiers. A task with a checker ([`checkers.py`](nesyplan/checkers.py)) is scored
symbolically from the final world state — deterministic, and the number that counts. Tasks
without one fall back to the LLM judge, which is why an ad-hoc `--task-prompt` works with
no code. The judge runs on every task regardless and its agreement with the checker is
recorded, so its reliability stays measurable rather than assumed. It is the noisiest
instrument here; do not build a claim on a judge-only number.

---

## Models

Short aliases resolve to full ids, and the *model* decides which endpoint is used
([`model_aliases.py`](nesyplan/model_aliases.py)). `qwen3-32b` (the default), `qwen3-30b-a3b`
and `qwen3-14b` are verified over OpenRouter — tool calls and a genuine reasoning
off-switch included.

That off-switch matters more than it looks: `reasoning_effort: "none"` is only a *hint*, and
a model that ignores it turns every reasoning-OFF configuration into a silent duplicate of
its reasoning-ON cousin. Probe before trusting a new model:

```bash
python3 -m nesyplan.probe_reasoning --models qwen3-14b,qwen3-32b
```

The `phoenix` / `kimi` / `command` / `merlin` aliases point at Aleph Alpha's internal
endpoint and are not reachable without access to it. Any OpenAI-compatible endpoint works —
set `PHOENIX_BASE_URL` and `BEARER_TOKEN` in `.env` and pass `--model` with its model id.

---

## Driving an external executor

If you *do* run a command server that speaks the contract in
[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md), every entry point can drive it instead of the
in-process model:

```bash
./scripts/run_web_demo.sh --backend sim --url http://localhost:8100
python3 -m nesyplan.eval --backend sim --url http://localhost:8100
```

Nothing above that boundary changes — the loop, the policies, the cache, the UI and the eval
are identical either way. That symmetry is the point: the same agent code that runs a
campaign in seconds on a laptop is what drove the physical cell.

![The physical cell: a UR5e with an RG2 gripper over the printed plate](images/real-robot-cell.png)

*The cell this world model was extracted from.* The printed plate is
[`images/plate-template-a2.pdf`](images/plate-template-a2.pdf) — the 6×6 building area and
the storage row, at A2. The agent only ever sees a 0–6 grid, so re-printing at a different
size changes nothing it reads.

---

## Repository layout

```
nesyplan/          the agent: loop, policies, cache, world model, eval, web UI
  webui/           the browser front end (Vue, vendored; no build step)
data/cubes.json    the cube set — one source of truth for state, prompt and drawing
scripts/           run_web_demo.sh · run_experiment.sh
docs/              ARCHITECTURE.md · EXPERIMENT.md
images/            UI screenshots, photographs of the cell, the printable plate
results/harness/   the shipped campaign's summary, results table and manifest
llm_logs/          your recorded sessions (plus the one shipped for `replay`)
```

A tour of how the pieces fit — the loop, the validators, the two backends, the free-axis
rule — is in **[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)**.

## Status and licence

Research code, published as a demonstrator rather than a library — the interfaces are not
stable and there are no unit tests.

Licensed under the **[Apache License 2.0](LICENSE)**.

The vendored `nesyplan/webui/vendor/vue.global.prod.js` is [Vue 3](https://vuejs.org),
MIT-licensed, and belongs to its authors.
