"""NeSyPlan Agent -- a neuro-symbolic agent harness for a cube-stacking robot.

An LLM plans; a symbolic executor owns the world. The executor validates every action
against the actual state (bounds, occupancy, support, gripper reachability) and either
performs it or REJECTS it with a reason. The model never moves anything directly -- it
proposes, and the world answers. That answer is the only feedback it gets.

What this package is for is measuring how the *orchestration around* that loop changes
the outcome. Two dimensions, both plain configuration of one loop (config.py):

  Dimension 1 -- invocation policy: when may the model think?
                 ONESHOT (plan once, execute blind) | ALWAYS (reason every turn)
                 | REASON_FIRST | PERIODIC(N) | SELF_TRIGGERED | ON_ERROR
                 (+ hard triggers on_failure / pre_done, escalation budget).
  Dimension 2 -- reasoning cache: what survives between turns?
                 NONE (traces discarded) | RAW (trace written into the assistant
                 content) | SUMMARY (a distilled note instead). One deliberation slot,
                 replace-on-update, injected late.

The second dimension exists because a reasoning trace is usually NOT carried into the
next request by the provider, so anything the model concluded is lost unless it writes
it into `content` itself -- which smaller models frequently do not do. SUMMARY supplies
that memory externally. docs/EXPERIMENT.md measures what it buys.

One source of truth throughout: the system prompt (prompts.py), the tool schemas
(tools.py), the coordinate system (environment.py), and one comparable log schema for
every mode (session_log.py).

Entry points:
    python3 -m nesyplan.web_demo      browser chat UI (the demonstrator)
    python3 -m nesyplan.demo          the same engine in the terminal
    python3 -m nesyplan.run "<task>"  one scripted episode
    python3 -m nesyplan.eval          a campaign: {tasks} x {configs}

Everything runs against an in-process world model by default (fake_robot.py) -- no
simulator, no containers, no robot. See docs/ARCHITECTURE.md.
"""
