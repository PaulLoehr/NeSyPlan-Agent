"""The unified episode loop -- every proposal mode as configuration of one path.

run_episode() dispatches on policy:
  - ONESHOT: one reasoning call -> parse the JSON plan -> execute it blindly over the
    executor (no feedback to the model), then done.
  - agentic (ALWAYS / REASON_FIRST / PERIODIC / SELF_TRIGGERED): the proposal's core
    loop -- should_reason() per turn; context = [system, task] + last-K history +
    cache slot (+ reflection); one tool call per turn; think() escalation; hard
    triggers (on_failure, pre_done); escalation budget; cache updated every reasoning
    turn. Reproduces the ReAct prototype at policy=ALWAYS, cache=NONE.

Structured, comparable logging is emitted alongside the verbatim conversation on
every turn (see nesyplan/session_log.py).
"""

import json
import re
import time

from nesyplan.cache import ReasoningCache
from nesyplan.config import CacheMode, Policy
from nesyplan.environment import concise_feedback, cube_ids, format_state, grid_units
from nesyplan.metrics import Metrics, normalize_usage
from nesyplan.policy import should_reason
from nesyplan.prompts import REFLECTION_INSTRUCTION, build_system_prompt
from nesyplan.tools import build_tools


def assistant_history(message, strip_content=False):
    """Assistant entry for the running history: keep content + tool_calls, drop reasoning.

    The reasoning trace is logged verbatim (via log.add of the raw message) but is NOT
    re-sent as part of history -- re-injection is the cache's job (proposal Dim. 2).

    strip_content=True additionally drops the assistant `content` from the re-sent
    history (the tool_calls are kept). This is the content-as-memory ablation: it
    removes exactly the durable plan a model like kimi writes into content, so we can
    measure whether that content is what keeps its later-turn reasoning short. The
    model's raw output (incl. any content) is still logged; only the echoed-back copy
    is stripped.
    """
    out = {'role': 'assistant', 'content': '' if strip_content else message.get('content')}
    if message.get('tool_calls'):
        out['tool_calls'] = message['tool_calls']
    return out


def _parse_args(call):
    try:
        return json.loads((call.get('function') or {}).get('arguments') or '{}')
    except ValueError:
        return {}


def _first_call(resp):
    calls = resp.tool_calls
    if not calls:
        return None
    call = calls[0]
    return {'name': (call.get('function') or {}).get('name', ''), 'arguments': _parse_args(call)}


def _last_k(rest, k):
    """Keep the last k assistant turns (with their trailing tool/user messages).

    Slices at an assistant boundary so every assistant->tool pairing stays intact
    (the OpenAI contract: each tool message answers the assistant right before it).
    """
    idxs = [i for i, m in enumerate(rest) if m.get('role') == 'assistant']
    if len(idxs) <= k:
        return rest
    return rest[idxs[-k]:]


def _assemble(messages, config, reflect):
    """Build one request: [system, task-seed] + pruned history (+ reflection).

    The cache no longer injects a separate "plan slot" -- it writes each turn's
    distilled reasoning into that turn's assistant `content` (see cache.py), so the
    persisted reasoning already lives in `messages` as natural history. The reflection
    instruction, being an act-now directive, stays last.
    """
    system, seed, rest = messages[0], messages[1], messages[2:]
    if config.context_k is not None:
        rest = _last_k(rest, config.context_k)
    assembled = [system, seed] + rest
    if reflect:
        assembled.append({'role': 'user', 'content': REFLECTION_INSTRUCTION})
    return assembled


def parse_plan(text):
    """Parse a one-shot JSON plan into a list of {name, arguments} pick/place steps."""
    text = (text or '').strip()
    fence = re.search(r'```(?:json)?\s*\n(.*?)```', text, re.DOTALL)
    if fence:
        text = fence.group(1).strip()
    try:
        data = json.loads(text)
    except ValueError:
        start, end = text.find('['), text.rfind(']')
        if start == -1 or end == -1 or end < start:
            raise ValueError('no JSON array found in the model output')
        data = json.loads(text[start:end + 1])
    if isinstance(data, dict):
        data = data.get('plan') or data.get('steps') or []
    steps = []
    for item in data:
        if not isinstance(item, dict):
            continue
        name = item.get('name')
        args = item.get('arguments') or item.get('args') or {}
        if name in ('pick', 'place', 'store'):
            steps.append({'name': name, 'arguments': args})
    if not steps:
        raise ValueError('plan contained no valid pick/place steps')
    return steps


def run_episode(task, config, robot, llm, log):
    """Run one episode to completion. Returns an exit code (0 done, 1 unfinished, 2 error)."""
    config.validate()
    state = robot.get_state()
    grid = grid_units(state)
    ids = cube_ids(state)
    metrics = Metrics()

    # The one-shot bakes the current layout into its system prompt (it plans blind in a
    # single shot, so nothing goes stale); the agentic prompt keeps the layout OUT of the
    # system prompt (it would go stale across turns) and receives it in the seed message.
    if config.policy == Policy.ONESHOT:
        seed = {'role': 'user', 'content': task}
    else:
        seed = {'role': 'user', 'content': f'{task}\n\nCurrent world state:\n{format_state(state)}'}
    messages = [
        {'role': 'system', 'content': build_system_prompt(state, config)},
        seed,
    ]
    for m in messages:
        log.add(m)
    log.track_conversation(messages)   # finish() snapshots this as model_view (post-backfill)

    try:
        if config.policy == Policy.ONESHOT:
            return _run_oneshot(config, robot, llm, log, metrics, messages)
        return _run_agentic(config, robot, llm, log, metrics, messages, ids, grid)
    except BaseException as exc:   # persist why the transcript ends, then re-raise
        log.set_metrics(metrics.finalize('error', len(log.data.get('turns', []))))
        log.finish(status='error', error=repr(exc))
        raise


def _oneshot_core(config, robot, llm, log, metrics, messages, base_turn=0, emit=None):
    """Plan once (blind) and execute the plan over the robot; record turns + metrics.

    Does NOT finish the log -- the caller does, so a multi-task demo can reuse one log.
    `base_turn` offsets the recorded turn numbers for successive tasks in a demo session.
    `emit` is an optional UI event sink (nesyplan/web_demo.py); None -> unchanged behaviour.
    Returns {exit_code, status, steps, error}.
    """
    def _emit(type, **data):
        if emit:
            emit({'type': type, **data})

    reasoning = config.reason_on_first   # --no-initial-reasoning -> plan without CoT (baseline)
    trigger = 'initial' if reasoning else None
    _emit('turn', step=base_turn + 1, reasoning=reasoning, trigger=trigger)
    t0 = time.monotonic()
    resp = llm.chat(messages, reasoning=reasoning)
    dt = int((time.monotonic() - t0) * 1000)
    log.add(resp.message)
    metrics.add_llm(resp.usage)
    metrics.add_trace(resp.trace)
    if reasoning:
        metrics.record_reasoning('initial')
    metrics.latency_ms_total += dt

    try:
        plan = parse_plan(resp.content)
    except ValueError as exc:
        log.add_turn({'turn': base_turn + 1, 'reasoning': reasoning, 'trigger': trigger, 'tool_call': None,
                      'note': f'plan parse failed: {exc}',
                      'usage': normalize_usage(resp.usage), 'latency_ms': dt})
        log.set_metrics(metrics.finalize('planning_failed', 0))
        print(f'One-shot planning failed: {exc}')
        _emit('note', text=f'one-shot planning failed: {exc}')
        return {'exit_code': 1, 'status': 'planning_failed', 'steps': 0, 'error': str(exc)}

    log.add_turn({'turn': base_turn + 1, 'reasoning': reasoning, 'trigger': trigger, 'tool_call': None,
                  'plan': plan, 'usage': normalize_usage(resp.usage), 'latency_ms': dt})
    print(f'One-shot plan: {len(plan)} steps')
    _emit('plan', steps=len(plan))

    # Execute blindly -- no feedback ever returns to the model (proposal Mode 0).
    for i, step in enumerate(plan, start=1):
        name, args = step['name'], step['arguments']
        _emit('tool_call', step=i, reasoning=False, name=name, args=args)
        fb = robot.send_command(name, **args)
        ok = bool(fb.get('ok'))
        if not ok:
            metrics.tool_failures += 1
        print(f'  [{i}/{len(plan)}] {name}({args}) -> '
              f'{"ok" if ok else "FAIL: " + str(fb.get("error"))}')
        _emit('feedback', ok=ok, message=fb.get('message'), error=fb.get('error'),
              held=(fb.get('observation') or {}).get('held'))
        log.add_turn({'turn': base_turn + 1 + i, 'reasoning': False, 'trigger': None,
                      'tool_call': {'name': name, 'arguments': args},
                      'feedback': concise_feedback(fb, config.feedback_level)})

    robot.send_command('done', reason='one-shot plan executed')
    log.set_metrics(metrics.finalize('done', len(plan)))
    print(f'One-shot done: executed {len(plan)} steps ({metrics.tool_failures} failed).')
    _emit('done', reason=f'one-shot plan executed ({len(plan)} steps, {metrics.tool_failures} failed)',
          steps=len(plan))
    return {'exit_code': 0, 'status': 'done', 'steps': len(plan), 'error': None}


def _run_oneshot(config, robot, llm, log, metrics, messages):
    """Single-episode one-shot: plan+execute, then finish the log. (Frozen behaviour.)"""
    result = _oneshot_core(config, robot, llm, log, metrics, messages)
    if result['status'] == 'planning_failed':
        log.finish(status='planning_failed', error=result['error'])
    else:
        log.finish(status='done', steps=result['steps'])
    return result['exit_code']


class AgenticSession:
    """The agentic loop as a reusable object: run one task at a time over a PERSISTENT
    conversation.

    run_task() appends the task (with the current world state) and runs turns until
    done()/max_steps, then returns -- leaving `messages`, `cache` and the think budget
    intact so the NEXT task continues the same conversation. It records metrics + turns
    but does NOT finish the log, so the caller controls the log lifecycle.

    run_episode() uses it for a single task (behaviour identical to the frozen loop);
    the demo (nesyplan/demo.py) constructs one session and calls run_task() repeatedly
    for a continuous chat over one evolving world (no reset between tasks).

    compact_history (the demo turns it on) is the cross-task context engineering: the
    per-turn reasoning/tool trail governs the LIVE loop, but once a task ends run_task()
    collapses that trail to a clean user->assistant pair (the task text + the done()
    reason as assistant content), so only that "chat" carries into the next task. Off in
    the eval's single-episode path, which needs the full model_view. See _compact().
    """

    def __init__(self, config, robot, llm, log, metrics, ids, grid, messages,
                 compact_history=False, on_cache_write=None, emit=None, on_compact=None):
        self.config = config
        self.robot = robot
        self.llm = llm
        self.log = log
        self.metrics = metrics
        self.ids = ids
        self.grid = grid
        self.messages = messages       # [system, seed, ...]; grows across tasks
        self.cache = ReasoningCache(config.cache_mode, llm)
        self.think_calls = 0           # escalation budget spans the whole session
        self.compact_history = compact_history   # collapse each finished task to a clean chat pair
        # Optional display hook (the demo sets it): called on(cache_mode, prev_content,
        # new_content) whenever a cache turn OVERWRITES the assistant content, so a viewer can
        # show the text that is actually persisted (not just what the model itself wrote). None
        # in the eval path.
        self.on_cache_write = on_cache_write
        # Optional structured-event sink for a live UI (nesyplan/web_demo.py): called with a
        # dict {'type': ..., ...} at each progress point (tool_call, feedback, done, ...),
        # ALONGSIDE the existing prints. None in the terminal/eval paths, so their behaviour
        # is byte-for-byte unchanged.
        self.emit = emit
        # Optional hook called with the task's full turn trail JUST BEFORE _compact() throws it
        # away, so a UI can keep showing the finished loop instead of only its compacted pair.
        self.on_compact = on_compact

    def _emit(self, type, **data):
        """Fire one structured progress event to the optional UI sink (no-op if unset)."""
        if self.emit:
            self.emit({'type': type, **data})

    def run_task(self, task=None):
        """Run one task to done()/max_steps. If `task` is given, append it (with the CURRENT
        world state) as a new user turn first; otherwise the task is already seeded in
        `messages` (the single-episode path). Returns {exit_code, status, reason, steps};
        does NOT finish the log."""
        config = self.config
        seed_index = None
        if task is not None:
            current = self.robot.get_state()
            seed = {'role': 'user',
                    'content': f'{task}\n\nCurrent world state:\n{format_state(current)}'}
            seed_index = len(self.messages)   # where this task's turns begin (for _compact)
            self.messages.append(seed)
            self.log.add(seed)

        escalate = False
        forced_trigger = None
        last_feedback = None
        pre_done_used = False
        acted = False   # has a build action (pick/place/store) happened yet this task? (allow_chat)
        prose_turns = 0  # CONSECUTIVE turns where the model talked instead of calling a tool

        for step in range(1, config.max_steps + 1):
            reasoning, trigger = should_reason(config, step, escalate, last_feedback, forced_trigger)
            escalate = False
            forced_trigger = None
            if trigger == 'hard_failure':
                last_feedback = None   # consume the failure -> fire once, not every following turn

            budget_ok = (config.escalation_budget is None or self.think_calls < config.escalation_budget)
            include_think = (config.policy == Policy.SELF_TRIGGERED and not reasoning and budget_ok)
            tools = build_tools(self.ids, self.grid, include_think=include_think,
                                include_store=config.allow_store)

            reflect = (config.policy == Policy.PERIODIC and reasoning and trigger == 'scheduled')
            req_messages = _assemble(self.messages, config, reflect)

            # Announce the policy's verdict BEFORE the call, so a UI can label the reasoning
            # this turn produces with WHY it happened (the trigger) -- the quantity this whole
            # harness is about. Only the loop knows it; the LLM wrapper downstream does not.
            self._emit('turn', step=step, reasoning=reasoning, trigger=trigger)

            t0 = time.monotonic()
            resp = self.llm.chat(req_messages, tools=tools, reasoning=reasoning)
            dt = int((time.monotonic() - t0) * 1000)
            self.metrics.latency_ms_total += dt
            self.metrics.add_llm(resp.usage)
            # Count the trace the model ACTUALLY produced, not the one we asked for: the
            # endpoint returns no usage.reasoning_tokens (all None), and a model that cannot
            # honour reasoning=False still thinks (FINDINGS Finding A). Measuring the
            # returned trace is the only faithful cost signal for this dimension.
            self.metrics.add_trace(resp.trace)
            if reasoning:
                self.metrics.record_reasoning(trigger)

            asst_msg = assistant_history(resp.message, config.strip_content)
            self.messages.append(asst_msg)
            self.log.add(resp.message)   # raw -> keeps the model's reasoning (and any content)

            turn_rec = {'turn': step, 'reasoning': reasoning, 'trigger': trigger,
                        'usage': normalize_usage(resp.usage), 'latency_ms': dt}

            # Persist this turn's reasoning into the assistant `content` (the durable
            # channel that survives in re-sent history). Only on a plan-changing reasoning
            # turn: a done()/get_state() turn changes nothing worth carrying forward.
            # strip_content ablates the channel, so nothing is written then.
            #
            # KNOWN CONSEQUENCE (deliberate -- the eval campaigns in llm_logs/ were run under
            # this rule): the gate is the turn's TOOL CALL, not its position. A model that
            # orients itself with get_state() on turn 1 loses that turn's trace, so under
            # raw/summary the cache starts filling from turn 2. Sharpest under REASON_FIRST,
            # which reasons on turn 1 ONLY: if that one turn calls get_state(), the episode
            # caches nothing at all and reason_first_raw/_summary behave like cache=none.
            # Check `cache_update` per turn in the session log before reading such a run as
            # evidence about the cache dimension.
            decided = _first_call(resp)
            plan_changing = reasoning and decided and decided['name'] not in ('done', 'get_state')
            if plan_changing and config.cache_mode != CacheMode.NONE:
                text, cu_usage = self.cache.content_for(resp.trace, self.log)
                if text and not config.strip_content:
                    asst_msg['content'] = text
                    if self.on_cache_write:
                        self.on_cache_write(config.cache_mode, resp.content, text)
                self.metrics.cache_updates += 1
                turn_rec['cache_update'] = {'mode': config.cache_mode.value, 'content': text}
                if cu_usage:
                    self.metrics.add_llm(cu_usage, cache=True)

            tool_calls = resp.tool_calls
            if not tool_calls:
                said = (resp.content or '').strip()
                # Conversational turn: the model replied in prose instead of acting. With
                # allow_chat (the demo) this is a clean exit -- BUT ONLY before any build action:
                # NeSyPlan may greet, answer, or ask for a task without entering the loop. Once
                # it has started building (a pick/place/store this task), a no-tool-call turn is
                # NOT a valid finish -- it is nudged, so the build must be closed with done()
                # rather than trailing off in prose. The prose exit, via _compact, becomes the
                # assistant content carried into the chat.
                if config.allow_chat and said and not acted:
                    # The reply text is shown by the caller's view (SpinnerLLM's content block);
                    # here we just record it and end the turn.
                    turn_rec['tool_call'] = None
                    turn_rec['note'] = 'chat'
                    self.log.add_turn(turn_rec)
                    self.log.set_metrics(self.metrics.finalize('chat', step))
                    self._compact(task, seed_index, said)
                    self._emit('chat', text=said, steps=step)
                    return {'exit_code': 0, 'status': 'chat', 'reason': said, 'steps': step}
                # The model did not act. Count only turns where it actually SAID something
                # instead of acting -- a wholly empty reply is a transport glitch, not a refusal
                # to act, so that one always gets nudged.
                if said:
                    prose_turns += 1
                print(f'[{step}] model spoke without a tool call: {said[:200]}')
                # Give up instead of nudging forever: with max_no_tool_calls set (the demo), a
                # model that keeps narrating mid-build ends the task rather than spending the
                # remaining step budget on nudges. Its prose becomes the task's outcome text.
                if (said and config.max_no_tool_calls is not None
                        and prose_turns >= config.max_no_tool_calls):
                    turn_rec['tool_call'] = None
                    turn_rec['note'] = 'no_tool_call_giveup'
                    self.log.add_turn(turn_rec)
                    self.log.set_metrics(self.metrics.finalize('no_tool_call', step))
                    print(f'Stopping: replied without a tool call {prose_turns}x in a row.')
                    self._compact(task, seed_index, said)
                    self._emit('no_tool_call', text=said, steps=step, turns=prose_turns)
                    return {'exit_code': 1, 'status': 'no_tool_call', 'reason': said, 'steps': step}
                # Otherwise (eval baseline, or an empty reply): nudge back toward one tool call.
                self._emit('note', text='model replied without a tool call — nudged to act')
                nudge = {'role': 'user',
                         'content': 'Call exactly one tool. Do not reply in prose.'}
                self.messages.append(nudge)
                self.log.add(nudge)
                turn_rec['tool_call'] = None
                turn_rec['note'] = 'no_tool_call'
                self.log.add_turn(turn_rec)
                continue

            prose_turns = 0   # it acted -> the give-up counter is about CONSECUTIVE prose turns
            call = tool_calls[0]
            name = (call.get('function') or {}).get('name', '')
            args = _parse_args(call)
            print(f'[{step}] {"(reason) " if reasoning else ""}-> '
                  f'{name}({", ".join(f"{k}={v}" for k, v in args.items())})')
            self._emit('tool_call', step=step, reasoning=reasoning, name=name, args=args)

            def record_tool(content):
                msg = {'role': 'tool', 'tool_call_id': call.get('id', ''), 'content': content}
                self.messages.append(msg)
                self.log.add(msg)

            def ack_skipped():
                for skipped in tool_calls[1:]:
                    msg = {'role': 'tool', 'tool_call_id': skipped.get('id', ''),
                           'content': json.dumps({'ok': False,
                               'error': 'skipped: only one action per turn is executed; '
                                        'resend this after reading the previous feedback'})}
                    self.messages.append(msg)
                    self.log.add(msg)

            turn_rec['tool_call'] = {'name': name, 'arguments': args}

            if name == 'think':
                self.think_calls += 1
                self.metrics.think_calls += 1
                escalate = True
                reason = args.get('reason', '')
                record_tool(json.dumps({'ok': True,
                                        'message': f'deliberation granted (reason: {reason})'}))
                ack_skipped()
                self.log.add_turn(turn_rec)
                self._emit('feedback', ok=True, message=f'deliberation granted (reason: {reason})')
                continue

            if name == 'get_state':
                current = self.robot.get_state()
                self.metrics.get_state_calls += 1
                record_tool(format_state(current))
                ack_skipped()
                print('    (state queried)')
                self.log.add_turn(turn_rec)
                self._emit('feedback', ok=True, message='state queried')
                continue

            if name == 'done':
                if config.pre_done_check and not reasoning and not pre_done_used:
                    pre_done_used = True
                    forced_trigger = 'pre_done'
                    record_tool(json.dumps({'ok': False,
                        'error': 'verify first: re-check that the task is fully and correctly '
                                 'complete (order, positions, count) before calling done().'}))
                    ack_skipped()
                    turn_rec['rejected'] = 'pre_done'
                    print('    (done rejected -- pre-done verification forced)')
                    self.log.add_turn(turn_rec)
                    self._emit('feedback', ok=False,
                               error='done rejected — verify the task is complete first')
                    continue
                reason = args.get('reason', '')
                fb = self.robot.send_command('done', reason=reason)
                record_tool(json.dumps(concise_feedback(fb, config.feedback_level)))
                turn_rec['feedback'] = concise_feedback(fb, config.feedback_level)
                self.log.add_turn(turn_rec)
                print(f'\nDONE: {reason}')
                self.log.set_metrics(self.metrics.finalize('done', step))
                # Models often write a user-facing summary AND call done() in the same turn
                # (kimi and phoenix both do). That prose, not the internal done() reason, is what
                # the conversation should carry forward -- and `spoke` tells a UI that the message
                # is already on screen, so it must not print the reason as a second reply.
                said = (resp.content or '').strip()
                self._compact(task, seed_index, said or reason or 'Task completed.')
                self._emit('done', reason=reason, steps=step, spoke=bool(said))
                return {'exit_code': 0, 'status': 'done', 'reason': reason, 'steps': step}

            # pick / place / store
            acted = True   # a build action -> from here a bare prose turn is nudged, not a chat exit
            fb = self.robot.send_command(name, **args)
            last_feedback = fb
            if not fb.get('ok'):
                self.metrics.tool_failures += 1
            _print_feedback(fb)
            # The MODEL sees the feedback at the configured informativeness level; the loop
            # itself keeps the raw dict (last_feedback above), so the on_error trigger fires
            # identically in every arm and only the readable text differs.
            record_tool(json.dumps(concise_feedback(fb, config.feedback_level)))
            ack_skipped()
            turn_rec['feedback'] = concise_feedback(fb, config.feedback_level)
            self.log.add_turn(turn_rec)
            self._emit('feedback', ok=bool(fb.get('ok')), message=fb.get('message'),
                       error=fb.get('error'), held=(fb.get('observation') or {}).get('held'))

        self.log.set_metrics(self.metrics.finalize('max_steps', config.max_steps))
        print(f'\nReached max steps ({config.max_steps}) without done(). Stopping.')
        self._compact(task, seed_index,
                      f'(Stopped after {config.max_steps} steps without finishing this task.)')
        self._emit('max_steps', steps=config.max_steps)
        return {'exit_code': 1, 'status': 'max_steps', 'reason': None, 'steps': config.max_steps}

    def _compact(self, task, seed_index, summary):
        """Collapse the just-finished task's turns in the PERSISTENT history down to a clean
        user->assistant pair: the task text (WITHOUT its volatile world-state block) and one
        assistant message whose content is the outcome summary (the done() reason).

        This is the demo's cross-task context engineering. The turn-by-turn reasoning/tool
        trail -- including any reasoning the cache wrote into assistant content -- drives the
        LIVE loop, but once a task ends only this clean "chat" carries into the next task:
        the next task is re-seeded with a FRESH world-state snapshot, so none of that trail is
        needed to continue, and the full trail is still preserved verbatim in the session log
        (log.add keeps its own list, so trimming self.messages does not touch it).

        No-op unless compact_history is set (the eval's single-episode path keeps the full
        model_view) or when there is no seed to compact (task is None)."""
        if not self.compact_history or seed_index is None:
            return
        if self.on_compact:   # hand the trail over before it is replaced (see __init__)
            self.on_compact([dict(m) for m in self.messages[seed_index:]])
        self.messages[seed_index:] = [
            {'role': 'user', 'content': task},
            {'role': 'assistant', 'content': summary},
        ]


def _run_agentic(config, robot, llm, log, metrics, messages, ids, grid):
    """Single-episode agentic run: one task (already seeded in `messages`), then finish
    the log. Behaviour identical to the pre-refactor loop (verified via the eval)."""
    session = AgenticSession(config, robot, llm, log, metrics, ids, grid, messages)
    result = session.run_task()   # task already seeded by run_episode
    if result['status'] == 'done':
        log.finish(status='done', reason=result['reason'] or '', steps=result['steps'])
    else:
        log.finish(status=result['status'], steps=result['steps'])   # max_steps / no_tool_call / chat
    return result['exit_code']


def _print_feedback(feedback):
    if feedback.get('ok'):
        print(f'    ok: {feedback.get("message", "")}')
    else:
        print(f'    ERROR: {feedback.get("error", "")}')
    print(f'    held: {(feedback.get("observation") or {}).get("held")}')
