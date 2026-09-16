"""Aggregate a campaign's results.jsonl into a human-readable summary.md.

Reads the flat, one-row-per-run index that eval.py streams out and produces:
  1. a success matrix (config x task) -- the at-a-glance "what worked",
  2. a per-config roll-up (success rate over tasks, mean tokens split, steps, latency)
     sorted cheapest-first -- a first read on the cost/robustness Pareto front.

Runnable standalone on any campaign dir:  python3 -m nesyplan.aggregate results/<campaign>
Also called at the end of an eval run. Stdlib only; no pandas dependency.
"""

import json
import os
import sys


def load_rows(campaign_dir):
    path = os.path.join(campaign_dir, 'results.jsonl')
    rows = []
    if not os.path.isfile(path):
        return rows
    with open(path, encoding='utf-8') as fh:
        for line in fh:
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except ValueError:
                    pass
    return rows


def _mark(success):
    if success is True:
        return 'PASS'
    if success is False:
        return 'fail'
    return '?'


def _mean(vals):
    vals = [v for v in vals if isinstance(v, (int, float))]
    return sum(vals) / len(vals) if vals else 0.0


def _table(headers, rows):
    """Render a GitHub-flavored markdown table with padded columns."""
    widths = [len(h) for h in headers]
    for r in rows:
        for i, cell in enumerate(r):
            widths[i] = max(widths[i], len(str(cell)))
    def fmt(cells):
        return '| ' + ' | '.join(str(c).ljust(widths[i]) for i, c in enumerate(cells)) + ' |'
    sep = '| ' + ' | '.join('-' * widths[i] for i in range(len(headers))) + ' |'
    return '\n'.join([fmt(headers), sep] + [fmt(r) for r in rows])


def collect_axes(rows):
    """First-seen-ordered configs + tasks (matches run order = ladder/matrix) + difficulty map."""
    configs, tasks, task_diff = [], [], {}
    for r in rows:
        if r.get('config') not in configs:
            configs.append(r.get('config'))
        t = r.get('task')
        if t not in tasks:
            tasks.append(t)
            task_diff[t] = r.get('difficulty', '')
    return configs, tasks, task_diff


def cell_map(rows):
    """(config, task) -> list of runs (multiple only when seeds/reps > 1)."""
    cell = {}
    for r in rows:
        cell.setdefault((r.get('config'), r.get('task')), []).append(r)
    return cell


def rollup(rows, configs=None):
    """Per-config roll-up dicts (success rate, mean token split, steps, latency), cheapest-first.

    Pure data -- shared by build_summary (markdown) and nesyplan.viewer (HTML) so the
    cost/robustness numbers are computed in exactly one place.
    """
    if configs is None:
        configs, _, _ = collect_axes(rows)
    roll = []
    for cfg in configs:
        cfg_rows = [r for r in rows if r.get('config') == cfg]
        n = len(cfg_rows)
        passed = sum(1 for r in cfg_rows if r.get('success') is True)
        confs = [r.get('judge_confidence') for r in cfg_rows]
        roll.append({
            'config': cfg,
            'success': f'{passed}/{n}' if n else '0/0',
            'success_rate': (passed / n) if n else 0.0,
            'conf': _mean(confs) if any(c is not None for c in confs) else None,
            'tok_total': _mean([r.get('tokens_total') for r in cfg_rows]),
            'tok_reason': _mean([r.get('tokens_reasoning') for r in cfg_rows]),
            # The reasoning cost that actually carries signal: the provider never reports
            # reasoning_tokens (tok_reason is 0 in every campaign to date), so this is the
            # trace-length proxy from nesyplan/metrics.py.
            'tok_reason_est': _mean([r.get('tokens_reasoning_est') for r in cfg_rows]),
            'tok_cache': _mean([r.get('tokens_cache_injection') for r in cfg_rows]),
            'reason_turns': _mean([r.get('reasoning_turns_total') for r in cfg_rows]),
            'steps': _mean([r.get('steps') for r in cfg_rows]),
            'tool_failures': _mean([r.get('tool_failures') for r in cfg_rows]),
            'latency_s': _mean([r.get('latency_ms_total') for r in cfg_rows]) / 1000.0,
        })
    roll.sort(key=lambda d: d['tok_total'])   # cheapest-first: a glance at the frontier
    return roll


def cell_text(runs):
    """Matrix cell: PASS/fail/? for a single run, or passed/total when reps > 1."""
    if not runs:
        return ''
    succ = [x.get('success') for x in runs]
    if len(runs) == 1:
        return _mark(succ[0])
    passed = sum(1 for s in succ if s is True)
    return f'{passed}/{len(runs)}'


def _model_section(rows, heading):
    """Success matrix + roll-up for ONE model's rows."""
    configs, tasks, task_diff = collect_axes(rows)
    cell = cell_map(rows)

    # --- table 1: success matrix (config x task) ---
    headers = ['config'] + [f'{t} ({task_diff[t]})' for t in tasks]
    matrix = [[cfg] + [cell_text(cell.get((cfg, t), [])) for t in tasks] for cfg in configs]

    # --- table 2: per-config roll-up ---
    roll = rollup(rows, configs)
    roll_rows = [[
        d['config'], d['success'], f'{d["success_rate"]*100:.0f}%',
        f'{d["tok_total"]:.0f}', f'{d["tok_reason_est"]:.0f}', f'{d["tok_cache"]:.0f}',
        f'{d["reason_turns"]:.1f}', f'{d["steps"]:.1f}', f'{d["tool_failures"]:.1f}',
        f'{d["latency_s"]:.1f}',
    ] for d in roll]
    roll_headers = ['config', 'success', 'rate', 'tok/tot', 'tok/reason~',
                    'tok/cache', 'reason/turns', 'steps', 'fails', 'lat(s)']

    total = len(rows)
    passed = sum(1 for r in rows if r.get('success') is True)
    return [
        heading,
        '',
        f'{total} runs · {passed}/{total} passed ({(passed / total * 100):.0f}%) · '
        f'{len(configs)} configs × {len(tasks)} tasks',
        '',
        _table(headers, matrix),
        '',
        _table(roll_headers, roll_rows),
        '',
    ]


def _scoring_note(rows):
    """One line on which scorer decided, and how often the judge disagreed with the checker."""
    scorers = {}
    for r in rows:
        s = r.get('scorer')
        if s:
            scorers[s] = scorers.get(s, 0) + 1
    if not scorers:
        return []
    parts = ', '.join(f'{v} {k}' for k, v in sorted(scorers.items()))
    agree = [r.get('scorer_agree') for r in rows if r.get('scorer_agree') is not None]
    line = f'Scored by: {parts}.'
    if agree:
        n_agree = sum(1 for a in agree if a)
        line += (f' Judge agreed with the symbolic checker on {n_agree}/{len(agree)} '
                 f'code-scored runs ({n_agree / len(agree) * 100:.0f}%).')
    return ['', line, '']


def build_summary(campaign_dir, rows=None):
    rows = rows if rows is not None else load_rows(campaign_dir)
    campaign = os.path.basename(os.path.normpath(campaign_dir))
    if not rows:
        return f'# Eval summary: {campaign}\n\n_No results yet._\n'

    # Models are reported SEPARATELY. Pooling them into one matrix (the earlier behaviour)
    # made a "2/3" cell read as three reps of one model when it was really three different
    # models -- see results/smalleval_3models/summary.md, whose header names one model for
    # 48 runs spanning three.
    models = []
    for r in rows:
        m = r.get('model') or '?'
        if m not in models:
            models.append(m)

    total = len(rows)
    passed_total = sum(1 for r in rows if r.get('success') is True)
    out = [
        f'# Eval summary: {campaign}',
        '',
        f'{len(models)} model(s) · {total} runs · {passed_total}/{total} passed '
        f'({(passed_total / total * 100):.0f}%)',
    ]
    out += _scoring_note(rows)
    out += [
        'Each section is ONE model: success matrix (config × task), then the per-config '
        'roll-up sorted cheapest-first. `tok/reason~` is the trace-length ESTIMATE of '
        'reasoning tokens (the endpoint reports none); `tok/cache` is summarizer overhead; '
        '`fails` is rejected actions per run.',
        '',
    ]
    for m in models:
        mrows = [r for r in rows if (r.get('model') or '?') == m]
        out += _model_section(mrows, f'## `{m}`')
    return '\n'.join(out)


def write_summary(campaign_dir, rows=None):
    text = build_summary(campaign_dir, rows=rows)
    path = os.path.join(campaign_dir, 'summary.md')
    with open(path, 'w', encoding='utf-8') as fh:
        fh.write(text)
    return path


def main(argv=None):
    argv = argv if argv is not None else sys.argv[1:]
    if not argv:
        print('usage: python3 -m nesyplan.aggregate <campaign_dir>', file=sys.stderr)
        return 2
    campaign_dir = argv[0]
    if not os.path.isdir(campaign_dir):
        print(f'error: not a directory: {campaign_dir}', file=sys.stderr)
        return 2
    path = write_summary(campaign_dir)
    print(build_summary(campaign_dir))
    print(f'\n[wrote {path}]')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
