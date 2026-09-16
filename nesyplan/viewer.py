"""Render an eval campaign into ONE self-contained report.html for reading traces.

The campaign layer already writes machine-readable artifacts (results.jsonl,
summary.md, per-run JSON transcripts, per-run PNGs). This turns them into a single
human-facing page you open in a browser -- the piece a LangFuse-style UI would give
you, but with zero infrastructure and no data leaving the machine:

  - a campaign header (model, backend, git sha, pass rate) + the success matrix and
    the cost-sorted per-config roll-up (the SAME numbers as summary.md -- both call
    nesyplan.aggregate, so they never drift);
  - one card per run: the final-state PNG (embedded), the judge verdict, and a
    STRUCTURED turn timeline (reasoning on/off + trigger, tool call, feedback, tokens,
    latency) with each turn's raw reasoning trace expandable inline, plus the full
    verbatim conversation (agent / cache-summarizer / judge sub-calls labeled);
  - a COMPARE panel: pick any two runs and read their timelines side by side (e.g.
    react vs oneshot on the same task).

Everything is inlined (CSS, a little JS, PNGs as data: URIs), so report.html is a
single portable file -- copy it anywhere, or publish it as an artifact. Stdlib only;
matplotlib is NOT needed (the PNGs were rendered at eval time).

Runnable standalone on any campaign dir:  python3 -m nesyplan.viewer results/<campaign>
Also written automatically at the end of an eval run (best-effort).
"""

import base64
import html
import json
import os
import sys

from nesyplan.aggregate import cell_map, cell_text, collect_axes, load_rows, rollup

JUDGE_MARKER = '[judge call'
SUMMARIZER_MARKER = '[cache summarizer call]'


# --- small helpers -----------------------------------------------------------

def _esc(text):
    return html.escape('' if text is None else str(text))


def _read_json(path):
    try:
        with open(path, encoding='utf-8') as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return None


def _data_uri_png(path):
    """Base64 data: URI for a PNG, so the report is a single self-contained file."""
    try:
        with open(path, 'rb') as fh:
            return 'data:image/png;base64,' + base64.b64encode(fh.read()).decode('ascii')
    except OSError:
        return None


def _get_trace(message):
    """The reasoning trace on an assistant message (provider-specific field), or ''."""
    for key in ('reasoning', 'reasoning_content'):
        val = (message or {}).get(key)
        if val:
            return val
    return ''


def _fmt_call(tc):
    """Render a tool call as name(k=v, ...)."""
    if not tc:
        return ''
    name = tc.get('name', '')
    args = tc.get('arguments') or {}
    inner = ', '.join(f'{k}={v}' for k, v in args.items())
    return f'{name}({inner})'


# --- transcript structure ----------------------------------------------------

def partition_messages(messages):
    """Tag each logged message with its channel: 'agent' | 'summarizer' | 'judge'.

    The C2 cache summarizer logs a 3-message side-call (system marker + user + assistant)
    inline right after a reasoning turn; the judge logs its 3-message call at the very
    end. Both are transparency artifacts, NOT part of what the agent saw -- separate them
    so the agent conversation reads cleanly and the sub-calls are clearly labeled.
    """
    out, i, n = [], 0, len(messages or [])
    while i < n:
        m = messages[i]
        content = m.get('content')
        marker = content if isinstance(content, str) else ''
        if m.get('role') == 'system' and marker.startswith(JUDGE_MARKER):
            out.extend(('judge', jm) for jm in messages[i:])   # judge = the tail
            break
        if m.get('role') == 'system' and marker.startswith(SUMMARIZER_MARKER):
            block = messages[i:i + 3]                          # system + user + assistant
            out.extend(('summarizer', sm) for sm in block)
            i += len(block)
            continue
        out.append(('agent', m))
        i += 1
    return out


def attach_reasoning(turns, partitioned):
    """Attach each LLM-call turn's raw reasoning/content from the agent assistant messages.

    A turn is an LLM call iff it carries a `usage` block (agentic turns all do; a
    one-shot run's blind execution steps do not). Agent assistant messages line up with
    those turns in order, so we walk them with a single pointer -- robust across modes,
    and a length mismatch simply leaves later turns without an inline trace.
    """
    assistants = [m for ch, m in partitioned if ch == 'agent' and m.get('role') == 'assistant']
    p = 0
    enriched = []
    for t in turns or []:
        t = dict(t)
        if 'usage' in t and p < len(assistants):
            msg = assistants[p]
            p += 1
            t['_reasoning'] = _get_trace(msg)
            t['_content'] = msg.get('content') if isinstance(msg.get('content'), str) else ''
        enriched.append(t)
    return enriched


# --- HTML fragments ----------------------------------------------------------

def _badge(text, kind=''):
    return f'<span class="badge {kind}">{_esc(text)}</span>'


def _mark_kind(success):
    return 'pass' if success is True else 'fail' if success is False else 'unk'


def _mark_label(success):
    return 'PASS' if success is True else 'fail' if success is False else '?'


def render_timeline(turns):
    """The structured per-turn timeline (the comparable view), reasoning expandable inline."""
    rows = []
    for t in turns:
        num = t.get('turn')
        reasoning = t.get('reasoning')
        trigger = t.get('trigger')
        rlabel = (f'reason · {trigger}' if trigger else 'reason') if reasoning else 'act'
        rkind = 'reason' if reasoning else 'act'

        parts = [f'<div class="t-head"><span class="t-num">#{_esc(num)}</span>'
                 f'{_badge(rlabel, rkind)}']

        if t.get('plan') is not None:   # one-shot planning turn
            steps = ' → '.join(_fmt_call(s) for s in t['plan'])
            parts.append(f'{_badge("plan", "act")}<code class="call">{_esc(steps)}</code>')
        else:
            call = _fmt_call(t.get('tool_call'))
            if call:
                parts.append(f'<code class="call">{_esc(call)}</code>')
            elif t.get('note'):
                parts.append(f'<span class="note">{_esc(t["note"])}</span>')
        if t.get('rejected'):
            parts.append(_badge(f'rejected: {t["rejected"]}', 'fail'))

        u = t.get('usage') or {}
        if u or t.get('latency_ms') is not None:
            meta = []
            if u.get('total') is not None:
                meta.append(f'{u.get("total")} tok')
            if u.get('reasoning'):
                meta.append(f'{u.get("reasoning")} reason')
            if t.get('latency_ms') is not None:
                meta.append(f'{t["latency_ms"]} ms')
            parts.append(f'<span class="meta">{_esc(" · ".join(meta))}</span>')
        parts.append('</div>')

        fb = t.get('feedback')
        if fb:
            if fb.get('ok'):
                parts.append(f'<div class="fb ok">ok: {_esc(fb.get("message", ""))}</div>')
            else:
                parts.append(f'<div class="fb err">ERROR: {_esc(fb.get("error", ""))}</div>')

        trace = t.get('_reasoning')
        content = t.get('_content')
        if trace or content:
            body = ''
            if trace:
                body += f'<div class="rlabel">reasoning</div><pre>{_esc(trace)}</pre>'
            if content and content.strip():
                body += f'<div class="rlabel">message</div><pre>{_esc(content)}</pre>'
            parts.append(f'<details class="reason-box"><summary>reasoning trace</summary>'
                         f'{body}</details>')

        rows.append(f'<div class="turn {rkind}">' + ''.join(parts) + '</div>')
    return '<div class="timeline">' + ''.join(rows) + '</div>'


def render_raw(partitioned):
    """Verbatim conversation dump, each message labeled by channel + role."""
    blocks = []
    for ch, m in partitioned:
        role = m.get('role', '?')
        chtag = '' if ch == 'agent' else f'<span class="ch {ch}">{_esc(ch)}</span>'
        head = f'<div class="raw-head">{chtag}<span class="role {role}">{_esc(role)}</span></div>'
        body = ''
        trace = _get_trace(m)
        if trace:
            body += f'<div class="rlabel">reasoning</div><pre>{_esc(trace)}</pre>'
        content = m.get('content')
        if isinstance(content, str) and content.strip():
            body += f'<pre>{_esc(content)}</pre>'
        for tc in (m.get('tool_calls') or []):
            fn = tc.get('function') or {}
            body += f'<pre class="tc">→ {_esc(fn.get("name", ""))}({_esc(fn.get("arguments", ""))})</pre>'
        blocks.append(f'<div class="raw-msg">{head}{body}</div>')
    return ('<details class="raw"><summary>full verbatim conversation '
            '(agent · summarizer · judge)</summary>' + ''.join(blocks) + '</details>')


def render_run_card(row, campaign_dir):
    """One run: header stats + PNG + judge verdict + timeline + raw conversation."""
    task = row.get('task')
    config = row.get('config')
    rep = row.get('rep', 0)
    rid = row.get('session_id') or f'{task}__{config}__{rep}'
    success = row.get('success')

    header = [f'<span class="run-title">{_esc(task)}</span>'
              f'<span class="run-cfg">{_esc(config)}</span>',
              _badge(_mark_label(success), _mark_kind(success))]
    conf = row.get('judge_confidence')
    if conf is not None:
        header.append(_badge(f'conf {conf:.2f}', 'info'))
    for label, key, suffix in (('outcome', 'outcome', ''), ('steps', 'steps', ''),
                               ('tok', 'tokens_total', ''), ('reason-tok', 'tokens_reasoning', ''),
                               ('reason-turns', 'reasoning_turns_total', ''),
                               ('fails', 'tool_failures', '')):
        v = row.get(key)
        if v not in (None, ''):
            header.append(_badge(f'{label} {v}{suffix}', 'stat'))
    lat = row.get('latency_ms_total')
    if lat is not None:
        header.append(_badge(f'{lat / 1000.0:.1f} s', 'stat'))

    body = []
    img = row.get('image_path')
    uri = _data_uri_png(os.path.join(campaign_dir, img)) if img else None
    if uri:
        body.append(f'<img class="state" src="{uri}" alt="final state">')

    detail = row.get('judge_reasoning') or row.get('check_detail')
    if detail:
        body.append(f'<div class="judge"><b>judge:</b> {_esc(detail)}</div>')
    if row.get('error'):
        body.append(f'<div class="fb err">run error: {_esc(row["error"])}</div>')

    session = _read_json(os.path.join(campaign_dir, row['log_path'])) if row.get('log_path') else None
    if session:
        turns = attach_reasoning(session.get('turns') or [], partition_messages(session.get('messages') or []))
        timeline = render_timeline(turns)
        body.append(f'<div class="content-row"><div class="tl-wrap">{timeline}</div></div>')
        body.append(render_raw(partition_messages(session.get('messages') or [])))
    else:
        timeline = '<div class="note">transcript not found</div>'
        body.append(timeline)

    card = (f'<section class="run" id="run-{_esc(rid)}">'
            f'<div class="run-head">{"".join(header)}</div>'
            f'<div class="run-body">{"".join(body)}</div></section>')
    return rid, card, timeline


# --- page assembly -----------------------------------------------------------

def _overview(rows):
    configs, tasks, task_diff = collect_axes(rows)
    cell = cell_map(rows)

    # success matrix
    head = '<tr><th>config \\ task</th>' + ''.join(
        f'<th>{_esc(t)}<span class="diff">{_esc(task_diff[t])}</span></th>' for t in tasks) + '</tr>'
    body = []
    for cfg in configs:
        tds = [f'<th class="rowhdr">{_esc(cfg)}</th>']
        for t in tasks:
            runs = cell.get((cfg, t), [])
            txt = cell_text(runs)
            kind = _mark_kind(runs[0].get('success')) if len(runs) == 1 else 'unk'
            anchor = runs[0].get('session_id') if runs else None
            inner = f'<a href="#run-{_esc(anchor)}">{_esc(txt)}</a>' if anchor else _esc(txt)
            tds.append(f'<td class="cell {kind}">{inner}</td>')
        body.append('<tr>' + ''.join(tds) + '</tr>')
    matrix = f'<table class="matrix">{head}{"".join(body)}</table>'

    # roll-up (cheapest-first) -- same computation summary.md uses
    # tok/reason~ is the trace-length estimate (nesyplan/metrics.py): the endpoint returns no
    # reasoning_tokens, so the raw tok_reason column is 0 for every run ever recorded.
    cols = [('config', 'config'), ('success', 'success'), ('rate', None), ('judge-conf', None),
            ('tok/tot', 'tok_total'), ('tok/reason~', 'tok_reason_est'), ('tok/cache', 'tok_cache'),
            ('reason/turns', 'reason_turns'), ('steps', 'steps'), ('fails', 'tool_failures'),
            ('lat(s)', 'latency_s')]
    rhead = '<tr>' + ''.join(f'<th>{_esc(c)}</th>' for c, _ in cols) + '</tr>'
    rbody = []
    for d in rollup(rows, configs):
        cells = [
            _esc(d['config']), _esc(d['success']), f'{d["success_rate"] * 100:.0f}%',
            (f'{d["conf"]:.2f}' if d['conf'] is not None else '-'),
            f'{d["tok_total"]:.0f}', f'{d["tok_reason_est"]:.0f}', f'{d["tok_cache"]:.0f}',
            f'{d["reason_turns"]:.1f}', f'{d["steps"]:.1f}', f'{d["tool_failures"]:.1f}',
            f'{d["latency_s"]:.1f}']
        rbody.append('<tr>' + ''.join(f'<td>{c}</td>' for c in cells) + '</tr>')
    roll = f'<table class="roll">{rhead}{"".join(rbody)}</table>'
    return matrix, roll


CSS = """
:root { --bg:#fff; --fg:#1a1a1a; --muted:#666; --line:#e2e2e2; --card:#fafafa;
  --pass:#1a7f37; --passbg:#e8f6ec; --fail:#c1272d; --failbg:#fdeaea; --unk:#8a8a8a; --unkbg:#f0f0f0;
  --reason:#5a3fbf; --reasonbg:#efeaff; --act:#0b6bcb; --actbg:#e8f1fd; --warn:#a86500; --warnbg:#fdf3e0; }
@media (prefers-color-scheme: dark) {
  :root { --bg:#16171a; --fg:#e6e6e6; --muted:#9aa0a6; --line:#2c2f36; --card:#1d1f24;
    --passbg:#153021; --failbg:#3a1c1e; --unkbg:#26282d; --reasonbg:#241d3d; --actbg:#132840; --warnbg:#332612; } }
* { box-sizing:border-box; }
body { margin:0; font:14px/1.5 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif;
  color:var(--fg); background:var(--bg); }
.wrap { max-width:1200px; margin:0 auto; padding:24px; }
h1 { font-size:22px; margin:0 0 4px; }
h2 { font-size:16px; margin:28px 0 10px; padding-bottom:4px; border-bottom:1px solid var(--line); }
.sub { color:var(--muted); margin-bottom:8px; }
code, pre { font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace; }
table { border-collapse:collapse; font-size:13px; margin:6px 0 14px; }
th, td { border:1px solid var(--line); padding:5px 9px; text-align:left; }
.matrix td.cell { text-align:center; font-weight:600; }
.matrix a { text-decoration:none; color:inherit; display:block; }
.cell.pass { background:var(--passbg); color:var(--pass); }
.cell.fail { background:var(--failbg); color:var(--fail); }
.cell.unk  { background:var(--unkbg);  color:var(--unk); }
.rowhdr { background:var(--card); }
.diff { color:var(--muted); font-weight:400; margin-left:5px; font-size:11px; }
.roll tr:first-child { font-weight:600; }
.badge { display:inline-block; padding:1px 7px; border-radius:10px; font-size:11px; font-weight:600;
  margin-right:5px; background:var(--unkbg); color:var(--fg); white-space:nowrap; }
.badge.pass { background:var(--passbg); color:var(--pass); }
.badge.fail { background:var(--failbg); color:var(--fail); }
.badge.unk  { background:var(--unkbg);  color:var(--unk); }
.badge.info { background:var(--actbg);  color:var(--act); }
.badge.stat { background:transparent; color:var(--muted); border:1px solid var(--line); }
.badge.reason { background:var(--reasonbg); color:var(--reason); }
.badge.act { background:var(--actbg); color:var(--act); }
.badge.warn { background:var(--warnbg); color:var(--warn); }
.run { border:1px solid var(--line); border-radius:8px; margin:14px 0; overflow:hidden; }
.run-head { background:var(--card); padding:9px 12px; display:flex; align-items:center; flex-wrap:wrap; gap:4px 0;
  border-bottom:1px solid var(--line); }
.run-title { font-weight:700; margin-right:8px; }
.run-cfg { color:var(--muted); margin-right:12px; }
.run-body { padding:12px; }
img.state { max-width:340px; float:right; margin:0 0 10px 16px; border:1px solid var(--line); border-radius:6px; }
.judge { margin:2px 0 12px; padding:8px 10px; background:var(--card); border-radius:6px; }
.timeline { display:flex; flex-direction:column; gap:6px; }
.turn { border-left:3px solid var(--line); padding:5px 10px; background:var(--card); border-radius:0 6px 6px 0; }
.turn.reason { border-left-color:var(--reason); }
.turn.act { border-left-color:var(--act); }
.t-head { display:flex; align-items:center; flex-wrap:wrap; gap:6px; }
.t-num { color:var(--muted); font-weight:700; min-width:34px; }
.call { background:var(--bg); border:1px solid var(--line); border-radius:5px; padding:1px 6px; }
.meta { color:var(--muted); font-size:12px; margin-left:auto; }
.note { color:var(--muted); font-style:italic; }
.fb { margin:5px 0 0 40px; font-size:13px; }
.fb.ok { color:var(--pass); }
.fb.err { color:var(--fail); }
.reason-box, .raw { margin-top:6px; }
.reason-box summary, .raw summary { cursor:pointer; color:var(--act); font-size:12px; }
.rlabel { color:var(--muted); font-size:11px; text-transform:uppercase; margin:6px 0 2px; letter-spacing:.04em; }
pre { white-space:pre-wrap; word-break:break-word; background:var(--bg); border:1px solid var(--line);
  border-radius:6px; padding:8px 10px; margin:2px 0; font-size:12px; max-height:420px; overflow:auto; }
.raw-msg { border-top:1px dashed var(--line); padding:6px 0; }
.raw-head { display:flex; gap:6px; align-items:center; margin-bottom:2px; }
.role { font-weight:700; font-size:12px; }
.ch { font-size:11px; padding:0 6px; border-radius:8px; background:var(--warnbg); color:var(--warn); }
.ch.judge { background:var(--reasonbg); color:var(--reason); }
.tc { color:var(--act); }
.cmp { display:flex; gap:14px; align-items:flex-start; }
.cmp > div { flex:1; min-width:0; }
select { font:13px inherit; padding:4px 6px; border:1px solid var(--line); border-radius:6px;
  background:var(--bg); color:var(--fg); }
.cmp-controls { display:flex; gap:16px; align-items:center; margin-bottom:10px; flex-wrap:wrap; }
"""

COMPARE_JS = """
const T = __TIMELINES__;
function render(side){
  const id = document.getElementById('sel-'+side).value;
  document.getElementById('col-'+side).innerHTML = T[id] || '<div class="note">—</div>';
}
window.addEventListener('DOMContentLoaded', () => { render('a'); render('b'); });
"""


def build_report(campaign_dir):
    rows = load_rows(campaign_dir)
    campaign = os.path.basename(os.path.normpath(campaign_dir))
    manifest = _read_json(os.path.join(campaign_dir, 'manifest.json')) or {}

    if not rows:
        return (f'<div class="wrap"><h1>Eval report: {_esc(campaign)}</h1>'
                f'<p class="sub">No results yet.</p></div>')

    total = len(rows)
    passed = sum(1 for r in rows if r.get('success') is True)
    model = manifest.get('model') or next((r.get('model') for r in rows if r.get('model')), '?')
    meta_bits = [f'{total} runs', f'{passed}/{total} passed ({passed / total * 100:.0f}%)']
    for label, key in (('model', None), ('backend', 'backend'), ('target', 'target'),
                       ('scorer', 'scorer'), ('git', 'git_sha')):
        if key and manifest.get(key):
            v = manifest[key]
            meta_bits.append(f'{label} {v[:10]}' if key == 'git_sha' else f'{label} {v}')
    sub = f'model <code>{_esc(model)}</code> · ' + ' · '.join(_esc(b) for b in meta_bits)

    matrix, roll = _overview(rows)

    cards, timelines, options = [], {}, []
    for r in rows:
        rid, card, timeline = render_run_card(r, campaign_dir)
        cards.append(card)
        timelines[rid] = timeline
        model_short = (r.get('model') or '').split('/')[-1]
        label = (f'{model_short} · ' if model_short else '') + f'{r.get("task")} × {r.get("config")}'
        if r.get('rep'):
            label += f' (rep {r.get("rep")})'
        options.append((rid, label))

    opts_html = ''.join(f'<option value="{_esc(rid)}">{_esc(lbl)}</option>' for rid, lbl in options)
    sel_b_default = options[1][0] if len(options) > 1 else options[0][0]
    opts_b_html = ''.join(
        f'<option value="{_esc(rid)}"{" selected" if rid == sel_b_default else ""}>{_esc(lbl)}</option>'
        for rid, lbl in options)

    # Embed the timelines for the compare panel; guard against </script> in content.
    tl_json = json.dumps(timelines).replace('</', '<\\/')
    js = COMPARE_JS.replace('__TIMELINES__', tl_json)

    compare = (
        '<h2>Compare two runs</h2>'
        '<div class="cmp-controls">'
        f'<label>A: <select id="sel-a" onchange="render(\'a\')">{opts_html}</select></label>'
        f'<label>B: <select id="sel-b" onchange="render(\'b\')">{opts_b_html}</select></label>'
        '</div>'
        '<div class="cmp"><div id="col-a"></div><div id="col-b"></div></div>')

    return (
        f'<div class="wrap">'
        f'<h1>Eval report: {_esc(campaign)}</h1><p class="sub">{sub}</p>'
        f'<h2>Success matrix (config × task)</h2>{matrix}'
        f'<h2>Per-config roll-up (cheapest first)</h2>{roll}'
        f'{compare}'
        f'<h2>Runs ({total})</h2>{"".join(cards)}'
        f'</div>'
        f'<script>{js}</script>')


def build_page(campaign_dir):
    """Full standalone HTML document (used by the CLI / eval hook)."""
    campaign = os.path.basename(os.path.normpath(campaign_dir))
    return (f'<!doctype html><html lang="en"><head><meta charset="utf-8">'
            f'<meta name="viewport" content="width=device-width, initial-scale=1">'
            f'<title>Eval report: {_esc(campaign)}</title><style>{CSS}</style></head>'
            f'<body>{build_report(campaign_dir)}</body></html>')


def write_report(campaign_dir, filename='report.html'):
    path = os.path.join(campaign_dir, filename)
    with open(path, 'w', encoding='utf-8') as fh:
        fh.write(build_page(campaign_dir))
    return path


def main(argv=None):
    argv = argv if argv is not None else sys.argv[1:]
    if not argv:
        print('usage: python3 -m nesyplan.viewer <campaign_dir>', file=sys.stderr)
        return 2
    campaign_dir = argv[0]
    if not os.path.isdir(campaign_dir):
        print(f'error: not a directory: {campaign_dir}', file=sys.stderr)
        return 2
    path = write_report(campaign_dir)
    print(f'[wrote {path}]  open it in a browser')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
