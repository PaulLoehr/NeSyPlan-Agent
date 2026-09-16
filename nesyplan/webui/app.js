'use strict';
/* NeSyPlan agent demonstrator -- Vue 3 frontend (no build step; vendored global build).
   Bootstraps from GET /config, streams live agent events from GET /events (SSE) into a
   reactive chat, and POSTs tasks/commands. The transcript rebuilds from the server's
   replayed event history on every (re)connect -- so `busy`/`stopping` are derived from the
   task_start / stopping / task_end events, never from local optimism. */

const { createApp, ref, reactive, computed, onMounted, nextTick, watch } = Vue;

const CUBE_COLORS = { red: '#e5484d', green: '#46a758', blue: '#3e63dd', yellow: '#f5d90a',
  black: '#3a3f47', white: '#e8ebf0', orange: '#f76b15', purple: '#8e4ec6' };

function esc(s) { return (s == null ? '' : String(s)).replace(/[&<>]/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;' }[c])); }
/* Perceived brightness of a cube colour -- the white and yellow cubes need DARK digits to
   stay readable, every other one white ones. */
function isLightColor(hex) {
  const h = String(hex || '').replace('#', '');
  const n = parseInt(h.length === 3 ? h.replace(/./g, '$&$&') : h, 16) || 0;
  return (0.299 * ((n >> 16) & 255) + 0.587 * ((n >> 8) & 255) + 0.114 * (n & 255)) > 150;
}
function fmtTok(n) { n = n || 0; return n >= 1000 ? (n / 1000).toFixed(1) + 'k' : String(n); }
function fmtArgs(args) {
  args = args || {};
  return Object.keys(args).map(k => { let v = args[k]; if (typeof v === 'string') v = '"' + v + '"'; return k + '=' + v; }).join(', ');
}

/* Minimal, self-contained Markdown -> safe HTML (no external lib). Escapes HTML FIRST so
   model text can never inject markup; URLs are scheme-checked. @@...@@ ASCII placeholder
   sentinels are collision-proof for chat prose (e.g. "C4"/"L2" are never mis-restored). */
function renderMarkdown(src) {
  src = String(src == null ? '' : src);
  const fences = [];
  src = src.replace(/```[^\n]*\n?([\s\S]*?)```/g, (m, code) => {
    fences.push(code.replace(/\n+$/, ''));
    return `@@FENCE${fences.length - 1}@@`;
  });
  const lines = src.split('\n');
  const out = [];
  let para = [];
  const flush = () => { if (para.length) { out.push('<p>' + para.map(mdInline).join('<br>') + '</p>'); para = []; } };
  let i = 0;
  while (i < lines.length) {
    const line = lines[i];
    const fence = line.match(/^@@FENCE(\d+)@@$/);
    if (fence) { flush(); out.push('<pre><code>' + esc(fences[+fence[1]]) + '</code></pre>'); i++; continue; }
    if (/^\s*$/.test(line)) { flush(); i++; continue; }
    const h = line.match(/^(#{1,6})\s+(.*)$/);
    if (h) { flush(); const n = h[1].length; out.push(`<h${n}>` + mdInline(h[2]) + `</h${n}>`); i++; continue; }
    if (/^\s*([-*_])(\s*\1){2,}\s*$/.test(line)) { flush(); out.push('<hr>'); i++; continue; }
    if (/^\s*[-*+]\s+/.test(line)) {
      flush(); const items = [];
      while (i < lines.length && /^\s*[-*+]\s+/.test(lines[i])) { items.push(lines[i].replace(/^\s*[-*+]\s+/, '')); i++; }
      out.push('<ul>' + items.map(it => '<li>' + mdInline(it) + '</li>').join('') + '</ul>'); continue;
    }
    if (/^\s*\d+[.)]\s+/.test(line)) {
      flush(); const items = [];
      while (i < lines.length && /^\s*\d+[.)]\s+/.test(lines[i])) { items.push(lines[i].replace(/^\s*\d+[.)]\s+/, '')); i++; }
      out.push('<ol>' + items.map(it => '<li>' + mdInline(it) + '</li>').join('') + '</ol>'); continue;
    }
    if (/^\s*>\s?/.test(line)) {
      flush(); const q = [];
      while (i < lines.length && /^\s*>\s?/.test(lines[i])) { q.push(lines[i].replace(/^\s*>\s?/, '')); i++; }
      out.push('<blockquote>' + q.map(mdInline).join('<br>') + '</blockquote>'); continue;
    }
    para.push(line); i++;
  }
  flush();
  return out.join('\n').replace(/@@FENCE(\d+)@@/g, (m, n) => '<pre><code>' + esc(fences[+n]) + '</code></pre>');
}

function mdInline(text) {
  text = esc(text);
  const codes = [];
  text = text.replace(/`([^`]+)`/g, (m, c) => { codes.push(c); return `@@CODE${codes.length - 1}@@`; });
  const links = [];
  text = text.replace(/\[([^\]]+)\]\(([^)\s]+)\)/g, (m, t, u) => {
    const safe = /^(https?:|mailto:|\/|#)[^"'\s]*$/i.test(u) ? u : '#';
    links.push(`<a href="${safe}" target="_blank" rel="noopener">${t}</a>`);
    return `@@LINK${links.length - 1}@@`;
  });
  text = text
    .replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>')
    .replace(/__([^_]+)__/g, '<strong>$1</strong>')
    .replace(/\*([^*\n]+)\*/g, '<em>$1</em>')
    .replace(/(^|[^\w])_([^_\n]+)_(?=[^\w]|$)/g, '$1<em>$2</em>')
    .replace(/~~([^~]+)~~/g, '<del>$1</del>');
  return text
    .replace(/@@LINK(\d+)@@/g, (m, n) => links[+n])
    .replace(/@@CODE(\d+)@@/g, (m, n) => '<code>' + codes[+n] + '</code>');
}

/* ---------- components ---------- */

/* ---------- theme ----------
   Three states, because "follow the OS" has to stay reachable: auto -> light -> dark. `auto`
   removes the attribute and lets the stylesheet's `color-scheme: light dark` track the system.
   Applied HERE, at load, before Vue mounts, so a stored choice never flashes the other theme.
   The stylesheet does the rest: every colour is a light-dark() pair keyed on color-scheme. */
const THEMES = ['auto', 'light', 'dark'];
const THEME_KEY = 'nesyplan.theme';

function applyTheme(name) {
  if (name === 'auto') document.documentElement.removeAttribute('data-theme');
  else document.documentElement.setAttribute('data-theme', name);
}
function storedTheme() {
  try { const v = localStorage.getItem(THEME_KEY); return THEMES.includes(v) ? v : 'auto'; }
  catch (e) { return 'auto'; }   // storage can be blocked; the OS default is a fine fallback
}
applyTheme(storedTheme());

/* Which rails are unfolded. Same storage pattern as the theme. Defaults: the controls FOLDED
   (a demo is driven from the chat, and model/mode are set once), the world state OPEN (that is
   the thing to watch). The key is versioned: bumping it retires stored choices from the old
   defaults, so a changed default is actually seen instead of masked by localStorage. */
const PANELS_KEY = 'nesyplan.panels.v2';
const PANEL_DEFAULTS = { left: false, right: true };
function storedPanels() {
  try {
    const v = JSON.parse(localStorage.getItem(PANELS_KEY) || '{}');
    return { left: typeof v.left === 'boolean' ? v.left : PANEL_DEFAULTS.left,
             right: typeof v.right === 'boolean' ? v.right : PANEL_DEFAULTS.right };
  } catch (e) { return Object.assign({}, PANEL_DEFAULTS); }
}

/* Seconds -- this text is prose in the stream ("Thought for 7.7 s"), and a thinking time is
   never interesting to the millisecond. */
function fmtSecs(ms) {
  const s = (ms || 0) / 1000;
  return (s < 10 ? s.toFixed(1) : String(Math.round(s))) + ' s';
}

/* Why the loop reasoned on this turn (nesyplan/policy.py's trigger vocabulary). Reading these
   off the stream is the point of the demonstrator: they ARE the invocation policy at work. */
const TRIGGER_LABEL = {
  initial: 'first turn', scheduled: 'scheduled', hard_failure: 'after failure',
  self: 'self-requested', pre_done: 'before done()',
};
const TRIGGER_TITLE = {
  initial: 'Turn 1 of this task — the mode forces reasoning here.',
  scheduled: 'The mode schedules reasoning for this turn.',
  hard_failure: 'A tool failed — the on-error trigger forced reasoning.',
  self: 'The model asked for a thinking step itself, via think().',
  pre_done: 'done() arrived without reasoning — the finish is repeated with reasoning.',
};

const ChatItem = {
  props: ['item'],
  methods: {
    md(t) { return renderMarkdown(t); }, args(a) { return fmtArgs(a); },
    trigLabel(t) { return TRIGGER_LABEL[t] || t || ''; },
    trigTitle(t) { return TRIGGER_TITLE[t] || ''; },
    secs(ms) { return fmtSecs(ms); },
    toks(n) { return n >= 1000 ? (n / 1000).toFixed(1).replace('.0', '') + 'k' : String(n || 0); },
  },
  // One row per event: a dot in the left gutter, the text next to it, nothing around it. The
  // dots are joined by a hairline (see .item::before) so a task reads as one thread. What a dot
  // encodes: grey = the model (reasoning / prose), green = a tool, red = a tool that failed;
  // pulsing = still running, solid = finished. The task separator carries only the MODE -- the
  // task text itself is the user line right below it, and printing it twice was pure noise.
  template: `
    <div class="item ask" v-if="item.kind==='user'">
      <div class="txt user">{{ item.text }}</div></div>

    <div class="item msg" v-else-if="item.kind==='agent'">
      <i class="dot" :class="{ok:item.done, warn:item.warn}"></i>
      <div class="txt md" :class="{done:item.done, warn:item.warn}"
        :title="item.doneReason ? 'done(): ' + item.doneReason : undefined" v-html="md(item.text)"></div></div>

    <div class="item divider" v-else-if="item.kind==='divider'">
      <span v-if="item.task==='/cleanup'">Clean up</span><span class="mode-tag">{{ item.mode }}</span></div>

    <div class="item" v-else-if="item.kind==='think'">
      <i class="dot" :class="{pulse:item.pending}"></i>
      <div class="think-live" v-if="item.pending">{{ item.stopping ? 'stopping'
        : (item.reasoning ? 'Thinking' : 'Answering') }}
        <span class="elapsed">{{ item.elapsed }}</span></div>
      <details class="think" v-else>
        <summary><span class="th-lead">Thought for {{ secs(item.ms) }}</span><span class="chev">▶</span>
          <span class="th-meta">
            <span v-if="item.trigger" :class="item.trigger" :title="trigTitle(item.trigger)">{{ trigLabel(item.trigger) }}</span>
            <span v-if="item.tokens" :title="item.estimated ? 'estimated from the trace length (' + item.chars + ' characters) — this model reports no reasoning_tokens' : 'reasoning_tokens as reported by the API'">{{ item.estimated ? '~' : '' }}{{ toks(item.tokens) }} tokens</span>
            <span v-if="item.step" class="th-turn">Turn {{ item.step }}</span>
          </span></summary>
        <div class="think-body">{{ item.text }}</div></details></div>

    <div class="item" v-else-if="item.kind==='tool'">
      <i class="dot tool" :class="{pulse:item.ok===null, err:item.ok===false}"></i>
      <div class="tool">
        <div class="call"><span class="fn">{{ item.name }}</span><span class="args">({{ args(item.args) }})</span></div>
        <div class="result" :class="{err:item.ok===false}" v-if="item.result">{{ item.result }}</div></div></div>

    <div class="item" v-else-if="item.kind==='note'">
      <i class="dot meta"></i><div class="note" :class="{cache:item.cache}">{{ item.text }}</div></div>

    <!-- a cache write: the text this turn persisted into the assistant content. Rendered with
         the SAME .txt.md block the agent's prose uses, because from here on it IS that content
         -- a different font would suggest a different channel. Only the label is quiet. -->
    <div class="item" v-else-if="item.kind==='cache'">
      <i class="dot meta"></i>
      <div class="cache-write">
        <div class="cw-label" :title="item.overwrite
              ? 'Replaces the text the model had written into the content itself'
              : 'Written into the assistant content — the model reads it on the next turn'">{{
          (item.mode === 'raw' ? 'Reasoning trace' : 'Summary') + ' stored in content' }}</div>
        <div class="txt md" v-if="item.text" v-html="md(item.text)"></div>
      </div></div>

    <div class="item" v-else-if="item.kind==='chip'">
      <i class="dot" :class="item.variant"></i><div class="txt state" :class="item.variant">{{ item.text }}</div></div>
  `,
};

const StateView = {
  props: ['world'],
  computed: {
    grid() { return (this.world && this.world.grid_units) || 6; },
    cubes() { return (this.world && this.world.cubes) || {}; },
    held() { return this.world && this.world.held; },
    area() { return Object.keys(this.cubes).filter(id => (this.cubes[id] || {}).location === 'area').map(id => Object.assign({ id }, this.cubes[id])); },
    // EVERY cube gets a fixed slot in the store, in catalog order (the order the server sends
    // them in, which never changes). A cube that is currently on the plate or in the gripper
    // leaves its slot empty instead of letting the others slide over -- so the store is a place
    // you can read positionally, not a list that reshuffles on every pick.
    slots() {
      return Object.keys(this.cubes).map(id => {
        const loc = (this.cubes[id] || {}).location;
        return { id, present: loc !== 'area' && loc !== 'held' };
      });
    },
    // Grid lines must follow the world's actual grid, not a hardcoded 6.
    gridStyle() { return { '--cell': (100 / this.grid) + '%' }; },
  },
  methods: {
    color(id) { const c = this.cubes[id] || {}; return CUBE_COLORS[c.color] || '#888'; },
    num(id) { const n = (this.cubes[id] || {}).number; return n == null ? '' : n; },
    // Colour + readable digits for one cube -- shared by the plate, the gripper and storage,
    // so a cube looks the same wherever it currently is.
    faceStyle(id) {
      const bg = this.color(id), light = isLightColor(bg);
      return { background: bg, color: light ? '#10151c' : '#fff',
               textShadow: light ? 'none' : '0 1px 2px rgba(0,0,0,.6)' };
    },
    // The plate is a TOP-DOWN view of the building area and y grows toward the BACK of the
    // cell, so it has to be drawn bottom-up (y=0 at the bottom edge = front, nearest the
    // viewer). Drawing it top-down mirrors front and back.
    //
    // `--cx`/`--cy` is where the cube's cell centre is; the LEAN on top of it is CSS
    // (--stack-dx/--stack-dy) times `--lvl`. A cube cannot float, so its level alone says how
    // high it is: level 0 sits exactly on its cell, and every level above leans one more step
    // towards the back-right. Nothing here looks at what a cube is standing ON -- a pyramid, a
    // cube straddling two others, a plain tower all get the same treatment.
    //
    // A higher cube is never covered by a lower one (it is on top of everything under its
    // footprint), so height decides who draws over whom; between two cubes on the same level
    // the one nearer the FRONT wins, because the lean goes towards the back.
    cubeStyle(c) {
      const lvl = c.level || 0;
      return Object.assign({ '--cx': (c.x / this.grid * 100) + '%',
                             '--cy': ((this.grid - c.y) / this.grid * 100) + '%',
                             '--lvl': String(lvl),
                             zIndex: lvl * 100 + Math.round((this.grid - c.y) * 10) },
                           this.faceStyle(c.id));
    },
  },
  // Three labelled blocks, each label on its own line above what it describes -- and the whole
  // panel centred in its rail, because it is a display, not a list of controls.
  template: `
    <div class="state-view">
      <section class="sv-block">
        <h3>Gripper</h3>
        <div class="sv-row">
          <span class="cube-dot" :class="{empty: !held}" :style="held ? faceStyle(held) : null"
                :title="held || 'gripper empty'">{{ held ? num(held) : '' }}</span>
          <b v-if="held">{{ held }}</b>
        </div>
      </section>

      <section class="sv-block">
        <h3>Building area</h3>
        <div class="plate" :style="gridStyle">
          <div v-for="c in area" :key="c.id" class="plate-cube" :style="cubeStyle(c)"
               :title="c.id + '  (x' + c.x + ', y' + c.y + ', L' + c.level + ')'">
            <span v-if="c.number!=null">{{ c.number }}</span>
          </div>
        </div>
      </section>

      <section class="sv-block">
        <h3>Storage</h3>
        <div class="sv-row store">
          <span v-for="s in slots" :key="s.id" class="cube-dot" :class="{empty: !s.present}"
                :style="s.present ? faceStyle(s.id) : null"
                :title="s.present ? s.id : s.id + ' — not in storage'">{{ s.present ? num(s.id) : '' }}</span>
        </div>
      </section>
    </div>
  `,
};

/* ---------- root app ---------- */

createApp({
  components: { ChatItem, StateView },
  setup() {
    const items = ref([]);
    const cfg = reactive({ backend: '', replay: null, model: '', model_alias: '', models: [], model_ids: {},
      mode: '', modes: [], available_modes: [], window: 0, reasoning_optional: false,
      grid: 6, cubes: [] });
    const world = ref(null);
    const busy = ref(false);
    const stopping = ref(false);   // Stop pressed; the loop is unwinding (still busy)
    const conn = ref(false);
    // The conversation context of the last model call, split into the cross-task layer (`base`,
    // survives the task) and this task's loop (`loop`, compacted away at the end). Both are
    // ESTIMATES from demo.estimate_tokens -- the real per-call number is meters.ctxPeak.
    const ctx = reactive({ base: 0, loop: 0, loopMsgs: 0, split: false });
    const meters = reactive({ ctxPeak: 0, window: 0, carried: 0, total: 0, prompt: 0, completion: 0 });
    // Context inspector: the actual conversation behind the two bars, fetched from GET /context
    // on open and again after every turn while it stays open (see the watch below).
    const inspect = reactive({ open: false, tab: 'base', data: null, loading: false, error: '' });
    const taskText = ref('');
    const streamEl = ref(null);
    const taEl = ref(null);
    const menuIndex = ref(0);
    let lastSeq = 0, lastTool = null, activityTimer = null, pendingTurn = null, pendingLlm = null;

    const modeDesc = computed(() => { const m = cfg.modes.find(x => x.label === cfg.mode); return m ? m.desc : ''; });

    // Both rails can be folded away -- useful when demoing on a small screen or when the
    // conversation itself is the point. The choice is remembered like the theme; the header
    // buttons stay put, so a folded rail is always one click from coming back.
    const panels = reactive(storedPanels());
    function togglePanel(side) {
      panels[side] = !panels[side];
      try { localStorage.setItem(PANELS_KEY, JSON.stringify(panels)); } catch (e) { /* not persisted */ }
    }

    // Theme: the icon shows the CURRENT state, the tooltip names the next one -- otherwise a
    // one-button cycle is a guessing game.
    const theme = ref(storedTheme());
    const THEME_LABEL = { auto: 'follows the system', light: 'light', dark: 'dark' };
    const themeTitle = computed(() => {
      const next = THEMES[(THEMES.indexOf(theme.value) + 1) % THEMES.length];
      return `Theme: ${THEME_LABEL[theme.value]} — click for ${THEME_LABEL[next]}`;
    });
    function cycleTheme() {
      theme.value = THEMES[(THEMES.indexOf(theme.value) + 1) % THEMES.length];
      applyTheme(theme.value);
      try { localStorage.setItem(THEME_KEY, theme.value); } catch (e) { /* not persisted, fine */ }
    }

    // Top-right badge: WHAT is behind this session. `sim` drives an external executor where an
    // arm can actually move, so it stays loud; fake and replay are drab on purpose -- they must
    // never be mistaken for a live cell.
    const badge = computed(() => {
      const r = cfg.replay;
      if (r) return { cls: 'replay', text: 'REPLAY',
        title: `Replay of ${r.file} — task ${r.task || 0}/${r.tasks}, ${r.speed}×`
             + ` (recording of ${r.model || '?'}). No LLM, no robot.` };
      if (cfg.backend === 'fake') return { cls: 'fake', text: 'FAKE',
        title: 'Fake backend: world model in-process — no simulator, no arm motion' };
      return { cls: 'sim', text: 'SIM',
        title: 'External executor over HTTP — a simulated or real arm may actually move' };
    });

    // The start screen stays up until the session has REAL conversation in it. Notes and status
    // chips do not count -- otherwise the lone "Neue Session gestartet." note would hide it, and
    // /new would land on an empty stream instead of back on the intro.
    const CONVO_KINDS = ['user', 'agent', 'tool', 'think', 'divider'];
    const showIntro = computed(() => !items.value.some(it => CONVO_KINDS.includes(it.kind)));

    // The two layers as shares of the conversation context (100% = base + loop). Both are
    // estimates from the same formula, so the proportion is meaningful even though neither is
    // an exact token count.
    const ctxMix = computed(() => {
      const tot = (ctx.base || 0) + (ctx.loop || 0);
      if (!ctx.split || !tot) return { base: 0, loop: 0 };   // no call yet -> an empty track
      return { base: 100 * ctx.base / tot, loop: 100 * ctx.loop / tot };
    });
    const ctxTitle = computed(() => ctx.split
      ? `Session (across tasks, persists): ~${fmtTok(ctx.base)}\n`
        + `Current loop (${ctx.loopMsgs} messages, gets compacted): ~${fmtTok(ctx.loop)}\n\n`
        + `Estimated from the message text (~3.6 characters/token + 5 tokens per message).\n`
        + `Not included: the tool schemas (~780 tokens, re-sent every turn).\n`
        + `The real per-request figure is below, as the context peak.`
      : 'One-shot mode: no carried session.');

    // Chat slash-commands: mirror the sidebar buttons. Typed in the composer, they open a
    // filtered picker (menuItems) and execute via runSlash.
    const SLASH_CMDS = [
      { name: 'model', desc: 'switch model (context is kept)', opts: () => cfg.models },
      { name: 'mode', desc: 'switch mode', opts: () => cfg.modes.filter(x => x.available).map(x => x.label) },
      { name: 'new', desc: 'start a new session', opts: null },
      { name: 'cleanup', desc: 'all cubes back to storage', opts: null },
      { name: 'context', desc: "view the model's context", opts: null },
    ];
    const menuItems = computed(() => {
      const t = taskText.value;
      if (!t.startsWith('/')) return [];
      const m = t.match(/^\/(\S*)(\s+([\s\S]*))?$/);
      const cmd = (m[1] || '').toLowerCase();
      if (m[2] == null) {   // still typing the command word -> suggest commands
        return SLASH_CMDS.filter(c => c.name.startsWith(cmd)).map(c => ({
          label: '/' + c.name, hint: c.desc, fill: c.opts ? '/' + c.name + ' ' : null, exec: c.opts ? null : '/' + c.name }));
      }
      const c = SLASH_CMDS.find(x => x.name === cmd);   // command chosen -> suggest its options
      if (!c || !c.opts) return [];
      const q = (m[3] || '').toLowerCase();
      return c.opts().filter(o => o.toLowerCase().includes(q)).map(o => ({ label: o, hint: '', fill: null, exec: '/' + c.name + ' ' + o }));
    });

    // The stream follows the newest content unconditionally -- having scrolled up earlier does
    // not stop it (that used to be a "near the bottom?" test, which left the view stranded after
    // reading back). Driven by the content events only, never by the elapsed-time ticker: a timer
    // firing 10x/s would make scrolling up impossible instead of merely temporary.
    function scrollBottom() { nextTick(() => { const el = streamEl.value; if (el) el.scrollTop = el.scrollHeight; }); }
    function push(item) { items.value.push(item); scrollBottom(); }

    // The model call IS a stream item, not a separate status row: it appears the moment the
    // request goes out ("Denke nach", ticking) and turns into its own result in place
    // ("Thought for 7.7 s", the trace one click away). So the pulsing dot marking "busy"
    // sits exactly where the thing it describes will end up.
    function beginLlm(reasoning) {
      endLlm();
      const item = reactive({ kind: 'think', pending: true, reasoning: !!reasoning, elapsed: fmtSecs(0) });
      pendingLlm = item;
      push(item);
      const t0 = performance.now();
      activityTimer = setInterval(() => { item.elapsed = fmtSecs(performance.now() - t0); }, 100);
    }
    // Stop the ticker and, if the call produced no trace to show (reasoning was off, or the
    // model returned none), take the placeholder back out -- an empty "Antwortet" line left
    // standing is exactly the clutter this layout is meant to avoid.
    function endLlm() {
      if (activityTimer) { clearInterval(activityTimer); activityTimer = null; }
      if (pendingLlm && pendingLlm.pending) {
        const i = items.value.indexOf(pendingLlm);
        if (i >= 0) items.value.splice(i, 1);
      }
      pendingLlm = null;
    }
    // The trace arrived: turn the live line into the finished, clickable one, in place.
    function fillLlm(ev) {
      // trim: models routinely open a trace with a newline, and `white-space: pre-wrap` would
      // render it -- the leading dot would then sit alone on an empty first line.
      const facts = { text: (ev.text || '').trim(), ms: ev.ms, tokens: ev.tokens, chars: ev.chars,
                      estimated: ev.tokens_estimated !== false,
                      step: pendingTurn && pendingTurn.step,
                      trigger: pendingTurn && pendingTurn.trigger };
      if (activityTimer) { clearInterval(activityTimer); activityTimer = null; }
      // same as a tool result: the pending row turns into its result in place, no push
      if (pendingLlm) { Object.assign(pendingLlm, facts, { pending: false }); pendingLlm = null; scrollBottom(); }
      else push(Object.assign({ kind: 'think', pending: false }, facts));
      pendingTurn = null;
    }

    function applyFeedback(ev) {
      const msg = ev.ok ? (ev.message || 'ok') : (ev.error || 'Fehler');
      // the result fills the existing row in place, so it grows without a push -> scroll here too
      if (lastTool && lastTool.ok === null) { lastTool.ok = !!ev.ok; lastTool.result = msg; lastTool = null; scrollBottom(); }
      else { push({ kind: 'chip', variant: ev.ok ? 'done' : 'err', text: (ev.ok ? '✓ ' : '✕ ') + msg }); }
    }
    function resolveDone() { if (lastTool && lastTool.ok === null) { lastTool.ok = true; lastTool.result = 'done'; lastTool = null; } }

    // Mark the message the model just wrote as the closing one. A turn can carry BOTH a content
    // message and done() -- the prose already arrived as a `content` event, so flagging that
    // message beats appending a second one with the internal done() reason (which read like a
    // fresh agent turn). Returns false if this turn produced no message of its own.
    function markLastAgentDone(reason) {
      for (let i = items.value.length - 1; i >= 0; i--) {
        const it = items.value[i];
        if (it.kind === 'agent') { it.done = true; it.doneReason = reason || undefined; scrollBottom(); return true; }
        if (it.kind === 'user' || it.kind === 'divider' || it.kind === 'tool') return false;
      }
      return false;
    }

    function handle(ev) {
      switch (ev.type) {
        case 'user_msg':   push({ kind: 'user', text: ev.text }); break;
        case 'task_start': push({ kind: 'divider', task: ev.task, mode: ev.mode }); busy.value = true; stopping.value = false; break;
        case 'stopping':   stopping.value = true;
                           if (pendingLlm && pendingLlm.pending) pendingLlm.stopping = true;
                           push({ kind: 'note', text: 'Stop requested — the agent halts after the current step.' }); break;
        case 'llm_start':  beginLlm(ev.reasoning); break;
        case 'context':    Object.assign(ctx, { base: ev.base || 0, loop: ev.loop || 0,
                             loopMsgs: ev.loop_msgs || 0, split: ev.base != null }); break;
        case 'ctx_compact': push({ kind: 'note', cache: true,
                             text: `Current loop compacted: ~${fmtTok(ev.before)} → ~${fmtTok(ev.after)} tokens`
                                   + ` · session for the next task: ~${fmtTok(ev.carried)}` }); break;
        // Informational, not a failure: the model answered in prose instead of calling done().
        // Its message is already on screen right above, so a neutral note is enough (a warn chip
        // next to a perfectly good completion message reads like something broke).
        case 'no_tool_call': push({ kind: 'note',
                             text: 'Task ended — the agent replied without calling done().' }); break;
        // The loop's per-turn verdict, sent BEFORE the call: kept until the reasoning it
        // explains arrives (a reasoning-OFF turn simply never claims it).
        case 'turn':       pendingTurn = { step: ev.step, trigger: ev.trigger }; break;
        case 'reasoning':  fillLlm(ev); break;
        case 'content':    endLlm(); push({ kind: 'agent', text: ev.text }); break;
        case 'llm_end':    endLlm(); break;
        case 'tool_call':  endLlm(); if (ev.name !== 'done') { push(reactive({ kind: 'tool', name: ev.name, args: ev.args || {}, ok: null, result: null })); lastTool = items.value[items.value.length - 1]; } break;
        case 'feedback':   applyFeedback(ev); break;
        // What the turn wrote into the assistant content -- shown verbatim under both cache
        // modes, since that text is what the model reads on its next turn.
        case 'cache_write': push({ kind: 'cache', mode: ev.mode, overwrite: !!ev.overwrite,
                                   text: (ev.text || '').trim() }); break;
        case 'note':       push({ kind: 'note', text: ev.text }); break;
        case 'done':       resolveDone();
                           if (ev.spoke && markLastAgentDone(ev.reason)) break;
                           push({ kind: 'agent', done: true, text: ev.reason || 'Task complete.' }); break;
        case 'chat':       break;   // the reply already arrived as a `content` event
        case 'max_steps':  push({ kind: 'agent', warn: true, text: 'I tried to solve your task but did not quite get there. Feel free to try again, or give me a new task.' }); break;
        case 'meters':     Object.assign(meters, { ctxPeak: ev.context_peak, window: ev.window, carried: ev.carried,
                             total: (ev.tokens || {}).total || 0, prompt: (ev.tokens || {}).prompt || 0, completion: (ev.tokens || {}).completion || 0 }); break;
        case 'mode_changed':  cfg.mode = ev.mode; push({ kind: 'note', text: `Mode → ${ev.mode}` }); break;
        case 'model_changed': push({ kind: 'note', text: `Model → ${ev.alias} (context is kept)` }); loadConfig(); break;
        case 'session_reset': items.value = []; busy.value = false; stopping.value = false;
                           Object.assign(ctx, { base: 0, loop: 0, loopMsgs: 0, split: false });
                           Object.assign(meters, { ctxPeak: 0, carried: 0, total: 0, prompt: 0, completion: 0 });
                           push({ kind: 'note', text: 'New session started.' }); break;
        case 'state':      world.value = ev.state; break;
        case 'error':      endLlm(); push({ kind: 'chip', variant: 'err', text: ev.text }); break;
        // A task can end with a tool call still spinning (abort, error) -- close it out so no
        // spinner is left behind, then unlock the composer.
        case 'task_end':   if (lastTool && lastTool.ok === null) { lastTool.ok = false; lastTool.result = ev.status === 'aborted' ? 'abgebrochen' : 'kein Ergebnis'; }
                           busy.value = false; stopping.value = false; endLlm(); lastTool = null; break;
      }
    }

    // One line per layer -- the long explanations read like documentation in a place where the
    // conversation right below IS the explanation.
    const LAYER_DESC = {
      base: 'System prompt and the session history between user and agent.',
      loop: 'Current context of the agent loop including tool calls and reasoning content '
          + '(if selected).',
    };
    const layer = computed(() => (inspect.data && inspect.data[inspect.tab]) || { tokens: 0, messages: [] });
    // The inspector borrows the chat's dots, so a role reads the same in both places: hollow for
    // the frame (system), grey for the model, green for a tool result. Unlike the chat the human
    // turn gets one too (blue) -- here every message is a row of ONE thread, and a row without a
    // dot would break the line that ties them together.
    const CTX_DOT = { system: 'meta', user: 'ask', tool: 'tool' };   // assistant: the plain grey dot
    function ctxDot(role) { return CTX_DOT[role] || ''; }
    async function loadInspect() {
      inspect.loading = true; inspect.error = '';
      try { const r = await fetch('/context'); inspect.data = await r.json(); }
      catch (e) { inspect.error = 'Context unavailable: ' + e; }
      inspect.loading = false;
    }
    function openInspect(tab) { inspect.tab = tab || 'base'; inspect.open = true; loadInspect(); }
    // No refresh button: the view refetches itself. The `context` event carries the new split
    // after every model call, so watching those numbers re-reads the conversation once per turn
    // while the inspector is open -- which is what makes "it is always current" true.
    watch(() => [ctx.base, ctx.loop, ctx.loopMsgs].join('/'),
          () => { if (inspect.open) loadInspect(); });
    function closeInspect() { inspect.open = false; }

    async function loadConfig() { const r = await fetch('/config'); Object.assign(cfg, await r.json()); }
    async function fetchState() { try { const r = await fetch('/state'); const s = await r.json(); if (!s.error) world.value = s; } catch (_) {} }

    function connect() {
      const es = new EventSource('/events');
      es.onopen = () => { conn.value = true; items.value = []; lastSeq = 0; lastTool = null; endLlm(); fetchState(); };
      es.onerror = () => { conn.value = false; };
      es.onmessage = (e) => {
        let ev; try { ev = JSON.parse(e.data); } catch (_) { return; }
        if (ev.seq && ev.seq <= lastSeq) return;
        if (ev.seq) lastSeq = ev.seq;
        handle(ev);
      };
    }

    async function command(payload) {
      const r = await fetch('/command', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(payload) });
      return r.json().catch(() => ({}));
    }
    async function sendTask(t) {
      const r = await fetch('/task', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ task: t }) });
      if (r.status === 409) push({ kind: 'note', text: 'Please wait — a task is still running.' });
    }
    async function onModel(e) { const r = await command({ cmd: 'model', model: e.target.value }); if (r && r.ok === false) push({ kind: 'chip', variant: 'err', text: r.error || 'switching the model failed' }); }
    async function onMode(e) { const r = await command({ cmd: 'mode', mode: e.target.value }); if (r && r.ok === false) push({ kind: 'chip', variant: 'err', text: r.error || 'switching the mode failed' }); }
    // Stop the running task. The server keeps the session busy until the loop has actually
    // unwound (it aborts before the next robot move), so we stay in the `stopping` state and
    // wait for task_end rather than unlocking the composer optimistically.
    // A REFUSED stop must never look like a successful one: only 'idle' (the server says nothing
    // runs) unlocks the composer; anything else -- notably an older server process that does not
    // know cmd:'stop' -- says so loudly, because the agent IS still working.
    async function onStop() {
      if (!busy.value) return;
      stopping.value = true;
      const r = await command({ cmd: 'stop' });
      if (!r || r.ok !== false) return;
      stopping.value = false;
      if (r.error === 'idle') { busy.value = false; return; }
      push({ kind: 'chip', variant: 'err',
             text: 'Stop failed (' + (r.error || '?') + ') — the agent keeps running. '
                 + 'Is the server still on old code? Then restart web_demo.' });
    }
    async function onCleanup() { if (busy.value) return; const r = await command({ cmd: 'cleanup' }); if (r && r.ok === false) push({ kind: 'note', text: 'Cannot clean up — something is still running.' }); }
    async function onNewSession() { const r = await command({ cmd: 'new_session' }); if (r && r.ok === false) push({ kind: 'note', text: 'Cannot start a new session — something is still running.' }); }

    function runSlash(text) {
      const m = text.match(/^\/(\S*)\s*([\s\S]*)$/);
      const cmd = (m[1] || '').toLowerCase();
      const arg = (m[2] || '').trim();
      if (cmd === 'new') { onNewSession(); return; }
      if (cmd === 'cleanup') { onCleanup(); return; }
      if (cmd === 'context') { openInspect(arg === 'loop' ? 'loop' : 'base'); return; }
      if (cmd === 'model') {
        if (cfg.models.includes(arg)) command({ cmd: 'model', model: arg }).then(r => { if (r && r.ok === false) push({ kind: 'chip', variant: 'err', text: r.error || 'switching the model failed' }); });
        else push({ kind: 'note', text: `/model: unknown model "${arg || '—'}" — available: ${cfg.models.join(', ')}` });
        return;
      }
      if (cmd === 'mode') {
        const avail = cfg.modes.filter(x => x.available).map(x => x.label);
        if (avail.includes(arg)) command({ cmd: 'mode', mode: arg }).then(r => { if (r && r.ok === false) push({ kind: 'chip', variant: 'err', text: r.error || 'switching the mode failed' }); });
        else push({ kind: 'note', text: `/mode: "${arg || '—'}" not available — ${avail.join(', ')}` });
        return;
      }
      push({ kind: 'note', text: `Unknown command /${cmd}. Available: /model, /mode, /new, /cleanup, /context` });
    }
    function applySuggestion(s) {
      if (!s) return;
      if (s.fill != null) { taskText.value = s.fill; menuIndex.value = 0; nextTick(() => { if (taEl.value) taEl.value.focus(); }); }
      else if (s.exec != null) { runSlash(s.exec); taskText.value = ''; menuIndex.value = 0; }
    }
    function onSlash() { if (busy.value) return; if (!taskText.value.startsWith('/')) taskText.value = '/'; menuIndex.value = 0; nextTick(() => { if (taEl.value) taEl.value.focus(); }); }

    // Repeated Enter while the agent works must not stack the same hint over and over.
    const BUSY_NOTE = 'The agent is still working — the task was not sent.';
    // Put the caret back in the composer. Called after every action that could take it away,
    // so typing the next task never needs a click first.
    function focusComposer() {
      nextTick(() => { const ta = taEl.value; if (ta && !ta.disabled) ta.focus(); });
    }
    async function submit() {
      const t = taskText.value.trim();
      if (!t) return;
      // While a task runs the server would answer 409 anyway. Say so instead of swallowing the
      // keystroke, and KEEP the text -- retyping it after the task ends would be the real
      // annoyance. The composer stays focused either way.
      if (busy.value) {
        const last = items.value[items.value.length - 1];
        if (!(last && last.kind === 'note' && last.text === BUSY_NOTE)) push({ kind: 'note', text: BUSY_NOTE });
        focusComposer();
        return;
      }
      taskText.value = '';
      if (t.startsWith('/')) runSlash(t); else sendTask(t);
      focusComposer();
    }
    function onKeydown(e) {
      const items = menuItems.value;
      if (items.length) {
        if (e.key === 'ArrowDown') { e.preventDefault(); menuIndex.value = (menuIndex.value + 1) % items.length; return; }
        if (e.key === 'ArrowUp') { e.preventDefault(); menuIndex.value = (menuIndex.value - 1 + items.length) % items.length; return; }
        if (e.key === 'Tab') { e.preventDefault(); applySuggestion(items[Math.min(menuIndex.value, items.length - 1)]); return; }
        if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); applySuggestion(items[Math.min(menuIndex.value, items.length - 1)]); return; }
        if (e.key === 'Escape') { e.preventDefault(); taskText.value = ''; return; }
      }
      if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); submit(); }
    }
    function resizeTa() { const ta = taEl.value; if (!ta) return; ta.style.height = 'auto'; ta.style.height = ta.scrollHeight + 'px'; }
    function grow() { menuIndex.value = 0; resizeTa(); }
    // Re-fit the box on programmatic value changes too (slash-fill, submit-clear), where the
    // native `input` event does not fire.
    watch(taskText, () => nextTick(resizeTa));
    // A finished task, a mode switch, a cleanup -- all end with the user wanting to type again.
    watch(busy, running => { if (!running) focusComposer(); });

    onMounted(async () => {
      await loadConfig(); await fetchState(); connect();
      focusComposer();
      window.addEventListener('keydown', e => { if (e.key === 'Escape' && inspect.open) closeInspect(); });
    });

    return { cfg, badge, theme, themeTitle, cycleTheme, panels, togglePanel,
      items, world, busy, stopping, conn, ctx, ctxMix, ctxTitle, meters, showIntro,
      taskText, streamEl, taEl, menuItems, menuIndex, modeDesc, fmtTok, Math, submit, onModel,
      onMode, onCleanup, onNewSession, onStop, onSlash, applySuggestion, onKeydown, grow,
      inspect, layer, LAYER_DESC, openInspect, closeInspect, loadInspect, ctxDot };
  },
  template: `
  <header class="topbar">
    <div class="brand"><i class="logo" role="img" aria-label="NeSyPlan"></i><span class="name">NeSyPlan</span></div>
    <div class="topbar-right">
      <button class="theme-btn" @click="cycleTheme" :title="themeTitle" :aria-label="themeTitle">
        <svg v-if="theme==='auto'" viewBox="0 0 24 24" width="15" height="15" fill="none"
             stroke="currentColor" stroke-width="1.7" aria-hidden="true">
          <rect x="3" y="4.5" width="18" height="12" rx="2"/><path d="M9 20h6"/></svg>
        <svg v-else-if="theme==='light'" viewBox="0 0 24 24" width="15" height="15" fill="none"
             stroke="currentColor" stroke-width="1.7" aria-hidden="true">
          <circle cx="12" cy="12" r="4.2"/><path d="M12 2.5v2.2M12 19.3v2.2M2.5 12h2.2M19.3 12h2.2
                   M5.3 5.3l1.6 1.6M17.1 17.1l1.6 1.6M18.7 5.3l-1.6 1.6M6.9 17.1l-1.6 1.6"/></svg>
        <svg v-else viewBox="0 0 24 24" width="15" height="15" fill="none"
             stroke="currentColor" stroke-width="1.7" aria-hidden="true">
          <path d="M20 14.6A8.6 8.6 0 1 1 9.4 4a7 7 0 0 0 10.6 10.6Z"/></svg>
      </button>
      <div class="target-badge" :class="badge.cls" :title="badge.title">{{ badge.text }}</div>
      <div class="conn" :class="conn ? 'up' : 'down'" :title="conn ? 'Connected' : 'Connection lost'"><i></i></div>
    </div>
  </header>

  <div class="layout" :class="{'no-left': !panels.left, 'no-right': !panels.right}">
    <aside class="sidebar" :class="{folded: !panels.left}">
      <!-- The rail's own fold control, in both states: chevron outward while open, the rail's
           content icon (faders = the controls) while folded. Folded the rail keeps its width in
           the grid as a strip, so the chat never reflows. -->
      <div class="rail-head">
        <button class="rail-stub" @click="togglePanel('left')"
                :title="(panels.left ? 'Collapse controls' : 'Expand controls') + ' (model, mode, context)'"
                :aria-label="panels.left ? 'Collapse controls' : 'Expand controls'"
                :aria-expanded="panels.left">
          <svg v-if="panels.left" viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="currentColor"
               stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
            <path d="M14 7.5 9.5 12l4.5 4.5"/><path d="M5.5 6.5v11" stroke-width="1.5"/></svg>
          <svg v-else viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="currentColor"
               stroke-width="1.7" stroke-linecap="round" aria-hidden="true">
            <path d="M4 7h11M19 7h1M4 12h4M12 12h8M4 17h9M17 17h3"/>
            <circle cx="17" cy="7" r="1.9"/><circle cx="10" cy="12" r="1.9"/><circle cx="15" cy="17" r="1.9"/></svg>
        </button>
      </div>
      <template v-if="panels.left">
      <!-- two groups, each opened by a heading with a hairline: what you SET, and what you RUN -->
      <div class="rail-group">
        <h2 class="group-head">Settings</h2>
        <section class="field"><label>Model</label>
          <!-- the wrapper draws the chevron inside the field: appearance:none removes the native
               one, and without a replacement the select read as plain text -->
          <div class="select-wrap">
            <select :value="cfg.model_alias" @change="onModel" :disabled="busy">
              <option v-for="a in cfg.models" :key="a" :value="a" :title="cfg.model_ids[a] || a">{{ a }}</option>
            </select>
          </div></section>

        <section class="field"><label>Mode</label>
          <div class="select-wrap">
            <select :value="cfg.mode" @change="onMode" :disabled="busy">
              <option v-for="m in cfg.modes" :key="m.label" :value="m.label" :disabled="!m.available">{{ m.label + (m.available ? '' : '  (reasoning cannot be turned off)') }}</option>
            </select>
          </div>
          <p class="hint">{{ modeDesc }}</p></section>
      </div>

      <div class="rail-group">
        <h2 class="group-head">Commands</h2>
        <section class="controls">
          <button class="btn" @click="onCleanup" :disabled="busy">Clean up<span class="kbd">/cleanup</span></button>
          <button class="btn" @click="onNewSession" :disabled="busy">New session<span class="kbd">/new</span></button>
        </section>
      </div>

      <!-- pushed to the bottom of the rail: the meters are reference, not controls -->
      <section class="meters push">
        <h3>Context</h3>
        <div class="ctx-stack" :title="ctxTitle">
          <i class="seg base" :style="{width: ctxMix.base + '%'}"
             @click="openInspect('base')" title="View the session"></i>
          <i class="seg loop" :style="{width: ctxMix.loop + '%'}"
             @click="openInspect('loop')" title="View the current loop"></i>
        </div>
        <template v-if="ctx.split">
          <div class="meter-row clickable" @click="openInspect('base')" title="View the session as a chat">
            <span><i class="key base"></i>Session <em>persists</em></span><b>~{{ fmtTok(ctx.base) }} ›</b></div>
          <div class="meter-row clickable" @click="openInspect('loop')" title="View the current loop as a chat">
            <span><i class="key loop"></i>Current loop</span><b>~{{ fmtTok(ctx.loop) }} ›</b></div>
          <div class="meter-row sum"><span>= total</span><b>~{{ fmtTok(ctx.base + ctx.loop) }}</b></div>
          <div class="meter-row sub"><span>Loop messages</span><b>{{ ctx.loopMsgs }}</b></div>
        </template>
        <p class="hint tiny" v-else>{{ (cfg.mode || '').startsWith('oneshot')
          ? 'One-shot mode: one call, no carried session.'
          : 'No model call in this session yet.' }}</p>
      </section>

      <section class="meters">
        <h3>Tokens</h3>
        <div class="meter-row"><span>total</span><b>{{ fmtTok(meters.total) }}</b></div>
        <div class="meter-row sub"><span>in · out</span><b>{{ fmtTok(meters.prompt) + ' · ' + fmtTok(meters.completion) }}</b></div>
        <div class="meter-row sub" title="Largest request in this session, real prompt_tokens">
          <span>Context peak</span><b>{{ meters.window ? fmtTok(meters.ctxPeak) + '/' + fmtTok(meters.window) : fmtTok(meters.ctxPeak) }}</b></div>
      </section>
      </template>
    </aside>

    <main class="chat">
      <div class="stream" ref="streamEl">
        <div v-if="showIntro" class="intro">
          <div class="intro-logo" role="img" aria-label="NeSyPlan"></div><h2>Give NeSyPlan a task</h2>
          <p>e.g. "put the red cube in the centre" or "build a tower of three from red, green, blue".</p>
        </div>
        <chat-item v-for="(it, idx) in items" :key="idx" :item="it"></chat-item>
      </div>

      <form class="composer" @submit.prevent="submit">
        <div class="composer-box">
          <div v-if="menuItems.length" class="slash-menu">
            <div v-for="(s, i) in menuItems" :key="i" class="slash-item" :class="{active: i === menuIndex}"
                 @mousedown.prevent="applySuggestion(s)" @mouseenter="menuIndex = i">
              <span class="slash-label">{{ s.label }}</span><span class="slash-hint">{{ s.hint }}</span>
            </div>
          </div>
          <button type="button" class="icon-btn slash-btn" @click="onSlash" :disabled="busy" title="Insert a command">/</button>
          <textarea ref="taEl" class="composer-ta" v-model="taskText" @keydown="onKeydown" @input="grow" rows="1"
            :placeholder="busy ? (stopping ? 'stopping…' : 'agent is working…')
              : (cfg.replay ? 'Replay: Enter plays the next recorded task…' : 'Enter a task…')"></textarea>
          <button v-if="!busy" type="submit" class="send" aria-label="Send" title="Send">
            <svg viewBox="0 0 24 24" width="18" height="18" aria-hidden="true"><path d="M12 4l-8 8h5v8h6v-8h5z" fill="currentColor"/></svg>
          </button>
          <button v-else type="button" class="send stop" :class="{pending: stopping}" @click="onStop"
            :aria-label="stopping ? 'stopping' : 'Stop'" :title="stopping ? 'stopping — halts after the current step' : 'Stop the task'">
            <svg viewBox="0 0 24 24" width="18" height="18" aria-hidden="true"><rect x="7" y="7" width="10" height="10" rx="2" fill="currentColor"/></svg>
          </button>
        </div>
        <p class="composer-note">
          NeSyPlan is an AI agent and controls a UR5e robot arm. Language models can make
          mistakes.
        </p>
      </form>
    </main>

    <!-- right rail: the live world. Its own side, because it is what you WATCH while the agent
         works, not something you operate -- the left rail is for operating. -->
    <aside class="sidebar right" :class="{folded: !panels.right}">
      <div class="rail-head">
        <button class="rail-stub" @click="togglePanel('right')"
                :title="panels.right ? 'Collapse state' : 'Expand state'"
                :aria-label="panels.right ? 'Collapse state' : 'Expand state'"
                :aria-expanded="panels.right">
          <svg v-if="panels.right" viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="currentColor"
               stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
            <path d="M10 7.5 14.5 12 10 16.5"/><path d="M18.5 6.5v11" stroke-width="1.5"/></svg>
          <!-- folded: the building area in miniature -- says what is behind the strip -->
          <svg v-else viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="currentColor"
               stroke-width="1.6" aria-hidden="true">
            <rect x="4" y="4" width="16" height="16" rx="2"/><path d="M4 9.3h16M4 14.7h16M9.3 4v16M14.7 4v16"/>
            <rect x="9.9" y="9.9" width="4.2" height="4.2" fill="currentColor" stroke="none"/></svg>
        </button>
      </div>
      <state-view v-if="panels.right" :world="world"></state-view>
    </aside>
  </div>

  <div class="modal-backdrop" v-if="inspect.open" @mousedown.self="closeInspect">
    <div class="modal ctx-modal">
      <header>
        <div class="ctx-tabs">
          <button :class="{active: inspect.tab==='base'}" @click="inspect.tab='base'">
            <i class="key base"></i>Session
            <span v-if="inspect.data">{{ inspect.data.base.messages.length }} · ~{{ fmtTok(inspect.data.base.tokens) }}</span>
          </button>
          <button :class="{active: inspect.tab==='loop'}" @click="inspect.tab='loop'">
            <i class="key loop"></i>Current loop
            <span v-if="inspect.data">{{ inspect.data.loop.messages.length }} · ~{{ fmtTok(inspect.data.loop.tokens) }}</span>
          </button>
        </div>
        <div class="ctx-actions">
          <button class="icon-btn" @click="closeInspect" title="Close (Esc)">✕</button>
        </div>
      </header>
      <div class="modal-body ctx-body">
        <p class="hint tiny layer-desc">{{ LAYER_DESC[inspect.tab] }}</p>
        <p class="ctx-banner" v-if="layer.compacted">
This task has finished — you are seeing the current loop exactly as it was last sent to the
          model. All that remains of it in the session is the compacted pair (below). It is
          discarded when the next task starts.
        </p>
        <p class="empty-hint" v-if="inspect.error">{{ inspect.error }}</p>
        <p class="empty-hint" v-else-if="inspect.loading && !inspect.data">loading…</p>
        <p class="empty-hint" v-else-if="inspect.data && inspect.data.oneshot">
          One-shot mode: no carried context.
        </p>
        <p class="empty-hint" v-else-if="!layer.messages.length">
          Empty — {{ inspect.tab==='loop' ? 'no task has run yet.' : 'no finished task in the session yet.' }}
        </p>
        <!-- the context read as what it is: a conversation, drawn like the chat itself -- one
             thread, a dot per message (colour = role), joined by the same hairline -->
        <div class="ctx-thread">
          <div v-for="(m, i) in layer.messages" :key="i" class="item">
            <i class="dot" :class="ctxDot(m.role)"></i>
            <div class="ctx-role">{{ m.role }} <span class="idx">· #{{ i+1 }}</span></div>
            <!-- trim: contents routinely start with a newline, which pre-wrap would render as an
                 empty first line. Never folded -- the point of the inspector is the full text. -->
            <div class="txt" :class="{user: m.role==='user'}"
                 v-if="(m.content || '').trim()">{{ m.content.trim() }}</div>
            <div class="tool" v-for="(c, j) in (m.tool_calls || [])" :key="j">
              <div class="call"><span class="fn">{{ c.name }}</span>({{ c.arguments }})</div>
            </div>
          </div>
        </div>
        <div class="ctx-compacted" v-if="layer.compacted_to">
          <div class="ctx-compacted-head">Appended to the session
            (~{{ fmtTok(layer.compacted_tokens) }} instead of ~{{ fmtTok(layer.tokens) }})</div>
          <div class="ctx-thread">
            <div v-for="(m, i) in layer.compacted_to" :key="'c'+i" class="item">
              <i class="dot" :class="ctxDot(m.role)"></i>
              <div class="ctx-role">{{ m.role }}</div>
              <div class="txt" :class="{user: m.role==='user'}"
                   v-if="(m.content || '').trim()">{{ m.content.trim() }}</div>
            </div>
          </div>
        </div>
      </div>
    </div>
  </div>
  `,
}).mount('#app');
