// Strata workbench client: runs, live events, explorer, inbox, seismic viewer with overlays.
(function () {
const $ = id => document.getElementById(id);
const S = { cfg: null, run: null, info: null, events: [], ws: null, meta: null, section: null, hz: null, ties: [],
            view: 'inline', idx: { inline: 1120, xline: 1800, time: 2500 }, mapMode: 'twt', sel: null };
const api = async (p, o) => { const r = await fetch(p, o); if (!r.ok) throw new Error(await r.text()); return r.json(); };
const fmt = (v, d = 1) => v == null || !isFinite(v) ? '–' : Number(v).toFixed(d);
const esc = s => String(s ?? '').replace(/[&<>]/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;' }[c]));
const b64 = (s, T) => { const b = atob(s), u = new Uint8Array(b.length); for (let i = 0; i < b.length; i++) u[i] = b.charCodeAt(i); return new T(u.buffer); };

// ── init ────────────────────────────────────────────────────────────────────
async function init() {
  S.cfg = await api('/api/config');
  $('brief').value = S.cfg.default_brief;
  $('model').innerHTML = S.cfg.models.map(m => `<option ${m === S.cfg.default_model ? 'selected' : ''}>${m}</option>`).join('');
  $('autonomy').innerHTML = S.cfg.autonomy.map(a => `<option ${a === 'gates_only' ? 'selected' : ''}>${a}</option>`).join('');
  if (!S.cfg.anthropic_key) $('model').title = 'ANTHROPIC_API_KEY is not set: only the "reference" workflow (no LLM) will run';
  $('runBtn').onclick = startRun;
  $('runSelect').onchange = () => openRun($('runSelect').value);
  document.querySelectorAll('#viewSeg button').forEach(b => b.onclick = () => setView(b.dataset.v));
  document.querySelectorAll('#mapSeg button').forEach(b => b.onclick = () => { S.mapMode = b.dataset.m; press('#mapSeg', b); draw(); });
  $('slider').oninput = () => { S.idx[S.view] = +$('slider').value; $('sliderVal').textContent = S.idx[S.view]; draw(); clearTimeout(S.t); S.t = setTimeout(loadSection, 120); };
  ['ovHz', 'ovTies'].forEach(id => $(id).onchange = draw);
  $('hzVersion').onchange = () => loadHorizon(+$('hzVersion').value);
  new ResizeObserver(draw).observe($('view'));
  if (S.cfg.data_available) { S.meta = await api('/api/seismic/meta'); setView('inline'); }
  else $('hud').textContent = 'Norne data drive not mounted';
  await refreshRuns(true);
}
function press(sel, btn) { document.querySelectorAll(sel + ' button').forEach(b => b.setAttribute('aria-pressed', b === btn)); }

async function refreshRuns(openLatest) {
  const runs = await api('/api/runs');
  $('runSelect').innerHTML = runs.map(r => `<option value="${r.run_id}">${r.run_id} · ${r.model || ''} · ${r.status || (r.active ? 'running' : '')}</option>`).join('');
  if (openLatest && runs.length) openRun(runs[0].run_id);
}

async function startRun() {
  $('runBtn').disabled = true;
  try {
    const r = await api('/api/runs', { method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ brief: $('brief').value, target: $('target').value, model: $('model').value, autonomy: $('autonomy').value }) });
    await refreshRuns(false); $('runSelect').value = r.run_id; await openRun(r.run_id);
  } catch (e) { alertBox(e.message); } finally { $('runBtn').disabled = false; }
}
function alertBox(m) { $('events').insertAdjacentHTML('afterbegin', `<div class="ev run_error"><span class="k">error</span><span class="b" style="grid-column:2/5">${esc(m)}</span></div>`); }

async function openRun(id) {
  if (S.ws) { S.ws.close(); S.ws = null; }
  S.run = id; S.events = []; S.hz = null; S.ties = [];
  $('events').innerHTML = '';
  const evs = await api(`/api/runs/${id}/events`); evs.forEach(addEvent);
  await refreshInfo();
  const proto = location.protocol === 'https:' ? 'wss' : 'ws';
  const ws = new WebSocket(`${proto}://${location.host}/api/runs/${id}/ws`);
  ws.onmessage = m => { const e = JSON.parse(m.data); if (e.seq > (S.events.at(-1)?.seq || 0)) { addEvent(e); onLive(e); } };
  S.ws = ws;
  clearInterval(S.poll); S.poll = setInterval(refreshInfo, 3000);
}

function onLive(e) {
  if (['publish', 'verdict', 'card', 'decision', 'status', 'score', 'run_end', 'export'].includes(e.kind)) refreshInfo();
}

async function refreshInfo() {
  if (!S.run) return;
  const info = await api(`/api/runs/${S.run}`); S.info = info;
  const st = info.status || 'idle'; $('state').className = 'state ' + st; $('state').textContent = st.toUpperCase();
  renderExplorer(); renderInbox(); renderVerdicts(); renderAgents(); renderBench();
  const hz = info.tree.filter(o => o.type === 'HorizonPicks');
  const latest = hz.length ? Math.max(...hz.map(o => o.version)) : 0;
  if (latest && (!S.hz || S.hzLatest !== latest)) { S.hzLatest = latest; await loadHorizon(latest); }
  if (!S.ties.length && info.tree.some(o => o.type === 'WellTies')) { S.ties = await api(`/api/runs/${S.run}/ties`); draw(); }
  if (['done', 'failed'].includes(st)) clearInterval(S.poll);
}

// ── panels ──────────────────────────────────────────────────────────────────
function renderExplorer() {
  const tree = S.info.tree; if (!tree.length) { $('explorer').innerHTML = '<div class="empty">Store is empty.</div>'; return; }
  const groups = { Input: ['ProjectSettings', 'WellTies'], Interpretation: ['HorizonPicks', 'FaultSticks'], Models: ['Grid', 'Property', 'Volumes'] };
  let h = '';
  for (const [g, types] of Object.entries(groups)) {
    const objs = tree.filter(o => types.includes(o.type)); if (!objs.length) continue;
    h += `<div class="grp">${g.toUpperCase()}</div>`;
    h += objs.map(o => `<div class="obj ${S.sel === o.id + o.version ? 'sel' : ''}" data-id="${o.id}" data-v="${o.version}"><span>${esc(o.name)} <span class="v">${o.type} v${o.version}</span></span><span class="badge ${o.status}">${o.status}</span></div>`).join('');
  }
  $('explorer').innerHTML = h;
  $('explorer').querySelectorAll('.obj').forEach(el => el.onclick = () => inspect(el.dataset.id, +el.dataset.v));
}

function inspect(id, v) {
  const o = S.info.tree.find(x => x.id === id && x.version === v); if (!o) return;
  S.sel = id + v; renderExplorer();
  if (o.type === 'HorizonPicks') loadHorizon(v);
  $('inspector').innerHTML = `<div class="kv"><span>object</span><span>${esc(o.id)} v${o.version}</span><span>status</span><span>${o.status}</span>
    <span>producer</span><span>${esc(o.producer)}</span><span>confidence</span><span>${fmt(o.confidence, 3)}</span><span>inputs</span><span>${esc(o.inputs.join(', ') || '–')}</span>
    <span>summary</span><span>${esc(o.summary)}</span></div>
    <div class="grp">TOOL CALLS (PROVENANCE)</div>` +
    (o.tool_calls.map(t => `<div class="tc"><b>${esc(t.tool)}</b> <span class="p">${esc(JSON.stringify(t.params))}</span></div>`).join('') || '<div class="empty">none</div>') +
    `<div class="grp">VERDICTS</div>` + (o.verdicts.map(vd => `<div class="vd ${vd.passed ? 'pass' : 'fail'}"><div class="who">${esc(vd.verifier)} · round ${vd.round}</div>${esc(vd.summary)}</div>`).join('') || '<div class="empty">none</div>');
}

function renderInbox() {
  const cards = S.info.cards; $('inboxCount').textContent = cards.length ? `${cards.length} waiting` : '';
  if (!cards.length) { $('inbox').innerHTML = '<div class="empty">No decisions waiting.</div>'; return; }
  $('inbox').innerHTML = cards.map(c => `<div class="card"><h4>${esc(c.title)}</h4><pre>${esc(c.body)}</pre><div class="btns">${c.options.map(o => `<button data-c="${c.id}" data-o="${esc(o)}" class="${o === c.default ? 'def' : ''}">${esc(o)}</button>`).join('')}</div></div>`).join('');
  $('inbox').querySelectorAll('button').forEach(b => b.onclick = async () => {
    await api(`/api/runs/${S.run}/cards/${b.dataset.c}`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ choice: b.dataset.o }) });
    refreshInfo();
  });
}

function renderVerdicts() {
  const v = S.events.filter(e => e.kind === 'verdict');
  $('verdicts').innerHTML = v.length ? v.map(e => `<div class="vd ${e.passed ? 'pass' : 'fail'}"><div class="who">${esc(e.agent)} on ${esc(e.object)} v${e.version} · round ${e.round} · ${e.passed ? 'PASS' : 'REJECT'}</div>${esc(e.summary)}${(e.findings || []).map(f => `<div class="muted">· ${esc(f)}</div>`).join('')}</div>`).join('') : '<div class="empty">No verdicts yet.</div>';
}

function renderAgents() {
  const ag = {};
  for (const e of S.events) {
    if (!e.agent || e.agent.startsWith('orchestrator') || e.agent === 'scorer' || e.agent === 'human') continue;
    const a = ag[e.agent] ||= { on: false, tin: 0, tout: 0, runs: 0, tools: 0 };
    if (e.kind === 'agent_start') a.on = true;
    if (e.kind === 'agent_end' || e.kind === 'agent_error') { a.on = false; a.runs++; a.tin += e.usage?.input_tokens || 0; a.tout += e.usage?.output_tokens || 0; }
    if (e.kind === 'tool') a.tools++;
  }
  $('agents').innerHTML = Object.entries(ag).map(([n, a]) => `<div class="agent"><span class="dot ${a.on ? 'on' : ''}"></span><span>${esc(n)}</span><span class="tk">${a.runs}× · ${a.tools} tools · ${(a.tin / 1000).toFixed(1)}k/${(a.tout / 1000).toFixed(1)}k tok</span></div>`).join('') || '<div class="empty">No agents yet.</div>';
}

function renderBench() {
  const sc = S.info.score; $('benchPanel').hidden = !sc; if (!sc) return;
  const hz = sc.stages.HZ;
  let h = `<div class="big">${sc.checks_passed}/${sc.checks_total} <span class="muted" style="font-size:12px">checks · composite ${fmt(sc.composite, 3)}</span></div>`;
  for (const [k, st] of Object.entries(sc.stages)) {
    h += `<div class="grp">${esc(st.name.toUpperCase())} · ${esc(st.outcome)}</div>`;
    h += Object.values(st.metrics).filter(m => m.primary).map(m => `<div class="chk"><span>${esc(m.label)}</span><span class="${m.pass ? 'ok' : 'no'}">${m.value == null ? '–' : (m.unit === 'frac' ? fmt(m.value * 100, 1) + ' %' : fmt(m.value, 2) + ' ' + m.unit)} ${m.pass ? '✓' : '✗'}</span></div>`).join('');
  }
  if (S.info.report) h += `<div style="margin-top:8px"><a href="${S.info.report}" target="_blank" style="color:var(--amber)">open full visual report ↗</a></div>`;
  $('bench').innerHTML = h;
}

function addEvent(e) {
  S.events.push(e);
  const time = new Date(e.t * 1000).toLocaleTimeString([], { hour12: false });
  const body = { ...e }; ['seq', 't', 'kind', 'agent'].forEach(k => delete body[k]);
  let txt = e.kind === 'tool' ? `${e.tool} ${JSON.stringify(e.params)} → ${JSON.stringify(e.diagnostics).slice(0, 260)}`
    : e.kind === 'agent_end' ? `done in ${e.seconds}s · tokens ${e.usage?.input_tokens}/${e.usage?.output_tokens} · ${JSON.stringify(e.output).slice(0, 200)}`
    : JSON.stringify(body).slice(0, 400);
  $('events').insertAdjacentHTML('afterbegin', `<div class="ev ${e.kind}"><span class="k">${time}</span><span class="a">${esc(e.agent)}</span><span class="k">${e.kind}</span><span class="b" title="${esc(txt)}">${esc(txt)}</span></div>`);
  $('evCount').textContent = `${S.events.length} events`;
  if (e.kind === 'verdict') renderVerdicts();
  if (['agent_start', 'agent_end', 'tool'].includes(e.kind)) renderAgents();
}

// ── viewer ──────────────────────────────────────────────────────────────────
function setView(v) {
  S.view = v; press('#viewSeg', document.querySelector(`#viewSeg [data-v=${v}]`));
  $('mapSeg').hidden = v !== 'map'; $('slider').hidden = $('sliderVal').hidden = v === 'map';
  if (!S.meta) return;
  const r = v === 'inline' ? S.meta.ilines : v === 'xline' ? S.meta.xlines : [S.meta.t0, S.meta.t0 + S.meta.dt * (S.meta.nt - 1)];
  $('slider').min = r[0]; $('slider').max = r[1]; $('slider').step = v === 'time' ? S.meta.dt : 1;
  $('slider').value = S.idx[v] = Math.min(Math.max(S.idx[v], r[0]), r[1]); $('sliderVal').textContent = S.idx[v];
  if (v === 'map') draw(); else loadSection();
}

async function loadSection() {
  if (S.view === 'map' || !S.meta) return;
  const r = await api(`/api/seismic/section?kind=${S.view}&idx=${S.idx[S.view]}`);
  S.section = { ...r, arr: b64(r.data, Int8Array), kind: S.view, idx: S.idx[S.view] }; draw();
}

async function loadHorizon(v) {
  try {
    const r = await api(`/api/runs/${S.run}/horizon?version=${v}`);
    S.hz = { v: r.version, ni: r.shape[0], nj: r.shape[1], twt: b64(r.twt, Float32Array), conf: b64(r.conf, Float32Array) };
    $('hzVersion').innerHTML = r.versions.map(x => `<option ${x === r.version ? 'selected' : ''}>${x}</option>`).join('');
    draw();
  } catch (e) { /* no horizon yet */ }
}

const seis = v => { const t = v / 127; return t >= 0 ? [255 * (1 - 0.23 * t), 255 * (1 - 0.73 * t), 255 * (1 - 0.76 * t)] : [255 * (1 + 0.75 * t), 255 * (1 + 0.55 * t), 255 * (1 + 0.28 * t)]; };
const seqc = t => { t = Math.max(0, Math.min(1, t)); const st = [[253, 231, 37], [94, 201, 98], [33, 145, 140], [59, 82, 139], [68, 1, 84]]; const k = Math.min(3, Math.floor(t * 4)), f = t * 4 - k; return st[k].map((a, i) => a + (st[k + 1][i] - a) * f); };
const confc = c => c >= 0.9 ? '#3fa57d' : c >= 0.75 ? '#e3a23c' : c > 0 ? '#d95757' : '#8a8f94';

function draw() {
  const cv = $('cv'), box = $('view'), dpr = devicePixelRatio || 1;
  cv.width = box.clientWidth * dpr; cv.height = box.clientHeight * dpr;
  const g = cv.getContext('2d'); g.setTransform(dpr, 0, 0, dpr, 0, 0); g.fillStyle = '#06090c'; g.fillRect(0, 0, box.clientWidth, box.clientHeight);
  if (S.view === 'map') return drawMap(g, box.clientWidth, box.clientHeight);
  const s = S.section; if (!s) return;
  const W = box.clientWidth, Hh = box.clientHeight, img = g.createImageData(s.w, s.h);
  for (let i = 0; i < s.arr.length; i++) { const c = seis(s.arr[i]); img.data.set([c[0], c[1], c[2], 255], i * 4); }
  const off = new OffscreenCanvas(s.w, s.h); off.getContext('2d').putImageData(img, 0, 0);
  g.imageSmoothingEnabled = s.kind === 'time'; g.drawImage(off, 0, 0, W, Hh);
  const m = S.meta, sx = W / s.w, sy = Hh / s.h;
  const ty = t => (t - m.t0) / m.dt * sy;
  let hud = s.kind === 'time' ? `time slice ${s.idx} ms` : `${s.kind} ${s.idx}`;
  if ($('ovHz').checked && S.hz && s.kind !== 'time') {
    g.lineWidth = 2;
    const n = s.kind === 'inline' ? S.hz.nj : S.hz.ni, fixed = s.idx - (s.kind === 'inline' ? m.ilines[0] : m.xlines[0]);
    let prev = null;
    for (let k = 0; k < n; k++) {
      const ii = s.kind === 'inline' ? fixed * S.hz.nj + k : k * S.hz.nj + fixed, t = S.hz.twt[ii];
      if (!isFinite(t)) { prev = null; continue; }
      const x = (k + 0.5) * sx, y = ty(t);
      if (prev) { g.strokeStyle = confc(S.hz.conf[ii]); g.beginPath(); g.moveTo(prev[0], prev[1]); g.lineTo(x, y); g.stroke(); }
      prev = [x, y];
    }
    hud += ` · horizon v${S.hz.v}`;
  }
  if ($('ovTies').checked && S.ties.length && s.kind !== 'time') {
    for (const w of S.ties) {
      const near = s.kind === 'inline' ? Math.abs(w.il - s.idx) <= 3 : Math.abs(w.xl - s.idx) <= 3; if (!near || w.twt == null) continue;
      const x = ((s.kind === 'inline' ? w.xl - m.xlines[0] : w.il - m.ilines[0]) + 0.5) * sx, y = ty(w.twt);
      g.strokeStyle = '#e8eef2'; g.lineWidth = 1; g.beginPath(); g.moveTo(x, y - 14); g.lineTo(x, y + 14); g.stroke();
      g.fillStyle = '#e3a23c'; g.beginPath(); g.arc(x, y, 4, 0, 7); g.fill();
      g.font = '10px ui-monospace, monospace'; g.fillStyle = '#e8eef2'; g.fillText(w.well.replace('6608/10-', ''), x + 6, y - 6);
    }
  }
  $('hud').textContent = hud;
  $('legend').innerHTML = `<span>horizon confidence</span><span style="color:#3fa57d">≥0.9</span><span style="color:#e3a23c">≥0.75</span><span style="color:#d95757">&lt;0.75</span><span style="color:#8a8f94">filled</span><span>● well tie</span>`;
}

function drawMap(g, W, Hh) {
  if (!S.hz) { $('hud').textContent = 'no horizon yet'; $('legend').innerHTML = ''; return; }
  const { ni, nj, twt, conf } = S.hz, img = new ImageData(nj, ni);
  let lo = Infinity, hi = -Infinity; for (const t of twt) if (isFinite(t)) { lo = Math.min(lo, t); hi = Math.max(hi, t); }
  for (let i = 0; i < ni; i++) for (let j = 0; j < nj; j++) {
    const k = i * nj + j, p = ((ni - 1 - i) * nj + j) * 4, t = twt[k];
    let c = [20, 24, 30];
    if (isFinite(t)) {
      if (S.mapMode === 'twt') c = seqc((t - lo) / (hi - lo || 1));
      else if (S.mapMode === 'conf') c = conf[k] > 0 ? seqc(1 - conf[k]) : [138, 143, 148];
      else { const tn = j + 1 < nj ? twt[k + 1] : NaN, tu = i + 1 < ni ? twt[k + nj] : NaN; c = (Math.abs(tn - t) > 12 || Math.abs(tu - t) > 12) ? [217, 87, 87] : [40, 48, 56]; }
    }
    img.data.set([c[0], c[1], c[2], 255], p);
  }
  const off = new OffscreenCanvas(nj, ni); off.getContext('2d').putImageData(img, 0, 0);
  const sc = Math.min(W / nj, Hh / ni), ox = (W - nj * sc) / 2, oy = (Hh - ni * sc) / 2;
  g.imageSmoothingEnabled = false; g.drawImage(off, ox, oy, nj * sc, ni * sc);
  if ($('ovTies').checked) for (const w of S.ties) {
    if (w.il == null) continue; const x = ox + (w.xl - S.meta.xlines[0] + 0.5) * sc, y = oy + (ni - 1 - (w.il - S.meta.ilines[0]) + 0.5) * sc;
    g.fillStyle = '#e3a23c'; g.beginPath(); g.arc(x, y, 3.5, 0, 7); g.fill();
  }
  $('hud').textContent = `horizon v${S.hz.v} · map (IL up, XL right)`;
  $('legend').innerHTML = S.mapMode === 'twt' ? `<span>${fmt(lo, 0)} ms</span><canvas class="rmp" id="rmp" width="120" height="9"></canvas><span>${fmt(hi, 0)} ms</span>`
    : S.mapMode === 'conf' ? `<span>conf 1</span><canvas class="rmp" id="rmp" width="120" height="9"></canvas><span>0</span><span style="color:#8a8f94">grey = filled</span>`
    : `<span style="color:#d95757">■ jump &gt; 12 ms (cycle skip / fault)</span>`;
  const r = document.getElementById('rmp'); if (r) { const q = r.getContext('2d'); for (let x = 0; x < 120; x++) { const c = seqc(x / 119); q.fillStyle = `rgb(${c})`; q.fillRect(x, 0, 1, 9); } }
}

init();
})();
