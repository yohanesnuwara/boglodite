/* Boglodite console frontend — vanilla JS, no build step. */
"use strict";

const $ = (id) => document.getElementById(id);

const state = {
  ws: null,
  running: false,
  meta: null,
  skills: [],           // [{dir,name,description,path}]
  selectedSkills: new Set(),
  currentAssistant: null, // element receiving streaming text
  outputsPoll: null,
  toolChips: new Map(),   // tool call id -> chip element
  lastToolChip: null,     // fallback pairing when no id
  counts: { agent: 0, proc: 0 },
  devstamp: null,
};

/* ── minimal offline markdown renderer (agent messages) ──────
   HTML-escapes first (XSS-safe), then supports: fenced code blocks,
   inline code, #–#### headings, bold, italic, lists, http(s) links. */
function escHtml(s) {
  return s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
}

function mdInline(s) {
  return s
    .replace(/`([^`]+)`/g, "<code>$1</code>")
    .replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>")
    .replace(/(^|[\s(])\*([^*\n]+)\*/g, "$1<em>$2</em>")
    .replace(/\[([^\]]+)\]\((https?:\/\/[^)\s]+)\)/g,
      '<a href="$2" target="_blank" rel="noopener">$1</a>');
}

function renderMarkdown(text) {
  const src = escHtml(text);
  const out = [];
  const lines = src.split("\n");
  let i = 0, list = null, para = [];

  const flushPara = () => {
    if (para.length) { out.push("<p>" + para.map(mdInline).join("<br>") + "</p>"); para = []; }
  };
  const flushList = () => {
    if (list) { out.push(`<${list.tag}>` + list.items.map((x) => `<li>${mdInline(x)}</li>`).join("") + `</${list.tag}>`); list = null; }
  };

  while (i < lines.length) {
    const line = lines[i];
    if (line.startsWith("```")) {               // fenced code block
      flushPara(); flushList();
      const buf = []; i++;
      while (i < lines.length && !lines[i].startsWith("```")) buf.push(lines[i++]);
      i++;
      out.push('<pre class="md-code">' + buf.join("\n") + "</pre>");
      continue;
    }
    const h = line.match(/^(#{1,4})\s+(.*)/);
    if (h) { flushPara(); flushList(); out.push(`<h${h[1].length + 2}>${mdInline(h[2])}</h${h[1].length + 2}>`); i++; continue; }
    const ul = line.match(/^\s*[-*]\s+(.*)/);
    const ol = line.match(/^\s*\d+[.)]\s+(.*)/);
    if (ul || ol) {
      flushPara();
      const tag = ul ? "ul" : "ol";
      if (!list || list.tag !== tag) { flushList(); list = { tag, items: [] }; }
      list.items.push((ul || ol)[1]); i++; continue;
    }
    if (!line.trim()) { flushPara(); flushList(); i++; continue; }
    flushList(); para.push(line); i++;
  }
  flushPara(); flushList();
  return out.join("");
}

function setAgentBody(el, rawText) {
  el.dataset.raw = rawText;
  el.querySelector(".msg-body").innerHTML = renderMarkdown(rawText);
}

/* ── helpers ─────────────────────────────────────────────── */
function chatEl(cls, html) {
  const wrap = document.createElement("div");
  wrap.className = "msg " + cls;
  const body = document.createElement("div");
  body.className = "msg-body";
  if (typeof html === "string") body.textContent = html;
  else body.appendChild(html);
  if (cls.startsWith("tool") || cls === "info" || cls === "err" || cls === "summary") {
    wrap.innerHTML = "";
    wrap.append(...(typeof html === "string" ? [document.createTextNode(html)] : [html]));
  } else {
    wrap.appendChild(body);
  }
  $("chatScroll").appendChild(wrap);
  $("chatScroll").scrollTop = $("chatScroll").scrollHeight;
  return wrap;
}

function bumpCount(which) {
  state.counts[which] += 1;
  const el = which === "agent" ? $("agentCount") : $("procCount");
  el.textContent = state.counts[which];
  const tab = document.querySelector(`.log-tab[data-tab="${which}"]`);
  if (!tab.classList.contains("active")) tab.classList.add("unseen");
}

function appendPre(pre, line, cls) {
  const span = document.createElement("span");
  if (cls) span.className = cls;
  span.textContent = line + "\n";
  pre.appendChild(span);
  while (pre.childNodes.length > 4000) pre.removeChild(pre.firstChild);
  if ($("autoScroll").checked) pre.scrollTop = pre.scrollHeight;
}

function addLog(line, cls) {
  appendPre($("logScroll"), line, cls);
  bumpCount("agent");
}

function addProc(line) {
  const isHead = line.startsWith("── ");
  appendPre($("procScroll"), line, isHead ? "p-head" : "");
  bumpCount("proc");
}

function setRunning(running) {
  state.running = running;
  $("agentState").textContent = running ? "RUNNING" : "IDLE";
  $("agentState").className = "state " + (running ? "running" : "idle");
  $("stopBtn").disabled = !running;
  $("sendBtn").disabled = running;
  if (running) {
    if (!state.outputsPoll) state.outputsPoll = setInterval(refreshOutputs, 4000);
  } else if (state.outputsPoll) {
    clearInterval(state.outputsPoll);
    state.outputsPoll = null;
  }
}

/* ── meta + seismic viewer ───────────────────────────────── */
async function loadMeta() {
  const r = await fetch("/api/meta");
  state.meta = await r.json();
  $("sessionId").textContent = (state.meta.session_id || "").slice(0, 8);

  if (!state.meta.copilot_found) {
    chatEl("err", `Copilot CLI ('${state.meta.copilot_bin}') not found on PATH — ` +
      "install it or launch with --copilot-bin.");
  }
  if (state.meta.segy_error) {
    $("inputPlaceholder").textContent = state.meta.segy_error;
    return;
  }
  const s = state.meta.segy;
  $("inputPlaceholder").textContent = `Survey loaded: ${s.file}`;
  applyAxisRange();
  treeInitDefaults();
  loadSlice();
}

function axisRange() {
  const s = state.meta && state.meta.segy;
  if (!s) return null;
  return s[$("axisSelect").value];
}

function applyAxisRange() {
  const r = axisRange();
  if (!r) return;
  const inp = $("sliceValue");
  inp.min = r.min; inp.max = r.max; inp.step = r.step;
  const v = parseInt(inp.value, 10);
  if (isNaN(v) || v < r.min || v > r.max) {
    inp.value = Math.round((r.min + r.max) / 2 / r.step) * r.step;
  }
}

async function loadSlice() {
  const r = axisRange();
  if (!r) return;
  const axis = $("axisSelect").value;
  const value = parseInt($("sliceValue").value, 10);
  $("inputTag").textContent = `// ${axis.toUpperCase()} ${value} — loading…`;
  try {
    const resp = await fetch(`/api/slice?axis=${axis}&value=${value}`);
    if (!resp.ok) throw new Error(await resp.text());
    const snapped = resp.headers.get("X-Slice-Value");
    if (snapped) $("sliceValue").value = snapped;
    const blob = await resp.blob();
    const img = $("inputImg");
    if (img.src) URL.revokeObjectURL(img.src);
    img.src = URL.createObjectURL(blob);
    img.classList.remove("hidden");
    $("inputPlaceholder").classList.add("hidden");
    $("inputTag").textContent = `// ${axis.toUpperCase()} ${snapped || value}`;
    treeSetSeismic(axis, parseInt(snapped || value, 10));
  } catch (e) {
    $("inputTag").textContent = `// ${axis.toUpperCase()} ${value} — failed`;
    addLog("slice load failed: " + e.message, "l-err");
  }
}

function stepSlice(dir) {
  const r = axisRange();
  if (!r) return;
  const inp = $("sliceValue");
  inp.value = Math.min(r.max, Math.max(r.min, parseInt(inp.value, 10) + dir * r.step));
  loadSlice();
}

/* ── outputs / result pane ───────────────────────────────── */
async function refreshOutputs(selectName) {
  const r = await fetch("/api/outputs");
  const { files } = await r.json();
  const sel = $("outputSelect");
  const prev = selectName || (state.resultShown ? sel.value : "");
  sel.innerHTML = "";
  sel.appendChild(new Option(files.length ? "— select an output —" : "— no outputs yet —", ""));
  for (const f of files) sel.appendChild(new Option(f.name, f.name));
  sel.value = files.some((f) => f.name === prev) ? prev : "";
  // Only display something if a file was explicitly requested (new output
  // from a turn, or the user's current selection) — never on plain boot.
  if (sel.value) showOutput(sel.value);
}

async function showOutput(name) {
  if (!name) {
    state.resultShown = false;
    tree.checkedResult = null;
    if (typeof renderTree === "function") renderTree();
    $("resultImg").classList.add("hidden");
    $("resultPlaceholder").classList.remove("hidden");
    $("resultTag").textContent = "// outputs/";
    return;
  }
  $("resultTag").textContent = `// outputs/${name} — loading…`;
  try {
    const resp = await fetch(`/api/output/view?name=${encodeURIComponent(name)}&t=${Date.now()}`,
                             { cache: "no-store" });
    if (!resp.ok) throw new Error(await resp.text());
    const blob = await resp.blob();
    const img = $("resultImg");
    if (img.src) URL.revokeObjectURL(img.src);
    img.src = URL.createObjectURL(blob);
    img.classList.remove("hidden");
    $("resultPlaceholder").classList.add("hidden");
    $("resultTag").textContent = `// outputs/${name}`;
    state.resultShown = true;
    treeSyncResult(name);
  } catch (e) {
    $("resultTag").textContent = `// outputs/${name} — cannot render`;
    addLog("output render failed: " + e.message, "l-err");
  }
}

/* ── models + skills ─────────────────────────────────────── */
async function loadModels() {
  const r = await fetch("/api/models");
  const { models, default: def } = await r.json();
  const sel = $("modelSelect");
  for (const m of models) sel.appendChild(new Option(m, m));
  if (def) sel.value = def;
}

async function loadSkills() {
  const r = await fetch("/api/skills");
  state.skills = (await r.json()).skills;
  const menu = $("skillsMenu");
  menu.innerHTML = "";
  const note = document.createElement("div");
  note.className = "sk-desc";
  note.style.padding = "6px 8px";
  note.textContent = "Selected skills are sent to the agent once per session.";
  menu.appendChild(note);
  for (const sk of state.skills) {
    const label = document.createElement("label");
    const cb = document.createElement("input");
    cb.type = "checkbox";
    cb.value = sk.path;
    cb.checked = true;                       // all skills loaded by default
    state.selectedSkills.add(sk.path);
    cb.addEventListener("change", () => {
      cb.checked ? state.selectedSkills.add(sk.path) : state.selectedSkills.delete(sk.path);
      updateSkillsBtn();
    });
    const name = document.createElement("div");
    name.className = "sk-name";
    name.textContent = "/" + sk.name;
    const desc = document.createElement("div");
    desc.className = "sk-desc";
    desc.textContent = sk.description;
    label.append(cb, document.createTextNode(" "), name, desc);
    menu.appendChild(label);
  }
  updateSkillsBtn();
  if (!state.skills.length) {
    menu.innerHTML = '<div class="sk-desc" style="padding:8px">No skills/*/SKILL.md found.</div>';
  }
}

function updateSkillsBtn() {
  const n = state.selectedSkills.size;
  $("skillsBtn").textContent = n ? `${n} loaded ▾` : "none loaded ▾";
}

/* ── websocket chat ──────────────────────────────────────── */
function connect() {
  const proto = location.protocol === "https:" ? "wss" : "ws";
  const ws = new WebSocket(`${proto}://${location.host}/ws`);
  state.ws = ws;
  ws.onmessage = (e) => handleEvent(JSON.parse(e.data));
  ws.onclose = () => {
    setRunning(false);
    addLog("── websocket closed, reconnecting in 2 s ──", "l-err");
    setTimeout(connect, 2000);
  };
}

function handleEvent(ev) {
  switch (ev.type) {
    case "status":
      setRunning(ev.state === "running");
      if (ev.session_id) $("sessionId").textContent = ev.session_id.slice(0, 8);
      if (ev.reset) chatEl("info", "New session started — context cleared.");
      break;
    case "assistant": {
      if (ev.delta) {
        if (state.currentAssistant) {
          setAgentBody(state.currentAssistant,
                       (state.currentAssistant.dataset.raw || "") + ev.text);
        } else {
          state.currentAssistant = chatEl("agent", "");
          setAgentBody(state.currentAssistant, ev.text);
        }
      } else if (state.currentAssistant) {
        // A full (non-delta) message after streaming is authoritative —
        // replace the streamed bubble rather than duplicating it.
        setAgentBody(state.currentAssistant, ev.text);
        state.currentAssistant = null;
      } else {
        setAgentBody(chatEl("agent", ""), ev.text);
      }
      $("chatScroll").scrollTop = $("chatScroll").scrollHeight;
      break;
    }
    case "reasoning":
      addLog("· " + ev.text);
      break;
    case "tool": {
      if (ev.phase === "end") {
        // Pair with the start chip (by id, else the most recent open chip).
        const chip = state.toolChips.get(ev.id) || state.lastToolChip;
        if (chip && !chip.classList.contains("tool-done") && !chip.classList.contains("tool-fail")) {
          chip.classList.add(ev.ok ? "tool-done" : "tool-fail");
          const n = chip.querySelector(".tname");
          if (n) n.textContent = (ev.ok ? "✔ " : "✗ ") + n.textContent.replace(/^[▸✔✗] /, "");
          if (ev.detail) {
            const d = chip.querySelector(".tdetail");
            const short = ev.detail.split("\n")[0].slice(0, 120);
            if (d && short) d.textContent = "  → " + short;
          }
          state.toolChips.delete(ev.id);
          if (state.lastToolChip === chip) state.lastToolChip = null;
          break;
        }
        // No start seen — fall through and render a standalone end chip.
      }
      const frag = document.createElement("span");
      const n = document.createElement("span");
      n.className = "tname";
      n.textContent = (ev.phase === "end" ? (ev.ok ? "✔ " : "✗ ") : "▸ ") + ev.name;
      frag.appendChild(n);
      const d = document.createElement("span");
      d.className = "tdetail";
      if (ev.detail) d.textContent = "  " + ev.detail.split("\n")[0].slice(0, 160);
      frag.appendChild(d);
      const chip = chatEl("tool" + (ev.phase === "end" ? (ev.ok ? " tool-done" : " tool-fail") : ""), frag);
      if (ev.phase === "start") {
        if (ev.id) state.toolChips.set(ev.id, chip);
        state.lastToolChip = chip;
      }
      state.currentAssistant = null;
      break;
    }
    case "proc":
      for (const line of ev.text.split("\n")) addProc(line);
      break;
    case "summary":
      chatEl("summary", ev.text);
      break;
    case "info":
      chatEl("info", ev.text);
      break;
    case "error":
      chatEl("err", ev.text);
      break;
    case "log":
      addLog(ev.line, ev.stream === "stderr" ? "l-err" : ev.line.startsWith("$") ? "l-cmd" : "");
      break;
    case "done":
      chatEl("info", ev.code === 0 ? "— turn complete —" : `— agent exited with code ${ev.code} —`);
      state.currentAssistant = null;
      break;
    case "ui_command":
      handleUiCommand(ev);
      break;
    case "outputs_changed":
      if (ev.new && ev.new.length) {
        chatEl("info", "New outputs: " + ev.new.join(", "));
        refreshInterpretations().then(() => refreshOutputs(ev.new[0]));
      } else {
        refreshOutputs();
      }
      break;
  }
}

function send() {
  const text = $("chatInput").value.trim();
  if (!text || state.running || !state.ws || state.ws.readyState !== 1) return;
  chatEl("user", text);
  state.currentAssistant = null;
  state.toolChips.clear();
  state.lastToolChip = null;
  state.ws.send(JSON.stringify({
    type: "chat",
    text,
    model: $("modelSelect").value || null,
    skills: [...state.selectedSkills],
  }));
  $("chatInput").value = "";
}

/* ── local provider (Run Locally) ─────────────────────────── */
const provider = { enabled: false, host: "LM Studio", base_url: "", model: "" };

async function loadProvider() {
  Object.assign(provider, await (await fetch("/api/provider")).json());
  $("localToggle").checked = provider.enabled;
  $("providerUrl").value = provider.base_url;
  if (provider.model) {
    $("providerModel").innerHTML = "";
    $("providerModel").appendChild(new Option(provider.model, provider.model));
    $("providerSave").disabled = false;
  }
}

function openProviderModal() {
  $("providerUrl").value = provider.base_url || "http://192.168.56.1:1234/v1";
  $("providerStatus").textContent = "";
  $("providerStatus").className = "field-note";
  $("providerModal").classList.remove("hidden");
}

function closeProviderModal(revertToggle) {
  $("providerModal").classList.add("hidden");
  if (revertToggle) $("localToggle").checked = provider.enabled;
}

async function detectModels() {
  const base = $("providerUrl").value.trim();
  const status = $("providerStatus");
  status.textContent = "Detecting…";
  status.className = "field-note";
  const { models, error } = await (await fetch(
    `/api/provider/models?base_url=${encodeURIComponent(base)}`)).json();
  const sel = $("providerModel");
  sel.innerHTML = "";
  if (error || !models.length) {
    sel.appendChild(new Option("— none found —", ""));
    status.textContent = error || "The server responded but lists no models.";
    status.className = "field-note err";
    $("providerSave").disabled = true;
    return;
  }
  for (const m of models) sel.appendChild(new Option(m, m));
  if (models.includes(provider.model)) sel.value = provider.model;
  status.textContent = `Found ${models.length} model(s).`;
  status.className = "field-note ok";
  $("providerSave").disabled = false;
}

async function saveProvider() {
  const body = {
    enabled: true,
    host: $("providerHost").value,
    base_url: $("providerUrl").value.trim(),
    model: $("providerModel").value,
  };
  Object.assign(provider, await (await fetch("/api/provider", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  })).json());
  closeProviderModal(false);
  $("localToggle").checked = true;
  // Sync the topbar model picker with the local server's models.
  const sel = $("modelSelect");
  sel.innerHTML = "";
  for (const opt of $("providerModel").options) {
    if (opt.value) sel.appendChild(new Option(opt.value, opt.value));
  }
  sel.value = provider.model;
  chatEl("info", `Running locally via ${provider.host} — ${provider.model} ` +
    "(applies from the next turn; set-copilot-env.sh updated).");
}

async function disableProvider() {
  Object.assign(provider, await (await fetch("/api/provider", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ enabled: false }),
  })).json());
  const sel = $("modelSelect");
  sel.innerHTML = "";
  sel.appendChild(new Option("provider default", ""));
  chatEl("info", "Run Locally off — using Copilot's own model routing from the next turn.");
}

$("localToggle").addEventListener("change", (e) =>
  e.target.checked ? openProviderModal() : disableProvider());
$("providerClose").addEventListener("click", () => closeProviderModal(true));
$("providerModal").addEventListener("click", (e) => {
  if (e.target === $("providerModal")) closeProviderModal(true);
});
$("providerDetect").addEventListener("click", detectModels);
$("providerSave").addEventListener("click", saveProvider);

/* ── log tabs ────────────────────────────────────────────── */
document.querySelectorAll(".log-tab").forEach((tab) =>
  tab.addEventListener("click", () => {
    document.querySelectorAll(".log-tab").forEach((t) => t.classList.remove("active"));
    tab.classList.add("active");
    tab.classList.remove("unseen");
    const isAgent = tab.dataset.tab === "agent";
    $("logScroll").classList.toggle("hidden", !isAgent);
    $("procScroll").classList.toggle("hidden", isAgent);
  }));

/* ── hot reload ──────────────────────────────────────────── */
async function pollDevstamp() {
  try {
    const { stamp } = await (await fetch("/api/devstamp")).json();
    if (state.devstamp === null) state.devstamp = stamp;
    else if (stamp > state.devstamp) location.reload();
  } catch (e) { /* server restarting (e.g. --dev reload) — retry */ }
}
setInterval(() => { if (!document.hidden) pollDevstamp(); }, 2500);

/* ── wiring ──────────────────────────────────────────────── */
$("sendBtn").addEventListener("click", send);
$("chatInput").addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); send(); }
});
$("stopBtn").addEventListener("click", () =>
  state.ws && state.ws.send(JSON.stringify({ type: "stop" })));
$("newSessionBtn").addEventListener("click", () =>
  state.ws && state.ws.send(JSON.stringify({ type: "new_session" })));

$("loadSlice").addEventListener("click", loadSlice);
$("axisSelect").addEventListener("change", () => { applyAxisRange(); loadSlice(); });
$("sliceValue").addEventListener("keydown", (e) => { if (e.key === "Enter") loadSlice(); });
$("sliceMinus").addEventListener("click", () => stepSlice(-1));
$("slicePlus").addEventListener("click", () => stepSlice(+1));

$("outputSelect").addEventListener("change", (e) => showOutput(e.target.value));
$("refreshOutputs").addEventListener("click", () => refreshOutputs());
$("clearLog").addEventListener("click", () => {
  $("logScroll").innerHTML = "";
  $("procScroll").innerHTML = "";
  state.counts = { agent: 0, proc: 0 };
  $("agentCount").textContent = "";
  $("procCount").textContent = "";
});

$("skillsBtn").addEventListener("click", (e) => {
  e.stopPropagation();
  $("skillsMenu").classList.toggle("hidden");
});
document.addEventListener("click", (e) => {
  if (!$("skillsMenu").contains(e.target)) $("skillsMenu").classList.add("hidden");
});

document.querySelectorAll(".hint").forEach((el) =>
  el.addEventListener("click", () => { $("chatInput").value = el.dataset.fill; $("chatInput").focus(); }));

/* draggable dividers */
function makeDragger(divider, before, horizontalProp) {
  divider.addEventListener("pointerdown", (e) => {
    e.preventDefault();
    divider.setPointerCapture(e.pointerId);
    const move = (ev) => {
      const rect = before.parentElement.getBoundingClientRect();
      const frac = Math.min(0.85, Math.max(0.12, (ev.clientY - rect.top) / rect.height));
      if (horizontalProp === "grow") {
        before.style.flex = `1 1 ${frac * 100}%`;
        before.nextElementSibling.nextElementSibling.style.flex = `1 1 ${(1 - frac) * 100}%`;
      } else {
        // log divider: size the pane BELOW
        const below = divider.nextElementSibling;
        below.style.flex = `0 0 ${Math.max(64, rect.bottom - ev.clientY)}px`;
      }
    };
    const up = () => {
      divider.removeEventListener("pointermove", move);
      divider.removeEventListener("pointerup", up);
    };
    divider.addEventListener("pointermove", move);
    divider.addEventListener("pointerup", up);
  });
}
makeDragger($("divider"), $("paneInput"), "grow");
makeDragger($("logDivider"), $("chatPane"), "fixed");

/* boot */
connect();
loadMeta();
loadModels();
loadSkills();
loadProvider();
refreshOutputs();
/* ═════════════════════════════════════════════════════════════
   OBJECT PANEL — Petrel-style tree
   Seismic > Survey (editable) > Inline / Crossline / Time slice
   Interpretation > Facies | Fault > entries discovered in outputs/
   ═════════════════════════════════════════════════════════════ */
const tree = {
  surveyName: "F3 seismic",
  expanded: { seismic: true, survey: true, interp: true, facies: true, fault: true },
  seismic: { inline: null, xline: null, time: null, active: null },
  interp: { facies: [], fault: [] },
  checkedResult: null,          // filename currently shown in the result pane
};

const AXIS_LABEL = { inline: "Inline", xline: "Crossline", time: "Time slice" };

const TICONS = {
  seismic: '<svg viewBox="0 0 15 15"><path d="M2 5l5-3 6 3v5l-6 3-5-3z" fill="#2e7d8c" stroke="#4db6c9" stroke-width="0.8"/><path d="M2 5l5 3 6-3M7 8v7" stroke="#4db6c9" stroke-width="0.8" fill="none"/></svg>',
  survey:  '<svg viewBox="0 0 15 15"><path d="M2 5l5-3 6 3v5l-6 3-5-3z" fill="#2e8c50" stroke="#57c983" stroke-width="0.8"/><path d="M2 5l5 3 6-3M7 8v7" stroke="#57c983" stroke-width="0.8" fill="none"/></svg>',
  inline:  '<svg viewBox="0 0 15 15"><rect x="6" y="1.5" width="3" height="12" fill="#3f72b8" stroke="#7aa6dd" stroke-width="0.7" transform="skewY(-8)"/></svg>',
  xline:   '<svg viewBox="0 0 15 15"><rect x="1.5" y="6" width="12" height="3" fill="#2e8c50" stroke="#57c983" stroke-width="0.7" transform="skewX(-8)"/></svg>',
  time:    '<svg viewBox="0 0 15 15"><path d="M2 8l5-2.5 6 2.5-5 2.5z" fill="#b8860b" stroke="#e3a23c" stroke-width="0.7"/></svg>',
  interp:  '<svg viewBox="0 0 15 15"><path d="M1.5 4h4l1.5 2h6.5v6h-12z" fill="#5a4a2e" stroke="#a08040" stroke-width="0.7"/></svg>',
  facies:  '<svg viewBox="0 0 15 15"><rect x="2" y="3" width="11" height="3" fill="#c4453c"/><rect x="2" y="6" width="11" height="3" fill="#3f72b8"/><rect x="2" y="9" width="11" height="3" fill="#2e8c50"/></svg>',
  fault:   '<svg viewBox="0 0 15 15"><rect x="2" y="3" width="11" height="9" fill="#20262d"/><path d="M4 3l3 9M8 3l3 9" stroke="#c4453c" stroke-width="1.4"/></svg>',
};

function tRow({ depth, twisty, expandKey, checkbox, checked, icon, label, count,
                branch, onCheck, onOpenLabel, editable }) {
  const row = document.createElement("div");
  row.className = "trow" + (checked ? " checked-row" : "");
  const tw = document.createElement("span");
  tw.className = "twisty" + (twisty ? "" : " leaf");
  tw.textContent = twisty ? (tree.expanded[expandKey] ? "▾" : "▸") : "";
  if (twisty) tw.addEventListener("click", () => {
    tree.expanded[expandKey] = !tree.expanded[expandKey]; renderTree();
  });
  row.appendChild(tw);
  if (checkbox) {
    const cb = document.createElement("input");
    cb.type = "checkbox"; cb.className = "tcheck"; cb.checked = !!checked;
    cb.addEventListener("change", () => onCheck && onCheck(cb.checked));
    row.appendChild(cb);
  }
  const ic = document.createElement("span");
  ic.className = "ticon"; ic.innerHTML = TICONS[icon] || "";
  row.appendChild(ic);
  const lb = document.createElement("span");
  lb.className = "tlabel" + (branch ? " branch" : "") + (editable ? " editable" : "");
  lb.textContent = label;
  if (onOpenLabel) { lb.style.cursor = "pointer"; lb.addEventListener("click", () => onOpenLabel()); }
  if (editable) lb.addEventListener("dblclick", () => startSurveyEdit(lb));
  row.appendChild(lb);
  if (count !== undefined) {
    const c = document.createElement("span");
    c.className = "tcount"; c.textContent = `(${count})`;
    row.appendChild(c);
  }
  return row;
}

function startSurveyEdit(lb) {
  lb.classList.add("editing");
  lb.contentEditable = "true";
  lb.focus();
  document.getSelection().selectAllChildren(lb);
  const commit = async () => {
    lb.contentEditable = "false"; lb.classList.remove("editing");
    const name = lb.textContent.trim() || "F3 seismic";
    tree.surveyName = name;
    await fetch("/api/state", { method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ survey_name: name }) });
    renderTree();
  };
  lb.addEventListener("blur", commit, { once: true });
  lb.addEventListener("keydown", (e) => {
    if (e.key === "Enter") { e.preventDefault(); lb.blur(); }
    if (e.key === "Escape") { lb.textContent = tree.surveyName; lb.blur(); }
  });
}

function renderTree() {
  const root = $("objectTree");
  root.innerHTML = "";

  /* ── Seismic ─────────────────────────────────────────── */
  root.appendChild(tRow({ twisty: true, expandKey: "seismic", icon: "seismic",
                          label: "Seismic", branch: true }));
  const seisKids = document.createElement("div");
  seisKids.className = "tchildren" + (tree.expanded.seismic ? "" : " collapsed");
  seisKids.appendChild(tRow({ twisty: true, expandKey: "survey", icon: "survey",
                              label: tree.surveyName, branch: true, editable: true }));
  const survKids = document.createElement("div");
  survKids.className = "tchildren" + (tree.expanded.survey ? "" : " collapsed");
  for (const axis of ["inline", "xline", "time"]) {
    const v = tree.seismic[axis];
    survKids.appendChild(tRow({
      checkbox: true, checked: tree.seismic.active === axis,
      icon: axis, label: AXIS_LABEL[axis] + (v !== null ? " " + v : ""),
      onCheck: (on) => on ? openSeismicFromTree(axis) : renderTree(),
      onOpenLabel: () => openSeismicFromTree(axis),
    }));
  }
  seisKids.appendChild(survKids);
  root.appendChild(seisKids);

  /* ── Interpretation ──────────────────────────────────── */
  root.appendChild(tRow({ twisty: true, expandKey: "interp", icon: "interp",
                          label: "Interpretation", branch: true }));
  const interpKids = document.createElement("div");
  interpKids.className = "tchildren" + (tree.expanded.interp ? "" : " collapsed");
  for (const kind of ["facies", "fault"]) {
    const entries = tree.interp[kind] || [];
    interpKids.appendChild(tRow({
      twisty: true, expandKey: kind, icon: kind,
      label: kind === "facies" ? "Facies" : "Fault",
      branch: true, count: entries.length,
    }));
    const kids = document.createElement("div");
    kids.className = "tchildren" + (tree.expanded[kind] ? "" : " collapsed");
    if (!entries.length) {
      const empty = document.createElement("div");
      empty.className = "tempty";
      empty.textContent = "no results in outputs/ yet";
      kids.appendChild(empty);
    }
    for (const e of entries) {
      const isChecked = e.files.includes(tree.checkedResult);
      kids.appendChild(tRow({
        checkbox: true, checked: isChecked, icon: e.axis,
        label: `${AXIS_LABEL[e.axis]} ${e.value}` + (e.inferred ? " ~" : ""),
        onCheck: (on) => on ? openInterpretation(kind, e)
                            : (tree.checkedResult = null, showOutput(""), renderTree()),
        onOpenLabel: () => openInterpretation(kind, e),
      }));
    }
    interpKids.appendChild(kids);
  }
  root.appendChild(interpKids);
}

/* ── tree actions ────────────────────────────────────────── */
function openSeismicFromTree(axis) {
  $("axisSelect").value = axis;
  applyAxisRange();
  if (tree.seismic[axis] !== null) $("sliceValue").value = tree.seismic[axis];
  loadSlice();
}

function openInterpretation(kind, entry) {
  tree.checkedResult = entry.preferred;
  const sel = $("outputSelect");
  if ([...sel.options].some((o) => o.value === entry.preferred)) sel.value = entry.preferred;
  showOutput(entry.preferred);
  // QC workflow: bring the matching seismic slice into the input pane too.
  $("axisSelect").value = entry.axis;
  applyAxisRange();
  $("sliceValue").value = entry.value;
  loadSlice();
  renderTree();
}

function treeSetSeismic(axis, value) {
  tree.seismic[axis] = value;
  tree.seismic.active = axis;
  renderTree();
}

function treeSyncResult(name) {
  for (const kind of ["facies", "fault"]) {
    for (const e of tree.interp[kind]) {
      if (e.files.includes(name)) { tree.checkedResult = e.preferred; renderTree(); return; }
    }
  }
  tree.checkedResult = name;   // not an interpretation — nothing checked in tree
  renderTree();
}

async function refreshInterpretations() {
  try {
    tree.interp = await (await fetch("/api/interpretations")).json();
  } catch (e) { /* server restarting */ }
  renderTree();
}

function treeInitDefaults() {
  const s = state.meta && state.meta.segy;
  if (!s) { renderTree(); return; }
  const mid = (r) => Math.round((r.min + r.max) / 2 / r.step) * r.step;
  if (tree.seismic.inline === null) tree.seismic.inline = parseInt($("sliceValue").value, 10) || mid(s.inline);
  if (tree.seismic.xline === null) tree.seismic.xline = mid(s.xline);
  if (tree.seismic.time === null) tree.seismic.time = mid(s.time);
  renderTree();
}

/* ── agent-driven UI commands ───────────────────────────── */
async function handleUiCommand(ev) {
  if (ev.action !== "open") return;
  const axisName = AXIS_LABEL[ev.axis] || ev.axis;
  if (ev.target === "seismic") {
    $("axisSelect").value = ev.axis;
    applyAxisRange();
    $("sliceValue").value = ev.value;
    loadSlice();
    chatEl("info", `⚙ Agent opened seismic · ${axisName} ${ev.value}`);
    return;
  }
  await refreshInterpretations();
  const entry = (tree.interp[ev.target] || []).find(
    (e) => e.axis === ev.axis && e.value === ev.value);
  if (entry) {
    openInterpretation(ev.target, entry);
    chatEl("info", `⚙ Agent opened ${ev.target} · ${axisName} ${ev.value}`);
  } else {
    chatEl("info", `⚙ Agent asked to open ${ev.target} at ${axisName} ${ev.value}, ` +
      "but no matching result exists in outputs/.");
  }
}

/* ── panel divider drag ─────────────────────────────────── */
$("panelDivider").addEventListener("pointerdown", (e) => {
  e.preventDefault();
  const div = $("panelDivider");
  div.setPointerCapture(e.pointerId);
  const move = (ev) => {
    const w = Math.min(420, Math.max(170, ev.clientX));
    $("objectPanel").style.flex = `0 0 ${w}px`;
  };
  const up = () => {
    div.removeEventListener("pointermove", move);
    div.removeEventListener("pointerup", up);
  };
  div.addEventListener("pointermove", move);
  div.addEventListener("pointerup", up);
});

/* ── tree boot ──────────────────────────────────────────── */
(async function initTree() {
  try {
    const st = await (await fetch("/api/state")).json();
    tree.surveyName = st.survey_name || "F3 seismic";
  } catch (e) { /* defaults */ }
  await refreshInterpretations();
})();
