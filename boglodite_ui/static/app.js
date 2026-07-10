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
          state.currentAssistant.querySelector(".msg-body").textContent += ev.text;
        } else {
          state.currentAssistant = chatEl("agent", ev.text);
        }
      } else if (state.currentAssistant) {
        // A full (non-delta) message after streaming is authoritative —
        // replace the streamed bubble rather than duplicating it.
        state.currentAssistant.querySelector(".msg-body").textContent = ev.text;
        state.currentAssistant = null;
      } else {
        chatEl("agent", ev.text);
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
    case "outputs_changed":
      if (ev.new && ev.new.length) {
        chatEl("info", "New outputs: " + ev.new.join(", "));
        refreshOutputs(ev.new[0]);
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
