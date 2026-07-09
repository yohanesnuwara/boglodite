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
};

/* ── helpers ─────────────────────────────────────────────── */
function chatEl(cls, html) {
  const wrap = document.createElement("div");
  wrap.className = "msg " + cls;
  const body = document.createElement("div");
  body.className = "msg-body";
  if (typeof html === "string") body.textContent = html;
  else body.appendChild(html);
  if (cls === "tool" || cls === "info" || cls === "err") {
    wrap.innerHTML = "";
    wrap.append(...(typeof html === "string" ? [document.createTextNode(html)] : [html]));
  } else {
    wrap.appendChild(body);
  }
  $("chatScroll").appendChild(wrap);
  $("chatScroll").scrollTop = $("chatScroll").scrollHeight;
  return wrap;
}

function addLog(line, cls) {
  const span = document.createElement("span");
  if (cls) span.className = cls;
  span.textContent = line + "\n";
  const log = $("logScroll");
  log.appendChild(span);
  while (log.childNodes.length > 4000) log.removeChild(log.firstChild);
  if ($("autoScroll").checked) log.scrollTop = log.scrollHeight;
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
  const prev = selectName || sel.value;
  sel.innerHTML = "";
  if (!files.length) {
    sel.appendChild(new Option("— no outputs yet —", ""));
    return;
  }
  for (const f of files) sel.appendChild(new Option(f.name, f.name));
  sel.value = files.some((f) => f.name === prev) ? prev : files[0].name;
  if (sel.value) showOutput(sel.value);
}

async function showOutput(name) {
  if (!name) return;
  $("resultTag").textContent = `// outputs/${name} — loading…`;
  try {
    const resp = await fetch(`/api/output/view?name=${encodeURIComponent(name)}`);
    if (!resp.ok) throw new Error(await resp.text());
    const blob = await resp.blob();
    const img = $("resultImg");
    if (img.src) URL.revokeObjectURL(img.src);
    img.src = URL.createObjectURL(blob);
    img.classList.remove("hidden");
    $("resultPlaceholder").classList.add("hidden");
    $("resultTag").textContent = `// outputs/${name}`;
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
  for (const sk of state.skills) {
    const label = document.createElement("label");
    const cb = document.createElement("input");
    cb.type = "checkbox";
    cb.value = sk.path;
    cb.addEventListener("change", () => {
      cb.checked ? state.selectedSkills.add(sk.path) : state.selectedSkills.delete(sk.path);
      const n = state.selectedSkills.size;
      $("skillsBtn").textContent = n ? `${n} loaded ▾` : "none loaded ▾";
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
  if (!state.skills.length) {
    menu.innerHTML = '<div class="sk-desc" style="padding:8px">No skills/*/SKILL.md found.</div>';
  }
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
      const frag = document.createElement("span");
      const n = document.createElement("span");
      n.className = "tname";
      n.textContent = (ev.phase === "end" ? "✔ " : "▸ ") + ev.name;
      frag.appendChild(n);
      if (ev.detail) frag.appendChild(document.createTextNode("  " + ev.detail));
      chatEl("tool", frag);
      state.currentAssistant = null;
      break;
    }
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
  state.ws.send(JSON.stringify({
    type: "chat",
    text,
    model: $("modelSelect").value || null,
    skills: [...state.selectedSkills],
  }));
  $("chatInput").value = "";
}

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
$("clearLog").addEventListener("click", () => ($("logScroll").innerHTML = ""));

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
refreshOutputs();
