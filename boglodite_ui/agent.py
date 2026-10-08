"""Copilot CLI bridge for the Boglodite UI.

The UI does NOT replace the Copilot CLI agent stack — it drives it.
Each chat turn spawns one non-interactive run:

    copilot -p "<prompt>" \
        --session-id <uuid>       # same UUID every turn = persistent context
        --allow-all-tools         # required for non-interactive mode
        --output-format json      # JSONL event stream on stdout
        --no-ask-user             # agent can't block waiting for a TTY
        [--model <model>]

The fixed --session-id gives multi-turn memory: Copilot creates the session
on the first turn and resumes it (full history, loaded skills, cwd trust)
on every subsequent turn. All of .github/copilot-instructions.md, skills/,
and BYOK env vars (COPILOT_PROVIDER_BASE_URL / COPILOT_MODEL from
set-copilot-env.sh) apply unchanged.

Observed JSONL schema (Copilot CLI ≥ 0.0.3xx): objects like
    {"type": "tool.execution_started", "data": {...}, "id": "...",
     "parentId": "...", "timestamp": "...", "ephemeral": true|false}
    {"type": "message....", ...}
    {"type": "result", "exitCode": 0, "usage": {"premiumRequests": N, ...}}
Field names drift between versions, so classification is heuristic and
tolerant: every raw line is always forwarded to the Agent log verbatim,
and only confidently-recognised events are promoted to chat / process-log
entries. Adjust `classify_event` if a CLI update changes the schema.

Two extra behaviours make agent-run subprocesses first-class citizens:
  * MATPLOTLIBRC + PYTHONUNBUFFERED are injected into the copilot
    environment, so plots written by sandbox scripts inherit the console's
    dark theme and their stdout is unbuffered (live logs).
  * Text output found inside tool events (shell stdout, script logs) is
    emitted as {"type": "proc"} events, which the UI routes to the
    PROCESS log tab — separate from the raw agent JSONL.
"""

from __future__ import annotations

import asyncio
import json
import os
import shlex
import shutil
import uuid
from dataclasses import dataclass, field
from typing import Any, AsyncIterator

_DARK_RC = os.path.join(os.path.dirname(__file__), "dark.matplotlibrc")


# ── generic JSON spelunking ──────────────────────────────────────────────────
def _deep_find(obj: Any, keys: tuple[str, ...], depth: int = 3,
               want_str: bool = True) -> Any:
    """Breadth-first search for the first value under any of `keys`."""
    frontier = [obj]
    for _ in range(depth):
        nxt = []
        for node in frontier:
            if isinstance(node, dict):
                for k in keys:
                    if k in node and node[k] is not None:
                        v = node[k]
                        if not want_str or isinstance(v, str):
                            return v
                nxt.extend(node.values())
            elif isinstance(node, list):
                nxt.extend(node)
        frontier = nxt
    return None


def _content_to_text(content: Any) -> str | None:
    """Flatten OpenAI/Copilot-style content (str or list of blocks)."""
    if isinstance(content, str):
        return content if content.strip() else None
    if isinstance(content, list):
        parts = []
        for blk in content:
            if isinstance(blk, str):
                parts.append(blk)
            elif isinstance(blk, dict):
                t = blk.get("text") or blk.get("content")
                if isinstance(t, str):
                    parts.append(t)
        joined = "".join(parts)
        return joined if joined.strip() else None
    return None


_NAME_KEYS = ("toolName", "tool_name", "toolTitle", "tool_title", "name", "tool")
_ARG_KEYS = ("arguments", "args", "input", "rawInput", "parameters", "params",
             "command", "query")
_OUT_KEYS = ("output", "stdout", "result", "content", "text", "detail",
             "resultText", "toolResult")
_ID_KEYS = ("toolCallId", "tool_call_id", "callId", "call_id", "invocationId",
            "id")


def classify_event(obj: dict) -> list[dict]:
    """Map one Copilot JSONL object to zero or more UI events.

    UI event shapes:
        {"type": "assistant", "text": str, "delta": bool}
        {"type": "reasoning", "text": str}
        {"type": "tool", "id": str, "name": str, "detail": str,
         "phase": "start"|"end", "ok": bool}
        {"type": "proc", "text": str}           # subprocess/tool output
        {"type": "summary", "text": str}        # end-of-turn usage summary
        {"type": "info", "text": str}
    """
    events: list[dict] = []
    etype = str(obj.get("type") or obj.get("event") or obj.get("kind") or "").lower()
    role = str(obj.get("role") or "").lower()
    data = obj.get("data") if isinstance(obj.get("data"), dict) else {}

    # Transient UI bookkeeping (spinners, background-task counters, …):
    # keep in the raw Agent log only — never promote to chat.
    if obj.get("ephemeral") is True:
        return events
    if etype.startswith(("session.", "heartbeat", "ping", "usage.")):
        return events

    # ── end-of-turn summary ─────────────────────────────────────────────
    if etype in ("result", "turn.result", "run.result") or (
        "exitCode" in obj and "usage" in obj
    ):
        usage = obj.get("usage") or data.get("usage") or {}
        bits = []
        code = obj.get("exitCode", data.get("exitCode"))
        if code is not None:
            bits.append("exit 0 ✓" if code == 0 else f"exit {code} ✗")
        if usage.get("premiumRequests") is not None:
            bits.append(f"{usage['premiumRequests']} model requests")
        ms = usage.get("totalApiDurationMs")
        if ms:
            bits.append(f"{ms / 1000:.1f}s model time")
        cc = usage.get("codeChanges") or {}
        if cc.get("filesModified"):
            bits.append(f"{len(cc['filesModified'])} file(s) modified")
        if bits:
            events.append({"type": "summary", "text": " · ".join(bits)})
        return events

    # ── tool lifecycle ──────────────────────────────────────────────────
    if "tool" in etype:
        name = _deep_find(obj, _NAME_KEYS) or "tool"
        # Pairing id: prefer the call id inside the payload (identical on the
        # start and end events) over the per-event envelope id.
        call_id = (_deep_find(data, _ID_KEYS) or obj.get("parentId")
                   or obj.get("id") or "")
        is_end = any(w in etype for w in
                     ("end", "complete", "finish", "result", "done", "output"))
        ok = True
        for src in (obj, data):
            if src.get("error") or src.get("isError") or src.get("success") is False:
                ok = False

        if is_end:
            out = _deep_find(obj, _OUT_KEYS, want_str=False)
            out_text = _content_to_text(out) or (out if isinstance(out, str) else None)
            detail = (out_text or "").strip()
            events.append({"type": "tool", "id": str(call_id), "name": str(name),
                           "detail": detail[:400], "phase": "end", "ok": ok})
            if out_text and out_text.strip():
                head = out_text.strip()[:2000]
                # Skip binary-ish payloads (e.g. a base64 PNG the agent just
                # viewed): one giant "word" with no whitespace is not a log.
                if not (len(head) >= 2000 and " " not in head and "\n" not in head):
                    events.append({"type": "proc",
                                   "text": f"── {name} output ──\n"
                                           f"{out_text.strip()[:8000]}"})
        else:
            args = _deep_find(obj, _ARG_KEYS, want_str=False)
            if isinstance(args, (dict, list)):
                detail = json.dumps(args, ensure_ascii=False)
            else:
                detail = str(args) if args is not None else ""
            # Skip anonymous, empty tool chatter (permission checks etc.)
            if name == "tool" and not detail:
                return events
            events.append({"type": "tool", "id": str(call_id), "name": str(name),
                           "detail": detail[:400], "phase": "start", "ok": True})
        return events

    # ── reasoning ───────────────────────────────────────────────────────
    if "reason" in etype or "think" in etype:
        text = _content_to_text(_deep_find(obj, ("content", "text"), want_str=False))
        if text:
            events.append({"type": "reasoning", "text": text})
        return events

    # ── assistant / message text ────────────────────────────────────────
    msg = obj.get("message") if isinstance(obj.get("message"), dict) else {}
    delta = obj.get("delta") if isinstance(obj.get("delta"), dict) else {}
    text = (
        _content_to_text(delta.get("content")) or
        (delta.get("text") if isinstance(delta.get("text"), str) else None) or
        _content_to_text(msg.get("content")) or
        _content_to_text(obj.get("content")) or
        (obj.get("text") if isinstance(obj.get("text"), str) else None) or
        _content_to_text(data.get("content")) or
        (data.get("text") if isinstance(data.get("text"), str) else None)
    )
    if text is not None:
        role = role or str(msg.get("role") or data.get("role") or "").lower()
        if role in ("user", "system"):
            return events  # echo of our own prompt — skip
        is_delta = bool(delta) or "delta" in etype or "chunk" in etype
        events.append({"type": "assistant", "text": text, "delta": is_delta})
        return events

    # ── errors ──────────────────────────────────────────────────────────
    if "error" in etype or "warn" in etype:
        err = _deep_find(obj, ("message", "error", "text")) or \
            json.dumps(obj, ensure_ascii=False)[:300]
        events.append({"type": "info", "text": f"⚠ {err}"})
    return events


# ── runner ───────────────────────────────────────────────────────────────────
@dataclass
class AgentSession:
    """One persistent Copilot session; runs one turn at a time."""

    copilot_bin: str = "copilot"
    cwd: str = "."
    session_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    model: str | None = None
    extra_args: list[str] = field(default_factory=list)
    env_overrides: dict[str, str] = field(default_factory=dict)
    env_unset: list[str] = field(default_factory=list)

    _proc: asyncio.subprocess.Process | None = None

    def resolve_bin(self) -> str | None:
        return shutil.which(self.copilot_bin) or (
            self.copilot_bin if os.path.isfile(self.copilot_bin) else None
        )

    def new_session(self):
        self.session_id = str(uuid.uuid4())

    def build_cmd(self, prompt: str) -> list[str]:
        cmd = [
            self.copilot_bin,
            "-p", prompt,
            "--session-id", self.session_id,
            "--allow-all-tools",
            "--output-format", "json",
            "--no-ask-user",
            "--no-color",
            "--no-auto-update",
        ]
        if self.model:
            cmd += ["--model", self.model]
        if os.environ.get("BOGLODITE_DISABLE_MCP", "").strip().lower() in {"1", "true", "yes", "on"}:
            cmd += ["--disable-mcp-server=boglodite-seismic"]
        cmd += self.extra_args
        return cmd

    def build_env(self) -> dict[str, str]:
        env = dict(os.environ)
        # Dark-theme every matplotlib plot the agent's subprocesses produce,
        # and unbuffer python stdout so `tee outputs/run.log` streams live.
        env.setdefault("MATPLOTLIBRC", _DARK_RC)
        env.setdefault("PYTHONUNBUFFERED", "1")
        # The UI runs Copilot in non-interactive prompt mode. Opt in to the
        # repository-scoped MCP server because prompt mode cannot display the
        # normal first-use workspace trust prompt. The server itself is defined
        # in this repository under .github/mcp.json.
        env.setdefault("GITHUB_COPILOT_PROMPT_MODE_WORKSPACE_MCP", "true")
        for k in self.env_unset:
            env.pop(k, None)
        env.update(self.env_overrides)
        return env

    @property
    def running(self) -> bool:
        return self._proc is not None and self._proc.returncode is None

    async def stop(self):
        if self.running:
            try:
                self._proc.terminate()
                try:
                    await asyncio.wait_for(self._proc.wait(), timeout=5)
                except asyncio.TimeoutError:
                    self._proc.kill()
            except ProcessLookupError:
                pass

    async def run_turn(self, prompt: str) -> AsyncIterator[dict]:
        """Spawn one copilot turn; yield UI events until the process exits."""
        if self.resolve_bin() is None:
            yield {"type": "error",
                   "text": (f"Copilot CLI not found ('{self.copilot_bin}'). "
                            "Install it (npm install -g @github/copilot) or set "
                            "BOGLODITE_COPILOT_BIN.")}
            return

        cmd = self.build_cmd(prompt)
        yield {"type": "log", "line": "$ " + " ".join(shlex.quote(c) for c in cmd[:1])
               + f" -p <prompt> --session-id {self.session_id[:8]}… "
               + " ".join(cmd[4:])}

        self._proc = await asyncio.create_subprocess_exec(
            *cmd,
            cwd=self.cwd,
            env=self.build_env(),
            stdin=asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            # Copilot emits tool events that can embed entire files (e.g. a
            # base64 PNG when the agent views a plot) as ONE JSONL line —
            # multi-megabyte lines that overflow asyncio's 64 KB default and
            # deadlock the pipe. Raise the limit generously.
            limit=64 * 1024 * 1024,
        )

        queue: asyncio.Queue[dict | None] = asyncio.Queue()

        async def read_line(stream) -> bytes:
            """readline that survives lines longer than the stream limit."""
            try:
                return await stream.readline()
            except (asyncio.LimitOverrunError, ValueError):
                chunks = []
                while True:
                    try:
                        chunks.append(await stream.readuntil(b"\n"))
                        break
                    except asyncio.LimitOverrunError as e:
                        chunks.append(await stream.read(e.consumed))
                    except asyncio.IncompleteReadError as e:
                        chunks.append(e.partial)
                        break
                return b"".join(chunks)

        async def pump(stream, is_stderr: bool):
            try:
                while True:
                    raw = await read_line(stream)
                    if not raw:
                        break
                    line = raw.decode("utf-8", errors="replace").rstrip("\n")
                    if not line.strip():
                        continue
                    # Forward a truncated copy to the UI log (a multi-MB
                    # base64 blob would choke the browser) …
                    shown = line if len(line) <= 4000 else \
                        line[:4000] + f" …[+{len(line) - 4000} chars truncated]"
                    await queue.put({"type": "log", "line": shown,
                                     "stream": "stderr" if is_stderr else "stdout"})
                    if not is_stderr:
                        # … but classify the FULL line.
                        try:
                            obj = json.loads(line)
                        except (json.JSONDecodeError, ValueError):
                            continue
                        if isinstance(obj, dict):
                            for ev in classify_event(obj):
                                await queue.put(ev)
            except Exception as e:  # noqa: BLE001 — a reader crash must never
                await queue.put({"type": "log",   # deadlock the turn
                                 "line": f"[reader error] {e}",
                                 "stream": "stderr"})

        pumps = [
            asyncio.create_task(pump(self._proc.stdout, False)),
            asyncio.create_task(pump(self._proc.stderr, True)),
        ]

        async def finish():
            await asyncio.gather(*pumps, return_exceptions=True)
            code = await self._proc.wait()
            await queue.put({"type": "done", "code": code})
            await queue.put(None)

        fin = asyncio.create_task(finish())
        try:
            while True:
                item = await queue.get()
                if item is None:
                    break
                yield item
        finally:
            fin.cancel()
            for t in pumps:
                t.cancel()
            self._proc = None
