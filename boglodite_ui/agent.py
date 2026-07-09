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

Every raw stdout/stderr line is forwarded to the UI Log panel verbatim.
Lines that parse as JSON are additionally classified into chat-level events
(assistant text, tool calls) with tolerant heuristics, so schema drift in
the CLI degrades gracefully to "still visible in the log" instead of
breaking the UI.
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


# ── JSONL classification ─────────────────────────────────────────────────────
def _first_str(*vals) -> str | None:
    for v in vals:
        if isinstance(v, str) and v.strip():
            return v
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
                t = _first_str(blk.get("text"), blk.get("content"))
                if t:
                    parts.append(t)
        joined = "".join(parts)
        return joined if joined.strip() else None
    return None


def classify_event(obj: dict) -> list[dict]:
    """Map one Copilot JSONL object to zero or more UI events.

    UI event shapes:
        {"type": "assistant", "text": str, "delta": bool}
        {"type": "reasoning", "text": str}
        {"type": "tool", "name": str, "detail": str, "phase": "start"|"end"}
        {"type": "info", "text": str}
    """
    events: list[dict] = []
    etype = str(_first_str(obj.get("type"), obj.get("event"), obj.get("kind")) or "").lower()
    role = str(obj.get("role") or "").lower()

    # nested payload conventions: {"type": ..., "data": {...}} / {"message": {...}}
    data = obj.get("data") if isinstance(obj.get("data"), dict) else {}
    msg = obj.get("message") if isinstance(obj.get("message"), dict) else {}
    delta = obj.get("delta") if isinstance(obj.get("delta"), dict) else {}

    # tool call?
    tool_name = _first_str(
        obj.get("tool_name"), obj.get("tool"),
        data.get("tool_name"), data.get("tool"),
        (obj.get("name") if "tool" in etype else None),
        (data.get("name") if "tool" in etype else None),
    )
    if tool_name or "tool" in etype:
        args = None
        for src in (obj, data):
            for key in ("arguments", "input", "args", "parameters", "command"):
                if src.get(key) is not None:
                    args = src[key]
                    break
            if args is not None:
                break
        detail = args if isinstance(args, str) else (
            json.dumps(args, ensure_ascii=False)[:400] if args is not None else ""
        )
        result = _content_to_text(obj.get("result")) or _content_to_text(data.get("result"))
        phase = "end" if (result is not None or "result" in etype or "end" in etype
                          or "complet" in etype) else "start"
        events.append({
            "type": "tool",
            "name": tool_name or "tool",
            "detail": (result or detail or "")[:600],
            "phase": phase,
        })
        return events

    # reasoning?
    if "reason" in etype or "think" in etype:
        text = (_content_to_text(obj.get("content")) or _first_str(obj.get("text"))
                or _content_to_text(data.get("content")) or _first_str(data.get("text")))
        if text:
            events.append({"type": "reasoning", "text": text})
        return events

    # assistant / message text?
    text = (
        _content_to_text(delta.get("content")) or _first_str(delta.get("text"))
        or _content_to_text(msg.get("content")) or _first_str(msg.get("text"))
        or _content_to_text(obj.get("content")) or _first_str(obj.get("text"))
        or _content_to_text(data.get("content")) or _first_str(data.get("text"))
    )
    if text is not None:
        if role in ("user", "system"):
            return events  # echo of our own prompt — skip
        is_delta = bool(delta) or "delta" in etype or "chunk" in etype
        events.append({"type": "assistant", "text": text, "delta": is_delta})
        return events

    # informative lifecycle events (session start, model, usage, errors)
    if any(k in etype for k in ("error", "warn")):
        err = _first_str(obj.get("error"), data.get("error"),
                         json.dumps(obj, ensure_ascii=False)[:400])
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

    _proc: asyncio.subprocess.Process | None = None
    _first_turn: bool = True

    def resolve_bin(self) -> str | None:
        return shutil.which(self.copilot_bin) or (
            self.copilot_bin if os.path.isfile(self.copilot_bin) else None
        )

    def new_session(self):
        self.session_id = str(uuid.uuid4())
        self._first_turn = True

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
        cmd += self.extra_args
        return cmd

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
        env = dict(os.environ)
        env.update(self.env_overrides)

        yield {"type": "log", "line": "$ " + " ".join(shlex.quote(c) for c in cmd[:1])
               + f" -p <prompt> --session-id {self.session_id[:8]}… "
               + " ".join(cmd[4:])}

        self._proc = await asyncio.create_subprocess_exec(
            *cmd,
            cwd=self.cwd,
            env=env,
            stdin=asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        self._first_turn = False

        queue: asyncio.Queue[dict | None] = asyncio.Queue()

        async def pump(stream, is_stderr: bool):
            while True:
                raw = await stream.readline()
                if not raw:
                    break
                line = raw.decode("utf-8", errors="replace").rstrip("\n")
                if not line.strip():
                    continue
                await queue.put({"type": "log", "line": line,
                                 "stream": "stderr" if is_stderr else "stdout"})
                if not is_stderr:
                    try:
                        obj = json.loads(line)
                    except (json.JSONDecodeError, ValueError):
                        continue
                    if isinstance(obj, dict):
                        for ev in classify_event(obj):
                            await queue.put(ev)

        pumps = [
            asyncio.create_task(pump(self._proc.stdout, False)),
            asyncio.create_task(pump(self._proc.stderr, True)),
        ]

        async def finish():
            await asyncio.gather(*pumps)
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
