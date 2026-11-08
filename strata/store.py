"""Project store (blackboard): typed, versioned, immutable objects plus an event log.

Layout per run (docs/SWARM_ARCHITECTURE.md §3, §7.1):

    runs/<run_id>/
      manifest.json
      events.jsonl
      store/<Type>/<name>/v<k>/meta.json + payload files
      candidate/            benchmark export (§5.0 contract)
"""

from __future__ import annotations

import asyncio
import json
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Literal

from pydantic import BaseModel, Field

from strata.config import RUNS_DIR

Status = Literal["draft", "verified", "rejected", "frozen", "superseded"]


class ToolCall(BaseModel):
    tool: str
    params: dict[str, Any] = {}
    diagnostics: dict[str, Any] = {}


class Verdict(BaseModel):
    verifier: str
    passed: bool
    summary: str
    findings: list[str] = []
    evidence: dict[str, Any] = {}
    round: int = 1


class ObjectMeta(BaseModel):
    id: str                                  # "<Type>:<name>"
    type: str
    name: str
    version: int
    producer: str
    inputs: list[str] = []                   # "<id>@v<k>"
    tool_calls: list[ToolCall] = []
    crs: str = "ED50 / UTM zone 32N"
    domain: Literal["time", "depth", "none"] = "none"
    unit: str = ""
    confidence: float | None = None
    status: Status = "draft"
    verdicts: list[Verdict] = []
    scenario: str | None = None
    summary: str = ""
    payload: list[str] = []                  # file names inside the version folder
    created: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    @property
    def ref(self) -> str:
        return f"{self.id}@v{self.version}"


class Store:
    """One run's blackboard. Thread-safe enough for a single asyncio loop."""

    def __init__(self, run_id: str | None = None, root: Path = RUNS_DIR):
        self.run_id = run_id or time.strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:4]
        self.dir = root / self.run_id
        (self.dir / "store").mkdir(parents=True, exist_ok=True)
        (self.dir / "candidate").mkdir(exist_ok=True)
        self.events_file = self.dir / "events.jsonl"
        self._subscribers: list[asyncio.Queue] = []
        self._seq = sum(1 for _ in self.events_file.open()) if self.events_file.exists() else 0

    # ── events ───────────────────────────────────────────────────────────────
    def emit(self, kind: str, agent: str = "", **data) -> dict:
        self._seq += 1
        ev = {"seq": self._seq, "t": time.time(), "kind": kind, "agent": agent, **data}
        with self.events_file.open("a") as f:
            f.write(json.dumps(ev, default=str) + "\n")
        for q in list(self._subscribers):
            q.put_nowait(ev)
        return ev

    def subscribe(self) -> asyncio.Queue:
        q: asyncio.Queue = asyncio.Queue()
        self._subscribers.append(q)
        return q

    def unsubscribe(self, q: asyncio.Queue) -> None:
        if q in self._subscribers:
            self._subscribers.remove(q)

    def events(self) -> list[dict]:
        if not self.events_file.exists():
            return []
        return [json.loads(line) for line in self.events_file.open()]

    # ── objects ──────────────────────────────────────────────────────────────
    def _obj_dir(self, type_: str, name: str) -> Path:
        return self.dir / "store" / type_ / name

    def versions(self, type_: str, name: str) -> list[int]:
        d = self._obj_dir(type_, name)
        return sorted(int(p.name[1:]) for p in d.glob("v*")) if d.exists() else []

    def publish(self, type_: str, name: str, producer: str, write: Callable[[Path], list[str]] | None = None,
                **meta) -> ObjectMeta:
        """Create a new immutable version. `write(dir)` writes payload files and returns their names."""
        v = (self.versions(type_, name) or [0])[-1] + 1
        d = self._obj_dir(type_, name) / f"v{v}"
        d.mkdir(parents=True)
        payload = write(d) if write else []
        m = ObjectMeta(id=f"{type_}:{name}", type=type_, name=name, version=v, producer=producer,
                       payload=payload, **meta)
        (d / "meta.json").write_text(m.model_dump_json(indent=2))
        for old in self.versions(type_, name)[:-1]:          # previous versions become superseded
            om = self.get(type_, name, old)
            if om.status in ("draft", "verified", "rejected"):
                self._update(om, status="superseded")
        self.emit("publish", producer, object=m.id, version=v, status=m.status, summary=m.summary,
                  confidence=m.confidence)
        return m

    def get(self, type_: str, name: str, version: int | None = None) -> ObjectMeta:
        vs = self.versions(type_, name)
        if not vs:
            raise KeyError(f"{type_}:{name} not in store")
        v = version or vs[-1]
        return ObjectMeta.model_validate_json((self._obj_dir(type_, name) / f"v{v}" / "meta.json").read_text())

    def path(self, m: ObjectMeta, file: str | None = None) -> Path:
        d = self._obj_dir(m.type, m.name) / f"v{m.version}"
        return d / (file or (m.payload[0] if m.payload else ""))

    def _update(self, m: ObjectMeta, **changes) -> ObjectMeta:
        m = m.model_copy(update=changes)
        (self._obj_dir(m.type, m.name) / f"v{m.version}" / "meta.json").write_text(m.model_dump_json(indent=2))
        return m

    def add_verdict(self, m: ObjectMeta, verdict: Verdict) -> ObjectMeta:
        status = "verified" if verdict.passed else "rejected"
        m = self._update(self.get(m.type, m.name, m.version), verdicts=[*m.verdicts, verdict], status=status)
        self.emit("verdict", verdict.verifier, object=m.id, version=m.version, passed=verdict.passed,
                  summary=verdict.summary, findings=verdict.findings, round=verdict.round)
        return m

    def set_status(self, m: ObjectMeta, status: Status, by: str = "orchestrator", **extra) -> ObjectMeta:
        m = self._update(self.get(m.type, m.name, m.version), status=status, **extra)
        self.emit("status", by, object=m.id, version=m.version, status=status)
        return m

    def tree(self) -> list[dict]:
        out = []
        for meta in sorted((self.dir / "store").glob("*/*/v*/meta.json")):
            m = ObjectMeta.model_validate_json(meta.read_text())
            out.append({"id": m.id, "type": m.type, "name": m.name, "version": m.version, "status": m.status,
                        "producer": m.producer, "summary": m.summary, "confidence": m.confidence,
                        "domain": m.domain, "verdicts": [v.model_dump() for v in m.verdicts],
                        "tool_calls": [t.model_dump() for t in m.tool_calls], "inputs": m.inputs})
        return out

    def write_manifest(self, **fields) -> None:
        p = self.dir / "manifest.json"
        cur = json.loads(p.read_text()) if p.exists() else {"run_id": self.run_id}
        cur.update(fields)
        p.write_text(json.dumps(cur, indent=2, default=str))

    def manifest(self) -> dict:
        p = self.dir / "manifest.json"
        return json.loads(p.read_text()) if p.exists() else {"run_id": self.run_id}
