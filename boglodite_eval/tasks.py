"""Fixed benchmark tasks used by the Boglodite research evaluation.

The benchmark is deliberately small: six interpreter-like tasks spanning the
capabilities demonstrated in the paper.  The task definitions are data, not
hard-coded scoring logic, so they can be edited in evaluation/tasks.json before
an experiment is frozen.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

ToolName = Literal["faultseg", "malenov"]


@dataclass(frozen=True)
class Task:
    id: str
    tool: ToolName
    prompt: str
    orientation: str
    coordinate: int
    candidate_files: dict[str, str]
    reference_files: dict[str, str]

    @classmethod
    def from_dict(cls, obj: dict) -> "Task":
        return cls(
            id=str(obj["id"]),
            tool=str(obj["tool"]).lower(),
            prompt=str(obj["prompt"]),
            orientation=str(obj.get("orientation", "inline")),
            coordinate=int(obj["coordinate"]),
            candidate_files=dict(obj["candidate_files"]),
            reference_files=dict(obj.get("reference_files", obj["candidate_files"])),
        )


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def default_task_file() -> Path:
    return repo_root() / "evaluation" / "tasks.json"


def load_tasks(path: str | Path | None = None) -> dict[str, Task]:
    p = Path(path) if path else default_task_file()
    raw = json.loads(p.read_text(encoding="utf-8"))
    tasks = [Task.from_dict(x) for x in raw["tasks"]]
    return {t.id: t for t in tasks}
