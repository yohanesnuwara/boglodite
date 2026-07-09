"""Configuration for the Boglodite UI.

Everything is resolved relative to the repo root (the directory you launch
`boglodite` from), mirroring the conventions in .github/copilot-instructions.md:

    data/     seismic input data
    outputs/  agent-generated results
    skills/   SKILL.md registry
    models/   trained weights

Environment overrides:
    BOGLODITE_SEGY         absolute/relative path to the SEGY volume to display
    BOGLODITE_COPILOT_BIN  copilot executable (default: "copilot")
    BOGLODITE_PORT         UI port (default: 8265)
"""

from __future__ import annotations

import glob
import os
from dataclasses import dataclass, field


def _find_default_segy(repo_root: str) -> str | None:
    """Prefer the documented F3 path, else first *.segy/*.sgy under data/."""
    f3 = os.path.join(
        repo_root,
        "data",
        "Dutch F3 seismic data",
        "Dutch Government_F3_entire_8bit seismic.segy",
    )
    if os.path.isfile(f3):
        return f3
    for pat in ("data/**/*.segy", "data/**/*.sgy", "data/**/*.SEGY", "data/**/*.SGY"):
        hits = sorted(glob.glob(os.path.join(repo_root, pat), recursive=True))
        if hits:
            return hits[0]
    return None


@dataclass
class Config:
    repo_root: str = field(default_factory=os.getcwd)
    port: int = int(os.environ.get("BOGLODITE_PORT", "8265"))
    copilot_bin: str = os.environ.get("BOGLODITE_COPILOT_BIN", "copilot")

    @property
    def data_dir(self) -> str:
        return os.path.join(self.repo_root, "data")

    @property
    def outputs_dir(self) -> str:
        return os.path.join(self.repo_root, "outputs")

    @property
    def skills_dir(self) -> str:
        return os.path.join(self.repo_root, "skills")

    @property
    def segy_path(self) -> str | None:
        env = os.environ.get("BOGLODITE_SEGY")
        if env:
            p = env if os.path.isabs(env) else os.path.join(self.repo_root, env)
            return p if os.path.isfile(p) else None
        return _find_default_segy(self.repo_root)


CONFIG = Config()
