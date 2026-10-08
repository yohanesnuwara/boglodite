"""Programmatic workspace bootstrap for Boglodite.

The pip-installable packages (``boglodite_ui`` / ``boglodite_mcp``) are only the
thin orchestration layer. A *runnable* agent also needs a full working tree:
the ``skills/``, ``.github/mcp.json`` + ``.github/copilot-instructions.md``
harness, the ``sandbox/`` adapters, the external tool repos under ``tools/``,
the F3 SEG-Y data, and the pre-trained model weights. None of that ships in the
wheel, so this module recreates it on demand — the code equivalent of the
``initiate_boglodite`` skill.

Typical use from a notebook::

    from boglodite_ui.bootstrap import ensure_workspace
    from boglodite_ui.agent import AgentSession

    ws = ensure_workspace(device="cpu")          # clone + initiate (idempotent)
    session = AgentSession(cwd=ws.path, env_overrides=ws.env)

Every step is guarded by a sentinel check, so re-running is cheap and safe.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

# ── asset manifest ────────────────────────────────────────────────────────────
REPO_URL = "https://github.com/yohanesnuwara/boglodite"
DEFAULT_REF = "main"
DEFAULT_WORKSPACE = Path.home() / ".cache" / "boglodite" / "workspace"

# External tool repositories cloned into tools/ (name -> git URL).
TOOL_REPOS = {
    "MalenoV": "https://github.com/bolgebrygg/MalenoV",
    "facies_net": "https://github.com/crild/facies_net",
    "faultSeg": "https://github.com/xinwucwp/faultSeg",
}
OPTIONAL_TOOL_REPOS = {
    "seismicfoundationmodel": "https://github.com/shenghanlin/seismicfoundationmodel",
}

# Google-Drive folders downloaded with gdown (url, destination-relative-to-root).
DATA_DRIVE = (
    "https://drive.google.com/drive/folders/"
    "0B7brcf-eGK8CbGhBdmZoUnhiTWs?resourcekey=0-0ZhV_OJ3TKN1ShFAGcrOzQ"
)
FAULTSEG_MODEL_DRIVE = (
    "https://drive.google.com/drive/folders/1q8sAoLJgbhYHRubzyqMi9KkTeZWXWtNd"
)

# Sentinels (relative to workspace root) that mean "this asset is present".
DATA_SENTINEL = Path("data") / "Dutch F3 seismic data" / "Dutch Government_F3_entire_8bit seismic.segy"
FAULTSEG_MODEL_SENTINEL = Path("models") / "faultSeg_model" / "model" / "fseg-60.hdf5"
MALENOV_MODEL_SENTINEL = Path("models") / "F3_multiclass_model.h5"
FAULTSEG_ADAPTER = Path("sandbox") / "FaultSeg" / "predict_only_fault.py"
MALENOV_ADAPTER = Path("sandbox") / "MalenoV" / "predict_only_facies_stable.py"


@dataclass
class Workspace:
    """Result of :func:`ensure_workspace`."""

    path: Path
    ref: str
    device: str
    extra: str
    env: dict[str, str] = field(default_factory=dict)
    ready: dict[str, bool] = field(default_factory=dict)
    missing: list[str] = field(default_factory=list)
    copilot_bin: str | None = None

    @property
    def fully_ready(self) -> bool:
        return not self.missing and self.copilot_bin is not None


# ── helpers ───────────────────────────────────────────────────────────────────
def _log(msg: str, quiet: bool) -> None:
    if not quiet:
        print(f"[boglodite-init] {msg}", flush=True)


def _run(cmd: list[str], cwd: Path | None = None, quiet: bool = False) -> None:
    _log("$ " + " ".join(cmd), quiet)
    subprocess.run(cmd, cwd=str(cwd) if cwd else None, check=True)


def _detect_device(device: str) -> str:
    """Resolve ``auto`` to ``gpu``/``cpu`` by probing for an NVIDIA GPU."""
    if device != "auto":
        return device
    smi = shutil.which("nvidia-smi")
    if smi:
        try:
            subprocess.run([smi], check=True, capture_output=True)
            return "gpu"
        except Exception:
            pass
    return "cpu"


def _gdown_folder(url: str, dest: Path, quiet: bool) -> None:
    """Download a Drive folder with gdown via an ephemeral uv tool run."""
    dest.mkdir(parents=True, exist_ok=True)
    _run(["uv", "tool", "run", "gdown", "--folder", url, "-O", str(dest)],
         quiet=quiet)


# ── main entry point ──────────────────────────────────────────────────────────
def ensure_workspace(
    workspace: str | os.PathLike | None = None,
    ref: str | None = None,
    repo_url: str | None = None,
    device: str = "auto",
    include_optional_tools: bool = False,
    run_sync: bool = True,
    download_data: bool = True,
    download_models: bool = True,
    quiet: bool = False,
) -> Workspace:
    """Clone and initialise a runnable Boglodite workspace (idempotent).

    Parameters
    ----------
    workspace:
        Target directory. Defaults to ``$BOGLODITE_HOME`` or
        ``~/.cache/boglodite/workspace``.
    ref:
        Git ref (branch/tag/commit) to clone. Defaults to ``$BOGLODITE_REF`` or
        ``main``.
    repo_url:
        Boglodite repository URL (defaults to ``$BOGLODITE_REPO_URL`` or the
        canonical GitHub URL). A local path works too, for offline testing.
    device:
        ``auto`` (probe for an NVIDIA GPU), ``cpu`` (force CPU, no CUDA wheels),
        or ``gpu``.
    include_optional_tools:
        Also clone large/secondary tool repos (e.g. seismicfoundationmodel).
    run_sync:
        Run ``uv sync`` with the chosen inference extra in the workspace.
    download_data / download_models:
        Fetch the F3 SEG-Y volume / pre-trained weights if missing.
    quiet:
        Suppress progress logging.

    Returns
    -------
    Workspace
        Paths, resolved device/extra, the environment overrides to pass to
        :class:`~boglodite_ui.agent.AgentSession`, and a readiness report.
    """
    root = Path(workspace or os.environ.get("BOGLODITE_HOME") or DEFAULT_WORKSPACE).expanduser()
    ref = ref or os.environ.get("BOGLODITE_REF") or DEFAULT_REF
    repo_url = repo_url or os.environ.get("BOGLODITE_REPO_URL") or REPO_URL
    device = _detect_device(device)
    extra = "models" if device == "gpu" else "models-cpu"

    # 1. Clone the full working tree (skills + harness + adapters) if absent.
    if not (root / ".git").exists():
        root.parent.mkdir(parents=True, exist_ok=True)
        _log(f"cloning {repo_url}@{ref} -> {root}", quiet)
        _run(["git", "clone", "--branch", ref, repo_url, str(root)], quiet=quiet)
    else:
        _log(f"workspace already present at {root}", quiet)

    # 2. Ensure the runtime directory layout exists.
    for sub in ("data", "sandbox", "tools", "outputs", "models"):
        (root / sub).mkdir(parents=True, exist_ok=True)

    # 3. Clone external tool repos into tools/ (skip any already cloned).
    repos = dict(TOOL_REPOS)
    if include_optional_tools:
        repos.update(OPTIONAL_TOOL_REPOS)
    for name, url in repos.items():
        target = root / "tools" / name
        if (target / ".git").exists() or target.exists():
            _log(f"tool present: tools/{name}", quiet)
            continue
        _run(["git", "clone", url, str(target)], quiet=quiet)

    # 4. Build the inference virtual environment (TensorFlow lives here, NOT in
    #    the caller's env) with the device-appropriate extra.
    if run_sync:
        _log(f"uv sync --extra {extra} (device={device})", quiet)
        _run(["uv", "sync", "--extra", extra], cwd=root, quiet=quiet)

    # 5. Download data + model weights (best-effort, sentinel-guarded).
    if download_data and not (root / DATA_SENTINEL).exists():
        _log("downloading F3 SEG-Y volume (~1.2 GB)…", quiet)
        _gdown_folder(DATA_DRIVE, root / "data", quiet)
    if download_models and not (root / FAULTSEG_MODEL_SENTINEL).exists():
        _log("downloading FaultSeg model weights…", quiet)
        _gdown_folder(FAULTSEG_MODEL_DRIVE, root / "models" / "faultSeg_model", quiet)

    # 6. Build the readiness report from sentinels.
    ready = {
        "data": (root / DATA_SENTINEL).exists(),
        "faultseg_model": (root / FAULTSEG_MODEL_SENTINEL).exists(),
        "faultseg_adapter": (root / FAULTSEG_ADAPTER).exists(),
        "malenov_model": (root / MALENOV_MODEL_SENTINEL).exists(),
        "malenov_adapter": (root / MALENOV_ADAPTER).exists(),
        "skills": (root / "skills").is_dir(),
        "mcp_config": (root / ".github" / "mcp.json").exists(),
    }
    missing = [k for k, ok in ready.items() if not ok]

    # The MalenoV multiclass weights have no public auto-download; flag clearly.
    if not ready["malenov_model"]:
        _log("note: models/F3_multiclass_model.h5 not found — FaultSeg works; "
             "MalenoV facies classification needs this file added manually.", quiet)

    env: dict[str, str] = {}
    if device == "cpu":
        # Propagated via AgentSession to the copilot -> MCP -> inference
        # subprocess, forcing TensorFlow onto the CPU.
        env["CUDA_VISIBLE_DEVICES"] = ""

    copilot_bin = shutil.which("copilot")
    if copilot_bin is None:
        _log("note: the `copilot` CLI was not found on PATH — install it with "
             "`npm install -g @github/copilot` to run agent conversations.", quiet)

    ws = Workspace(path=root, ref=ref, device=device, extra=extra, env=env,
                   ready=ready, missing=missing, copilot_bin=copilot_bin)
    _log(f"workspace ready at {root} (device={device}; "
         f"missing={missing or 'none'})", quiet)
    return ws


def main(argv: list[str] | None = None) -> int:
    """``boglodite-init`` console entry point."""
    p = argparse.ArgumentParser(
        prog="boglodite-init",
        description="Clone and initialise a runnable Boglodite workspace.")
    p.add_argument("--workspace", help="target directory "
                   "(default: $BOGLODITE_HOME or ~/.cache/boglodite/workspace)")
    p.add_argument("--ref", help="git ref to clone (default: $BOGLODITE_REF or main)")
    p.add_argument("--repo-url", help="boglodite repo URL or local path")
    p.add_argument("--device", choices=("auto", "cpu", "gpu"), default="auto")
    p.add_argument("--optional-tools", action="store_true",
                   help="also clone large/secondary tool repos")
    p.add_argument("--no-sync", action="store_true", help="skip uv sync")
    p.add_argument("--no-data", action="store_true", help="skip data download")
    p.add_argument("--no-models", action="store_true", help="skip model download")
    p.add_argument("--quiet", action="store_true")
    args = p.parse_args(argv)

    ws = ensure_workspace(
        workspace=args.workspace,
        ref=args.ref,
        repo_url=args.repo_url,
        device=args.device,
        include_optional_tools=args.optional_tools,
        run_sync=not args.no_sync,
        download_data=not args.no_data,
        download_models=not args.no_models,
        quiet=args.quiet,
    )
    print(f"\nWorkspace : {ws.path}")
    print(f"Device    : {ws.device}  (uv extra: {ws.extra})")
    print(f"Copilot   : {ws.copilot_bin or 'NOT FOUND — npm install -g @github/copilot'}")
    print("Readiness :")
    for k, ok in ws.ready.items():
        print(f"  {'✓' if ok else '✗'} {k}")
    return 0 if ws.fully_ready else 1


if __name__ == "__main__":
    sys.exit(main())
