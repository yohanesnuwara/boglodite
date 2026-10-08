"""Rubric implementation for the compact Boglodite paper experiment.

Top-level outcome:
  correct         output exists and passes preregistered fidelity criteria
  silent_failure  output is well formed but fails numerical fidelity criteria
  overt_failure   required output missing/unreadable or wrong geometry

The evaluator keeps the scientific score separate from agent-process metadata
(e.g., human interventions and tool-call counts), so the same rubric can score
runs launched from the UI or directly from Copilot CLI.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from .metrics import array_summary, categorical_metrics, mae, nrms_percent, pearson, rmse, structural_similarity
from .tasks import Task


DEFAULT_THRESHOLDS = {
    "faultseg": {
        "max_nrms_percent": 1.0,
        "min_correlation": 0.999,
        "max_mae": 0.005,
        "probability_min": -1e-5,
        "probability_max": 1.00001,
    },
    "malenov": {
        "min_voxel_agreement": 0.995,
        "min_macro_f1": 0.99,
        "min_mean_iou": 0.98,
        "probability_sum_tolerance": 0.02,
    },
}


@dataclass
class RunMetadata:
    task_id: str
    condition: str
    replicate: int
    human_interventions: int = 0
    agent_log: str | None = None
    notes: str = ""


def load_thresholds(path: str | Path | None = None) -> dict[str, Any]:
    if path is None:
        return DEFAULT_THRESHOLDS
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    return raw.get("thresholds", raw)


def _load_npy(path: Path) -> tuple[np.ndarray | None, str | None]:
    if not path.exists():
        return None, "missing"
    try:
        return np.load(path, allow_pickle=False), None
    except Exception as exc:
        return None, f"unreadable: {type(exc).__name__}: {exc}"


def _tool_log_evidence(path: str | Path | None, expected_tool: str) -> dict[str, Any]:
    if not path:
        return {"available": False}
    p = Path(path)
    if not p.exists():
        return {"available": False, "error": f"log not found: {p}"}
    text = p.read_text(encoding="utf-8", errors="replace")
    # Accept raw MCP name, Copilot-rendered server prefix, or adapter name.
    patterns = {
        "faultseg": [r"run_faultseg", r"boglodite-seismic-run_faultseg", r"FaultSeg"],
        "malenov": [r"run_malenov", r"boglodite-seismic-run_malenov", r"MalenoV"],
    }
    hits = sum(len(re.findall(pat, text, flags=re.I)) for pat in patterns[expected_tool])
    toolish = len(re.findall(r'"type"\s*:\s*"[^"]*tool[^"]*"|boglodite-seismic-run_', text, flags=re.I))
    return {
        "available": True,
        "expected_tool_mentions": hits,
        "tool_event_mentions": toolish,
        "expected_tool_observed": hits > 0,
    }


def _score_faultseg(task: Task, candidate_dir: Path, reference_dir: Path, thr: dict[str, Any]) -> dict[str, Any]:
    key = "prediction"
    cpath = candidate_dir / task.candidate_files[key]
    rpath = reference_dir / task.reference_files[key]
    cand, cerr = _load_npy(cpath)
    ref, rerr = _load_npy(rpath)
    if rerr:
        raise FileNotFoundError(f"Reference for {task.id} is {rerr}: {rpath}")
    if cerr:
        return {"outcome": "overt_failure", "reason": cerr, "candidate": str(cpath), "reference": str(rpath)}
    assert cand is not None and ref is not None
    if cand.shape != ref.shape:
        return {
            "outcome": "overt_failure",
            "reason": "shape_mismatch",
            "candidate_summary": array_summary(cand),
            "reference_summary": array_summary(ref),
        }

    metrics = {
        "nrms_percent": nrms_percent(cand, ref),
        "correlation": pearson(cand, ref),
        "mae": mae(cand, ref),
        "rmse": rmse(cand, ref),
        "ssim": structural_similarity(cand, ref),
    }
    summary = array_summary(cand)
    probability_valid = (
        summary["finite_fraction"] == 1.0
        and summary["min"] is not None and summary["min"] >= float(thr["probability_min"])
        and summary["max"] is not None and summary["max"] <= float(thr["probability_max"])
    )
    passed = (
        math.isfinite(metrics["nrms_percent"]) and metrics["nrms_percent"] <= float(thr["max_nrms_percent"])
        and math.isfinite(metrics["correlation"]) and metrics["correlation"] >= float(thr["min_correlation"])
        and math.isfinite(metrics["mae"]) and metrics["mae"] <= float(thr["max_mae"])
        and probability_valid
    )
    return {
        "outcome": "correct" if passed else "silent_failure",
        "metrics": metrics,
        "candidate_summary": summary,
        "reference_summary": array_summary(ref),
        "scientific_invariants": {"shape_match": True, "probability_range_valid": probability_valid},
        "thresholds": thr,
    }


def _score_malenov(task: Task, candidate_dir: Path, reference_dir: Path, thr: dict[str, Any]) -> dict[str, Any]:
    ccls_path = candidate_dir / task.candidate_files["classes"]
    rcls_path = reference_dir / task.reference_files["classes"]
    cand, cerr = _load_npy(ccls_path)
    ref, rerr = _load_npy(rcls_path)
    if rerr:
        raise FileNotFoundError(f"Reference for {task.id} is {rerr}: {rcls_path}")
    if cerr:
        return {"outcome": "overt_failure", "reason": cerr, "candidate": str(ccls_path), "reference": str(rcls_path)}
    assert cand is not None and ref is not None
    if cand.shape != ref.shape:
        return {
            "outcome": "overt_failure",
            "reason": "shape_mismatch",
            "candidate_summary": array_summary(cand),
            "reference_summary": array_summary(ref),
        }

    metrics = categorical_metrics(cand, ref)
    prob_info: dict[str, Any] = {"available": False}
    prob_name = task.candidate_files.get("probabilities")
    ref_prob_name = task.reference_files.get("probabilities")
    if prob_name and ref_prob_name:
        cp, cperr = _load_npy(candidate_dir / prob_name)
        rp, rperr = _load_npy(reference_dir / ref_prob_name)
        if cperr is None and rperr is None and cp is not None and rp is not None and cp.shape == rp.shape:
            prob_mae = mae(cp, rp)
            sums = np.sum(cp, axis=-1)
            sum_error = float(np.nanmax(np.abs(sums - 1.0))) if sums.size else float("nan")
            prob_info = {
                "available": True,
                "mae_vs_reference": prob_mae,
                "max_softmax_sum_error": sum_error,
                "softmax_valid": bool(np.isfinite(sum_error) and sum_error <= float(thr["probability_sum_tolerance"])),
                "summary": array_summary(cp),
            }

    passed = (
        metrics["voxel_agreement"] >= float(thr["min_voxel_agreement"])
        and metrics["macro_f1"] >= float(thr["min_macro_f1"])
        and metrics["mean_iou"] >= float(thr["min_mean_iou"])
        and (not prob_info.get("available") or bool(prob_info.get("softmax_valid")))
    )
    return {
        "outcome": "correct" if passed else "silent_failure",
        "metrics": metrics,
        "probabilities": prob_info,
        "candidate_summary": array_summary(cand),
        "reference_summary": array_summary(ref),
        "scientific_invariants": {"shape_match": True},
        "thresholds": thr,
    }


def score_run(
    task: Task,
    candidate_dir: str | Path,
    reference_dir: str | Path,
    metadata: RunMetadata,
    thresholds: dict[str, Any] | None = None,
) -> dict[str, Any]:
    cdir = Path(candidate_dir)
    rdir = Path(reference_dir)
    thresholds = thresholds or DEFAULT_THRESHOLDS
    if task.tool == "faultseg":
        scientific = _score_faultseg(task, cdir, rdir, thresholds["faultseg"])
    elif task.tool == "malenov":
        scientific = _score_malenov(task, cdir, rdir, thresholds["malenov"])
    else:
        raise ValueError(f"Unsupported tool: {task.tool}")

    return {
        "schema_version": 1,
        "scored_at": datetime.now(timezone.utc).isoformat(),
        "task": {
            "id": task.id,
            "tool": task.tool,
            "orientation": task.orientation,
            "coordinate": task.coordinate,
            "prompt": task.prompt,
        },
        "metadata": asdict(metadata),
        "scientific_score": scientific,
        "process_score": {
            "human_interventions": metadata.human_interventions,
            "tool_evidence": _tool_log_evidence(metadata.agent_log, task.tool),
        },
        "primary_outcome": scientific["outcome"],
    }
