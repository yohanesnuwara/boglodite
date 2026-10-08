"""Command-line interface for the compact Boglodite evaluation framework."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import sys
from pathlib import Path

from .scoring import RunMetadata, load_thresholds, score_run
from .tasks import load_tasks, repo_root


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def cmd_list(args: argparse.Namespace) -> int:
    tasks = load_tasks(args.tasks)
    for t in tasks.values():
        print(f"{t.id:4s}  {t.tool:8s}  {t.orientation:9s} {t.coordinate:<5d}  {t.prompt}")
    return 0


def cmd_reference(args: argparse.Namespace) -> int:
    tasks = load_tasks(args.tasks)
    task = tasks[args.task]
    src = Path(args.source_dir)
    dst = Path(args.reference_dir) / task.id
    dst.mkdir(parents=True, exist_ok=True)
    manifest = {"task": task.id, "files": {}}
    for key, name in task.candidate_files.items():
        p = src / name
        if not p.exists():
            # PNG is optional for scoring and need not block a numerical reference.
            if p.suffix.lower() == ".png":
                continue
            raise FileNotFoundError(f"Required reference source missing: {p}")
        out = dst / task.reference_files.get(key, name)
        shutil.copy2(p, out)
        manifest["files"][key] = {"name": out.name, "sha256": _sha256(out)}
    (dst / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Frozen reference for {task.id}: {dst}")
    return 0


def cmd_score(args: argparse.Namespace) -> int:
    tasks = load_tasks(args.tasks)
    task = tasks[args.task]
    ref_dir = Path(args.reference_dir) / task.id
    meta = RunMetadata(
        task_id=task.id,
        condition=args.condition,
        replicate=args.replicate,
        human_interventions=args.human_interventions,
        agent_log=args.agent_log,
        notes=args.notes or "",
    )
    result = score_run(task, args.candidate_dir, ref_dir, meta, load_thresholds(args.thresholds))
    outdir = Path(args.run_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    out = outdir / f"{task.id}__{args.condition}__r{args.replicate:02d}.json"
    out.write_text(json.dumps(result, indent=2, allow_nan=True), encoding="utf-8")
    print(json.dumps({
        "task": task.id,
        "condition": args.condition,
        "replicate": args.replicate,
        "outcome": result["primary_outcome"],
        "score_file": str(out),
    }, indent=2))
    return 0 if result["primary_outcome"] == "correct" else 2


def _flatten_result(obj: dict) -> dict:
    sci = obj.get("scientific_score", {})
    met = sci.get("metrics", {})
    proc = obj.get("process_score", {})
    tool_ev = proc.get("tool_evidence", {})
    return {
        "task": obj.get("task", {}).get("id"),
        "tool": obj.get("task", {}).get("tool"),
        "condition": obj.get("metadata", {}).get("condition"),
        "replicate": obj.get("metadata", {}).get("replicate"),
        "outcome": obj.get("primary_outcome"),
        "human_interventions": proc.get("human_interventions"),
        "expected_tool_observed": tool_ev.get("expected_tool_observed"),
        "nrms_percent": met.get("nrms_percent"),
        "correlation": met.get("correlation"),
        "mae": met.get("mae"),
        "voxel_agreement": met.get("voxel_agreement"),
        "macro_f1": met.get("macro_f1"),
        "mean_iou": met.get("mean_iou"),
        "cohen_kappa": met.get("cohen_kappa"),
    }


def cmd_summarize(args: argparse.Namespace) -> int:
    run_dir = Path(args.run_dir)
    rows = []
    for path in sorted(run_dir.glob("*.json")):
        try:
            rows.append(_flatten_result(json.loads(path.read_text(encoding="utf-8"))))
        except Exception as exc:
            print(f"warning: skipped {path}: {exc}", file=sys.stderr)
    if not rows:
        print(f"No score files found in {run_dir}", file=sys.stderr)
        return 1

    # Per-condition primary outcome counts.
    conditions = sorted({str(r["condition"]) for r in rows})
    summary = {}
    for cond in conditions:
        subset = [r for r in rows if r["condition"] == cond]
        counts = {k: sum(r["outcome"] == k for r in subset) for k in ("correct", "silent_failure", "overt_failure")}
        n = len(subset)
        summary[cond] = {
            "n": n,
            **counts,
            "success_rate": counts["correct"] / n if n else None,
            "silent_failure_rate": counts["silent_failure"] / n if n else None,
            "overt_failure_rate": counts["overt_failure"] / n if n else None,
            "mean_human_interventions": sum(int(r["human_interventions"] or 0) for r in subset) / n if n else None,
        }

    out_csv = Path(args.csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader(); w.writerows(rows)

    out_json = Path(args.summary_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps({"conditions": summary, "n_runs": len(rows)}, indent=2), encoding="utf-8")
    print(json.dumps({"conditions": summary, "n_runs": len(rows), "csv": str(out_csv), "summary_json": str(out_json)}, indent=2))
    return 0


def cmd_manifest(args: argparse.Namespace) -> int:
    root = repo_root()
    paths = [
        root / "pyproject.toml",
        root / ".github" / "copilot-instructions.md",
        root / ".github" / "mcp.json",
        root / "boglodite_mcp" / "tools.py",
        root / "sandbox" / "FaultSeg" / "predict_only_fault.py",
        root / "sandbox" / "MalenoV" / "predict_only_facies_stable.py",
    ]
    # Add large scientific inputs when they exist locally.
    paths += [
        root / "data" / "Dutch F3 seismic data" / "Dutch Government_F3_entire_8bit seismic.segy",
        root / "models" / "faultSeg_model" / "model" / "fseg-60.hdf5",
        root / "models" / "F3_multiclass_model.h5",
    ]
    manifest = {"files": []}
    for p in paths:
        if p.exists() and p.is_file():
            manifest["files"].append({"path": str(p.relative_to(root)), "sha256": _sha256(p), "size": p.stat().st_size})
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Wrote reproducibility manifest: {out}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="boglodite-eval", description="Boglodite compact research evaluation rubric")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("list", help="List frozen benchmark tasks")
    s.add_argument("--tasks", default=None)
    s.set_defaults(func=cmd_list)

    s = sub.add_parser("freeze-reference", help="Copy canonical numerical outputs into evaluation/references/<TASK>")
    s.add_argument("--task", required=True)
    s.add_argument("--source-dir", default="outputs")
    s.add_argument("--reference-dir", default="evaluation/references")
    s.add_argument("--tasks", default=None)
    s.set_defaults(func=cmd_reference)

    s = sub.add_parser("score", help="Score one agent run against its frozen canonical reference")
    s.add_argument("--task", required=True)
    s.add_argument("--condition", required=True, choices=("bare", "boglodite", "boglodite-no-mcp", "other"))
    s.add_argument("--replicate", type=int, required=True)
    s.add_argument("--candidate-dir", default="outputs")
    s.add_argument("--reference-dir", default="evaluation/references")
    s.add_argument("--run-dir", default="evaluation/runs")
    s.add_argument("--human-interventions", type=int, default=0)
    s.add_argument("--agent-log", default=None)
    s.add_argument("--notes", default="")
    s.add_argument("--thresholds", default=None)
    s.add_argument("--tasks", default=None)
    s.set_defaults(func=cmd_score)

    s = sub.add_parser("summarize", help="Aggregate all scored runs into paper-ready CSV + JSON counts")
    s.add_argument("--run-dir", default="evaluation/runs")
    s.add_argument("--csv", default="evaluation/summary.csv")
    s.add_argument("--summary-json", default="evaluation/summary.json")
    s.set_defaults(func=cmd_summarize)

    s = sub.add_parser("manifest", help="Hash code/data/model inputs used by the experiment")
    s.add_argument("--output", default="evaluation/reproducibility_manifest.json")
    s.set_defaults(func=cmd_manifest)
    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
