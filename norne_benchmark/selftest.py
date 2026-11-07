"""Self-test of the Norne scorer v2: perturbed copies of TRUTH must score as expected.

Each case is a candidate directory under outputs/norne_scores/selftest/<case>/
built from the reference with one controlled defect. The expected behaviour is
asserted, so a scorer bug (or a TRUTH-reading bug) fails loudly.

    uv run --no-sync python -m norne_benchmark.score --selftest
"""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

import numpy as np

from norne_benchmark import baseline as B
from norne_benchmark import io
from norne_benchmark import score as S

ROOT = S.OUT / "selftest"


def _link(src: Path, dst: Path) -> None:
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    os.symlink(src.resolve(), dst)


def reference_facies(ref: S.Reference) -> np.ndarray:
    """FACIES of the most common realization (R2 = the model shared by 78 of 120), in ATW ordering."""
    cp_raw, kws = io.read_cornerpoint(io.ENS_DIR / "FullNorne_R2.GRDECL")
    _, rev_j = B._canonicalise(cp_raw, ref.grid.cp)
    f = io.to_ijk(kws["FACIES"])
    return f[:, ::-1, :] if rev_j else f


def build_cases(ref: S.Reference) -> dict[str, Path]:
    if ROOT.exists():
        shutil.rmtree(ROOT)
    g = ref.grid
    props = {k: g.props[k] for k in ("PORO", "NTG", "PERMX")}
    props["FACIES"] = np.nan_to_num(reference_facies(ref), nan=0.0)
    raw = io.read_fault_sticks(io.NORNE_ROOT / "FAULTS/kseim_r06_faults_depth")
    faults = raw.rename(columns={"name": "fault"})[["x", "y", "z", "stick", "fault"]]
    h06 = io.read_horizon_points(io.NORNE_ROOT / "horisons/tral+lmig32r06_fxy+Not_2_Fm_Top+TIME_TDQ+2006+input_geomodel")
    h03 = io.read_horizon_points(io.NORNE_ROOT / "horisons/Top_Not_2_2003_Time.txt")
    cases: dict[str, Path] = {}

    def new(name: str) -> Path:
        d = ROOT / name
        d.mkdir(parents=True)
        cases[name] = d
        return d

    d0 = new("reference")
    io.write_grdecl(d0 / "grid.grdecl", g.cp, g.act, props)
    faults.to_csv(d0 / "faults.csv", index=False)
    h06.rename(columns={"z": "twt"})[["il", "xl", "twt"]].to_csv(d0 / "horizon_not2_time.csv", index=False)

    def clone(name, keep=("grid.grdecl", "faults.csv", "horizon_not2_time.csv")):
        d = new(name)
        for f in keep:
            _link(d0 / f, d / f)
        return d

    rest = ("faults.csv", "horizon_not2_time.csv")
    d = clone("shift_10m_deeper", rest)
    c = g.cp.coord.copy(); c[..., [2, 5]] += 10
    io.write_grdecl(d / "grid.grdecl", io.CornerPoint(c, g.cp.zcorn + 10), g.act, props)

    d = clone("y_negated", rest)
    c = g.cp.coord.copy(); c[..., [1, 4]] *= -1
    io.write_grdecl(d / "grid.grdecl", io.CornerPoint(c, g.cp.zcorn), g.act, props)

    d = clone("j_reversed_order", rest)
    io.write_grdecl(d / "grid.grdecl", io.CornerPoint(g.cp.coord[::-1].copy(), g.cp.zcorn[:, :, ::-1, ::-1].copy()),
                    g.act[:, ::-1, :], {k: v[:, ::-1, :] for k, v in props.items()})

    d = clone("poro_minus_10pct", rest)
    io.write_grdecl(d / "grid.grdecl", g.cp, g.act, {**props, "PORO": props["PORO"] * 0.9})

    d = clone("fault_offset_20m", rest)               # columns I ≤ 23 dropped 20 m: a new fault and wrong throws
    z = g.cp.zcorn.copy(); z[..., :23, :] += 20
    io.write_grdecl(d / "grid.grdecl", io.CornerPoint(g.cp.coord, z), g.act, props)

    d = clone("facies_all_one_code", rest)
    io.write_grdecl(d / "grid.grdecl", g.cp, g.act, {**props, "FACIES": np.zeros_like(props["FACIES"])})

    d = clone("drop_10_faults", ("grid.grdecl", "horizon_not2_time.csv"))
    big = ref.fault_points.name.value_counts().index[:10]
    faults[~faults.fault.isin(big)].to_csv(d / "faults.csv", index=False)

    d = clone("horizon_2003_interp", ("grid.grdecl", "faults.csv"))
    h03.rename(columns={"z": "twt"})[["il", "xl", "twt"]].to_csv(d / "horizon_not2_time.csv", index=False)

    d = new("overt_bad_files")
    (d / "grid.grdecl").write_text("SPECGRID\n 2 2 1 1 F /\nCOORD\n0 0 0 0 0 1 /\n")
    faults[["x", "y", "z"]].to_csv(d / "faults.csv", index=False)
    return cases


def _v(res, stage, key):
    return res["stages"][stage]["metrics"][key]["value"]


def checks(r: dict[str, dict]) -> list[tuple[str, bool, str]]:
    out = []

    def chk(name, ok, detail):
        out.append((name, bool(ok), detail))

    ref = r["reference"]
    chk("reference: all 8 stages correct", all(d["outcome"] == "correct" for d in ref["stages"].values()) and len(ref["stages"]) == 8,
        " ".join(f"{s}:{d['outcome'][:7]}" for s, d in ref["stages"].items()))
    chk("reference: blind-well MAE is the truth-vs-truth floor (2–8 m)", 2 <= _v(ref, "STRUCT", "well_mae") <= 8, f"{_v(ref, 'STRUCT', 'well_mae'):.2f} m")
    chk("reference: throw MAE 0, F1 1, compartments identical",
        _v(ref, "STRUCT", "throw_mae") < 1e-6 and _v(ref, "FLT", "fault_f1") == 1 and _v(ref, "CONN", "compartment_ari") > 0.999,
        f"throw {_v(ref, 'STRUCT', 'throw_mae'):.3f} · F1 {_v(ref, 'FLT', 'fault_f1'):.3f} · ARI {_v(ref, 'CONN', 'compartment_ari'):.3f}")

    jr = r["j_reversed_order"]
    chk("J-reversed ordering scores exactly like reference",
        all(abs(jr["stages"][s]["score"] - ref["stages"][s]["score"]) < 1e-3 for s in ref["stages"]),
        " ".join(f"{s}:{jr['stages'][s]['score']:.3f}" for s in jr["stages"]))

    sh = r["shift_10m_deeper"]
    chk("shift 10 m: surface MAE ≈ 10 m", 9.5 < _v(sh, "STRUCT", "surface_mae") < 10.5, f"{_v(sh, 'STRUCT', 'surface_mae'):.2f} m")
    chk("shift 10 m: thickness and throw unchanged", _v(sh, "STRAT", "thickness_mae") < 0.05 and _v(sh, "STRUCT", "throw_mae") < 0.05,
        f"thickness {_v(sh, 'STRAT', 'thickness_mae'):.3f} · throw {_v(sh, 'STRUCT', 'throw_mae'):.3f} m")
    chk("shift 10 m: STOIIP −5…−8 %, fails 5 % tolerance", -8 < _v(sh, "VOL", "stoiip_err") < -5, f"{_v(sh, 'VOL', 'stoiip_err'):+.2f} %")

    yn = r["y_negated"]
    chk("Y negated: structure, petrophysics, volumes, connectivity all silent failures",
        all(yn["stages"][s]["outcome"] == "silent_failure" for s in ("STRUCT", "PETRO", "VOL", "CONN")),
        " ".join(f"{s}:{yn['stages'][s]['outcome']}" for s in ("STRUCT", "PETRO", "VOL", "CONN")))

    po = r["poro_minus_10pct"]
    chk("porosity −10 %: porosity MAE fails, log k untouched",
        _v(po, "PETRO", "poro_mae") > 0.02 and _v(po, "PETRO", "logk_mae") < 1e-6,
        f"poro MAE {_v(po, 'PETRO', 'poro_mae'):.4f} · logk MAE {_v(po, 'PETRO', 'logk_mae'):.4f}")
    chk("porosity −10 %: PV and STOIIP −10 %", abs(_v(po, "VOL", "pv_err") + 10) < 0.1 and abs(_v(po, "VOL", "stoiip_err") + 10) < 0.1,
        f"PV {_v(po, 'VOL', 'pv_err'):+.2f} % · STOIIP {_v(po, 'VOL', 'stoiip_err'):+.2f} %")

    fo = r["fault_offset_20m"]
    chk("20 m offset: throw MAE detects it (> 2 m)", _v(fo, "STRUCT", "throw_mae") > 2, f"{_v(fo, 'STRUCT', 'throw_mae'):.2f} m")
    chk("20 m offset: structural stage not correct", fo["stages"]["STRUCT"]["outcome"] != "correct", fo["stages"]["STRUCT"]["outcome"])

    fa = r["facies_all_one_code"]
    chk("single facies code: facies stage silent failure", fa["stages"]["FAC"]["outcome"] == "silent_failure",
        f"{_v(fa, 'FAC', 'proportion_outside'):.3f} outside P10–P90")

    df = r["drop_10_faults"]
    chk("10 largest faults dropped: recall and point-F1 fall, precision stays 1",
        _v(df, "FLT", "fault_recall") < 0.9 and _v(df, "FLT", "point_f1") < 0.8 and _v(df, "FLT", "fault_precision") == 1,
        f"recall {_v(df, 'FLT', 'fault_recall'):.2f} · point F1 {_v(df, 'FLT', 'point_f1'):.2f} · fault F1 {_v(df, 'FLT', 'fault_f1'):.2f}")

    hz = r["horizon_2003_interp"]
    chk("2003 horizon: partial coverage ⇒ silent failure", hz["stages"]["HZ"]["outcome"] == "silent_failure",
        f"coverage {_v(hz, 'HZ', 'coverage'):.3f}")

    ob = r["overt_bad_files"]
    chk("malformed files: overt failures", all(d["outcome"] == "overt_failure" for d in ob["stages"].values()),
        " ".join(f"{s}:{d['outcome'][:5]}" for s, d in ob["stages"].items()))
    return out


def run() -> bool:
    from norne_benchmark import report
    ref = S.Reference()
    ctx = report.Ctx(ref)
    cases = build_cases(ref)
    results = {}
    for name, d in cases.items():
        res = S.score_candidate(d, ref)
        (d / "score.json").write_text(json.dumps(res, indent=2))
        report.build_report(d, ctx, res)
        results[name] = res
        print(f"{name:22s} {S.summary_line(res)}")
    print()
    rows = checks(results)
    for name, ok, detail in rows:
        print(f"  {'PASS' if ok else 'FAIL'}  {name:70s} {detail}")
    n_ok = sum(ok for _, ok, _ in rows)
    print(f"\n{n_ok}/{len(rows)} checks passed")
    (ROOT / "selftest.json").write_text(json.dumps(
        {"checks": [{"name": n, "ok": ok, "detail": d} for n, ok, d in rows],
         "composites": {k: v["composite"] for k, v in results.items()}}, indent=2))
    print("leaderboard →", report.build_index(ROOT, title="Scorer self-test fixtures — not agent runs"))
    return n_ok == len(rows)


if __name__ == "__main__":
    raise SystemExit(0 if run() else 1)
