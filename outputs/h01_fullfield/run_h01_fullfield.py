"""Run the H01 horizon agent (user's strata-agent-work/03_h01_horizon_agent, unchanged) on every inline.
Inputs changed for an honest whole-cube run:
  - seeds from WELL TIES (never the reference): inlines with a tie within ±2 inlines are seeded from it;
    other inlines from the neighbouring finished inline's confident pick (line-to-line propagation);
  - fault traces: variant A = input-only fault map (discontinuity, vertical traces per section);
                  variant B = tral_r06_faults (FAULTS/ folder = reference interpretation; NOT benchmark-valid).
Outputs per inline: DP from one seed, agent path, agent flag (0 seed block, 1 clean crossing, 2 flagged)."""
import os, sys, glob, json, time, numpy as np, pandas as pd
SRC = os.environ["H01_SRC"]; sys.path.insert(0, SRC)
from hz2d import Line, candidates, track
from agent import Agent, Faults
from strata.data import load_cube

T_FROM, T_TO = 2250.0, 2750.0
RUN = "20261109-200906-a55c"
OUT = "outputs/h01_fullfield"

def section(cube, i):
    t = cube.t0 + cube.dt * np.arange(cube.shape[2]); kk = (t >= T_FROM) & (t <= T_TO)
    A = np.asarray(cube.data[i])[:, kk].T.astype(float)
    return Line(A, t[kk], cube.xlines.astype(int), float(np.sqrt((A ** 2).mean())))

def faults_from_map(L, mask_row):
    out, j, n = {}, 0, len(mask_row)
    while j < n:
        if mask_row[j]:
            s = j
            while j < n and mask_row[j]: j += 1
            x = L.xl[(s + j - 1) // 2] + 0.0
            out[f"map_{int(x)}"] = np.array([[x, T_FROM], [x, T_TO]])
        else:
            j += 1
    return out

_TRAL = {}
def faults_tral(IL):
    import fault_section as FS
    if "T" not in _TRAL:
        T = FS.faults(FS.FAULT_FILE, "tral_"); _TRAL["T"] = {k: v for k, v in T.items() if not k.startswith("DepthTDQ")}
    real = FS.faults
    FS.faults = lambda *a, **k: _TRAL["T"]
    try:
        return FS.section_faults(IL)
    finally:
        FS.faults = real

def run_line(cube, i, seed_xl, seed_t, variant, fmask):
    L = section(cube, i); C = candidates(L, -1)
    secF = faults_from_map(L, fmask[i]) if variant == "A" else faults_tral(int(cube.ilines[i]))
    j0 = int(np.searchsorted(L.xl, seed_xl))
    dp = track(L, C, {j0: float(seed_t)})
    A = Agent(L, C, Faults(L, secF)); p = A.run(int(seed_xl), float(seed_t))
    return dp, p, A.flag.copy(), A.log

def run_all(variant):
    cube = load_cube(); ni, nj = cube.shape[:2]
    fmask = np.load(glob.glob(f"runs/{RUN}/store/FaultMap/reservoir/v*/faults.npz")[-1])["mask"]
    ties = pd.read_csv(glob.glob(f"runs/{RUN}/store/WellTies/*/v*/ties.csv")[-1]); ties = ties[ties.in_cube & ties.twt.notna()]
    DP = np.full((ni, nj), np.nan, np.float32); AG = np.full((ni, nj), np.nan, np.float32); FL = np.full((ni, nj), -1, np.int8)
    seeds, logs = {}, {}
    for i in range(ni):                                    # inlines seeded directly from a well tie (±2 inlines)
        il = int(cube.ilines[i]); near = ties[(ties.il - il).abs() <= 2]
        if len(near):
            r = near.iloc[(near.il - il).abs().argsort()].iloc[0]
            seeds[i] = (int(round(r.xl)), float(r.twt), f"well {r.well}")
    order = sorted(range(ni), key=lambda i: min(abs(i - k) for k in seeds))   # outward from well inlines
    t0 = time.time()
    for n, i in enumerate(order):
        if i not in seeds:                                 # carry the horizon from the finished neighbour
            k = next(k for k in (i - 1, i + 1) if 0 <= k < ni and (FL[k] >= 0).any())
            sxl = seeds[k][0]; ok = np.where(np.isfinite(AG[k]) & (FL[k] < 2))[0]
            if not len(ok):
                ok = np.where(np.isfinite(DP[k]))[0]; src = DP[k]
            else:
                src = AG[k]
            j = ok[np.argmin(np.abs(cube.xlines[ok] - sxl))]
            seeds[i] = (int(cube.xlines[j]), float(src[j]), f"from inline {int(cube.ilines[k])}")
        sxl, st, why = seeds[i]
        import signal
        def _timeout(*_): raise TimeoutError("line exceeded 30 s")
        signal.signal(signal.SIGALRM, _timeout); signal.alarm(30); tl = time.time()
        try:
            dp, p, fl, log = run_line(cube, i, sxl, st, variant, fmask)
        except Exception as e:
            signal.alarm(0)
            logs[int(cube.ilines[i])] = [{"error": f"{type(e).__name__}: {e}"}]
            print(f"  inline {int(cube.ilines[i])}: {type(e).__name__} after {time.time() - tl:.0f} s", flush=True); continue
        signal.alarm(0)
        if time.time() - tl > 5:
            print(f"  inline {int(cube.ilines[i])}: slow, {time.time() - tl:.0f} s", flush=True)
        DP[i], AG[i] = dp, p; FL[i] = np.where(np.isfinite(p), fl, -1)
        logs[int(cube.ilines[i])] = log
        if n % 20 == 0:
            print(f"  {n}/{ni} inlines, {time.time() - t0:.0f} s", flush=True)
    np.savez_compressed(f"{OUT}/h01_{variant}.npz", dp=DP, agent=AG, flag=FL)
    json.dump({"seeds": {int(cube.ilines[i]): s for i, s in seeds.items()}, "logs": logs},
              open(f"{OUT}/h01_{variant}_log.json", "w"), default=float)
    print(f"variant {variant} done in {time.time() - t0:.0f} s; well-seeded inlines {sum('well' in s[2] for s in seeds.values())}")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(); ap.add_argument("--variant", default="A"); ap.add_argument("--test", type=int, default=0)
    a = ap.parse_args()
    if not a.test:
        run_all(a.variant); sys.exit(0)
    cube = load_cube()
    fmask = np.load(glob.glob(f"runs/{RUN}/store/FaultMap/reservoir/v*/faults.npz")[-1])["mask"]
    ties = pd.read_csv(glob.glob(f"runs/{RUN}/store/WellTies/*/v*/ties.csv")[-1]); ties = ties[ties.in_cube & ties.twt.notna()]
    if a.test:
        i = a.test - int(cube.ilines[0]); r = ties.iloc[(ties.il - a.test).abs().argsort()].iloc[0]
        t0 = time.time(); dp, p, fl, log = run_line(cube, i, int(round(r.xl)), float(r.twt), a.variant, fmask)
        print(f"inline {a.test} variant {a.variant}: seed {r.well} xl {int(round(r.xl))} t {r.twt:.1f} | {time.time()-t0:.1f} s | "
              f"agent picks {np.isfinite(p).sum()} (flag2 {int((fl==2).sum())}) | dp picks {np.isfinite(dp).sum()} | review {sum('review' in q for q in log)}")

