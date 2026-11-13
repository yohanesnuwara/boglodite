"""Inline section: reference Not 2 top (truth, plotting only) vs agent horizons. Usage: python plot_inline_true_vs_agents.py [inline]"""
import sys, glob, json
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from norne_benchmark import score as S
from strata.data import load_cube
from strata.tools import horizon as H

IL = int(sys.argv[1]) if len(sys.argv) > 1 else 1120
RUNS = [("20261108-175254-abd5", "Sonnet 5.5 · old tools (39.0 ms)", "#e8743b"),
        ("20261108-191746-623a", "Scripted · new tools (20.6 ms)", "#3ba7e8"),
        ("20261108-193053-2632", "Opus 5.5 · new tools (21.3 ms)", "#2fd17a"),
        ("RETRACK:outputs/strata_dev/retracked_20261108-191746-623a_v3.npz", "Scripted + deviation repair (19.4 ms)", "#ff2d2d")]
SEEDS = "outputs/strata_dev/retrack_seeds_20261108-191746-623a_v3.json"

def frozen(run):
    if run.startswith("RETRACK:"):
        return run.split(":", 1)[1]
    for p in sorted(glob.glob(f"runs/{run}/store/HorizonPicks/Not2_top/v*/meta.json")):
        if json.load(open(p)).get("status") == "frozen":
            return p.replace("meta.json", "horizon.npz")
    return sorted(glob.glob(f"runs/{run}/store/HorizonPicks/Not2_top/v*/horizon.npz"))[-1]

c = load_cube(); i = IL - int(c.ilines[0]); xl = c.xlines
sec = np.asarray(c.data[i]).T
ref = S.Reference().horizon; r = ref[ref.il == IL].sort_values("xl")
fm = np.load("runs/20261108-193053-2632/store/FaultMap/reservoir/v1/faults.npz")["mask"][i]
t = c.t0 + c.dt * np.arange(c.shape[2])
lo, hi = np.nanmin(r.twt) - 150, np.nanmax(r.twt) + 120
kk = (t >= lo) & (t <= hi)

fig, (a, b) = plt.subplots(2, 1, figsize=(15, 9.5), gridspec_kw={"height_ratios": [3, 1.2]}, sharex=True)
v = 2.5 * c.rms
a.imshow(sec[kk], aspect="auto", cmap="gray_r", vmin=-v, vmax=v,
         extent=[xl[0] - .5, xl[-1] + .5, t[kk][-1], t[kk][0]])
for x in xl[fm]:
    a.axvline(x, color="#c03ad8", alpha=.07, lw=1)
a.plot(r.xl, r.twt, color="#ffd400", lw=3.2, label="Reference Not 2 top (truth)", zorder=5)
mref = dict(zip(r.xl.astype(int), r.twt))
for run, lab, col in RUNS:
    hz = H.load(frozen(run)); y = hz.twt[i]
    a.plot(xl, y, color=col, lw=1.6, label=lab, zorder=6)
    d = np.array([y[j] - mref.get(int(x), np.nan) for j, x in enumerate(xl)])
    ok = np.isfinite(d)
    b.plot(xl, d, color=col, lw=1.3, label=f"{lab.split(' (')[0]} · MAE on this inline {np.nanmean(np.abs(d)):.1f} ms")
sd = [p for p in json.load(open(SEEDS)) if abs(p["il"] - IL) <= 2]
if sd:
    a.scatter([p["xl"] for p in sd], [p["twt"] for p in sd], s=55, c="#ff2d2d", edgecolors="white", linewidths=1.2,
              zorder=8, label=f"repair seeds (inline/crossline consensus, IL {IL}±2)")
a.set_ylabel("TWT (ms)"); a.set_ylim(hi, lo)
a.set_title(f"Inline {IL} · Not 2 Fm top: reference vs agent horizons (purple bands = fault map)")
a.legend(loc="lower left", fontsize=9, framealpha=.85)
b.axhline(0, color="k", lw=.8); b.axhspan(-15, 15, color="#2fd17a", alpha=.12, label="±15 ms tolerance")
b.set_ylabel("agent − truth (ms)"); b.set_xlabel("Crossline"); b.set_ylim(-120, 120)
b.legend(loc="lower left", fontsize=8, ncol=2)
a.set_xlim(r.xl.min() - 40, r.xl.max() + 40)
plt.tight_layout()
out = f"outputs/strata_dev/inline{IL}_true_vs_agents_repair.png"
plt.savefig(out, dpi=110); print(out)
