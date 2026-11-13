"""Map view of H01 results vs reference (reporting only), layout as v3_vs_ncs_maps.png."""
import sys, json, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from norne_benchmark import score as S
from strata.data import load_cube
from strata.tools import horizon as H
V = sys.argv[1] if len(sys.argv) > 1 else "A"
LAB = {"A": "faults from input fault map (benchmark-valid)", "B": "faults from tral_r06 (reference interpretation)"}[V]
c = load_cube(); rh = S.Reference().horizon; ref = S.Reference(); i0, j0 = int(c.ilines[0]), int(c.xlines[0])
T = np.full(c.shape[:2], np.nan); ii = (rh.il.values - i0).astype(int); jj = (rh.xl.values - j0).astype(int)
ok = (ii >= 0) & (ii < T.shape[0]) & (jj >= 0) & (jj < T.shape[1]); T[ii[ok], jj[ok]] = rh.twt.values[ok]
z = np.load(f"outputs/h01_fullfield/h01_{V}.npz"); fl = z["flag"]
hs = {"DP from one seed": z["dp"], "Agent · confident": np.where(fl < 2, z["agent"], np.nan),
      "Agent · flagged for review": np.where(fl == 2, z["agent"], np.nan), "Agent · all picks": z["agent"]}
inside = np.isfinite(T)
def sc(h):
    m = {x.key: x.value for x in S.score_horizons(ref, H.to_table(c, H.Horizon(h.astype(np.float32), np.ones_like(h, np.float32)))).metrics}
    e = (h - T)[inside]; f = np.isfinite(e)
    return {"scorer_mae": round(m["time_mae"], 2) if m["time_mae"] is not None else None, "coverage": round(m["coverage"], 3),
            "map_mae": round(float(np.mean(np.abs(e[f]))), 1) if f.any() else None,
            "within15": round(float(np.mean(np.abs(e[f]) <= 15)), 3) if f.any() else None, "picked_of_ref": round(float(f.mean()), 3)}
stats = {k: sc(h) for k, h in hs.items()}
for k, v in stats.items(): print(V, k, v)
json.dump(stats, open(f"outputs/h01_fullfield/h01_{V}_scores.json", "w"), indent=1)
rows = np.where(inside.any(1))[0]; cols = np.where(inside.any(0))[0]
r0, r1 = max(rows.min() - 5, 0), min(rows.max() + 5, T.shape[0]); c0, c1 = max(cols.min() - 5, 0), min(cols.max() + 5, T.shape[1])
ext = [j0 + c0, j0 + c1, i0 + r1, i0 + r0]; S_ = np.s_[r0:r1, c0:c1]
fig, ax = plt.subplots(2, 4, figsize=(25, 13), constrained_layout=True)
vmin, vmax = np.nanpercentile(T, 1), np.nanpercentile(T, 99)
im = ax[0][0].imshow(T[S_], cmap="viridis_r", vmin=vmin, vmax=vmax, extent=ext, aspect="auto", interpolation="nearest")
ax[0][0].set_title("True Not 2 top (reference)")
for a, k in zip(ax[0][1:], ("DP from one seed", "Agent · confident", "Agent · flagged for review")):
    a.imshow(np.where(inside, hs[k], np.nan)[S_], cmap="viridis_r", vmin=vmin, vmax=vmax, extent=ext, aspect="auto", interpolation="nearest")
    s = stats[k]; a.set_title(f"{k}\nMAE {s['map_mae']} ms · ±15 ms: {s['within15']:.0%} · picked {s['picked_of_ref']:.0%} of area")
fig.colorbar(im, ax=ax[0], shrink=.8, label="TWT (ms)")
for a, k in zip(ax[1], ("DP from one seed", "Agent · confident", "Agent · flagged for review", "Agent · all picks")):
    e = np.where(inside, hs[k] - T, np.nan)
    im2 = a.imshow(e[S_], cmap="RdBu_r", vmin=-60, vmax=60, extent=ext, aspect="auto", interpolation="nearest")
    gap = inside & ~np.isfinite(hs[k])
    a.imshow(np.where(gap, 1.0, np.nan)[S_], cmap="Greys", vmin=0, vmax=1.5, extent=ext, aspect="auto", interpolation="nearest")
    s = stats[k]; a.set_title(f"{k}: agent − truth (grey = unpicked)\nscorer MAE {s['scorer_mae']} ms · coverage {s['coverage']:.0%}")
fig.colorbar(im2, ax=ax[1], shrink=.8, label="agent − truth (ms)")
for a in ax.ravel(): a.set_xlabel("Crossline"); a.set_ylabel("Inline")
fig.suptitle(f"H01 horizon agent on every inline — seeds from well ties, {LAB}", fontsize=14)
plt.savefig(f"outputs/h01_fullfield/h01_{V}_maps.png", dpi=85); print("saved")
