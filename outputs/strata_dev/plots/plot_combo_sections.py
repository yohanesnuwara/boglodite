"""Truth vs NCS level check vs NCS + deviation repair on three inlines (reporting only)."""
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from norne_benchmark import score as S
from strata.data import load_cube
from strata.tools import horizon as H
RUN = "20261108-201137-9b63"
c = load_cube(); rh = S.Reference().horizon; i0 = int(c.ilines[0]); t = c.t0 + c.dt * np.arange(c.shape[2])
hs = [(H.load(f"runs/{RUN}/store/HorizonPicks/Not2_top/v3/horizon.npz").twt, "#00b7ff", "Opus + repair (18.8 ms)"),
      (H.load(f"outputs/strata_dev/ncs_retracked_{RUN}_v3.npz").twt, "#ff9f1c", "+ NCS level check (15.7 ms)"),
      (H.load(f"outputs/strata_dev/ncs_then_repair_{RUN}_v3.npz").twt, "#ff2d2d", "+ NCS + repair (14.5 ms)")]
fig, axs = plt.subplots(3, 1, figsize=(15, 15))
for a, IL in zip(axs, (1080, 1120, 1160)):
    i = IL - i0; r = rh[rh.il == IL].sort_values("xl"); m = dict(zip(r.xl.astype(int), r.twt))
    lo, hi = np.nanmin(r.twt) - 120, np.nanmax(r.twt) + 80; kk = (t >= lo) & (t <= hi); v = 2.5 * c.rms
    a.imshow(np.asarray(c.data[i]).T[kk], aspect="auto", cmap="gray_r", vmin=-v, vmax=v, extent=[c.xlines[0] - .5, c.xlines[-1] + .5, t[kk][-1], t[kk][0]])
    a.plot(r.xl, r.twt, color="#ffd400", lw=3.2, label="True Not 2 top")
    for h, col, lab in hs:
        d = np.array([h[i, j] - m.get(int(x), np.nan) for j, x in enumerate(c.xlines)])
        a.plot(c.xlines, h[i], color=col, lw=1.4, label=f"{lab} · inline MAE {np.nanmean(np.abs(d)):.1f} ms")
    a.set_xlim(r.xl.min() - 40, r.xl.max() + 40); a.set_ylim(hi, lo); a.set_ylabel("TWT (ms)"); a.set_title(f"Inline {IL}")
    a.legend(loc="lower left", fontsize=9, framealpha=.9)
axs[-1].set_xlabel("Crossline"); plt.tight_layout(); plt.savefig("outputs/strata_dev/ncs_repair_sections.png", dpi=100); print("ok")
