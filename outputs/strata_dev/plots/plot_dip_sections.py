"""Truth vs dip-following horizon (+NCS, snapped) vs previous best, on several inlines (reporting only)."""
import sys, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from norne_benchmark import score as S
from strata.data import load_cube
from strata.tools import horizon as H
ILS = [int(a) for a in sys.argv[1:]] or [1224, 1080, 1120, 1160]
c = load_cube(); rh = S.Reference().horizon; i0 = int(c.ilines[0]); t = c.t0 + c.dt * np.arange(c.shape[2])
hs = [(H.load("runs/20261109-193635-2683/store/HorizonPicks/Not2_top/v1/horizon.npz").twt, "#7b2ff7", "Scripted NCS pipeline (14.7 ms)"),
      (H.load("outputs/strata_dev/dip_dip_plus_NCS_level_filled.npz").twt, "#ff2d2d", "Dip-following + NCS, snapped (15.2 ms)")]
fig, axs = plt.subplots(len(ILS), 1, figsize=(15, 4.6 * len(ILS)))
for a, IL in zip(np.atleast_1d(axs), ILS):
    i = IL - i0; r = rh[rh.il == IL].sort_values("xl"); m = dict(zip(r.xl.astype(int), r.twt))
    lo, hi = np.nanmin(r.twt) - 110, np.nanmax(r.twt) + 80; kk = (t >= lo) & (t <= hi); v = 2.5 * c.rms
    a.imshow(np.asarray(c.data[i]).T[kk], aspect="auto", cmap="RdBu", vmin=-v, vmax=v, extent=[c.xlines[0] - .5, c.xlines[-1] + .5, t[kk][-1], t[kk][0]])
    a.plot(r.xl, r.twt, color="k", lw=3, label="True Not 2 top")
    for h, col, lab in hs:
        d = np.array([h[i, j] - m.get(int(x), np.nan) for j, x in enumerate(c.xlines)])
        a.plot(c.xlines, h[i], color=col, lw=1.5, label=f"{lab} · inline MAE {np.nanmean(np.abs(d)):.1f} ms")
    a.set_xlim(r.xl.min() - 40, r.xl.max() + 40); a.set_ylim(hi, lo); a.set_ylabel("TWT (ms)"); a.set_title(f"Inline {IL}")
    a.legend(loc="lower left", fontsize=9, framealpha=.9)
np.atleast_1d(axs)[-1].set_xlabel("Crossline"); plt.tight_layout(); plt.savefig("outputs/strata_dev/dip_sections.png", dpi=90); print("ok")
