"""Inline: seismic + NCS P(below Not 2 top) + FM boundary + agent horizon (input-only view; no truth)."""
import sys, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from strata.data import load_cube
from strata.config import CACHE_DIR
from strata.tools import horizon as H
ILS = [int(a) for a in sys.argv[1:]] or [1080, 1120]
c = load_cube(); z = np.load(CACHE_DIR / "ncs3d_not2_prob.npz")
hz = H.load("runs/20261108-201137-9b63/store/HorizonPicks/Not2_top/v3/horizon.npz").twt
tn = c.t0 + c.dt * z["t_sample"]; xs = c.xlines[z["xl_sample"]]; t = c.t0 + c.dt * np.arange(c.shape[2])
fig, axs = plt.subplots(len(ILS), 1, figsize=(15, 5.2 * len(ILS)))
for a, IL in zip(np.atleast_1d(axs), ILS):
    i = IL - int(c.ilines[0]); gi = int(np.argmin(np.abs(z["il_sample"] - i)))
    kk = (t >= 2250) & (t <= 2700); v = 2.5 * c.rms
    a.imshow(np.asarray(c.data[i]).T[kk], aspect="auto", cmap="gray_r", vmin=-v, vmax=v, extent=[c.xlines[0], c.xlines[-1], 2700, 2250])
    im = a.imshow(z["P"][gi].T, aspect="auto", cmap="coolwarm", vmin=0, vmax=1, alpha=0.35, interpolation="bilinear",
                  extent=[xs[0], xs[-1], tn[-1], tn[0]])
    a.plot(c.xlines, z["surface"][i], color="#ffd400", lw=2.2, label="NCS package boundary (P=0.5 step)")
    a.plot(c.xlines, hz[i], color="#00e0ff", lw=1.4, label="Opus agent horizon (18.8 ms run)")
    a.set_ylim(2700, 2250); a.set_xlim(1400, 2150); a.set_title(f"Inline {IL} · NCS-3D P(below Not 2 top) — red = below, blue = above")
    a.set_ylabel("TWT (ms)"); a.legend(loc="lower left", fontsize=8)
plt.colorbar(im, ax=axs, shrink=.6, label="P(below)"); np.atleast_1d(axs)[-1].set_xlabel("Crossline")
out = "outputs/strata_dev/ncs_prob_sections.png"; plt.savefig(out, dpi=100, bbox_inches="tight"); print(out)
