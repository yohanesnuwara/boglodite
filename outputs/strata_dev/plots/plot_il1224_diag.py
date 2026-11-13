"""Inline 1224 diagnosis: well-guide surface ±35 ms window vs reflectors, agent picks, truth (reporting only)."""
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from norne_benchmark import score as S
from strata.data import load_cube
from strata.tools import horizon as H
IL = 1224
c = load_cube(); i = IL - int(c.ilines[0]); t = c.t0 + c.dt * np.arange(c.shape[2])
ties = pd.read_csv("runs/20261109-200906-a55c/store/WellTies/Not_2_NO_2006_Top/v1/ties.csv"); ties = ties[ties.in_cube]
g = H.guide_surface(c, [(r.il, r.xl, r.twt - 1.28) for r in ties.itertuples()])[i]
rh = S.Reference().horizon; r = rh[rh.il == IL].sort_values("xl")
v3 = H.load("runs/20261109-200906-a55c/store/HorizonPicks/Not2_top/v3/horizon.npz").twt[i]
ncs = H.load("runs/20261109-193635-2683/store/HorizonPicks/Not2_top/v1/horizon.npz").twt[i]
fig, a = plt.subplots(figsize=(16, 7)); kk = (t >= 2250) & (t <= 2650); v = 2.5 * c.rms
a.imshow(np.asarray(c.data[i]).T[kk], aspect="auto", cmap="RdBu", vmin=-v, vmax=v, extent=[c.xlines[0], c.xlines[-1], 2650, 2250])
a.fill_between(c.xlines, g - 35, g + 35, color="#ffaa00", alpha=.18, label="tracking window: well-interpolated guide ±35 ms")
a.plot(c.xlines, g, color="#ff8800", lw=1.5, ls="--", label="well-interpolated guide surface")
a.plot(c.xlines, v3, color="#00a651", lw=1.3, label="Opus v3 (greedy)")
a.plot(c.xlines, ncs, color="#7b2ff7", lw=1.3, label="Scripted NCS pipeline")
a.plot(r.xl, r.twt, color="k", lw=2.6, label="True Not 2 top")
d = ties.assign(dist=np.hypot(ties.il - IL, ties.xl - 0)); 
a.set_xlim(1700, 2200); a.set_ylim(2650, 2250); a.set_xlabel("Crossline"); a.set_ylabel("TWT (ms)")
a.set_title(f"Inline {IL}: the guide comes from wells far away (nearest tied well {np.min(np.abs(ties.il - IL)):.0f} inlines away)")
a.legend(loc="lower left", fontsize=8, framealpha=.9); plt.tight_layout()
plt.savefig("outputs/strata_dev/il1224_diagnosis.png", dpi=100); print("ok")
