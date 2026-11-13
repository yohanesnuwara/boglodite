"""Inline section: true Not 2 top vs one agent horizon. Usage: python plot_inline_simple.py <run_id> [inline]"""
import sys, glob, json, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from norne_benchmark import score as S
from strata.data import load_cube
from strata.tools import horizon as H

RUN = sys.argv[1]; IL = int(sys.argv[2]) if len(sys.argv) > 2 else 1120
metas = sorted(glob.glob(f"runs/{RUN}/store/HorizonPicks/Not2_top/v*/meta.json"))
path = next((m for m in metas if json.load(open(m)).get("status") == "frozen"), metas[-1]).replace("meta.json", "horizon.npz")
score = json.load(open(f"outputs/norne_scores/runs/{RUN}/score.json"))["stages"]["HZ"] if glob.glob(f"outputs/norne_scores/runs/{RUN}/score.json") else None
c = load_cube(); i = IL - int(c.ilines[0]); xl = c.xlines; t = c.t0 + c.dt * np.arange(c.shape[2])
ref = S.Reference().horizon; r = ref[ref.il == IL].sort_values("xl")
y = H.load(path).twt[i]
lo, hi = np.nanmin(r.twt) - 120, np.nanmax(r.twt) + 100; kk = (t >= lo) & (t <= hi)
fig, a = plt.subplots(figsize=(15, 6.5))
v = 2.5 * c.rms
a.imshow(np.asarray(c.data[i]).T[kk], aspect="auto", cmap="gray_r", vmin=-v, vmax=v, extent=[xl[0] - .5, xl[-1] + .5, t[kk][-1], t[kk][0]])
a.plot(r.xl, r.twt, color="#ffd400", lw=3, label="True Not 2 top (reference)")
m = dict(zip(r.xl.astype(int), r.twt)); d = np.array([y[j] - m.get(int(x), np.nan) for j, x in enumerate(xl)])
a.plot(xl, y, color="#ff2d2d", lw=1.6, label=f"Agent horizon (run {RUN}) · MAE on this inline {np.nanmean(np.abs(d)):.1f} ms")
a.set_xlim(r.xl.min() - 40, r.xl.max() + 40); a.set_ylim(hi, lo)
a.set_xlabel("Crossline"); a.set_ylabel("TWT (ms)")
mae = (S.score_horizons(S.Reference(), H.to_table(c, H.load(path))).metrics[0].value)
hz = f" · field-wide time MAE {mae:.1f} ms"
a.set_title(f"Inline {IL} · true horizon vs latest agent result{hz}")
a.legend(loc="lower left", framealpha=.9)
plt.tight_layout(); out = f"outputs/strata_dev/inline{IL}_true_vs_latest.png"; plt.savefig(out, dpi=110); print(out)
