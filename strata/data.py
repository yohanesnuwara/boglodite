"""INPUT data access for agents: the 2001 seismic cube and the non-blind wells.

Everything here reads INPUT-tier files only (strata.config.INPUT_DIRS) and drops the
blind wellbores. The seismic time window is cached locally as a .npy (derived
INPUT; nothing is ever written to the data drive).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from functools import cached_property, lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from norne_benchmark import io as nio
from strata.config import CACHE_DIR, SEISMIC_FILE, blind_wells, input_path

T_MIN_MS, T_MAX_MS = 1700.0, 3300.0        # reservoir-centred window kept in memory


# ── seismic ──────────────────────────────────────────────────────────────────
@dataclass
class Cube:
    data: np.ndarray            # (n_il, n_xl, n_t) float32
    ilines: np.ndarray
    xlines: np.ndarray
    t0: float                   # ms of sample 0 in the window
    dt: float                   # ms
    affine: np.ndarray          # (3, 2): [il, xl, 1] @ affine = [x, y]
    rms: float

    @property
    def shape(self):
        return self.data.shape

    def t_axis(self) -> np.ndarray:
        return self.t0 + self.dt * np.arange(self.data.shape[2])

    def ilxl_to_xy(self, il, xl) -> np.ndarray:
        il, xl = np.atleast_1d(il).astype(float), np.atleast_1d(xl).astype(float)
        return np.c_[il, xl, np.ones_like(il)] @ self.affine

    def xy_to_ilxl(self, x, y) -> np.ndarray:
        A = np.r_[self.affine[:2].T].reshape(2, 2)          # [x,y] = A @ [il,xl] + b
        b = self.affine[2]
        return np.linalg.solve(A, (np.c_[np.atleast_1d(x), np.atleast_1d(y)] - b).T).T

    def index(self, il: float, xl: float) -> tuple[int, int]:
        return int(round(il - self.ilines[0])), int(round(xl - self.xlines[0]))

    def inside(self, il, xl) -> np.ndarray:
        il, xl = np.asarray(il), np.asarray(xl)
        return (il >= self.ilines[0]) & (il <= self.ilines[-1]) & (xl >= self.xlines[0]) & (xl <= self.xlines[-1])


def _read_cube(path: Path) -> Cube:
    import segyio
    with segyio.open(path, "r") as f:
        ilines, xlines = np.array(f.ilines), np.array(f.xlines)
        samples = np.array(f.samples, float)
        dt = float(samples[1] - samples[0])
        k0, k1 = np.searchsorted(samples, T_MIN_MS), np.searchsorted(samples, T_MAX_MS, side="right")
        data = np.empty((len(ilines), len(xlines), k1 - k0), np.float32)
        for n, il in enumerate(ilines):
            data[n] = f.iline[il][:, k0:k1]
        F = segyio.TraceField
        rows = []
        n_tr = f.tracecount
        for i in sorted(set(range(0, n_tr, 997)) | {0, n_tr - 1}):
            h = f.header[i]
            sc = h[F.SourceGroupScalar]
            s = (1 / abs(sc)) if sc < 0 else (sc or 1)
            rows.append((h[F.INLINE_3D], h[F.CROSSLINE_3D], h[F.CDP_X] * s, h[F.CDP_Y] * s))
    g = np.array(rows, float)
    affine, *_ = np.linalg.lstsq(np.c_[g[:, :2], np.ones(len(g))], g[:, 2:], rcond=None)
    return Cube(data, ilines, xlines, float(samples[k0]), dt, affine, float(np.sqrt(np.mean(data[::4, ::4] ** 2))))


@lru_cache(maxsize=1)
def load_cube() -> Cube:
    """The 2001 cube's [1700, 3300] ms window, cached locally after the first read (~1–3 min)."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    npy, meta = CACHE_DIR / "norne2001_window.npy", CACHE_DIR / "norne2001_window.npz"
    if npy.exists() and meta.exists():
        m = np.load(meta)
        return Cube(np.load(npy), m["ilines"], m["xlines"], float(m["t0"]), float(m["dt"]),
                    m["affine"], float(m["rms"]))
    c = _read_cube(input_path(SEISMIC_FILE))
    np.save(npy, c.data)
    np.savez(meta, ilines=c.ilines, xlines=c.xlines, t0=c.t0, dt=c.dt, affine=c.affine, rms=c.rms)
    return Cube(np.load(npy), c.ilines, c.xlines, c.t0, c.dt, c.affine, c.rms)


# ── wells ────────────────────────────────────────────────────────────────────
def _norm_td_name(path: Path) -> str:
    m = re.match(r"norne-6608(\d\d)-(.+?)-td", path.name)
    return nio.norm_well(f"6608/{m.group(1)}-{m.group(2).replace('_', '')}") if m else path.stem


def read_td_tables() -> dict[str, list[dict]]:
    """{well key: [{source, twt (ms), depth (m)}, …]} from the OpenWorks time–depth exports.

    NOTE: the files say "Depth Mode: MD", but the depth column is TVD below the kelly
    bushing — verified 2026-11-08: read as TVD the tables agree to a median 6.9 ms at
    reservoir depth; read as MD through the deviated paths they disagree by a median 60 ms
    (up to 670 ms). Convert with TVDSS = depth − KB."""
    out: dict[str, list[dict]] = {}
    d = input_path("wells/time-depth")
    for f in sorted(d.glob("*-td.asc")):
        key = _norm_td_name(f)
        tables, cur = [], None
        for line in f.read_text(errors="replace").splitlines():
            tag = line[:4]
            if tag == "TDP1":
                cur = {"source": "", "pts": []}
                tables.append(cur)
            elif tag == "TDP3" and cur is not None:
                cur["source"] = line[4:].split()[0] if len(line[4:].split()) else ""
            elif tag == "TDP5" and cur is not None:
                v = line[4:].split()
                if len(v) >= 2:
                    cur["pts"].append((float(v[0]), float(v[1])))
        for t in tables:
            p = np.array(t["pts"])
            if len(p) >= 2 and np.all(np.diff(p[:, 1]) > 0):
                out.setdefault(key, []).append({"source": t["source"] or "unnamed", "twt": p[:, 0], "depth": p[:, 1]})
    return out


TD_PREFERENCE = ["td_vsp", "TIME-CKS", "td_par", "test:td_par", "td_par_prop", "test:td_par_prop", "tdpar_prop",
                 "bosd_TDmodel", "bosd_3pairs_td_par", "BOSD", "Sfinx", "unnamed", "Ricker_25Hz_Synthetic",
                 "RIcker_25Hz_Synthetic"]


class Wells:
    """Non-blind Norne wells: paths, formation tops (both pick generations), time–depth tables."""

    def __init__(self):
        self.blind = blind_wells()

    @cached_property
    def paths(self) -> dict[str, pd.DataFrame]:
        return {k: v for k, v in nio.read_position_logs().items() if k not in self.blind}

    @cached_property
    def picks(self) -> pd.DataFrame:
        p = nio.read_picks()
        return p[~p.wkey.isin(self.blind)].reset_index(drop=True)

    @cached_property
    def td(self) -> dict[str, list[dict]]:
        return {k: v for k, v in read_td_tables().items() if k not in self.blind}

    def marker_summary(self, pattern: str = "") -> pd.DataFrame:
        p = self.picks
        if pattern:
            p = p[p.marker.str.contains(pattern, case=False, regex=False)]
        g = p.groupby("marker").agg(wells=("wkey", "nunique"), tvdss_median=("tvdss", "median"),
                                    tvdss_min=("tvdss", "min"), tvdss_max=("tvdss", "max"))
        return g.sort_values("tvdss_median").reset_index()

    def kb(self, key: str) -> float:
        """Kelly-bushing elevation above sea level (m) from the deviation log's first row."""
        p = self.paths.get(key)
        return float(abs(p.tvdss.iloc[0])) if p is not None and p.md.iloc[0] == 0 else 22.0

    @cached_property
    def td_tvdss(self) -> list[dict]:
        """Every TD table as TWT vs TVDSS (depth column is TVD below KB → TVDSS = depth − KB),
        with the well location at the table's deepest point."""
        out = []
        for key, tables in self.td.items():
            path = self.paths.get(key)
            if path is None:
                continue
            kb = self.kb(key)
            for t in tables:
                z = t["depth"] - kb
                zmax = float(z[-1])
                i = int(np.argmin(np.abs(path.tvdss.values - zmax)))
                out.append({"well": key, "source": t["source"], "twt": t["twt"], "tvdss": z,
                            "x": float(path.x.iloc[i]), "y": float(path.y.iloc[i])})
        return out

    def td_table_qc(self, ref_tvdss: float = 2550.0, outlier_ms: float = 40.0) -> pd.DataFrame:
        """Each TD table compared with all other tables at a common depth: its own deepest
        point, capped at ref_tvdss (TVDSS). deviation = this table's TWT − median TWT of the
        other tables reaching that depth. Tables more than outlier_ms off are flagged — a
        time shift or a datum error in that table."""
        T = self.td_tvdss
        rows = []
        for n, t in enumerate(T):
            z = min(ref_tvdss, float(t["tvdss"][-1]))
            if z < 1500:
                rows.append({"table": f"{t['well']}:{t['source']}", "well": t["well"], "source": t["source"],
                             "eval_tvdss": round(z, 1), "twt": np.nan, "others_median": np.nan, "n_others": 0})
                continue
            mine = float(np.interp(z, t["tvdss"], t["twt"]))
            others = [float(np.interp(z, o["tvdss"], o["twt"])) for m, o in enumerate(T)
                      if m != n and o["well"] != t["well"] and o["tvdss"][0] <= z <= o["tvdss"][-1]]
            rows.append({"table": f"{t['well']}:{t['source']}", "well": t["well"], "source": t["source"],
                         "eval_tvdss": round(z, 1), "twt": mine,
                         "others_median": float(np.median(others)) if others else np.nan, "n_others": len(others)})
        df = pd.DataFrame(rows)
        df["deviation_ms"] = df.twt - df.others_median
        df["outlier"] = df.deviation_ms.abs() > outlier_ms
        return df.sort_values("deviation_ms", key=lambda c: -c.abs()).reset_index(drop=True)

    def tie_marker(self, marker: str, cube: Cube, preference: list[str] | None = None,
                   neighbours: int = 3, max_neighbour_km: float = 4.0,
                   exclude_sources: tuple[str, ...] = ("Ricker_25Hz_Synthetic", "RIcker_25Hz_Synthetic"),
                   exclude_tables: tuple[str, ...] = ()) -> pd.DataFrame:
        """Time of `marker` at every non-blind well. Method 'own': TVDSS → TWT with the well's own
        TD tables (all tables covering the pick; preferred source chosen). Method 'neighbour':
        if no own table covers the pick, TVDSS → TWT with the nearest wells' tables (TWT–TVDSS
        form) within max_neighbour_km; the nearest is used, the spread over up to `neighbours`
        wells is the uncertainty. Synthetic-seismogram tables are excluded by default: on
        Norne they are time-shifted by ~100–190 ms relative to checkshot/VSP-based tables."""
        pref = preference or TD_PREFERENCE
        rank = {s: i for i, s in enumerate(pref)}
        rows = []
        for r in self.picks[self.picks.marker == marker].itertuples():
            path = self.paths.get(r.wkey)
            if path is None or not np.isfinite(r.md):
                continue
            x, y = np.interp(r.md, path.md, path.x), np.interp(r.md, path.md, path.y)
            il, xl = cube.xy_to_ilxl(x, y)[0]
            times, method = {}, "own"
            for t in (t for t in self.td_tvdss if t["well"] == r.wkey):
                if (t["source"] not in exclude_sources and f"{r.wkey}:{t['source']}" not in exclude_tables
                        and t["tvdss"][0] <= r.tvdss <= t["tvdss"][-1]):
                    times.setdefault(t["source"], float(np.interp(r.tvdss, t["tvdss"], t["twt"])))
            if not times:
                method = "neighbour"
                best_per_well: dict[str, tuple[int, float, dict]] = {}
                for t in self.td_tvdss:
                    if (t["well"] == r.wkey or t["source"] in exclude_sources
                            or f"{t['well']}:{t['source']}" in exclude_tables
                            or not (t["tvdss"][0] <= r.tvdss <= t["tvdss"][-1])):
                        continue
                    d = float(np.hypot(t["x"] - x, t["y"] - y))
                    rk = rank.get(t["source"], 99)
                    if d <= max_neighbour_km * 1000 and (t["well"] not in best_per_well or rk < best_per_well[t["well"]][0]):
                        best_per_well[t["well"]] = (rk, d, t)
                for _, d, t in sorted(best_per_well.values(), key=lambda v: v[1])[:neighbours]:
                    times[f"{t['well']}:{t['source']}@{d / 1000:.1f}km"] = float(np.interp(r.tvdss, t["tvdss"], t["twt"]))
            if method == "own":
                chosen = next((s for s in pref if s in times), next(iter(times), None))
            else:
                chosen = next(iter(times), None)
            vals = np.array(list(times.values()))
            rows.append({"well": r.wkey, "md": r.md, "tvdss": r.tvdss, "x": x, "y": y, "il": il, "xl": xl,
                         "in_cube": bool(cube.inside(il, xl)), "method": method if times else "none",
                         "td_source": chosen, "twt": times.get(chosen, np.nan) if chosen else np.nan,
                         "n_tables": len(times), "table_spread_ms": float(np.ptp(vals)) if len(vals) > 1 else 0.0,
                         "all_tables": times})
        return pd.DataFrame(rows)
