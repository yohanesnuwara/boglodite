"""
Side-by-side comparison of FaultSeg fault probability and MalenoV facies
classification on inline 130 of the Dutch F3 dataset.

Loads pre-computed results from:
  outputs/F3_fault_inline130.npy   — (462, 384)  time × xlines 300-683
  outputs/F3_multi_class.npy       — (1, 891, 402) inline × xlines 330-1220 × time 124-1728ms

Plots three panels:
  1. Seismic amplitude (inline 130, full xline/time range)
  2. Fault probability overlay  (FaultSeg, xlines 300-683)
  3. Facies class map overlay    (MalenoV,  xlines 330-1220)

Output: outputs/F3_fault_vs_facies_inline130.png
"""

import os
import numpy as np
import segyio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap

# ── Paths ──────────────────────────────────────────────────────────────────────
SEGY_PATH   = "/workspace/boglodite/data/Dutch Government_F3_entire_8bit seismic.segy"
FAULT_PATH  = "/workspace/boglodite/outputs/F3_fault_inline130.npy"
FACIES_PATH = "/workspace/boglodite/outputs/F3_multi_class.npy"
OUT_PATH    = "/workspace/boglodite/outputs/F3_fault_vs_facies_inline130.png"

INLINE_TARGET = 130

# Coordinate extents (from each script's parameters)
# FaultSeg used inlines 100-227, xlines 300-683, all 462 time samples
FAULT_XL_START, FAULT_XL_END = 300, 683
FAULT_T_START,  FAULT_T_END  = 4.0, 1848.0

# MalenoV predicted xlines 330-1220, time 124-1728ms (after cube_incr=30 margin)
FACIES_XL_START, FACIES_XL_END = 330, 1220
FACIES_T_START,  FACIES_T_END  = 124.0, 1728.0

FACIES_NAMES = [
    "Else", "Grizzly", "High Amp Cont",
    "High Amp", "Low Amp Dips", "Low Amp",
    "Low Coherency", "Salt", "Steep Dips",
]
FACIES_COLORS = [
    "#aaaaaa", "#1f77b4", "#2ca02c",
    "#d62728", "#ff7f0e", "#9467bd",
    "#8c564b", "#e377c2", "#bcbd22",
]

# ── Load seismic inline 130 ────────────────────────────────────────────────────
print("Loading seismic inline 130 from SEGY ...")
with segyio.open(SEGY_PATH, "r", strict=False) as f:
    f.mmap()
    il_start = int(f.ilines[0])
    il_idx   = INLINE_TARGET - il_start        # 0-based: inline 130 → index 30
    seis     = f.iline[f.ilines[il_idx]]       # (n_xl, n_z)
seis = seis.astype(np.float32).T               # → (n_z, n_xl) = (462, 951)
print(f"  Seismic slice shape (time, xlines): {seis.shape}")

# ── Load pre-computed results ──────────────────────────────────────────────────
fault  = np.load(FAULT_PATH)                   # (462, 384)
facies = np.load(FACIES_PATH)[0]               # (891, 402) → xlines × time
facies = facies.T                              # → (402, 891) = time × xlines
print(f"  Fault  shape: {fault.shape}")
print(f"  Facies shape: {facies.shape}")

# ── Build RGBA overlay helpers ─────────────────────────────────────────────────
def fault_rgba(fp, cmap_name="hot_r"):
    cmap = plt.get_cmap(cmap_name)
    rgba = cmap(fp).astype(np.float32)
    rgba[..., 3] = fp                          # alpha = probability
    return rgba


def facies_rgba(cls_map, alpha=0.65):
    cmap  = ListedColormap(FACIES_COLORS)
    norm  = BoundaryNorm(np.arange(-0.5, 9), ncolors=9)
    mapper = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    rgba  = mapper.to_rgba(cls_map).astype(np.float32)
    rgba[..., 3] = alpha
    return rgba

# ── Plot ───────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(21, 8), sharey=False)
fig.suptitle(
    f"Dutch F3 — Inline {INLINE_TARGET}   |   Fault (FaultSeg) vs Facies (MalenoV)",
    fontsize=14,
)

# Full seismic extent for reference panel
full_extent = [300, 1250, 1848.0, 4.0]

# --- Panel 1: Seismic amplitude ---
ax = axes[0]
ax.imshow(seis, aspect="auto", cmap="gray", extent=full_extent)
ax.set_title("Seismic Amplitude", fontsize=12)
ax.set_xlabel("Xline")
ax.set_ylabel("Time (ms)")

# --- Panel 2: Fault probability overlay ---
ax = axes[1]
fault_extent = [FAULT_XL_START, FAULT_XL_END, FAULT_T_END, FAULT_T_START]
# Crop seismic to fault xline window (xlines 300-683 → indices 0-383)
seis_fault = seis[:, :384]
ax.imshow(seis_fault, aspect="auto", cmap="gray", extent=fault_extent)
ax.imshow(fault_rgba(fault), aspect="auto", extent=fault_extent)
sm = plt.cm.ScalarMappable(cmap="hot_r", norm=plt.Normalize(0, 1))
sm.set_array([])
plt.colorbar(sm, ax=ax, label="Fault probability", shrink=0.7)
ax.set_title("Fault Probability (FaultSeg)", fontsize=12)
ax.set_xlabel("Xline")
ax.set_ylabel("Time (ms)")

# --- Panel 3: Facies class overlay ---
ax = axes[2]
facies_extent = [FACIES_XL_START, FACIES_XL_END, FACIES_T_END, FACIES_T_START]
# Crop seismic to facies xline window (xlines 330-1220 → indices 30-920)
seis_facies = seis[30:432, 30:921]   # time 124-1728ms (indices 30..431), xlines 330-1220 (indices 30..920)
ax.imshow(seis_facies, aspect="auto", cmap="gray", extent=facies_extent)
ax.imshow(facies_rgba(facies), aspect="auto", extent=facies_extent)
cmap9  = ListedColormap(FACIES_COLORS)
norm9  = BoundaryNorm(np.arange(-0.5, 9), ncolors=9)
sm9    = plt.cm.ScalarMappable(cmap=cmap9, norm=norm9)
sm9.set_array([])
cbar   = plt.colorbar(sm9, ax=ax, shrink=0.7)
cbar.set_ticks(range(9))
cbar.set_ticklabels(FACIES_NAMES, fontsize=7)
ax.set_title("Facies Classification (MalenoV)", fontsize=12)
ax.set_xlabel("Xline")
ax.set_ylabel("Time (ms)")

plt.tight_layout()
plt.savefig(OUT_PATH, dpi=150, bbox_inches="tight")
print(f"\nSaved: {OUT_PATH}")
plt.close(fig)
print("Done.")
