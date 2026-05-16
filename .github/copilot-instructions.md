# Copilot Instructions — Boglodite

## Rules

### Install packages

Always install using `uv add` and not `pip install`.
If an instruction says `pip install`, use `uv add` instead.

### Tools

Agent has tools, which are repositories or packages that people have developed for geoscience before. Tools are located in the `/tools` folder inside the repo root directory.

### Data

Put any data for sandboxing in the `/data` folder inside the repo root directory.
If you need to download data from Google Drive, use `gdown`.

### Sandboxing code

If you need to write code in a sandbox, put it in the `/sandbox` folder inside the repo root directory.
Use `uv run python` to execute scripts.

### Outputs

Put any output files (plots, saved models, numpy arrays, CSVs, etc.) in the `/outputs` folder inside the repo root directory.

---


Boglodite is a geoscience agent that wraps 100+ open-source geoscience repositories, making them accessible through a skill-based workflow. Each supported tool lives in `tools/` with a corresponding `SKILL.md` in `skills/`.

---

## Commands

```bash
# Install a dependency (never use pip install)
uv add <package>

# Run a script
uv run python sandbox/my_script.py

# Download data from Google Drive
uv run gdown <url-or-folder-url> -O ./data/
```

There is no test suite or linter configured.

---

## Repository Layout

```
tools/<repo-name>/   # cloned 3rd-party geoscience repositories (git clone targets)
skills/<repo-name>/  # SKILL.md documentation for each tool (used by Copilot as skills)
sandbox/             # experimental and one-off scripts
data/                # seismic input data (SEGY, .dat, etc.) — large files, not committed
outputs/             # all generated files: plots, .npy arrays, saved models, CSVs
```

`main.py` is a stub entry point; the real work happens in `sandbox/` scripts.

---

## Key Conventions

### Package management
- **Always use `uv add`**, never `pip install`.
- Run scripts with **`uv run python`**, not `python3` or `python`.

### File placement
- Sandbox / prototype scripts → `sandbox/`
- All outputs (plots, `.npy`, `.h5`, `.csv`) → `outputs/`
- Data files (SEGY, `.dat`) → `data/`
- New tools cloned from GitHub → `tools/<repo-name>/`

### TensorFlow / Keras
All CNN code uses `tf_keras` (Keras 2 compatibility layer), not standalone `keras`. Set the env var before importing:

```python
os.environ["TF_USE_LEGACY_KERAS"] = "1"
import tf_keras as keras
```

### Seismic data
- SEGY files are opened with `segyio`. Use `strict=False` for non-standard files.
- `segyio.tools.cube(path)` returns shape `(n_inlines, n_xlines, n_samples)`.
- Normalise seismic to zero-mean / unit-variance before feeding to any CNN.

### FaultSeg input convention
FaultSeg's U-Net expects input transposed from segyio's natural order:
```python
gx = np.transpose(sub)   # (n_il, n_xl, n_z) → (n_z, n_xl, n_il)
```
The time/depth dimension must be padded to a multiple of 8 for the three MaxPooling3D(2) layers to have matching skip-connection sizes.

### Adding a new geoscience tool
Use the `add-geo-tool` skill: it clones the repo into `tools/`, inspects it, and writes `skills/<repo>/SKILL.md`. Invoke with:
```
/create-doc https://github.com/owner/repo
```

### Google Drive downloads
Use `gdown` (already in dependencies). For folders:
```bash
uv run gdown --folder '<drive-folder-url>' -O ./data/
```

---

## Existing Skills

| Skill name | Tool | Purpose |
|---|---|---|
| `malenov` | `tools/MalenoV` | 3D CNN seismic facies classification (voxel inputs, SEGY) |
| `facies_net` | `tools/facies_net` | Modular facies classification with augmentation + TensorBoard |
| `faultSeg` | `tools/faultSeg` | 3D U-Net automatic fault segmentation (Wu et al., 2019) |
| `add-geo-tool` | — | Clone a GitHub repo and generate its `SKILL.md` |
| `initiate-boglodite` | — | Bootstrap the repo structure and download F3 dataset |

Pre-trained FaultSeg weights live at `/data/faultSeg_model/model/` (downloaded separately via gdown from the Drive folder documented in `skills/faultSeg/SKILL.md`).

---

## F3 Dataset

The Dutch Government F3 seismic volume is the primary test dataset:
- Path: `data/Dutch Government_F3_entire_8bit seismic.segy`
- Inlines 100–750, xlines 300–1250, 462 time samples (4–1848 ms, step 4 ms)
- Download: `uv run gdown --folder "https://drive.google.com/drive/folders/0B7brcf-eGK8CbGhBdmZoUnhiTWs?resourcekey=0-0ZhV_OJ3TKN1ShFAGcrOzQ" -O ./data/`
