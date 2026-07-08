#!/bin/bash
# Run a MalenoV facies-segmentation script on the GPU.
#
# The RTX PRO 500 laptop GPU has compute capability 12.0a, which the installed
# TensorFlow build does not ship native kernels for, so the CUDA libraries that
# ship inside the uv venv (tensorflow[and-cuda]) must be on LD_LIBRARY_PATH for
# the GPU to be detected. This wrapper sets that up and runs the given script.
#
# Usage:
#   bash sandbox/MalenoV/run_facies_gpu.sh                         # runs the stable predictor
#   bash sandbox/MalenoV/run_facies_gpu.sh predict_only_facies_v1.py
#   bash sandbox/MalenoV/run_facies_gpu.sh train_seismic_facies.py
#
# To change which inline is predicted, edit SECTION_SEGY in
# sandbox/MalenoV/facies_common.py.

set -euo pipefail

REPO="/home/yohanuwa/projects/boglodite"
SCRIPT="${1:-predict_only_facies_stable.py}"

cd "$REPO"

# Put the venv-bundled NVIDIA CUDA libraries on the loader path.
LD_LIB=$(find .venv/lib/python3.12/site-packages/nvidia -maxdepth 2 -type d -name lib | paste -sd: -)
export LD_LIBRARY_PATH="${LD_LIB}:${LD_LIBRARY_PATH:-}"

exec uv run python "sandbox/MalenoV/${SCRIPT}"
