# Evaluation implementation added for the TLE study

The executable rubric is in `boglodite_eval/`; the frozen task design and user
workflow are in `evaluation/`.

Implemented pieces:

- six compact F3 tasks (3 FaultSeg + 3 MalenoV)
- primary conditions: `bare` and `boglodite`
- frozen canonical-reference creation with SHA256 hashes
- FaultSeg: NRMS, correlation, MAE, RMSE, SSIM, probability-range checks
- MalenoV: voxel agreement, macro F1, mean IoU, Cohen's kappa, softmax checks
- primary outcomes: `correct`, `silent_failure`, `overt_failure`
- human-intervention metadata
- optional expected-tool evidence from a Copilot JSONL log
- reproducibility manifest for code/data/model hashes
- aggregate CSV and JSON summaries for the Results section

See `evaluation/README.md` for the experiment procedure and exact commands.
