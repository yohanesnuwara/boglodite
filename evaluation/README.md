# Boglodite compact research evaluation

This folder implements the small experiment proposed for the TLE paper. It is
intentionally designed to be feasible in roughly two weeks while still giving a
controlled, falsifiable comparison.

## Primary research question

**Does a domain-specific agentic framework improve the fidelity with which a
general-purpose coding agent executes heterogeneous seismic interpretation
workflows from natural-language requests?**

The primary comparison is deliberately simple:

- **bare** — same coding agent/model, F3 data, trained weights, and original tool
  repositories, but no Boglodite domain instructions, skills, deterministic
  adapters, or seismic MCP tools.
- **boglodite** — same coding agent/model operating with the full Boglodite
  harness: domain instructions, skills, validation conventions, deterministic
  adapters, and typed seismic MCP tools.

The default benchmark has six tasks and three independent repetitions per
condition: **6 x 2 x 3 = 36 primary runs**. If time is tight, freeze four tasks
(F1, F2, M1, M2) before starting and report that reduced design honestly.

## Rubric

Every run receives one top-level outcome:

1. **Correct** — required numerical output exists, has the expected geometry,
   satisfies scientific invariants, and passes the preregistered numerical
   agreement thresholds against a frozen canonical execution.
2. **Silent numerical failure** — a well-formed output is produced but it fails
   the numerical fidelity criteria. This does *not* claim that a human would
   visually accept the result.
3. **Overt failure** — required output is absent/unreadable or has the wrong
   dimensions.

Secondary process fields record **human interventions** and, when a raw Copilot
JSONL log is supplied, whether the expected tool appears in the trajectory.

### FaultSeg metrics

- shape/orientation match
- NRMS (%)
- Pearson correlation
- MAE and RMSE
- SSIM for 2-D sections
- finite values and probability range [0, 1]

### MalenoV metrics

- class-map voxel agreement
- macro F1
- mean IoU
- Cohen's kappa
- optional probability MAE and softmax-sum validation

The default thresholds live in `evaluation/thresholds.json`. They should be
**frozen before agent scoring**. If repeated direct model executions show
non-negligible GPU numerical variation, widen them only from that repeatability
study, not after inspecting agent outcomes.

## 1. Freeze the experiment

List the tasks:

```bash
uv run boglodite-eval list
```

Generate a reproducibility manifest of code and any locally present F3/model
files:

```bash
uv run boglodite-eval manifest
```

Commit/tag the repository, `evaluation/tasks.json`, and
`evaluation/thresholds.json` before the first scored run.

## 2. Create canonical references

Run each requested workflow directly through the validated deterministic
adapter, not through a conversational agent. For example:

```bash
uv run python sandbox/FaultSeg/predict_only_fault.py --orientation inline --value 150
uv run boglodite-eval freeze-reference --task F1

uv run python sandbox/MalenoV/predict_only_facies_stable.py --inline 130
uv run boglodite-eval freeze-reference --task M1
```

Repeat for the benchmark tasks. Numerical references are copied to
`evaluation/references/<TASK>/` with SHA256 hashes.

**Important:** keep `evaluation/references/` outside the agent's readable
workspace during scored runs if the agent has unrestricted file access. The
reference is for the evaluator, not for the agent.

## 3. Run the two conditions

Use a fresh agent session and clean output directory for every run. Keep the
model, prompt, data, weights, and available compute fixed between conditions.
Randomize/interleave the order of conditions where practical.

For a run, archive its outputs into a run-specific directory, e.g.:

```text
evaluation/candidates/F1/bare/r01/
evaluation/candidates/F1/boglodite/r01/
```

If using Copilot CLI directly, save its raw JSONL trajectory too. Example:

```bash
copilot -p "Extract faults on inline 150 of the F3 seismic and save the numerical result and QC image." \
  --allow-all-tools --output-format json --no-ask-user \
  > evaluation/candidates/F1/boglodite/r01/agent.jsonl
```

For the **bare** condition, use a clean workspace that does not contain
Boglodite's `.github/copilot-instructions.md`, `skills/`, `.github/mcp.json`, or
validated `sandbox/` adapters. The raw third-party repositories, F3 data, and
weights should remain available. Do not approximate "bare" merely by disabling
MCP while leaving all procedural instructions visible.

### Helper scripts

Two helpers under `evaluation/scripts/` make this reproducible:

```bash
# 1. Build the clean bare workspace once (symlinks raw tools/data/weights only,
#    no harness). Re-run to refresh it.
evaluation/scripts/setup_bare_workspace.sh

# 2. Run a single task/condition/replicate. It launches Copilot with the frozen
#    prompt, saves the JSONL trajectory, and archives produced outputs into
#    evaluation/candidates/<TASK>/<condition>/rNN/ under the exact scorer names.
uv run python evaluation/scripts/run_task.py --task F1 --condition boglodite --replicate 1
uv run python evaluation/scripts/run_task.py --task F1 --condition bare      --replicate 1

# Inspect the exact command and archiving plan without launching the agent:
uv run python evaluation/scripts/run_task.py --task F1 --condition bare --replicate 1 --dry-run
```

The driver prints the ready-to-run `boglodite-eval score` command for each
completed run. `evaluation/workspaces/` is git-ignored.

## 4. Score a run

```bash
uv run boglodite-eval score \
  --task F1 \
  --condition boglodite \
  --replicate 1 \
  --candidate-dir evaluation/candidates/F1/boglodite/r01 \
  --agent-log evaluation/candidates/F1/boglodite/r01/agent.jsonl \
  --human-interventions 0 \
  --thresholds evaluation/thresholds.json
```

The command writes a structured JSON record to `evaluation/runs/`. Exit status
is 0 for a correct run and 2 for a silent/overt failure, which makes batch
scripts straightforward.

## 5. Summarize all runs

```bash
uv run boglodite-eval summarize
```

This writes:

- `evaluation/summary.csv` — one row per scored run for statistics/figures
- `evaluation/summary.json` — primary outcome counts and rates by condition

The paper's main table can be built directly from these fields:

| Condition | Correct | Silent failure | Overt failure | Human interventions |
|---|---:|---:|---:|---:|
| Bare agent | ... | ... | ... | ... |
| Boglodite | ... | ... | ... | ... |

## Optional small MCP ablation

Only if time permits, add a few runs with condition `boglodite-no-mcp`. Keep the
skills/instructions but disable the MCP server. Treat this as an architectural
case study (tool calls, source inspection, command construction), **not** as a
third full benchmark condition.

## Failure-cause coding

The automated evaluator assigns the top-level outcome. Afterward, inspect the
trajectory and assign one or more causal labels for the paper:

- E1 wrong tool
- E2 wrong seismic coordinate/section
- E3 axis/order error
- E4 normalization/scaling error
- E5 padding/window/geometry error
- E6 wrong model/weights
- E7 postprocessing/orientation error
- E8 runtime/dependency failure
- E9 unauthorized modification of data/model/tool source
- E10 other

Keep causal coding separate from the numerical outcome so the rubric remains
reproducible.
