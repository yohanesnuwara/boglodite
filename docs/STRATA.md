# Strata — geomodelling swarm (horizon slice)

Strata is the multi-agent geomodelling system described in
[`SWARM_ARCHITECTURE.md`](./SWARM_ARCHITECTURE.md), built with **Pydantic AI** and a
separate web workbench. The first slice implemented is **agentic horizon
interpretation** of the Not 2 Fm top on the Norne 2001 cube, scored by the benchmark in
[`NORNE_BENCHMARK.md`](./NORNE_BENCHMARK.md).

## Run it

```bash
# Norne drive mounted read-only at /mnt/t7 (see NORNE_BENCHMARK.md §2)
export ANTHROPIC_API_KEY=...            # for Claude agents; not needed for --model scripted

uv run --no-sync python -m strata.cli ui                 # workbench at http://127.0.0.1:8270
uv run --no-sync python -m strata.cli run --model scripted                       # deterministic baseline
uv run --no-sync python -m strata.cli run --model anthropic:claude-sonnet-5-5    # Claude agents
```

Models: any Pydantic AI model string (`anthropic:…`), an OpenAI-compatible local server
as `openai:<model>@http://localhost:1234/v1` (e.g. LM Studio), or `scripted` — a
deterministic policy that calls the same tools in a fixed order (the tool-only baseline).

The first use of the seismic reads the 2001 cube's 1700–3300 ms window from the drive
(~6 min) and caches it in `data/strata_cache/` (derived INPUT; nothing is written to the drive).

## What runs

```
data steward ─▶ G1 ─▶ well agent (tie) ─▶ horizon agent ─▶ well agent (verify)
                                              ▲                   │ reject (≤ 2 reworks)
                                              └──── findings ─────┘
                      still rejected ─▶ ledger scenario + decision card
                                     ─▶ G2 freeze ─▶ export candidate ─▶ scorer (separate process)
```

| Agent | Tools | Output |
|---|---|---|
| Data steward | `inspect_seismic`, `inspect_wells`, `publish_project_settings` | `ProjectSettings:norne` |
| Well agent (producer) | `list_markers`, `qc_td_tables`, `tie_marker`, `publish_ties` | `WellTies:<marker>` |
| Horizon agent | `seed_report`, `track_horizon`, `fill_and_smooth`, `publish_horizon` | `HorizonPicks:Not2_top` (versions) |
| Well agent (verifier) | `check_horizon_at_wells` | verdict on each horizon version |

- The **orchestrator** is deterministic Python: phases, the two-round rework rule, the
  uncertainty ledger, and decision cards (answered in the workbench inbox, or by a
  documented default in `autonomous` mode).
- Agents have **no file-system tools**. The data layer (`strata/data.py`) reads only INPUT
  folders (`strata/config.py: INPUT_DIRS`) and drops the blind wells.
- Agents load the `skills/geomodel-*` files as domain guidance.

## What is recorded (for resuming work)

Each run writes `runs/<run_id>/` (kept on disk, git-ignored):

| Path | Content |
|---|---|
| `manifest.json` | brief, model, autonomy, git revision, start/end, status |
| `events.jsonl` | every event: tool calls with parameters and diagnostics, publishes, verdicts, cards, decisions, token usage, scores |
| `store/<Type>/<name>/v<k>/` | every object version: `meta.json` (provenance, verdicts, status) and data files (`ties.csv`, `horizon.npz`, `picks.csv`, …) |
| `agents/NN_<agent>.json` | the full message transcript of every agent run |
| `candidate/` | the benchmark export |
| `RUN_LOG.md` | human-readable summary |

Scores and visual reports go to `outputs/norne_scores/runs/<run_id>/` (scorer side). Dev
experiments are recorded in `outputs/strata_dev/NOTES.md`.

## Findings so far (2026-11-08)

- **Time–depth tables store TVD below KB, not MD** (the file headers are wrong). Read as
  TVD, tables agree to a median 6.9 ms; read as MD they disagree by 60 ms (up to 670 ms).
  Synthetic-seismogram tables are unreliable.
- With own tables plus nearest-neighbour fallback, all 36 non-blind wellbores tie the
  Not 2 top (spread between tables: median 9.4 ms).
- The tied tops sit closer to a **peak** than a trough (median 7.2 vs 9.2 ms).
- Tracking the whole cube takes ~15 s. ~20,000 neighbour "jumps" (> 12 ms) remain —
  mostly faults, the next agent's job.
- **Scripted baseline** (run `20261108-171459-4789`): verifier rejected v1 (3 seeds, MAE
  19 ms at wells) and v2/v3 (all seeds, MAE 10.2 ms vs 9.4 ms tolerance) → ledger →
  accepted by default. Benchmark: Not 2 top **time MAE 34.0 ms ✗** (≤ 15 ms), **coverage
  98.5 % ✓** — the bar the LLM agents have to beat.

## Next

1. Claude-agent runs (needs `ANTHROPIC_API_KEY`); compare with the scripted baseline.
2. Fault agent: barriers for the tracker and the horizon–fault verification pair.
3. Depth conversion, then structural framework and gridding.
