---
name: geomodel-horizon-playbook
description: "Use when building a seismic horizon with several picking tools and you must decide which result to keep without access to the answer: start from the reference workflow, diagnose where the horizon is suspicious, apply the tool aimed at that cause, and judge changes with held-out wells, agreement between independent methods and local evidence — not with smoothness alone. Primary user: the horizon agent."
---

# Horizon picking playbook: diagnose → targeted tool → judge with independent evidence

**Used by:** horizon agent.

This is the working method that produced the reference workflow, written so an agent can follow it
without ever seeing the reference interpretation.

## 1. Start from the reference workflow

The reference workflow is a sensible default, not a law:

1. `seed_report` → choose the event (the synthetic-tie event outranks amplitude heuristics).
2. `track_horizon` (Viterbi, guide window ~35 ms, fault map on) — the base pick.
3. `repair_deviations` — fix local cycle skips / leaks.
4. `ncs_level_check` — fix wrong-level blocks (rock-character boundary from the foundation model).
5. `repair_deviations` again — clean spikes left by the level correction.
6. `fill_and_smooth` (small gaps), `compare_candidates`, publish.

Deviate from it when your diagnosis shows a problem it does not address — for example a horizon that
cuts across dipping reflectors far from wells (use `dip_guided_horizon`).

## 2. Diagnose before acting — find WHERE and WHY

Do not change parameters blindly. Look for these symptoms and their causes:

| Symptom (input evidence) | Likely cause | Tool aimed at it |
|---|---|---|
| Jump > 12 ms away from faults, event amplitude fades after it | picks left their reflector (cycle skip / leak) | `repair_deviations` |
| Large disagreement with the NCS boundary where it is confident | whole block on the wrong level (unmapped fault) | `ncs_level_check` |
| Picks cut across the local dip, zig-zag between cycles (high dip misfit locally) | tracking window centred on a surface that does not follow structure | `dip_guided_horizon` |
| Independent methods disagree in a region | ambiguous level — no method is trustworthy there alone | inspect; prefer the estimate backed by wells + NCS |

`method_agreement_check` lists the largest disagreement regions with each method's median time — that
is your map of where to look.

## 3. Judge with evidence that can actually tell right from wrong

Some checks are misleading in a faulted field. Measured on this project (input checks vs accuracy):

- **Smoothness checks reward the wrong answer.** Few off-fault jumps, low dip misfit and bright picks
  are typical of a horizon that stays on ONE reflector across faults the fault map misses — smooth,
  bright, well-tied, and on the wrong block. Do not optimise these on their own; never "tighten
  tracking until the fault check passes".
- **Well residuals barely discriminate** — every reasonable horizon passes through the tied tops.
- **Agreement with a confident NCS level** is the most informative single input check.

Use these, knowing their limits:

1. **`holdout_wells_check`** — replays the chain with groups of wells hidden and gives the error at the
   hidden wells after every step. It is a NEAR-WELL check: it catches steps that damage the horizon
   where wells are, but it cannot see the far-from-well blocks where the big errors are. On this project
   it ranked candidates roughly opposite to their real field-wide accuracy (simple tracking looks best
   at unseen wells, yet is worst across the field). Never use it to reject a level correction that acts
   far from wells; use it to make sure a step does not break the near-well picks.
2. **`method_agreement_check`** — the share of the horizon backed by ≥ 2 independent methods and the
   regions where they disagree. Useful to locate ambiguous areas; its large regions can include the
   edges of the survey outside the field, where the NCS boundary is weak.
3. **Local before/after** — judge a change where it acted (the flagged region), not by a field-wide
   average that dilutes it.
4. Verifier findings: read the reason, not just pass/fail. If the fault agent objects to steps that
   coincide with the NCS level or a likely fault zone, that is a known blind spot of the fault map —
   say so in your summary instead of removing a correct step.

No input check measures accuracy far from wells. Where the methods disagree far from wells, state it
as an open uncertainty (with the competing times) rather than pretending one check settles it.

## 4. Decide and explain

Keep a change only if the independent evidence (hold-out wells, method agreement, local evidence)
supports it. In the published summary state: the chain of steps, the hold-out error vs interpolation,
the agreement share, the regions that remain uncertain, and any verifier objection you consider a
blind spot and why. Unresolved regions belong in the uncertainty ledger, not hidden.
