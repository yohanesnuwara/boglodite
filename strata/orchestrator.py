"""Orchestrator for the horizon-interpretation slice (docs/SWARM_ARCHITECTURE.md §4).

    data steward ─▶ G1 ─▶ well agent (tie) ─▶ horizon agent ─▶ well agent (verify)
                                                   ▲                    │ reject (≤ 2 rounds)
                                                   └──── findings ──────┘
                                     still rejected ─▶ ledger scenario + decision card
                                                  ─▶ G2 freeze ─▶ export candidate ─▶ score (separate process)

Deterministic Python (no LLM) so runs are reproducible; decision cards are answered by a
human, or by a documented default rule in 'autonomous' mode. Everything is recorded in
runs/<run_id>/ (events.jsonl, store/, agents/ transcripts, candidate/, RUN_LOG.md).
"""

from __future__ import annotations

import asyncio
import json
import subprocess
import sys
import time
import uuid
from dataclasses import dataclass, field
from typing import Literal

from strata.agents import fault as FA
from strata.agents import horizon as HA
from strata.agents import steward as SA
from strata.agents import well as WA
from strata.agents.common import DEFAULT_MODEL, RunContext_, run_agent
from strata.config import REPO
from strata.store import Store, Verdict
from strata.tools import horizon as H

Autonomy = Literal["autonomous", "gates_only", "every_decision"]
MAX_ROUNDS = 2


@dataclass
class RunConfig:
    brief: str = "Interpret the top of the Not 2 Formation (Not 2 Fm top) as a seismic horizon in TWT."
    target: str = "Not 2"                 # marker hint for the well agent
    horizon_name: str = "Not2_top"
    model: str = DEFAULT_MODEL
    autonomy: Autonomy = "autonomous"
    score: bool = True


@dataclass
class Card:
    id: str
    title: str
    body: str
    options: list[str]
    default: str
    gate: str | None = None
    future: asyncio.Future | None = field(default=None, repr=False)


class Orchestrator:
    def __init__(self, cfg: RunConfig, store: Store | None = None):
        self.cfg = cfg
        self.store = store or Store()
        self.ctx = RunContext_(store=self.store)
        self.cards: dict[str, Card] = {}
        self.status = "created"

    # ── decision cards ───────────────────────────────────────────────────────
    async def ask(self, title: str, body: str, options: list[str], default: str, gate: str | None = None) -> str:
        card = Card(uuid.uuid4().hex[:8], title, body, options, default, gate)
        human = (self.cfg.autonomy == "every_decision") or (self.cfg.autonomy == "gates_only" and gate is not None)
        self.store.emit("card", "orchestrator", card=card.id, title=title, body=body, options=options,
                        default=default, gate=gate, waiting_for_human=human)
        if not human:
            self.store.emit("decision", "orchestrator(default)", card=card.id, choice=default, gate=gate,
                            rule="autonomous mode: documented default")
            return default
        card.future = asyncio.get_running_loop().create_future()
        self.cards[card.id] = card
        self.status = "waiting"
        choice = await card.future
        self.status = "running"
        self.store.emit("decision", "human", card=card.id, choice=choice, gate=gate)
        return choice

    def answer(self, card_id: str, choice: str) -> bool:
        card = self.cards.get(card_id)
        if not card or card.future is None or card.future.done() or choice not in card.options:
            return False
        card.future.set_result(choice)
        return True

    # ── the run ──────────────────────────────────────────────────────────────
    async def run(self) -> dict:
        c, s = self.cfg, self.store
        self.status = "running"
        s.write_manifest(brief=c.brief, target=c.target, model=c.model, autonomy=c.autonomy, tier="D2-horizon-slice",
                         started=time.strftime("%Y-%m-%d %H:%M:%S"), git=_git_rev())
        s.emit("run_start", "orchestrator", **{k: getattr(c, k) for k in ("brief", "model", "autonomy")})
        try:
            await self._run()
            self.status = "done"
        except Exception as e:
            self.status = "failed"
            s.emit("run_error", "orchestrator", error=f"{type(e).__name__}: {e}")
            raise
        finally:
            s.write_manifest(finished=time.strftime("%Y-%m-%d %H:%M:%S"), status=self.status)
            s.emit("run_end", "orchestrator", status=self.status)
            write_run_log(s)
        return s.manifest()

    async def _run(self):
        c, s, ctx = self.cfg, self.store, self.ctx
        s.emit("phase", "orchestrator", phase="G1 · data")
        audit = await run_agent(SA.build(c.model), ctx, "data_steward",
                                f"Study brief: {c.brief}\nAudit the input data and publish the project settings.", c.model)
        g1 = await self.ask("Gate G1 · data accepted?",
                            f"Data steward: {audit.summary}\nIssues: {audit.issues or 'none'}",
                            ["accept", "stop"], "accept" if audit.accepted else "stop", gate="G1")
        if g1 == "stop":
            return

        s.emit("phase", "orchestrator", phase="Well ties ‖ fault map (parallel)")
        ctx_f = RunContext_(store=s, state=ctx.state)          # shared state, separate agent identity
        tie, _faults = await asyncio.gather(
            run_agent(WA.build_tie(c.model, c.target), ctx, "well_agent",
                      f"Study brief: {c.brief}\nTie the corresponding formation top to seismic time at every well.", c.model),
            run_agent(FA.build_produce(c.model), ctx_f, "fault_agent",
                      f"Study brief: {c.brief}\nBuild the fault map of the reservoir interval.", c.model))
        s.emit("phase", "orchestrator", phase="Horizon interpretation")
        findings = ""
        verdicts = []
        for rnd in range(1, MAX_ROUNDS + 2):
            prompt = (f"Study brief: {c.brief}\nThe well agent tied marker '{ctx.state.get('marker', tie.marker)}'."
                      f" Publish the horizon as '{c.horizon_name}'.")
            if findings:
                prompt += f"\n\nThe well agent REJECTED your previous version. Its findings:\n{findings}\nRework the horizon."
            await run_agent(HA.build(c.model, c.horizon_name, rework=rnd > 1), ctx, "horizon_agent", prompt, c.model)
            m = s.get("HorizonPicks", c.horizon_name)
            ctx_v = RunContext_(store=s, state=ctx.state)
            vw, vf = await asyncio.gather(
                run_agent(WA.build_verify(c.model, c.horizon_name), ctx, "well_agent_verifier",
                          f"Verify HorizonPicks:{c.horizon_name} (version {m.version}) against your ties.", c.model),
                run_agent(FA.build_verify(c.model, c.horizon_name), ctx_v, "fault_agent_verifier",
                          f"Verify HorizonPicks:{c.horizon_name} (version {m.version}) against your fault map.", c.model))
            for who, v in (("well_agent", vw), ("fault_agent", vf)):
                m = s.add_verdict(s.get("HorizonPicks", c.horizon_name, m.version),
                                  Verdict(verifier=who, passed=v.passed, summary=v.summary, findings=v.findings,
                                          evidence={"bad_wells": v.bad_wells}, round=rnd))
            passed = vw.passed and vf.passed
            if not passed:                                   # status reflects the combined verdict
                m = s.set_status(m, "rejected")
            v = vw if not vw.passed else vf
            verdicts.append((m.version, v))
            if passed or rnd > MAX_ROUNDS:
                break
            findings = json.dumps({who: {"passed": vv.passed, "summary": vv.summary, "findings": vv.findings,
                                         "bad_wells": vv.bad_wells, "suggested_fix": vv.suggested_fix}
                                   for who, vv in (("well_agent", vw), ("fault_agent", vf))}, indent=1)

        final = s.get("HorizonPicks", c.horizon_name)
        final_passed = all(vd.passed for vd in final.verdicts if vd.round == final.verdicts[-1].round) if final.verdicts else False
        if not final_passed:
            s.emit("ledger", "orchestrator", scenario=f"{c.horizon_name}: unresolved after {MAX_ROUNDS} reworks",
                   object=final.id, versions=[vv for vv, _ in verdicts])
            choice = await self.ask("Horizon still rejected after two reworks",
                                    f"Latest verdict: {verdicts[-1][1].summary}\nFindings: {verdicts[-1][1].findings}",
                                    ["accept latest", "stop"], "accept latest")
            if choice == "stop":
                return
            s.set_status(final, "draft", scenario="unresolved verifier disagreement")

        g2 = await self.ask("Gate G2 · freeze interpretation?",
                            f"HorizonPicks:{c.horizon_name} v{final.version} — {final.summary}\n"
                            f"Verifier: {final.verdicts[-1].summary if final.verdicts else 'n/a'}",
                            ["freeze", "stop"], "freeze", gate="G2")
        if g2 == "stop":
            return
        final = s.set_status(s.get("HorizonPicks", c.horizon_name), "frozen")
        self.export_candidate(final)
        if c.score:
            await asyncio.to_thread(score_run, s)

    def export_candidate(self, m) -> None:
        hz = H.load(self.store.path(m, "horizon.npz"))
        H.to_table(self.ctx.cube, hz)[["il", "xl", "twt"]].to_csv(self.store.dir / "candidate" / "horizon_not2_time.csv",
                                                                  index=False)
        (self.store.dir / "candidate" / "manifest.json").write_text(json.dumps(
            {**self.store.manifest(), "exported": m.ref}, indent=2, default=str))
        self.store.emit("export", "orchestrator", object=m.ref, files=["horizon_not2_time.csv"])


def score_run(store: Store) -> dict | None:
    """Score the exported candidate in a separate process (TRUTH never enters the agent process)."""
    out = REPO / "outputs" / "norne_scores" / "runs" / store.run_id
    out.mkdir(parents=True, exist_ok=True)
    store.emit("scoring", "scorer", status="started")
    cmd = [sys.executable, "-m", "norne_benchmark.score", "--candidate", str(store.dir / "candidate"),
           "--out", str(out / "score.json")]
    p = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True)
    if p.returncode != 0:
        store.emit("scoring", "scorer", status="failed", error=p.stderr[-2000:])
        return None
    try:
        from norne_benchmark import report
        from norne_benchmark import score as S
        report.build_report(store.dir / "candidate", report.Ctx(S.Reference()),
                            json.loads((out / "score.json").read_text()), out / "report.html")
    except Exception as e:                       # the report is optional; the score is what matters
        store.emit("scoring", "scorer", status="report_failed", error=str(e))
    res = json.loads((out / "score.json").read_text())
    (store.dir / "score_location.txt").write_text(str(out))
    store.emit("score", "scorer", composite=res["composite"], checks=[res["checks_passed"], res["checks_total"]],
               stages={k: {m: v["value"] for m, v in d["metrics"].items()} for k, d in res["stages"].items()},
               path=str(out))
    return res


def write_run_log(store: Store) -> None:
    """Human-readable summary of the run (for resuming later)."""
    ev = store.events()
    man = store.manifest()
    lines = [f"# Run {store.run_id}", "", "```", json.dumps(man, indent=2, default=str), "```", "", "## Timeline", ""]
    tok_in = tok_out = 0
    for e in ev:
        k = e["kind"]
        if k == "agent_end":
            u = e.get("usage", {})
            tok_in += u.get("input_tokens", 0) or 0
            tok_out += u.get("output_tokens", 0) or 0
            lines.append(f"- **{e['agent']}** finished in {e.get('seconds')} s · tokens in/out {u.get('input_tokens')}/{u.get('output_tokens')}")
        elif k == "tool":
            lines.append(f"  - tool `{e['tool']}` {json.dumps(e.get('params', {}))[:200]}")
        elif k in ("publish", "verdict", "decision", "ledger", "export", "score", "phase", "run_error", "agent_error"):
            body = {x: e[x] for x in e if x not in ("seq", "t", "kind", "agent")}
            lines.append(f"- {k} · {e.get('agent', '')} · {json.dumps(body, default=str)[:400]}")
    lines += ["", f"Total tokens: in {tok_in}, out {tok_out}", ""]
    (store.dir / "RUN_LOG.md").write_text("\n".join(lines))


def _git_rev() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=REPO, capture_output=True, text=True).stdout.strip()
    except Exception:
        return ""
