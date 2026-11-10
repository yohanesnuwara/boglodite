"""Command line: `strata ui` (workbench) and `strata run` (headless run).

    uv run --no-sync python -m strata.cli ui [--port 8270]
    uv run --no-sync python -m strata.cli run [--model scripted|anthropic:claude-sonnet-5-5] [--no-score]
"""

from __future__ import annotations

import argparse
import asyncio
import json


def main() -> None:
    ap = argparse.ArgumentParser(prog="strata")
    sub = ap.add_subparsers(dest="cmd", required=True)
    u = sub.add_parser("ui", help="open the Strata workbench")
    u.add_argument("--port", type=int, default=8270)
    u.add_argument("--no-browser", action="store_true")
    r = sub.add_parser("run", help="run the horizon slice headless (autonomous)")
    r.add_argument("--model", default=None)
    r.add_argument("--brief", default=None)
    r.add_argument("--target", default=None)
    r.add_argument("--no-score", action="store_true")
    a = ap.parse_args()
    if a.cmd == "ui":
        from strata.ui.server import main as serve
        serve(a.port, not a.no_browser)
    else:
        from strata.orchestrator import Orchestrator, RunConfig
        cfg = RunConfig(autonomy="autonomous", score=not a.no_score)
        for k in ("model", "brief", "target"):
            if getattr(a, k):
                setattr(cfg, k, getattr(a, k))
        o = Orchestrator(cfg)
        print("run", o.store.run_id, "→", o.store.dir)
        man = asyncio.run(o.run())
        print(json.dumps(man, indent=2, default=str))
        print((o.store.dir / "RUN_LOG.md").read_text()[-3000:])


if __name__ == "__main__":
    main()
