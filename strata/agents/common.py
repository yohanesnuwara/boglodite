"""Shared agent machinery: run context (deps), model factory, scripted models,
skill loading, tool-call recording and the agent runner.

Every tool call is recorded twice: as an event in events.jsonl (live UI + audit) and
on the run context so the next published object carries its full tool provenance.
Every agent run saves its complete message transcript under runs/<id>/agents/.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import pandas as pd
from pydantic_ai import Agent
from pydantic_ai import messages as M
from pydantic_ai.models.function import AgentInfo, FunctionModel

from strata.config import REPO
from strata.store import Store, ToolCall

DEFAULT_MODEL = "anthropic:claude-opus-5-5"


@dataclass
class RunContext_:
    """Deps passed to every agent tool."""
    store: Store
    agent: str = ""
    state: dict[str, Any] = field(default_factory=dict)       # working objects shared between agents of a run
    calls: list[ToolCall] = field(default_factory=list)        # provenance for the next publish

    @property
    def cube(self):
        from strata.data import load_cube
        return load_cube()

    @property
    def wells(self):
        if "wells" not in self.state:
            from strata.data import Wells
            self.state["wells"] = Wells()
        return self.state["wells"]

    def record(self, tool: str, params: dict, diagnostics: dict) -> dict:
        """Log a tool call (event + provenance) and return the diagnostics for the model."""
        tc = ToolCall(tool=tool, params=_jsonable(params), diagnostics=_jsonable(diagnostics))
        self.calls.append(tc)
        self.store.emit("tool", self.agent, tool=tool, params=tc.params, diagnostics=_summarise(tc.diagnostics))
        return diagnostics

    def take_calls(self) -> list[ToolCall]:
        out, self.calls = self.calls, []
        return out


def _jsonable(x):
    return json.loads(json.dumps(x, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)))


def _summarise(d: dict, max_len: int = 1200) -> dict:
    s = json.dumps(d, default=str)
    return d if len(s) <= max_len else {"truncated": s[:max_len] + "…"}


def df_brief(df: pd.DataFrame, n: int = 60) -> list[dict]:
    """A DataFrame as a JSON-able list of records, rounded and truncated for the model."""
    return json.loads(df.head(n).round(3).to_json(orient="records"))


def skill(name: str) -> str:
    p = REPO / "skills" / name / "SKILL.md"
    txt = p.read_text()
    return txt.split("---", 2)[2].strip() if txt.startswith("---") else txt


# ── models ───────────────────────────────────────────────────────────────────
def make_model(spec: str, scripted_plan: Callable | None = None):
    """'scripted' → deterministic FunctionModel following the agent's plan (no LLM);
    'openai:<model>@<base_url>' → OpenAI-compatible server (e.g. LM Studio);
    anything else → a Pydantic AI model string such as 'anthropic:claude-sonnet-5-5'."""
    if spec == "scripted":
        if scripted_plan is None:
            raise ValueError("scripted model needs a plan")
        return FunctionModel(scripted_plan, model_name="scripted")
    if spec.startswith("openai:") and "@" in spec:
        from pydantic_ai.models.openai import OpenAIChatModel
        from pydantic_ai.providers.openai import OpenAIProvider
        name, base = spec[len("openai:"):].split("@", 1)
        return OpenAIChatModel(name, provider=OpenAIProvider(base_url=base, api_key="local"))
    return spec


def plan(steps: list[tuple[str, Callable[[list], dict] | dict]], final: Callable[[list], dict] | dict):
    """Build a FunctionModel plan: call `steps` in order (args may depend on earlier tool
    returns), then return the output. Used for the deterministic 'scripted' baseline."""
    def fn(messages: list[M.ModelMessage], info: AgentInfo) -> M.ModelResponse:
        returns = [p for m in messages for p in getattr(m, "parts", []) if isinstance(p, M.ToolReturnPart)]
        k = len(returns)
        if k < len(steps):
            name, args = steps[k]
            return M.ModelResponse(parts=[M.ToolCallPart(name, args(returns) if callable(args) else args)])
        out = final(returns) if callable(final) else final
        return M.ModelResponse(parts=[M.ToolCallPart(info.output_tools[0].name, out)])
    return fn


def tool_return(returns: list, name: str) -> Any:
    """Content of the latest tool return with the given tool name."""
    for p in reversed(returns):
        if p.tool_name == name:
            return p.content
    return None


# ── runner ───────────────────────────────────────────────────────────────────
async def run_agent(agent: Agent, ctx: RunContext_, name: str, prompt: str, model_spec: str,
                    usage_limit_requests: int = 40) -> Any:
    """Run one agent task: emits start/end events with token usage and saves the
    complete transcript to runs/<id>/agents/<n>_<name>.json."""
    from pydantic_ai.usage import UsageLimits
    ctx.agent = name
    ctx.store.emit("agent_start", name, prompt=prompt[:2000], model=model_spec)
    t0 = time.time()
    try:
        result = await agent.run(prompt, deps=ctx, usage_limits=UsageLimits(request_limit=usage_limit_requests))
    except Exception as e:
        ctx.store.emit("agent_error", name, error=f"{type(e).__name__}: {e}")
        raise
    u = result.usage
    d = ctx.store.dir / "agents"
    d.mkdir(exist_ok=True)
    n = len(list(d.glob("*.json"))) + 1
    (d / f"{n:02d}_{name}.json").write_bytes(result.all_messages_json())
    out = result.output
    ctx.store.emit("agent_end", name, seconds=round(time.time() - t0, 1),
                   usage={"input_tokens": u.input_tokens, "output_tokens": u.output_tokens,
                          "requests": u.requests, "tool_calls": u.tool_calls},
                   output=out.model_dump() if hasattr(out, "model_dump") else str(out))
    return out
