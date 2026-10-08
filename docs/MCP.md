# Boglodite MCP layer

Boglodite can expose its validated seismic operations as Model Context Protocol
(MCP) tools. The MCP layer does **not** replace the seismic skills or scientific
implementations. It provides a stable, typed interface between the general
purpose agent and the existing deterministic adapters.

## Architecture

```text
Interpreter request
      |
      v
Copilot CLI agent
      |
      v
Boglodite harness
(instructions + skills + constraints)
      |
      v
Boglodite MCP server
(typed tool interface)
      |
      +--> run_faultseg(...)
      |        |
      |        v
      |    sandbox/FaultSeg/predict_only_fault.py
      |
      +--> run_malenov(...)
               |
               v
           sandbox/MalenoV/predict_only_facies_stable.py
```

The MCP layer therefore standardizes **how** the agent invokes a capability.
The harness still determines **when and why** a capability should be used and
which geophysical rules must be respected.

## Exposed tools

| MCP tool | Purpose |
|---|---|
| `check_boglodite_readiness` | Check F3 data, model weights, and adapters without loading TensorFlow. |
| `inspect_seismic_volume` | Read SEG-Y geometry and coordinate ranges. |
| `run_faultseg` | Run the canonical FaultSeg adapter on inline, crossline, or time slice. |
| `run_malenov` | Run the canonical MalenoV adapter on a selected inline. |

## Copilot CLI integration

The committed `.github/mcp.json` config starts the local server with stdio:

```json
{
  "mcpServers": {
    "boglodite-seismic": {
      "type": "local",
      "command": "uv",
      "args": ["run", "--with", "mcp==2.0.0", "python", "-m", "boglodite_mcp.server"],
      "tools": ["*"]
    }
  }
}
```

After `uv sync`, start Copilot from the repository root and verify. The MCP SDK is added by `uv --with` at server startup, so it does not modify Boglodite's locked scientific environment:

```bash
copilot mcp list
copilot mcp get boglodite-seismic
```

For prompt-mode sessions (`copilot -p`), the Boglodite console opts into
workspace MCP loading so the project server is available to the non-interactive
agent process.

## Direct server test

```bash
uv run --with "mcp==2.0.0" python -m boglodite_mcp.server
```

This starts a stdio MCP server and waits for an MCP client. For interactive
inspection, use the MCP Inspector provided by the official SDK if installed.

## Research toggle

For evaluation runs where the MCP server must be disabled while retaining the
same repository, set:

```bash
export BOGLODITE_DISABLE_MCP=1
```

The Boglodite UI bridge will then start Copilot with
`--disable-mcp-server=boglodite-seismic`.
