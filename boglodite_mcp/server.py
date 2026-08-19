"""Boglodite seismic MCP server.

Run locally with:

    uv run python -m boglodite_mcp.server

GitHub Copilot CLI discovers this server automatically from .github/mcp.json.
The server uses stdio, so never print diagnostics to stdout from this module.
"""

from __future__ import annotations

from typing import Literal

from mcp.server import MCPServer

from .tools import inspect_f3_volume, readiness, run_faultseg as _run_faultseg
from .tools import run_malenov as _run_malenov

mcp = MCPServer("boglodite-seismic")


@mcp.tool()
def check_boglodite_readiness() -> dict:
    """Check whether the F3 data, trained models, and seismic adapters exist.

    Call this before a prediction when the local environment may not have been
    initialized. It does not load TensorFlow or the full seismic cube.
    """
    return readiness()


@mcp.tool()
def inspect_seismic_volume() -> dict:
    """Return F3 SEG-Y geometry and coordinate ranges without loading the cube.

    Use this to validate requested inline, crossline, or time coordinates before
    launching a computationally expensive interpretation task.
    """
    return inspect_f3_volume()


@mcp.tool()
async def run_faultseg(
    slice_type: Literal["inline", "xline", "timeslice"],
    coordinate: int,
) -> dict:
    """Run FaultSeg fault-probability prediction on the Dutch F3 survey.

    Args:
        slice_type: Seismic slice orientation: inline, xline, or timeslice.
        coordinate: SEG-Y coordinate. For timeslice this is time in milliseconds.

    Returns output paths plus execution evidence. The adapter enforces the
    existing Boglodite FaultSeg conditioning and model conventions.
    """
    return await _run_faultseg(slice_type, coordinate)


@mcp.tool()
async def run_malenov(inline: int) -> dict:
    """Run MalenoV 9-class seismic facies prediction on one Dutch F3 inline.

    Args:
        inline: SEG-Y inline coordinate to classify.

    Returns probability, class-map, and QC-figure paths plus execution evidence.
    The current validated MalenoV adapter supports inline prediction only.
    """
    return await _run_malenov(inline)


if __name__ == "__main__":
    mcp.run(transport="stdio")
