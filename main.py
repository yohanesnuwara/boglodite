"""Entry point — launches the Boglodite HITL console.

Equivalent to running the `boglodite` command:
    uv run boglodite            # after `uv sync`
    uv run python main.py       # this file
"""

from boglodite_ui.app import main

if __name__ == "__main__":
    main()
