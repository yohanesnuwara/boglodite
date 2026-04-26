# Rules

## Install packages

Always install using `uv add` and not `pip install`
If the instruction tells you to do `pip install`, use `uv add`

## Tools

Agent has tools, which are repositories or packages that people have developed for geoscience before. Tools are located in `/tools` folder inside the repo root directory.

## Data

Put any data for sandboxing in `/data` folder inside the repo root directory.

## Sandboxing code

If you need to write a code in a sandbo, put in `/sandbox` folder inside the repo root directory.
Use  `uv run python`.