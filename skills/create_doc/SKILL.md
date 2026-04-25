---
name: create-doc
description: "Use when you need to analyze a GitHub repository from its GitHub URL, clone it into tools/, summarize its purpose, inspect it meticulously script by script and function by function, explain how it works and how to run it, then create a new skills/<repo>/SKILL.md containing that repository-specific analysis. Typical use: /create-doc {github-url}."
---

# Create Repository Documentation Skill

## Purpose

This skill analyzes a repository from a user-provided GitHub URL and turns that analysis into a new repository-specific skill.

The generated output must be written to:

- `skills/<repository-name>/SKILL.md`

This is for cases where a user wants the workflow to clone another repository into `tools/` first and then create a reusable skill that explains:

- what the repository is for
- how it is structured
- how it works internally
- how to run the scripts, services, or applications it contains

## Invocation

Expected usage pattern:

```text
/create-doc {github-url}
```

Examples:

```text
/create-doc https://github.com/owner/some-repo
/create-doc https://github.com/owner/some-repo.git
```

## Required Outcome

Given a GitHub repository URL, this skill must:

1. Accept the GitHub repository URL.
2. Derive the repository name from that URL.
3. Clone the repository into `tools/<repository-name>`. Use `git clone {github-url}`.
4. Inspect the cloned repository contents directly.
5. Summarize what the repository is for.
6. Analyze the implementation meticulously, one important file at a time.
7. Go function by function or script by script where practical.
8. Explain execution flow and how components connect.
9. Determine how to run the repository from checked-in evidence.
10. Create a new folder under `skills/` named after the repository.
11. Write a new repository-specific `SKILL.md` in that folder.

## Operating Rules

### Stay grounded in the repository

- Do not invent features.
- Do not describe architecture that is not present.
- Do not infer behavior without evidence from code, metadata, or documentation.
- If the repository is incomplete, minimal, or scaffold-only, state that explicitly.

### Start from concrete control surfaces

Prioritize:

- `README.md`
- dependency and packaging files
- build or task files
- executable scripts
- application entrypoints
- CLI bootstrap modules
- test commands that reveal intended usage

Then step inward to the code that directly controls runtime behavior.

### Analyze in a strict order

Work in this sequence:

1. Purpose and top-level structure
2. Runtime and build metadata
3. Main entrypoints and executable scripts
4. Supporting modules and internal flow
5. Function-level behavior in important files
6. How to run the repository

### Spend depth where it matters

Give the deepest treatment to:

- files in `tools/`, `scripts/`, `bin/`, or equivalent folders
- main application entrypoints
- modules called directly by those entrypoints
- orchestration files that decide startup behavior

Use lighter treatment for:

- lockfiles
- generated files
- vendored code
- dependency directories
- build artifacts

## Recommended Workflow

### Step 1. Resolve the GitHub URL

- Accept the target GitHub repository URL.
- Confirm that it is a repository URL, not a local path.
- Derive the repository name from the URL.
- Define the clone destination as `tools/<repository-name>`.

If the target directory already exists:

- inspect whether it is the intended repository clone
- do not overwrite or delete existing contents blindly
- either reuse the existing clone if it matches the requested repository or state the conflict clearly

### Step 2. Clone into `tools/`

- Clone the repository into `/store005/yohanuwa/Code/boglodite/tools/<repository-name>`.
- Use the cloned checkout as the only source for subsequent analysis.
- Confirm the clone produced a plausible repository root.

Look for markers such as:

- `.git/`
- `README.md`
- `pyproject.toml`
- `package.json`
- `Cargo.toml`
- `go.mod`
- `Makefile`

### Step 3. Inventory the repository

Inspect the top-level structure and identify:

- documentation files
- dependency and packaging files
- source directories
- script directories
- test directories
- container or deployment files
- automation files

Only gather enough context to locate the controlling paths before going deeper.

### Step 4. Summarize repository purpose

Write a concise purpose summary based on:

- repository name
- README language
- package metadata
- exposed commands
- visible entrypoints
- dominant source layout

If the repository is a starter template, scaffold, or placeholder, say so directly.

### Step 5. Identify execution paths

Locate the files that directly control behavior, such as:

- `main.py`
- CLI modules
- app bootstrap files
- server entrypoints
- scheduled jobs
- utility scripts
- framework startup files

For each important path, trace inward to the functions or classes that actually perform work.

### Step 6. Analyze meticulously

For each important file:

- state its role
- explain how it is invoked
- summarize each important function or class
- explain inputs, outputs, side effects, and dependencies
- describe control flow in execution order where possible

If a file is small, prefer function-by-function analysis.

If a file is large, focus on public functions, orchestrators, and logic that materially affects runtime behavior.

### Step 7. Determine how to run the repository

Produce run instructions in decreasing confidence order:

1. Verified commands that were actually run successfully
2. Commands explicitly documented in the repository
3. Commands strongly implied by metadata or entrypoints

Include when relevant:

- runtime or language version requirements
- dependency installation steps
- environment setup
- local development commands
- one-off script commands
- expected output or observable behavior

If commands cannot be executed, still provide evidence-based instructions and label them as unverified.

### Step 8. Create the new skill

Create:

- `skills/<repository-name>/`
- `skills/<repository-name>/SKILL.md`

The new file must be repository-specific, not a generic template.

## Required Structure For The Generated Skill

The generated `skills/<repository-name>/SKILL.md` should usually follow this structure:

```md
---
name: <repository-name>
description: "Use when you need a repository-specific analysis of <repository-name>: summarize its purpose, explain each important script/module/function, describe execution flow, and show how to run it."
---

# <Repository Name> Repository Analysis

## Purpose

## Verified Repository Summary

## Repository Walkthrough

### <important file or module>

#### Function-by-function analysis

## Execution Flow

## How To Run The Repository

## Important Constraints Or Gaps

## Bottom Line
```

Adapt headings when needed, but keep the same logical order:

1. purpose
2. repository inventory
3. deep implementation analysis
4. runtime and execution
5. constraints, assumptions, or missing pieces

## Quality Bar For The Generated Skill

The generated skill must:

- be specific to the analyzed repository
- mention actual files and actual roles
- explain important functions or scripts in execution order
- describe how the system starts and what happens next
- include concrete run instructions
- distinguish verified facts from inferred guidance
- avoid generic filler

## Tooling Guidance

When carrying out this skill:

- use targeted file listing and reading to identify the main control paths quickly
- inspect nearby code around entrypoints before expanding scope
- validate the generated file after writing it
- if a local command can cheaply verify a run instruction, do that once

## Output Discipline

- Do not stop after summarizing the repository.
- Do not stop after analysis only.
- Always create the new skill folder and generated `SKILL.md`.
- If execution cannot be verified, still generate the new skill and clearly label the run instructions as evidence-based.

## Example Intent

If the user runs:

```text
/create-doc https://github.com/example/example-repo
```

the result should be:

- cloning into `tools/example-repo`
- analysis of the cloned repository
- creation of `skills/example-repo/SKILL.md`
- a repository-specific skill that can be reused later to understand that repository accurately and quickly
