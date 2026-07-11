# Contributing to Boglodite

Boglodite is meant to be a community-driven platform. Anyone can add their favorite open-source geoscience repository as a tool so that the Boglodite agent can use it.

## How to Add a New Tool

Follow these steps to add your favorite open-source repository to Boglodite:

### Step 1 — Open the Copilot CLI

Open a terminal in the Boglodite repository and launch the Copilot CLI:

```bash
copilot
```

### Step 2 — Add the skills folder (first time only)

If you haven't added the skills folder yet, register it so the agent can load all available skills:

```bash
skills add skills
```

### Step 3 — Add the tool with the agent

Use the `/add-geo-tool` command followed by the GitHub URL of the repository you want to add:

```bash
/add-geo-tool https://github.com/owner/repo-name
```

**Example:**

```bash
/add-geo-tool https://github.com/bolgebrygg/MalenoV
```

The Boglodite agent will automatically:

- Clone the repository into the `tools/` folder
- Inspect the repository contents
- Generate a new `skills/<repo-name>/SKILL.md` with a full analysis of the tool

### Step 4 — Test your new tool

Before submitting, it is recommended to test that the new skill works correctly. Invoke the newly created skill by its name:

```bash
/<your-new-tool>
```

For example, if you added a tool called `my-geo-tool`:

```bash
/my-geo-tool
```

Verify that the agent correctly understands the tool and can run it.

### Step 5 — Commit and open a Pull Request

Once you're happy with the result, commit the changes and open a Pull Request:

```bash
git add tools/<repo-name>/ skills/<repo-name>/
git commit -m "Add <repo-name> as a new tool"
git push origin your-branch-name
```

Then open a PR against the `main` branch of this repository.

---

## Guidelines

- Make sure the repository you are adding is **open-source** and relevant to **subsurface or geoscience** workflows.
- Test the skill end-to-end before submitting your PR.
- Add a row for your tool in the **Open Source Tools Gallery** table in `README.md`.
- Do not commit large data files (SEGY volumes, model weights). Use the `data/` and `models/` folders locally; they are excluded via `.gitignore`.

---

## Existing Tools

See the [Open Source Tools Gallery](./README.md#open-source-tools-gallery) in the README for the current list of tools.

---

## Questions or Ideas?

Open an [issue](https://github.com/yohanesnuwara/boglodite/issues) to suggest a new tool or discuss ideas for the project.
