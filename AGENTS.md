# AGENTS.md

## Git Account And Remote

- This repository must pull and push through the `ashishanand7` GitHub account.
- Use the SSH host alias configured on this Mac:
  `git@github-ashishanand7:ashishanand7/retrieval-head-atlas.git`
- Do not switch `origin` back to HTTPS for normal work. HTTPS currently uses a different GitHub auth path.
- Plain `git@github.com` authenticates with the default work SSH key on this machine, so prefer `github-ashishanand7` for this repo unless the local SSH config changes.
- Keep commit identity repo-local:
  `Ashish Anand <ashishanand7@users.noreply.github.com>`
- Verify account routing with:
  `ssh -T git@github-ashishanand7`

## Local Environment

- Use the repo-local virtual environment at `.venv` only for Codex/local notebook tooling.
- The main project code and GPU notebook execution are expected to run on the SageMaker notebook instance, not locally on this Mac.
- Activate it with:
  `source .venv/bin/activate`
- Notebook tooling expected in the venv:
  `jupytext`, `nbstripout`, and `ipykernel`.
- Recreate it if needed with:
  `python3 -m venv .venv && .venv/bin/python -m pip install -r requirements-dev.txt`

## Notebook Workflow

- Prefer normal Python scripts/modules as the primary Codex editing surface.
- Treat notebooks as optional wrappers or historical artifacts, not the main development surface.
- If a notebook must stay paired, edit the Jupytext `.py` file rather than editing `.ipynb` JSON directly.
- Keep paired notebooks and scripts in sync with:
  `jupytext --sync notebooks/*.ipynb`
- If editing a paired `.py` file directly, sync before running notebooks:
  `jupytext --sync notebooks/*.py`
- Avoid committing bulky notebook outputs. `nbstripout` is installed for this repo and should strip outputs from `.ipynb` files.
- When making notebook changes, inspect both the script diff and notebook metadata diff before committing.

## SageMaker Loop

- Mac/Codex side: edit locally, sync Jupytext pairs, commit, and push through the `ashishanand7` SSH remote.
- SageMaker side: pull the branch, run scripts/modules on the GPU instance, and use notebooks only when a notebook-specific workflow is required.
- For end-to-end remote operation, prefer a repeatable command path where Codex can SSH or SSM into SageMaker, pull the branch, launch the script, and tail logs.
- Remote logs should go under a stable ignored folder such as `logs/` so Codex can inspect run output without committing it.
- If an emergency fix is made directly on SageMaker, commit and push it there, then pull it back locally before continuing Codex work.
