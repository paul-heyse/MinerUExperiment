# How to work in this repo (for Codex / agents)

## Environment
- Use Python from `./.venv/bin/python` (micromamba).
- When installing Python packages, prefer `micromamba install -p ./.venv -c conda-forge` over `pip`.
- Load env vars from `.env` (direnv is configured). Do not print secrets.

## Commands & quality gates
- Run tests with `pytest -q`. Generate coverage with `pytest -q --maxfail=1 --disable-warnings`.
- Lint with `ruff check src tests` and format with `black .`.
- Before opening a PR, run tasks `lint` and `pytest` (both must pass).

## Git workflow
- Work on a branch named `cx/<short-task>`; make small, reviewed commits.
- Write conventional commit messages (e.g., `feat: add X`, `fix: handle Y`).
- When modifying multiple files, stage partial diffs and explain non-obvious changes in the commit body.

## Boundaries
- Stay inside the workspace unless explicitly approved.
- Do not run networked commands unless asked.
- Prefer creating new functions over editing large ones; keep functions under ~50 lines and add tests.

## Project layout
- Code in `src/<package>/`, tests in `tests/`, config in `.vscode/`, rules in `.cursor/rules/`.
