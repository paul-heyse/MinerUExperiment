#!/usr/bin/env bash
set -euo pipefail
# Ensure we're in project env when Codex runs
eval "$(micromamba shell hook -s bash)"
micromamba activate -p "${PWD}/.venv"
# Start Codex with cwd = repo root; let you pass args (e.g., "fix tests")
exec codex "$@"
