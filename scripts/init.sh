#!/usr/bin/env bash
set -euo pipefail

if [[ "${1-}" == "" ]]; then
  echo "Usage: scripts/init.sh <package_name> [python_version] [\"Project description\"]"
  exit 1
fi

PKG_NAME="$1"
PY_VER="${2:-3.12}"
DESC="${3:-A Python project.}"

# Rename package folder
if [[ -d "src/yourpkg" && ! -d "src/${PKG_NAME}" ]]; then
  git mv "src/yourpkg" "src/${PKG_NAME}" 2>/dev/null || mv "src/yourpkg" "src/${PKG_NAME}"
fi

# Replace occurrences in key files
sed -i "s/name = \"yourpkg\"/name = \"${PKG_NAME}\"/" pyproject.toml
sed -i "s/Project Title/${PKG_NAME}/" README.md
sed -i "s/from yourpkg/from ${PKG_NAME}/" tests/test_basic.py

# Update Python versions
sed -i "s/python=3\.12/python=${PY_VER}/" environment.yml
PYVER_SHORT="py$(echo "${PY_VER}" | tr -d '.')"
sed -i "s/target-version = \"py[0-9]\+\"/target-version = \"${PYVER_SHORT}\"/" pyproject.toml
sed -i "s/python_version = \"[0-9.]\+\"/python_version = \"${PY_VER}\"/" pyproject.toml

# Add description
sed -i "s/description = \".*\"/description = \"${DESC}\"/" pyproject.toml

# Create environment
if command -v micromamba >/dev/null 2>&1; then
  eval "$(micromamba shell hook -s bash)"
  micromamba create -y -p ./.venv -f environment.yml
else
  echo "micromamba not found; please install it and rerun scripts/init.sh"
  exit 2
fi

# Pre-commit hooks
if command -v pre-commit >/dev/null 2>&1; then
  pre-commit install || true
fi

# Direnv allow (optional)
if command -v direnv >/dev/null 2>&1; then
  direnv allow || true
fi

echo
echo "Initialized project:"
echo "  package: ${PKG_NAME}"
echo "  python : ${PY_VER}"
echo "Next steps:"
echo "  1) Open this folder in Cursor:  cursor ."
echo "  2) Verify the status bar shows .venv/bin/python"
echo "  3) Run tasks: pytest / lint / format"
