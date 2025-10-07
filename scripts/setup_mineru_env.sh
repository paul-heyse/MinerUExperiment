#!/usr/bin/env bash
set -euo pipefail

APT_PACKAGES=(libgl1-mesa-glx fonts-noto-cjk)
UV_BIN="${UV_BIN:-uv}"
PYTHON_BIN="${PYTHON_BIN:-python}"

log() {
  printf '[setup] %s\n' "$*" >&2
}

install_system_packages() {
  if ! command -v sudo >/dev/null 2>&1; then
    log "sudo not found; skipping automatic apt installation."
    log "Please install packages manually: sudo apt-get install ${APT_PACKAGES[*]}"
    return
  fi

  log "Updating apt cache..."
  sudo apt-get update -y
  log "Installing system dependencies: ${APT_PACKAGES[*]}"
  sudo apt-get install -y "${APT_PACKAGES[@]}"
}

install_python_packages_with_uv() {
  log "Installing MinerU and dependencies with uv."
  "$UV_BIN" pip install --upgrade "mineru[all]" "torch" "huggingface_hub" "modelscope"
}

install_python_packages_with_pip() {
  log "Installing MinerU and dependencies with pip."
  "$PYTHON_BIN" -m pip install --upgrade "mineru[all]" "torch" "huggingface_hub" "modelscope"
}

install_python_packages() {
  if command -v "$UV_BIN" >/dev/null 2>&1; then
    install_python_packages_with_uv
  else
    install_python_packages_with_pip
  fi
}

download_model() {
  log "Ensuring MinerU model is downloaded."
  "$PYTHON_BIN" - <<'PY'
from MinerUExperiment.mineru_config import ensure_model_downloaded

ensure_model_downloaded()
PY
}

verify_gpu() {
  log "Verifying CUDA availability and RTX 5090 detection."
  "$PYTHON_BIN" - <<'PY'
from MinerUExperiment.gpu_utils import verify_gpu

verify_gpu()
PY
}

main() {
  install_system_packages
  install_python_packages
  download_model
  verify_gpu
  log "MinerU environment setup complete."
}

main "$@"
