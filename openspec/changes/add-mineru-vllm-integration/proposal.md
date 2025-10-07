# Proposal: MinerU with vLLM Integration

## Why

The project requires a robust PDF-to-Markdown conversion foundation using MinerU's VLM backend accelerated by vLLM. This provides 20-30× speedup over standard Transformers inference and properly leverages the RTX 5090 GPU. Without this integration, the system cannot efficiently process PDFs or utilize the available high-performance hardware.

## What Changes

- Install and configure MinerU with vLLM extension (`mineru[all]`)
- Set up the vLLM embedded engine backend (`vlm-vllm-engine`)
- Configure GPU selection and model downloading from HuggingFace
- Create configuration management for MinerU settings (`~/mineru.json`)
- Implement basic single-PDF processing capability
- Add GPU warmup routine to optimize cold-start performance
- Force GPU utilization for all processing operations

## Impact

- Affected specs: **mineru-integration** (new capability)
- Affected code:
  - New module: `src/MinerUExperiment/mineru_config.py` - configuration management
  - New module: `src/MinerUExperiment/gpu_utils.py` - GPU warmup and utilities
  - Dependencies: `requirements.txt` or `pyproject.toml`
  - Config file: `~/mineru.json` (user home)
