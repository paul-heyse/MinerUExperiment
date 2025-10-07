# Proposal: Optimize GPU metrics sampling via NVML with CLI fallback

## Why

`MetricsCollector` shells out to `nvidia-smi` for every sample. On a 5s interval that incurs ~80-100ms per poll, blocks if
`nvidia-smi` is unavailable, and complicates container deployments that expose NVML but not the CLI. Switching to NVML (via
`pynvml`) provides near-zero-overhead access to GPU telemetry while retaining a CLI fallback for environments without NVML. The
change reduces monitoring overhead and broadens compatibility for benchmark mode.

## What Changes

- Introduce an NVML-based sampling path in `MetricsCollector` with graceful fallback to the existing `nvidia-smi` subprocess.
- Cache NVML handles, lazily initialize them, and ensure resources are released when `stop()` runs.
- Extend performance metrics tests to cover both NVML-present and CLI-fallback branches using fakes.
- Update project documentation to note the optional `pynvml` dependency for best-effort GPU metrics.

## Impact

- Affected spec: **performance-optimization** (GPU metrics requirement)
- Affected code: `src/MinerUExperiment/metrics.py`, potentially helper utilities, and tests exercising metrics sampling.
- Dependencies: Optional runtime dependency on `pynvml` when available.
