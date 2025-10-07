# Proposal: Add GPU-Aware Throttling to Batch Processor

## Why

- The resource monitor inside `BatchProcessor` only watches system RAM and CPU utilization, so workers keep launching jobs even when GPU memory or utilization is saturated, increasing the risk of vLLM OOM faults and slowdowns.【F:src/MinerUExperiment/batch_processor.py†L1149-L1193】
- GPU limits are validated just once at startup; there is no runtime feedback loop to adapt to temperature spikes or memory pressure that develops as PDFs vary in complexity.【F:src/MinerUExperiment/batch_processor.py†L1102-L1129】

## What Changes

- Sample GPU utilization, memory, and temperature during runs (via NVML or `nvidia-smi`) and feed the data into the existing throttle event alongside CPU/RAM checks.
- Add configurable thresholds for GPU memory %, utilization %, and temperature, with sensible defaults for the RTX 5090 profile set.
- Emit structured log/metric events when GPU pressure forces throttling so operators can tune profiles.
- Surface GPU headroom in progress summaries and performance reports to correlate throughput with accelerator load.

## Impact

- Specs: **batch-processing** (resource monitoring requirement gains GPU guardrails)
- Code: `batch_processor.py`, `metrics.py`, possibly new GPU sampling helper in `gpu_utils.py`
- Tests: extend resource-monitor unit tests and benchmark report expectations
