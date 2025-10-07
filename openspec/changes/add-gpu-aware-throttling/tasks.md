# Implementation Tasks

## 1. GPU Telemetry Collection

- [ ] 1.1 Introduce a reusable helper (NVML or `nvidia-smi`) that returns current GPU utilization %, memory %, and temperature.
- [ ] 1.2 Add configuration knobs for GPU sampling interval and thresholds (memory %, utilization %, temperature °C).
- [ ] 1.3 Extend `MetricsCollector` to store the additional GPU telemetry fields when benchmark mode is active.

## 2. Resource Monitor Integration

- [ ] 2.1 Update `_resource_monitor_loop` to consult GPU telemetry alongside CPU/RAM and toggle the throttle event when GPU thresholds are exceeded.
- [ ] 2.2 Include GPU headroom details in throttle/resume log messages and progress bar postfixes.
- [ ] 2.3 Ensure graceful handling when GPU telemetry is unavailable (e.g., NVML missing) without disabling CPU/RAM monitoring.

## 3. Reporting & CLI

- [ ] 3.1 Surface new GPU threshold flags in `scripts/process_batch.py` and document defaults per performance profile.
- [ ] 3.2 Augment the performance report aggregates with max/average GPU memory %, utilization %, and hottest temperature observed.
- [ ] 3.3 Update README/QUICKSTART guidance to explain GPU guardrails and tuning.

## 4. Validation

- [ ] 4.1 Unit test the telemetry helper using mocked NVML output and edge cases (no GPU, partial data).
- [ ] 4.2 Simulate GPU saturation in a controlled test to assert throttle/resume behavior affects worker dispatch.
- [ ] 4.3 Regenerate or update benchmark fixtures asserting GPU metrics appear in reports.
