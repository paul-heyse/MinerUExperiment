# Spec Delta: Performance Optimization

## MODIFIED Requirements

### Requirement: Performance Metrics Collection

#### Scenario: GPU utilization tracking (NVML preferred)

- **WHEN** the metrics sampler gathers GPU utilization
- **THEN** it SHALL prefer querying via NVML (`pynvml`) when available to avoid spawning external processes
- **AND** if NVML is unavailable, it SHALL fall back to `nvidia-smi` without interrupting sampling
- **AND** the collector SHALL continue reporting utilization, memory usage, and temperature in both paths
- **AND** shutting down the collector SHALL release NVML resources to prevent handle leaks
