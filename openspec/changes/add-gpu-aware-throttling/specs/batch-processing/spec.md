# Spec Delta: GPU-Aware Resource Monitoring

## MODIFIED Requirements

### Requirement: Resource Monitoring

#### Scenario: Throttle on GPU saturation
- **WHEN** GPU memory usage exceeds the configured threshold (default 92%) or GPU utilization exceeds the configured threshold (default 97%)
- **THEN** the system SHALL set the throttle event to pause new worker dispatch
- **AND** it SHALL log the GPU metrics that triggered the throttle
- **AND** it SHALL resume dispatch only after GPU metrics fall below the resume thresholds for at least one sampling interval

#### Scenario: Graceful degradation without telemetry
- **WHEN** GPU telemetry cannot be collected (e.g., NVML/nvidia-smi unavailable)
- **THEN** the system SHALL continue CPU/RAM monitoring
- **AND** it SHALL log a warning once per run indicating GPU guardrails are disabled

## ADDED Requirements

### Requirement: GPU Metrics Reporting

The batch processor SHALL expose GPU telemetry in progress output and benchmark reports.

#### Scenario: Progress visibility
- **WHEN** progress updates are emitted during a run
- **THEN** they SHALL include current GPU utilization and memory percentage when telemetry is available

#### Scenario: Benchmark aggregation
- **WHEN** `performance_report.json` is generated
- **THEN** it SHALL record average and peak GPU utilization %, memory %, and the highest temperature observed during the batch
