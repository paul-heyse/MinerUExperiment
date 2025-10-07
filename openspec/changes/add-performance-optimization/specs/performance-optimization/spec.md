# Spec Delta: Performance Optimization

## ADDED Requirements

### Requirement: vLLM Engine Optimization

The system SHALL configure vLLM engine parameters optimized for RTX 5090 GPU.

#### Scenario: GPU memory utilization

- **WHEN** vLLM engine is initialized
- **THEN** it SHALL set gpu_memory_utilization to 0.95 (use 95% of 24GB)
- **AND** it SHALL leave 5% for CUDA overhead

#### Scenario: Tensor parallelism

- **WHEN** running on single RTX 5090
- **THEN** tensor_parallel_size SHALL be set to 1
- **AND** data_parallel_size SHALL be set to 1

#### Scenario: KV cache configuration

- **WHEN** configuring block cache
- **THEN** block_size SHALL be set to 16 or 32
- **AND** swap_space SHALL be configured for documents exceeding GPU memory

#### Scenario: Data type optimization

- **WHEN** model supports it
- **THEN** dtype SHALL be set to bfloat16 for efficiency
- **AND** it SHALL fall back to float16 if bfloat16 is unavailable

### Requirement: Worker Process Configuration

The system SHALL optimize worker process configuration for AMD 9950x 16-core CPU.

#### Scenario: Optimal worker count

- **WHEN** running on 16-core CPU
- **THEN** default worker count SHALL be 14 (cores - 2)
- **AND** this SHALL be overridable via --workers argument
- **AND** maximum SHALL be capped at 14 to preserve system responsiveness

#### Scenario: CPU affinity

- **WHEN** workers are spawned
- **THEN** each worker process SHALL be pinned to specific CPU cores
- **AND** core assignments SHALL be distributed evenly
- **AND** system cores (0-1) SHALL be reserved for OS operations

#### Scenario: Thread control

- **WHEN** worker processes start
- **THEN** OMP_NUM_THREADS SHALL be set to 1 per worker
- **AND** MKL_NUM_THREADS SHALL be set to 1 per worker
- **AND** this SHALL prevent thread oversubscription

### Requirement: Memory Optimization

The system SHALL configure memory usage to leverage 192GB system RAM.

#### Scenario: Worker memory allocation

- **WHEN** 14 workers are active
- **THEN** each worker SHALL have access to 10-12GB RAM
- **AND** total worker memory SHALL not exceed 150GB
- **AND** 42GB SHALL be reserved for OS and buffers

#### Scenario: I/O buffer sizing

- **WHEN** reading or writing files
- **THEN** buffer size SHALL be set to 10MB or larger
- **AND** this SHALL reduce syscall overhead

#### Scenario: Memory pooling

- **WHEN** workers process multiple PDFs
- **THEN** memory SHALL be pooled and reused
- **AND** garbage collection SHALL be triggered between PDFs

### Requirement: Performance Profiles

The system SHALL provide three performance tuning profiles.

#### Scenario: Throughput profile

- **WHEN** --profile throughput is specified
- **THEN** worker count SHALL be set to 14
- **AND** GPU memory utilization SHALL be 0.95
- **AND** aggressive batching SHALL be enabled
- **AND** this profile optimizes for maximum PDFs/hour

#### Scenario: Balanced profile

- **WHEN** no profile is specified or --profile balanced
- **THEN** worker count SHALL be set to 12
- **AND** GPU memory utilization SHALL be 0.90
- **AND** this profile balances throughput and stability

#### Scenario: Latency profile

- **WHEN** --profile latency is specified
- **THEN** worker count SHALL be set to 4-6
- **AND** each PDF SHALL be prioritized for fastest individual processing
- **AND** this profile optimizes for minimum per-PDF latency

#### Scenario: Profile documentation

- **WHEN** --help is invoked
- **THEN** profile options SHALL be documented
- **AND** characteristics of each profile SHALL be explained

### Requirement: Performance Metrics Collection

The system SHALL collect and report performance metrics.

#### Scenario: Per-PDF timing

- **WHEN** a PDF is processed
- **THEN** the system SHALL record start and end timestamps
- **AND** it SHALL calculate processing duration in seconds
- **AND** it SHALL log this to metrics

#### Scenario: GPU utilization tracking

- **WHEN** processing is active
- **THEN** the system SHALL sample GPU utilization every 5 seconds
- **AND** it SHALL calculate average GPU utilization
- **AND** it SHALL report max GPU memory used

#### Scenario: System resource tracking

- **WHEN** batch processing runs
- **THEN** the system SHALL track CPU utilization per core
- **AND** it SHALL track system memory usage
- **AND** it SHALL track I/O wait time

#### Scenario: Throughput calculation

- **WHEN** batch processing completes
- **THEN** the system SHALL calculate total PDFs per hour
- **AND** it SHALL calculate average seconds per PDF
- **AND** it SHALL report these in summary

#### Scenario: Performance report generation

- **WHEN** processing completes
- **THEN** the system SHALL generate a performance report
- **AND** the report SHALL include all collected metrics
- **AND** the report SHALL be saved to MDFilesCreated/performance_report.json

### Requirement: I/O Optimization

The system SHALL optimize file I/O operations for high throughput.

#### Scenario: Buffered reading

- **WHEN** reading PDF files
- **THEN** the system SHALL use large read buffers (10MB+)
- **AND** it SHALL minimize syscalls

#### Scenario: Buffered writing

- **WHEN** writing Markdown files
- **THEN** the system SHALL use buffered writes
- **AND** it SHALL flush buffers at appropriate intervals

#### Scenario: Concurrent I/O

- **WHEN** multiple workers are active
- **THEN** I/O operations SHALL not block each other
- **AND** the system SHALL use OS-level I/O scheduling

### Requirement: System Stability

The system SHALL maintain stability under maximum load.

#### Scenario: Resource limits

- **WHEN** system resources approach limits
- **THEN** the system SHALL throttle worker spawning
- **AND** it SHALL log resource warnings
- **AND** it SHALL prevent OOM or system freeze

#### Scenario: Thermal monitoring

- **WHEN** GPU temperature is available
- **THEN** the system SHALL log GPU temperature
- **AND** it MAY throttle processing if temperature exceeds 85°C

#### Scenario: Error recovery

- **WHEN** a worker crashes due to resource exhaustion
- **THEN** the system SHALL restart that worker
- **AND** it SHALL continue processing remaining PDFs

### Requirement: Benchmarking Support

The system SHALL support performance benchmarking and comparison.

#### Scenario: Benchmark mode

- **WHEN** --benchmark flag is used
- **THEN** the system SHALL run in benchmark mode
- **AND** it SHALL collect detailed timing for each processing stage
- **AND** it SHALL output benchmark results in JSON format

#### Scenario: Comparison report

- **WHEN** multiple benchmark runs are performed
- **THEN** results SHALL be comparable
- **AND** the system SHALL support generating comparison reports

### Requirement: Configuration Validation

The system SHALL validate performance configurations before starting processing.

#### Scenario: Resource availability check

- **WHEN** batch processing is about to start
- **THEN** the system SHALL verify sufficient GPU memory
- **AND** it SHALL verify sufficient system RAM
- **AND** it SHALL verify requested worker count is feasible

#### Scenario: Incompatible settings

- **WHEN** performance settings are incompatible
- **THEN** the system SHALL raise a clear error
- **AND** it SHALL suggest corrected settings
- **AND** it SHALL not start processing

#### Scenario: Profile validation

- **WHEN** a performance profile is selected
- **THEN** the system SHALL validate the profile name
- **AND** it SHALL apply all profile settings atomically
- **AND** it SHALL log the active profile and settings
