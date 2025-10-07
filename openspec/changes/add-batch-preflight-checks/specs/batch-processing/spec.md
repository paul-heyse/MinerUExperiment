# Spec Delta: Batch Processing System

## ADDED Requirements

### Requirement: Environment Preflight Validation

The batch processor SHALL verify critical dependencies before starting worker processes.

#### Scenario: Missing MinerU CLI detection

- **WHEN** `BatchProcessor.run` is invoked and the configured MinerU CLI executable is not found or not executable
- **THEN** the setup phase SHALL raise a descriptive error before spawning workers
- **AND** it SHALL include remediation guidance in the log or exception message

#### Scenario: Missing GPU backend detection

- **WHEN** required Python modules (e.g., `torch`, `vllm`) cannot be imported
- **THEN** the preflight validation SHALL abort setup with a clear error indicating the missing dependency
- **AND** no worker processes SHALL be started

#### Scenario: Output directory validation

- **WHEN** the configured output directory is unwritable
- **THEN** the preflight SHALL raise an error and skip worker startup
- **AND** it SHALL log which path failed permission checks
