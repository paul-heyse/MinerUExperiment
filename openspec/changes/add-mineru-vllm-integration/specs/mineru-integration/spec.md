# Spec Delta: MinerU Integration

## ADDED Requirements

### Requirement: MinerU Installation

The system SHALL install MinerU with the vLLM extension module to enable GPU-accelerated PDF processing.

#### Scenario: Successful installation

- **WHEN** the installation script is run
- **THEN** MinerU with vLLM support is installed
- **AND** the `mineru` CLI command is available
- **AND** the `mineru[all]` package is present in the Python environment

#### Scenario: Missing dependencies

- **WHEN** system packages are missing (libgl1-mesa-glx or fonts)
- **THEN** the installation SHALL fail with a clear error message
- **AND** the error message SHALL list the missing dependencies

### Requirement: Model Download

The system SHALL download and configure the MinerU2.5-2509-1.2B model from HuggingFace.

#### Scenario: First-time model download

- **WHEN** the model is not present locally
- **THEN** the system SHALL download it from HuggingFace
- **AND** the model path SHALL be written to ~/mineru.json
- **AND** subsequent runs SHALL use the cached model

#### Scenario: Model source configuration

- **WHEN** MINERU_MODEL_SOURCE environment variable is set
- **THEN** the system SHALL use the specified model hub (huggingface or modelscope)

### Requirement: GPU Configuration

The system SHALL configure and enforce GPU usage for all MinerU processing operations.

#### Scenario: GPU detection

- **WHEN** the system initializes
- **THEN** it SHALL detect the RTX 5090 GPU
- **AND** it SHALL verify CUDA availability
- **AND** it SHALL log GPU memory and compute capability

#### Scenario: Force GPU usage

- **WHEN** MinerU processing is invoked
- **THEN** the system SHALL set CUDA_VISIBLE_DEVICES to GPU 0
- **AND** it SHALL use the vlm-vllm-engine backend
- **AND** it SHALL fail with a clear error if GPU is unavailable

### Requirement: GPU Warmup

The system SHALL perform a GPU warmup routine before processing PDFs to optimize performance.

#### Scenario: Cold start warmup

- **WHEN** the system starts processing for the first time
- **THEN** it SHALL run a dummy inference operation on the GPU
- **AND** the warmup SHALL complete within 10 seconds
- **AND** subsequent processing SHALL show improved latency

#### Scenario: Warmup failure

- **WHEN** GPU warmup fails
- **THEN** the system SHALL log a warning
- **AND** processing SHALL proceed without warmup

### Requirement: Configuration Management

The system SHALL manage MinerU configuration through a Python module and ~/mineru.json file.

#### Scenario: Load configuration

- **WHEN** the configuration module is initialized
- **THEN** it SHALL load settings from ~/mineru.json if present
- **AND** it SHALL use default settings if the file is missing
- **AND** it SHALL validate required fields (model paths, backend settings)

#### Scenario: Write configuration

- **WHEN** configuration is updated
- **THEN** it SHALL write changes to ~/mineru.json
- **AND** it SHALL preserve existing settings not being modified
- **AND** it SHALL create a backup before overwriting

### Requirement: Single PDF Processing

The system SHALL process a single PDF file using MinerU with vLLM acceleration.

#### Scenario: Successful processing

- **WHEN** a valid PDF path is provided
- **THEN** MinerU SHALL process the PDF using vlm-vllm-engine backend
- **AND** output files SHALL be created (Markdown, content_list.json, middle.json)
- **AND** the operation SHALL complete without errors

#### Scenario: Invalid PDF

- **WHEN** an invalid or corrupted PDF is provided
- **THEN** the system SHALL log the error
- **AND** it SHALL not crash
- **AND** it SHALL return a failure status with error details

#### Scenario: GPU memory overflow

- **WHEN** a PDF is too large for GPU memory
- **THEN** the system SHALL log the out-of-memory error
- **AND** it SHALL suggest reducing batch size or using CPU fallback
