# Implementation Tasks

## 1. Environment Setup

- [x] 1.1 Install system dependencies (libgl1-mesa-glx, fonts-noto-cjk)
- [x] 1.2 Install MinerU with vLLM extension via uv/pip
- [x] 1.3 Download MinerU2.5-2509-1.2B model from HuggingFace
- [x] 1.4 Verify CUDA availability and RTX 5090 detection

## 2. Configuration Module

- [x] 2.1 Create `src/MinerUExperiment/mineru_config.py`
- [x] 2.2 Implement config loading/writing for ~/mineru.json
- [x] 2.3 Add model path configuration
- [x] 2.4 Add GPU device selection (CUDA_VISIBLE_DEVICES)
- [x] 2.5 Add vLLM backend settings (data-parallel-size, etc.)

## 3. GPU Utilities Module

- [x] 3.1 Create `src/MinerUExperiment/gpu_utils.py`
- [x] 3.2 Implement GPU warmup function (dummy inference)
- [x] 3.3 Implement GPU availability check
- [x] 3.4 Add CUDA_VISIBLE_DEVICES environment enforcement

## 4. Basic Processing

- [x] 4.1 Create wrapper function for single-PDF MinerU invocation
- [x] 4.2 Implement subprocess call to `mineru` CLI with vlm-vllm-engine backend
- [x] 4.3 Add error handling and logging
- [x] 4.4 Verify output file creation

## 5. Testing

- [x] 5.1 Test with a sample PDF
- [x] 5.2 Verify GPU utilization during processing
- [x] 5.3 Confirm output files are generated correctly
- [x] 5.4 Validate warmup improves processing time
