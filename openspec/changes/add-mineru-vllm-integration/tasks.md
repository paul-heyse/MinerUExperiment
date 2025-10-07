# Implementation Tasks

## 1. Environment Setup

- [ ] 1.1 Install system dependencies (libgl1-mesa-glx, fonts-noto-cjk)
- [ ] 1.2 Install MinerU with vLLM extension via uv/pip
- [ ] 1.3 Download MinerU2.5-2509-1.2B model from HuggingFace
- [ ] 1.4 Verify CUDA availability and RTX 5090 detection

## 2. Configuration Module

- [ ] 2.1 Create `src/MinerUExperiment/mineru_config.py`
- [ ] 2.2 Implement config loading/writing for ~/mineru.json
- [ ] 2.3 Add model path configuration
- [ ] 2.4 Add GPU device selection (CUDA_VISIBLE_DEVICES)
- [ ] 2.5 Add vLLM backend settings (data-parallel-size, etc.)

## 3. GPU Utilities Module

- [ ] 3.1 Create `src/MinerUExperiment/gpu_utils.py`
- [ ] 3.2 Implement GPU warmup function (dummy inference)
- [ ] 3.3 Implement GPU availability check
- [ ] 3.4 Add CUDA_VISIBLE_DEVICES environment enforcement

## 4. Basic Processing

- [ ] 4.1 Create wrapper function for single-PDF MinerU invocation
- [ ] 4.2 Implement subprocess call to `mineru` CLI with vlm-vllm-engine backend
- [ ] 4.3 Add error handling and logging
- [ ] 4.4 Verify output file creation

## 5. Testing

- [ ] 5.1 Test with a sample PDF
- [ ] 5.2 Verify GPU utilization during processing
- [ ] 5.3 Confirm output files are generated correctly
- [ ] 5.4 Validate warmup improves processing time
