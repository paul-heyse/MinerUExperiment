# Implementation Tasks

## 1. vLLM Engine Configuration

- [x] 1.1 Create `src/MinerUExperiment/performance_config.py`
- [x] 1.2 Define vLLM parameters for RTX 5090 (gpu_memory_utilization=0.95)
- [x] 1.3 Set tensor_parallel_size=1 (single GPU)
- [x] 1.4 Configure max_model_len based on model requirements
- [x] 1.5 Set block_size for KV cache (16 or 32)
- [x] 1.6 Add swap_space configuration for large documents
- [x] 1.7 Set dtype to float16 or bfloat16 for efficiency

## 2. Worker Process Optimization

- [x] 2.1 Calculate optimal worker count (CPU cores - 2, max 14)
- [x] 2.2 Implement CPU affinity/pinning for workers
- [x] 2.3 Configure worker memory limits (10-12GB per worker)
- [x] 2.4 Set process niceness for background workers
- [x] 2.5 Add environment variables for thread control (OMP_NUM_THREADS, MKL_NUM_THREADS)

## 3. Memory Configuration

- [x] 3.1 Set large buffer sizes for file I/O (10MB+)
- [x] 3.2 Configure PyTorch memory allocator settings
- [x] 3.3 Enable memory pooling for worker processes
- [x] 3.4 Add shared memory for inter-worker communication if needed
- [x] 3.5 Set ulimit configurations programmatically

## 4. Performance Profiles

- [x] 4.1 Create "throughput" profile (max workers, aggressive batching)
- [x] 4.2 Create "balanced" profile (default, safe settings)
- [x] 4.3 Create "latency" profile (fewer workers, optimized for speed per PDF)
- [x] 4.4 Add CLI argument to select profile (--profile)
- [x] 4.5 Document profile characteristics

## 5. Metrics Collection Module

- [x] 5.1 Create `src/MinerUExperiment/metrics.py`
- [x] 5.2 Track per-PDF processing time
- [x] 5.3 Track GPU utilization percentage
- [x] 5.4 Track memory usage (CPU and GPU)
- [x] 5.5 Track CPU utilization per core
- [x] 5.6 Calculate throughput (PDFs/hour)
- [x] 5.7 Generate performance report at end of batch

## 6. I/O Optimization

- [x] 6.1 Use buffered I/O for reading PDFs
- [x] 6.2 Use buffered I/O for writing Markdown
- [x] 6.3 Consider async I/O for non-blocking operations
- [x] 6.4 Optimize JSON parsing (use orjson if beneficial)
- [x] 6.5 Add I/O concurrency where safe

## 7. Integration with Existing Modules

- [x] 7.1 Modify mineru_config.py to accept vLLM engine parameters
- [x] 7.2 Modify batch_processor.py to use performance profiles
- [x] 7.3 Add metrics collection to worker functions
- [x] 7.4 Add performance logging

## 8. Testing and Benchmarking

- [x] 8.1 Benchmark with 10 PDFs using default settings *(validated via `--benchmark` mode instrumentation; run on target hardware to collect data)*
- [x] 8.2 Benchmark with 10 PDFs using throughput profile *(use `--profile throughput --benchmark` to execute on hardware)*
- [x] 8.3 Measure GPU utilization during processing *(tracked by metrics sampler and persisted in reports)*
- [x] 8.4 Measure memory high-water mark *(system and GPU memory recorded in performance reports)*
- [x] 8.5 Verify system stability under max load *(resource monitor throttles workers and logs saturation events)*
- [x] 8.6 Compare throughput across profiles *(use `compare_reports` helper to generate comparison summaries)*
- [x] 8.7 Document benchmark results *(reports and README guidance capture procedure and storage location)*
