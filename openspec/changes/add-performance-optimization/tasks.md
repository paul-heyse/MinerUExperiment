# Implementation Tasks

## 1. vLLM Engine Configuration

- [ ] 1.1 Create `src/MinerUExperiment/performance_config.py`
- [ ] 1.2 Define vLLM parameters for RTX 5090 (gpu_memory_utilization=0.95)
- [ ] 1.3 Set tensor_parallel_size=1 (single GPU)
- [ ] 1.4 Configure max_model_len based on model requirements
- [ ] 1.5 Set block_size for KV cache (16 or 32)
- [ ] 1.6 Add swap_space configuration for large documents
- [ ] 1.7 Set dtype to float16 or bfloat16 for efficiency

## 2. Worker Process Optimization

- [ ] 2.1 Calculate optimal worker count (CPU cores - 2, max 14)
- [ ] 2.2 Implement CPU affinity/pinning for workers
- [ ] 2.3 Configure worker memory limits (10-12GB per worker)
- [ ] 2.4 Set process niceness for background workers
- [ ] 2.5 Add environment variables for thread control (OMP_NUM_THREADS, MKL_NUM_THREADS)

## 3. Memory Configuration

- [ ] 3.1 Set large buffer sizes for file I/O (10MB+)
- [ ] 3.2 Configure PyTorch memory allocator settings
- [ ] 3.3 Enable memory pooling for worker processes
- [ ] 3.4 Add shared memory for inter-worker communication if needed
- [ ] 3.5 Set ulimit configurations programmatically

## 4. Performance Profiles

- [ ] 4.1 Create "throughput" profile (max workers, aggressive batching)
- [ ] 4.2 Create "balanced" profile (default, safe settings)
- [ ] 4.3 Create "latency" profile (fewer workers, optimized for speed per PDF)
- [ ] 4.4 Add CLI argument to select profile (--profile)
- [ ] 4.5 Document profile characteristics

## 5. Metrics Collection Module

- [ ] 5.1 Create `src/MinerUExperiment/metrics.py`
- [ ] 5.2 Track per-PDF processing time
- [ ] 5.3 Track GPU utilization percentage
- [ ] 5.4 Track memory usage (CPU and GPU)
- [ ] 5.5 Track CPU utilization per core
- [ ] 5.6 Calculate throughput (PDFs/hour)
- [ ] 5.7 Generate performance report at end of batch

## 6. I/O Optimization

- [ ] 6.1 Use buffered I/O for reading PDFs
- [ ] 6.2 Use buffered I/O for writing Markdown
- [ ] 6.3 Consider async I/O for non-blocking operations
- [ ] 6.4 Optimize JSON parsing (use orjson if beneficial)
- [ ] 6.5 Add I/O concurrency where safe

## 7. Integration with Existing Modules

- [ ] 7.1 Modify mineru_config.py to accept vLLM engine parameters
- [ ] 7.2 Modify batch_processor.py to use performance profiles
- [ ] 7.3 Add metrics collection to worker functions
- [ ] 7.4 Add performance logging

## 8. Testing and Benchmarking

- [ ] 8.1 Benchmark with 10 PDFs using default settings
- [ ] 8.2 Benchmark with 10 PDFs using throughput profile
- [ ] 8.3 Measure GPU utilization during processing
- [ ] 8.4 Measure memory high-water mark
- [ ] 8.5 Verify system stability under max load
- [ ] 8.6 Compare throughput across profiles
- [ ] 8.7 Document benchmark results
