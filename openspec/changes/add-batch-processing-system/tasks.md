# Implementation Tasks

## 1. Worker Coordination Module

- [x] 1.1 Create `src/MinerUExperiment/worker_coordinator.py`
- [x] 1.2 Implement file-based locking mechanism (.lock files)
- [x] 1.3 Add function to claim a PDF for processing
- [x] 1.4 Add function to release a PDF after completion
- [x] 1.5 Add function to detect and clean stale locks
- [x] 1.6 Implement work queue discovery (scan PDFsToProcess)

## 2. Batch Processor Module

- [x] 2.1 Create `src/MinerUExperiment/batch_processor.py`
- [x] 2.2 Implement BatchProcessor class with worker pool
- [x] 2.3 Add worker process spawning (multiprocessing.Pool or ProcessPoolExecutor)
- [x] 2.4 Implement worker task: claim PDF, process, move output
- [x] 2.5 Add progress tracking (processed count, errors, time)
- [x] 2.6 Implement graceful shutdown on SIGINT/SIGTERM
- [x] 2.7 Add logging per worker and aggregated

## 3. Output Management

- [x] 3.1 Create MDFilesCreated directory if missing
- [x] 3.2 Implement file moving/copying from temp to MDFilesCreated
- [x] 3.3 Preserve intermediate files (content_list.json, middle.json)
- [x] 3.4 Add naming convention (preserve original PDF filename)

## 4. Error Handling and Retry

- [x] 4.1 Implement retry logic for transient failures (max 3 retries)
- [x] 4.2 Add dead-letter handling for permanently failed PDFs
- [x] 4.3 Log detailed error information per PDF
- [x] 4.4 Continue processing remaining PDFs on individual failures

## 5. Performance Configuration

- [x] 5.1 Auto-detect CPU core count (os.cpu_count())
- [x] 5.2 Set worker pool size (default: cpu_count - 2, configurable)
- [x] 5.3 Add memory monitoring to prevent OOM
- [x] 5.4 Implement backpressure if system load is high

## 6. CLI Entry Point

- [x] 6.1 Create `scripts/process_batch.py`
- [x] 6.2 Add argument parsing (--workers, --input-dir, --output-dir)
- [x] 6.3 Integrate with BatchProcessor
- [x] 6.4 Add status display (progress bar or periodic updates)

## 7. Testing

- [ ] 7.1 Test with 5+ sample PDFs
- [ ] 7.2 Verify no duplicate processing
- [ ] 7.3 Test graceful shutdown mid-processing
- [ ] 7.4 Verify all outputs in MDFilesCreated
- [ ] 7.5 Test retry logic with intentionally failing PDF
