# Proposal: Batch Processing System with Parallel Workers

## Why

Processing PDFs one-at-a-time is inefficient when dealing with large document collections. A batch processing system with coordinated parallel workers will maximize throughput by processing multiple PDFs concurrently while preventing duplicate work. With 192GB RAM and 16-core AMD 9950x, the system can handle multiple workers efficiently.

## What Changes

- Create batch processing orchestrator to scan PDFsToProcess directory
- Implement multi-worker parallel processing with file-based coordination
- Add worker coordination to prevent duplicate processing (lock files or queue)
- Implement progress tracking and status reporting
- Add error recovery and retry logic
- Configure worker pool sizing based on available CPU cores and memory
- Output all processed files to MDFilesCreated directory

## Impact

- Affected specs: **batch-processing** (new capability)
- Affected code:
  - New module: `src/MinerUExperiment/batch_processor.py` - orchestration and worker management
  - New module: `src/MinerUExperiment/worker_coordinator.py` - coordination logic
  - New script: `scripts/process_batch.py` - CLI entry point
- Dependencies: Requires **mineru-integration** capability
