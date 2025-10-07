# Proposal: Dynamic Batch Queue Tracking

## Why

The batch processor determines the total number of PDFs once at startup and stops the run when the original count has been processed. When new PDFs arrive (or stale locks are reclaimed) while the batch is running, workers are told to shut down even if there are still unprocessed files on disk. This prematurely ends long-running ingestion jobs and forces manual restarts to pick up the backlog.

## What Changes

- Rework batch orchestration to treat the input directory as a live queue until it is truly empty.
- Recompute remaining work as workers finish, growing the progress totals when new PDFs appear and only exiting after the queue has drained for a configurable quiet period.
- Surface the dynamic totals in progress output and the final summary so operators can see how many PDFs were discovered after startup.

## Impact

- Affected specs: **batch-processing** (existing capability)
- Affected code:
  - `src/MinerUExperiment/batch_processor.py`
  - `scripts/process_batch.py`
  - `tests/test_batch_processor_integration.py`
- Tooling: none (reuse existing CLI)
