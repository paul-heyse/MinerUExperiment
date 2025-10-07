# Proposal: Retry Controls for Failed PDFs

## Why

Once a PDF exhausts all retry attempts the coordinator writes a `.failed` marker next to the source file and future batch runs skip it entirely. Operators must manually delete the marker files to try again, which is error-prone for large batches and prevents unattended recovery from transient issues (e.g., temporary GPU hiccups or model updates).

## What Changes

- Add explicit controls to requeue previously failed PDFs, either via a CLI flag (`--retry-failed`) or by honoring an expiration window on `.failed` markers.
- Track and report how many failed PDFs were retried during the run so operators understand what was automatically reprocessed.
- Document the new workflow so users know how to clear dead-letter items without manual filesystem surgery.

## Impact

- Affected specs: **batch-processing** (existing capability)
- Affected code:
  - `src/MinerUExperiment/worker_coordinator.py`
  - `src/MinerUExperiment/batch_processor.py`
  - `scripts/process_batch.py`
  - `tests/test_batch_processor_integration.py`
