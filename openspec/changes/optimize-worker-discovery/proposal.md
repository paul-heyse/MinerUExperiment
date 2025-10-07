# Proposal: Optimize Worker Discovery and Continuous Intake

## Why

- `WorkerCoordinator.acquire_next` performs a full `pending_items()` scan on every claim attempt, forcing each worker to walk the entire PDF tree repeatedly and creating severe I/O overhead once thousands of PDFs accumulate.【F:src/MinerUExperiment/worker_coordinator.py†L56-L117】【F:src/MinerUExperiment/worker_coordinator.py†L220-L229】
- `BatchProcessor.run` snapshots the pending PDF list once during startup, so any PDFs dropped into the input directory after the run begins are never processed and progress totals become inaccurate.【F:src/MinerUExperiment/batch_processor.py†L630-L715】

## What Changes

- Introduce a shared discovery index that performs incremental scans on a configurable interval and feeds a queue instead of walking the directory tree for every worker claim.
- Add filesystem change detection (lightweight polling by mtime/hash) so newly added PDFs are enqueued while a batch run is active, and progress metrics adjust to the growing workload.
- Update worker coordination APIs to draw from the shared queue with lock awareness, and expose status telemetry (e.g., queue depth) for logging/metrics.
- Extend the batch processor progress reporting to recalculate totals based on dynamic intake and reflect late-arriving PDFs in status output and performance reports.

## Impact

- Specs: **batch-processing** (update discovery and progress requirements)
- Code: `worker_coordinator.py`, `batch_processor.py`, `metrics.py`, `process_batch.py`, related tests
- Tooling: potential lightweight helper for directory polling (new module or utility)
