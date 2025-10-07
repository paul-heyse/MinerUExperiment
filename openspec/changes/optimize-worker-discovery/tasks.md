# Implementation Tasks

## 1. Discovery Index Service

- [ ] 1.1 Design a thread-safe discovery manager that periodically rescans the input tree (configurable interval, default 30s).
- [ ] 1.2 Maintain a queue/set of unseen PDFs while respecting existing `.lock`, `.done`, and `.failed` markers.
- [ ] 1.3 Emit lightweight telemetry (queue depth, last scan duration) for consumption by the batch processor logger/metrics.

## 2. Worker Coordinator Integration

- [ ] 2.1 Extend `WorkerCoordinator` to optionally consume work items from the shared discovery manager instead of walking the directory each call.
- [ ] 2.2 Ensure stale-lock recovery cooperates with the queued workflow (requeue reclaimed PDFs, avoid duplicates).
- [ ] 2.3 Provide an API for enqueuing new PDFs discovered mid-run (e.g., direct drop or external trigger).

## 3. Batch Processor Updates

- [ ] 3.1 Instantiate the discovery manager when a batch run starts and share it with workers via multiprocessing-safe primitives.
- [ ] 3.2 Adjust progress tracking to handle dynamic totals (update `total` when new PDFs appear, emit logs when queue drains/refills).
- [ ] 3.3 Update benchmark metrics to capture total PDFs processed vs. total discovered over time.
- [ ] 3.4 Add configuration flags (`--rescan-interval`, `--max-pending-queue`) to `process_batch.py`.

## 4. Validation

- [ ] 4.1 Unit test discovery manager for large directory trees (simulate thousands of PDFs) to assert minimal rescan churn.
- [ ] 4.2 Integration test demonstrating ingestion of new PDFs during an active run (mock workers) and correct progress totals.
- [ ] 4.3 Performance smoke test comparing baseline vs. optimized discovery on synthetic directory (documented in README/metrics).
