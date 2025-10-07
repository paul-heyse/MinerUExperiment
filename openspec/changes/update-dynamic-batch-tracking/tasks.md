# Implementation Tasks

## 1. Batch Processor Orchestration

- [ ] 1.1 Track pending PDF count dynamically inside `BatchProcessor.run`.
- [ ] 1.2 Keep workers alive until the coordinator reports no pending PDFs for a configurable quiet period.
- [ ] 1.3 Expand summary bookkeeping to include PDFs discovered after startup.

## 2. Progress Reporting

- [ ] 2.1 Update progress bars and log messages to refresh the total/remaining counts when new PDFs appear.
- [ ] 2.2 Add configuration (and CLI flag) for the quiet-period timeout controlling when the batch is considered drained.

## 3. Testing & Documentation

- [ ] 3.1 Extend integration tests to cover PDFs added mid-run and ensure they are processed before shutdown.
- [ ] 3.2 Document the dynamic-queue behavior in README/CLI help so operators know late files are supported.
