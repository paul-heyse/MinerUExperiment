# Proposal: Add structured failure reporting to batch processor summary

## Why

When a PDF exhausts retries the batch processor only emits ad-hoc log lines before moving on. Operators must scrape logs to find
which documents failed and why, and there is no machine-readable artifact summarizing the errors. Producing a structured failure
report alongside the performance report would accelerate triage and enable automation to rerun specific PDFs.

## What Changes

- Capture permanent failure events inside the coordinator loop and aggregate document paths with their final error messages.
- Emit a `failed_documents.json` (or similar) artifact in the output directory summarizing each failed PDF and its attempts.
- Include the failure summary in the final console summary and return object for downstream tooling.
- Add tests covering runs with multiple permanent failures to assert the JSON artifact contents and summary logging.

## Impact

- Affected spec: **batch-processing**
- Affected code: `src/MinerUExperiment/batch_processor.py` and integration tests.
- Operational benefit: improves observability and automation for reruns.
