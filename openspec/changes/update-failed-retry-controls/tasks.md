# Implementation Tasks

## 1. Coordinator Enhancements

- [ ] 1.1 Accept options to include `.failed` PDFs when retrying.
- [ ] 1.2 Support age-based expiration for `.failed` markers so stale failures automatically requeue.
- [ ] 1.3 Expose metadata about retried failures for downstream reporting.

## 2. Batch Processor & CLI

- [ ] 2.1 Add configuration fields and CLI flags (`--retry-failed`, `--failed-expiry-hours`) to control retry behavior.
- [ ] 2.2 Update worker orchestration to mark retried items distinctly in progress and summaries.
- [ ] 2.3 Ensure permanent failures still produce `.failed` markers when retries are exhausted again.

## 3. Testing & Docs

- [ ] 3.1 Extend integration tests to cover reruns that automatically pick up `.failed` PDFs.
- [ ] 3.2 Document the new retry controls in README and CLI help text.
