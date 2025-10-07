# Tasks: Add structured failure reporting to batch processor summary

- [ ] Track permanent failure events (PDF path, attempts, error message) during the batch run.
- [ ] Write a JSON artifact summarizing failures to the output directory when any PDFs fail.
- [ ] Update the batch summary/log output to reference the failure report and include counts.
- [ ] Extend integration tests to assert the report contents for a run with multiple failures.
