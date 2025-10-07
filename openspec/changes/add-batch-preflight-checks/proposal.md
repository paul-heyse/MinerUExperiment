# Proposal: Add batch processor preflight validation and clearer failure handling

## Why

`BatchProcessor.run` only records an environment snapshot before launching workers. If the MinerU CLI is missing or the vLLM
backend cannot import (e.g., torch absent), workers fail repeatedly and consume retries before the operator learns the root cause.
We should fail fast with explicit diagnostics and avoid spawning workers when prerequisites are absent.

## What Changes

- Introduce a preflight validation step in `BatchProcessor` that checks for the MinerU CLI, verifies Python package dependencies
  (torch, vllm), and confirms output directories are writable before spawning workers.
- Surface actionable error messages and halt setup when validations fail instead of letting worker retries exhaust.
- Emit the validation results in setup progress output so operators can see what passed.
- Extend integration tests to cover scenarios where the CLI is missing or torch import fails, asserting that the processor aborts
  cleanly with helpful errors.

## Impact

- Affected spec: **batch-processing**
- Affected code: `src/MinerUExperiment/batch_processor.py` plus new test coverage.
- Operational benefit: reduces time to diagnose misconfiguration and prevents wasted worker cycles.
