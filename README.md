# MinerUExperiment (replace via scripts/init.sh)

Baseline template for Python projects in Cursor on Ubuntu.

## Quick start (per project)
1. Run `scripts/init.sh <package_name> [python_version] "Description"`.
2. Open folder in Cursor (`cursor .`).
3. Ensure interpreter shows `.venv/bin/python`.
4. Run target tasks: **pytest**, **lint**, **format**.

See `.cursor/rules`, `.vscode/*`, and `environment.yml` for configuration details.

## MinerU + vLLM Environment

Run `scripts/setup_mineru_env.sh` to install system dependencies, the `mineru[all]` package, and download
the `MinerU2.5-2509-1.2B` model. The script also verifies an RTX 5090 GPU is visible to CUDA. Once the
environment is ready you can validate end-to-end processing with:

```bash
scripts/validate_mineru_setup.py PDFsToProcess/sample.pdf --output-dir MDFilesCreated
```

The validator runs a cold and warm processing pass, captures GPU utilization snapshots, and reports
the observed speedup after warmup. For programmatic access you can also run:

```bash
python - <<'PY'
from pathlib import Path
from MinerUExperiment import (
    ensure_model_downloaded,
    load_config,
    process_pdf,
    validate_pdf_processing,
)

config = load_config()
ensure_model_downloaded(config=config)
result = process_pdf(Path("PDFsToProcess/sample.pdf"))
print(result)

report = validate_pdf_processing(Path("PDFsToProcess/sample.pdf"))
print(report.to_json())
PY
```

The `process_pdf` helper wraps the `mineru` CLI and enforces the `vlm-vllm-engine` backend.

### Batch Processor Preflight Checks

Before worker processes launch, the batch processor now performs preflight validation. It verifies that the
configured `mineru` CLI is executable, required Python modules (`torch`, `vllm`) import successfully, and the
output directory is writable. The setup progress bar reports each validation outcome so operators can quickly
diagnose failures. Resolve any reported issues before re-running the batch job to avoid exhausting worker
retries on misconfigurations.

## Performance Profiles

The batch processor exposes tuned profiles that map to the RTX 5090 + AMD 9950x hardware:

- `balanced` (default) – 12 workers, GPU memory utilization at 90% for stable day-to-day runs.
- `throughput` – 14 workers, 95% GPU utilization, aggressive batching for maximum PDFs/hour.
- `latency` – 6 workers, conservative GPU usage to minimize per-PDF turnaround time.

Invoke `scripts/process_batch.py --profile <balanced|throughput|latency>` to pick a profile, or
`--benchmark` to emit a detailed `performance_report.json` containing throughput, GPU, CPU, and memory
statistics collected during the run.

To compare multiple benchmark runs, load the generated reports with
`MinerUExperiment.metrics.compare_reports([...])` or run the helper via a short Python snippet to write a
`comparison.json` summary alongside your raw results.
