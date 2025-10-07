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

## Performance Profiles

The batch processor exposes tuned profiles that map to the RTX 5090 + AMD 9950x hardware:

- `balanced` (default) – 12 workers, GPU memory utilization at 90% for stable day-to-day runs. GPU guardrails throttle at 97% SM / 92% memory / 85 °C and resume at 90% / 85% / 80 °C.
- `throughput` – 14 workers, 95% GPU utilization, aggressive batching for maximum PDFs/hour. Guardrails pause at 99% SM / 95% memory / 87 °C to keep the RTX 5090 stable under full load.
- `latency` – 6 workers, conservative GPU usage to minimize per-PDF turnaround time with lower guardrail thresholds (95% SM / 88% memory / 83 °C).

Invoke `scripts/process_batch.py --profile <balanced|throughput|latency>` to pick a profile, or
`--benchmark` to emit a detailed `performance_report.json` containing throughput, GPU, CPU, and memory
statistics collected during the run.

You can override the GPU guardrails per run with flags such as `--gpu-memory-throttle`,
`--gpu-memory-resume`, `--gpu-util-throttle`, `--gpu-util-resume`, `--gpu-temp-throttle`,
`--gpu-temp-resume`, and `--gpu-monitor-interval`. Pass percentages (e.g. `--gpu-memory-throttle 90`)
or degrees Celsius for the temperature guardrails. Use these when experimenting with alternative GPUs
or cooling configurations.

To compare multiple benchmark runs, load the generated reports with
`MinerUExperiment.metrics.compare_reports([...])` or run the helper via a short Python snippet to write a
`comparison.json` summary alongside your raw results.
