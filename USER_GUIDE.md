# MinerU + vLLM Batch Processing System - User Guide

## Table of Contents

1. [Introduction](#introduction)
2. [System Overview](#system-overview)
3. [Installation](#installation)
4. [Quick Start](#quick-start)
5. [Detailed Usage](#detailed-usage)
6. [Performance Tuning](#performance-tuning)
7. [API Reference](#api-reference)
8. [Troubleshooting](#troubleshooting)
9. [Advanced Topics](#advanced-topics)
10. [Support](#support)

---

## 1. Introduction

### What is MinerUExperiment?

MinerUExperiment is a high-performance PDF-to-Markdown conversion system that leverages **MinerU 2.5+** with **vLLM acceleration** to transform complex academic PDFs into structured, RAG-ready Markdown with clear section hierarchies.

### Key Features

- **GPU-Accelerated Processing**: 20-30× speedup using vLLM with RTX 5090 (24GB VRAM)
- **Parallel Batch Processing**: Process multiple PDFs concurrently with 12-14 workers
- **Smart Coordination**: File-based locking prevents duplicate work across workers
- **Structured Output**: Enhanced Markdown with proper heading hierarchies (#, ##, ###)
- **Table & Equation Preservation**: Tables rendered as HTML, equations as LaTeX
- **RAG-Ready**: Clean section boundaries ideal for retrieval-augmented generation
- **Performance Profiles**: Throughput, balanced, and latency modes optimized for different use cases
- **Comprehensive Metrics**: Real-time monitoring of GPU, CPU, memory, and throughput

### Who Should Use This?

- Researchers processing large collections of academic PDFs
- Data scientists building RAG systems requiring structured documents
- Organizations converting document archives to searchable Markdown
- Anyone needing high-quality PDF-to-Markdown conversion at scale

### System Requirements

- **GPU**: NVIDIA GPU with CUDA 12.8+ (RTX 5090 recommended)
- **RAM**: 64GB minimum, 192GB recommended for optimal performance
- **CPU**: Multi-core processor (16 cores recommended)
- **OS**: Linux (Ubuntu 22.04+ recommended)
- **Python**: 3.10-3.13
- **Storage**: ~20GB for models, additional space for PDFs and outputs

---

## 2. System Overview

### Architecture

```
┌─────────────────┐
│ PDFsToProcess/  │ ──┐
│  ├── doc1.pdf   │   │
│  ├── doc2.pdf   │   │
│  └── docN.pdf   │   │
└─────────────────┘   │
                      ▼
        ┌──────────────────────────┐
        │  Batch Processor         │
        │  (Orchestrator)          │
        └──────────────────────────┘
                      │
          ┌───────────┴───────────┐
          │  Worker Coordinator   │
          │  (Lock-based)         │
          └───────────┬───────────┘
                      │
      ┌───────┬───────┼───────┬───────┐
      ▼       ▼       ▼       ▼       ▼
   Worker  Worker  Worker  Worker  Worker
     #1      #2      #3     ...     #14
      │       │       │       │       │
      └───────┴───────┴───────┴───────┘
                      │
                      ▼
        ┌──────────────────────────┐
        │  MinerU + vLLM Engine    │
        │  (GPU-accelerated)       │
        └──────────────────────────┘
                      │
                      ▼
        ┌──────────────────────────┐
        │  Enhanced Markdown       │
        │  Generator               │
        └──────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────┐
│ MDFilesCreated/                     │
│  ├── doc1/                          │
│  │   ├── doc1.md                    │
│  │   ├── content_list.json          │
│  │   ├── middle.json                │
│  │   └── artifacts/                 │
│  ├── doc2/                          │
│  └── docN/                          │
└─────────────────────────────────────┘
```

### Output Structure

For each input PDF `document.pdf`, the system generates:

```
MDFilesCreated/
└── document/
    ├── document.md              # Original MinerU Markdown
    ├── content_list.json        # Structured content blocks
    ├── middle.json              # Detailed layout tree
    ├── model.json               # Model metadata
    └── artifacts/               # Supporting files (images, tables, etc.)
        ├── document/
        │   └── vlm/
        │       └── images/
        │           ├── image_0.jpg
        │           └── image_N.jpg
        ├── layout.pdf           # Annotated PDF with layout detection
        └── origin.pdf           # Copy of original PDF
```

### Processing Pipeline

1. **Discovery**: Scan `PDFsToProcess/` for `.pdf` files
2. **Coordination**: Workers claim PDFs via lock files (`.pdf.lock`)
3. **Processing**: MinerU extracts content using vLLM-accelerated VLM
4. **Output Management**: Results moved to `MDFilesCreated/<pdf_name>/`
5. **Enhancement**: Post-process `content_list.json` for structured Markdown (optional)
6. **Metrics**: Collect performance data and generate reports

---

## 3. Installation

### Prerequisites Check

```bash
# Verify Python version (3.10-3.13)
python --version

# Check CUDA availability
nvidia-smi

# Verify GPU (should show your NVIDIA GPU)
nvidia-smi --query-gpu=name,memory.total --format=csv

# Check system resources
lscpu | grep -E "^CPU\(s\)|Model name"
free -h
```

### System Dependencies

```bash
# Update package lists
sudo apt update

# Install required system packages
sudo apt install -y \
    libgl1-mesa-glx \
    fonts-noto-core \
    fonts-noto-cjk \
    git \
    build-essential

# Refresh font cache
fc-cache -fv
```

### Python Environment Setup

#### Option 1: Using Project Environment

```bash
# Clone the repository
git clone https://github.com/paul-heyse/MinerUExperiment.git
cd MinerUExperiment

# Run the setup script
./scripts/setup_mineru_env.sh

# This script will:
# - Create a virtual environment (.venv)
# - Install mineru[all] with vLLM support
# - Download MinerU2.5-2509-1.2B model
# - Verify GPU availability
```

#### Option 2: Manual Installation

```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate

# Upgrade pip and install uv (fast installer)
pip install --upgrade pip uv

# Install MinerU with all dependencies
uv pip install -U "mineru[all]"

# Download models
export MINERU_MODEL_SOURCE=huggingface  # or modelscope
mineru-models-download
# Select: huggingface → all

# Install project in editable mode
pip install -e .
```

### Verify Installation

```bash
# Run validation script
python scripts/validate_mineru_setup.py PDFsToProcess/sample.pdf --output-dir MDFilesCreated

# Expected output:
# ✓ MinerU installation verified
# ✓ vLLM backend available
# ✓ GPU detected: NVIDIA RTX 5090
# ✓ Model loaded: MinerU2.5-2509-1.2B
# ✓ Cold processing time: ~45s
# ✓ Warm processing time: ~2s
# ✓ Speedup: 22.5×
```

### Configuration

The system automatically creates `~/mineru.json` with optimal settings:

```json
{
  "backend": {
    "name": "pipeline",
    "vllm": {
      "gpu_memory_utilization": 0.9,
      "tensor_parallel_size": 1,
      "data_parallel_size": 1,
      "max_model_len": 16384,
      "block_size": 16,
      "swap_space_mb": 8192,
      "dtype": "bfloat16"
    }
  },
  "models-dir": {
    "pipeline": "~/.cache/huggingface/hub/models--opendatalab--PDF-Extract-Kit-1.0/...",
    "vlm": "~/.cache/huggingface/hub/models--opendatalab--MinerU2.5-2509-1.2B/..."
  },
  "runtime": {
    "cuda_visible_devices": "0",
    "device": "cuda:0"
  }
}
```

---

## 4. Quick Start

### Single PDF Processing

```bash
# Process a single PDF directly with MinerU CLI
mineru -p PDFsToProcess/document.pdf -o MDFilesCreated

# Or use the Python API
python -c "
from pathlib import Path
from MinerUExperiment import process_pdf

result = process_pdf(Path('PDFsToProcess/document.pdf'))
print(f'Processed: {result}')
"
```

### Batch Processing

```bash
# Process all PDFs in PDFsToProcess/ with default settings (12 workers)
python scripts/process_batch.py

# Process with custom worker count
python scripts/process_batch.py --workers 14

# Process with custom directories
python scripts/process_batch.py \
  --input-dir /path/to/pdfs \
  --output-dir /path/to/outputs \
  --workers 14
```

### Monitor Progress

Open a second terminal and monitor system resources:

```bash
# Monitor GPU utilization
watch -n 1 nvidia-smi

# Monitor CPU and memory
htop

# Monitor detailed GPU stats
nvidia-smi dmon -s pucvmet
```

### View Results

```bash
# List generated Markdown files
ls -lh MDFilesCreated/*/*.md

# View a sample Markdown file
head -50 MDFilesCreated/document/document.md

# Check intermediate JSON files
ls -lh MDFilesCreated/*/content_list.json
ls -lh MDFilesCreated/*/middle.json
```

---

## 5. Detailed Usage

### Batch Processing CLI

#### Command-Line Arguments

```bash
python scripts/process_batch.py [OPTIONS]
```

**Input/Output:**

- `--input-dir PATH` - Directory containing PDFs (default: `PDFsToProcess`)
- `--output-dir PATH` - Directory for outputs (default: `MDFilesCreated`)

**Worker Configuration:**

- `--workers N` - Number of parallel workers (default: CPU cores - 2)
- `--max-retries N` - Max retry attempts per PDF (default: 3)
- `--retry-delay SEC` - Delay between retries in seconds (default: 10.0)

**Performance:**

- `--profile NAME` - Use performance profile: `balanced`, `throughput`, or `latency`
- `--cpu-mem-threshold PCT` - Pause workers when CPU > threshold (default: 0.90)
- `--mem-threshold PCT` - Pause workers when memory > threshold (default: 0.85)
- `--benchmark` - Enable detailed performance metrics collection

**MinerU Settings:**

- `--mineru-backend BACKEND` - MinerU backend (default: `pipeline`)
- `--mineru-cli PATH` - Path to mineru executable (default: `mineru`)
- `--mineru-extra-args ARGS` - Additional arguments passed to mineru

**Environment:**

- `--env KEY=VALUE` - Override environment variables (repeatable)

**Display:**

- `--no-progress` - Disable live progress updates
- `--quiet` - Suppress all output except errors

#### Examples

**Basic batch processing:**

```bash
# Use default settings (12 workers, balanced profile)
python scripts/process_batch.py
```

**High-throughput mode:**

```bash
# Maximum performance on RTX 5090 + AMD 9950x
python scripts/process_batch.py \
  --profile throughput \
  --workers 14
```

**Latency-optimized mode:**

```bash
# Minimize per-PDF processing time
python scripts/process_batch.py \
  --profile latency \
  --workers 6
```

**Benchmark mode:**

```bash
# Collect detailed metrics
python scripts/process_batch.py \
  --benchmark \
  --workers 14 \
  --profile throughput

# View results
cat MDFilesCreated/performance_metrics/performance_report.json | jq
```

**Custom configuration:**

```bash
# Process with custom directories and settings
python scripts/process_batch.py \
  --input-dir /data/research_papers \
  --output-dir /data/processed_markdown \
  --workers 10 \
  --max-retries 5 \
  --retry-delay 15.0 \
  --mem-threshold 0.80
```

**Environment overrides:**

```bash
# Override GPU memory settings
python scripts/process_batch.py \
  --env MINERU_VLLM_GPU_MEMORY_UTILIZATION=0.95 \
  --env CUDA_VISIBLE_DEVICES=0,1 \
  --workers 14
```

### Post-Processing Markdown

After batch processing, enhance Markdown with clearer hierarchies:

```bash
# Post-process all outputs
python scripts/postprocess_markdown.py MDFilesCreated

# Post-process specific directory
python scripts/postprocess_markdown.py /path/to/outputs

# Verbose mode
python scripts/postprocess_markdown.py MDFilesCreated --verbose
```

This creates `.structured.md` files with:

- Proper heading levels (#, ##, ###) mapped from `text_level`
- Tables preserved as HTML blocks
- Equations preserved as LaTeX ($$...$$)
- Image captions inline with images
- Normalized blank lines

### Python API

#### Basic Processing

```python
from pathlib import Path
from MinerUExperiment import process_pdf, load_config

# Load configuration
config = load_config()

# Process a single PDF
pdf_path = Path("PDFsToProcess/document.pdf")
result = process_pdf(pdf_path)
print(f"Output: {result}")
```

#### Batch Processing API

```python
from pathlib import Path
from MinerUExperiment.batch_processor import BatchProcessor, BatchProcessorConfig

# Configure batch processor
config = BatchProcessorConfig(
    input_dir=Path("PDFsToProcess"),
    output_dir=Path("MDFilesCreated"),
    workers=14,
    profile="throughput",
)

# Run batch processing
processor = BatchProcessor(config)
summary = processor.run()

# Display results
print(f"Processed: {summary.success_count}/{summary.total_count}")
print(f"Failed: {summary.failure_count}")
print(f"Duration: {summary.duration_seconds:.2f}s")
```

#### Enhanced Markdown Generation

```python
from pathlib import Path
from MinerUExperiment.markdown_builder import generate_structured_markdown

# Generate structured Markdown from content_list.json
content_list = Path("MDFilesCreated/document/content_list.json")
output = generate_structured_markdown(content_list)
print(f"Generated: {output}")
```

#### Validation

```python
from pathlib import Path
from MinerUExperiment import validate_pdf_processing

# Validate MinerU setup with a test PDF
pdf_path = Path("PDFsToProcess/sample.pdf")
report = validate_pdf_processing(pdf_path)

print(f"Cold processing: {report.cold_duration:.2f}s")
print(f"Warm processing: {report.warm_duration:.2f}s")
print(f"Speedup: {report.speedup:.1f}×")
print(f"GPU utilization: {report.gpu_util_cold:.1f}% → {report.gpu_util_warm:.1f}%")
```

---

## 6. Performance Tuning

### Performance Profiles

The system includes three pre-configured performance profiles optimized for different use cases:

#### Balanced (Default)

- **Workers**: 12
- **GPU Memory**: 90%
- **Use Case**: Stable day-to-day processing
- **Throughput**: ~30-40 PDFs/hour
- **CPU**: 70-80%
- **Memory**: 120-140GB

```bash
python scripts/process_batch.py --profile balanced
```

#### Throughput

- **Workers**: 14
- **GPU Memory**: 95%
- **Use Case**: Maximum PDFs/hour
- **Throughput**: ~40-50 PDFs/hour
- **CPU**: 80-90%
- **Memory**: 140-160GB

```bash
python scripts/process_batch.py --profile throughput --workers 14
```

#### Latency

- **Workers**: 6
- **GPU Memory**: 85%
- **Use Case**: Fastest per-PDF processing
- **Throughput**: ~20-30 PDFs/hour
- **Per-PDF**: 1-2 minutes
- **CPU**: 40-50%
- **Memory**: 60-80GB

```bash
python scripts/process_batch.py --profile latency --workers 6
```

### Resource Monitoring

The batch processor automatically monitors and adjusts based on:

- **CPU Usage**: Pauses workers if CPU > 90%
- **Memory Usage**: Pauses workers if memory > 85%
- **GPU Memory**: Configured per profile
- **Stale Locks**: Cleaned automatically after 30 minutes

### Performance Metrics

With `--benchmark` flag, detailed metrics are collected:

```json
{
  "summary": {
    "total_pdfs": 100,
    "successful": 98,
    "failed": 2,
    "duration_seconds": 7200,
    "throughput_pdfs_per_hour": 49.0
  },
  "resource_utilization": {
    "gpu_util_avg": 94.5,
    "gpu_util_max": 98.2,
    "cpu_util_avg": 85.3,
    "memory_used_avg_gb": 145.2,
    "memory_used_peak_gb": 158.7
  },
  "timing": {
    "avg_pdf_duration_seconds": 72.0,
    "min_pdf_duration_seconds": 45.2,
    "max_pdf_duration_seconds": 180.5
  }
}
```

### Optimization Tips

**Maximize Throughput:**

- Use `throughput` profile
- Set workers to CPU cores (or cores - 2)
- Ensure adequate cooling for sustained GPU load
- Use fast SSD for input/output directories

**Minimize Per-PDF Latency:**

- Use `latency` profile
- Reduce worker count to 4-6
- Process smaller batches

**Optimize for Stability:**

- Use `balanced` profile
- Lower `--cpu-mem-threshold` to 0.85
- Lower `--mem-threshold` to 0.80
- Monitor `nvidia-smi` during first few PDFs

**Handle Large PDFs:**

- Increase `--retry-delay` to 20-30 seconds
- Increase `--max-retries` to 5
- Lower worker count temporarily
- Consider processing large PDFs separately

---

## 7. API Reference

### Configuration Classes

#### `BatchProcessorConfig`

```python
@dataclass
class BatchProcessorConfig:
    input_dir: Path              # Directory with input PDFs
    output_dir: Path             # Directory for outputs
    workers: int                 # Number of parallel workers
    max_retries: int = 3         # Max retry attempts
    retry_delay: float = 10.0    # Delay between retries (seconds)
    stale_lock_timeout: float = 1800.0  # 30 minutes
    mineru_backend: str = "pipeline"
    mineru_cli: str = "mineru"
    mineru_extra_args: List[str] = field(default_factory=list)
    env_overrides: Dict[str, str] = field(default_factory=dict)
    mem_threshold: float = 0.85
    cpu_pause_threshold: float = 0.90
    profile: str = "balanced"
    gpu_memory_utilization: float = 0.90
    # ... additional fields
```

#### `BatchSummary`

```python
@dataclass
class BatchSummary:
    total_count: int             # Total PDFs discovered
    success_count: int           # Successfully processed
    failure_count: int           # Permanently failed
    duration_seconds: float      # Total processing time
    start_time: float            # Unix timestamp
    end_time: float              # Unix timestamp
```

### Core Functions

#### `process_pdf(pdf_path: Path) -> Path`

Process a single PDF using MinerU with vLLM backend.

**Parameters:**

- `pdf_path`: Path to input PDF file

**Returns:**

- Path to output directory

**Example:**

```python
from pathlib import Path
from MinerUExperiment import process_pdf

output = process_pdf(Path("document.pdf"))
print(f"Output directory: {output}")
```

#### `load_config() -> MinerUConfig`

Load MinerU configuration from `~/mineru.json`.

**Returns:**

- Configuration object

**Example:**

```python
from MinerUExperiment import load_config

config = load_config()
print(f"Backend: {config.backend.name}")
print(f"Model path: {config.models_dir.vlm}")
```

#### `validate_pdf_processing(pdf_path: Path) -> ValidationReport`

Validate MinerU setup by processing a PDF twice (cold and warm).

**Parameters:**

- `pdf_path`: Path to test PDF

**Returns:**

- Validation report with timing and GPU metrics

**Example:**

```python
from pathlib import Path
from MinerUExperiment import validate_pdf_processing

report = validate_pdf_processing(Path("test.pdf"))
print(f"Speedup: {report.speedup:.1f}×")
```

#### `generate_structured_markdown(content_list_path: Path) -> Path`

Generate enhanced Markdown from MinerU's content_list.json.

**Parameters:**

- `content_list_path`: Path to `*_content_list.json` file

**Returns:**

- Path to generated `.structured.md` file

**Example:**

```python
from pathlib import Path
from MinerUExperiment.markdown_builder import generate_structured_markdown

output = generate_structured_markdown(
    Path("MDFilesCreated/document/content_list.json")
)
print(f"Generated: {output}")
```

### Batch Processor

#### `BatchProcessor.run() -> BatchSummary`

Run batch processing on all PDFs in input directory.

**Example:**

```python
from pathlib import Path
from MinerUExperiment.batch_processor import BatchProcessor, BatchProcessorConfig

config = BatchProcessorConfig(
    input_dir=Path("PDFsToProcess"),
    output_dir=Path("MDFilesCreated"),
    workers=12,
)

processor = BatchProcessor(config)
summary = processor.run()

print(f"Success: {summary.success_count}/{summary.total_count}")
print(f"Time: {summary.duration_seconds:.1f}s")
```

### Performance Profiles

#### `apply_profile_to_config(config: BatchProcessorConfig, profile_name: str) -> BatchProcessorConfig`

Apply a performance profile to configuration.

**Parameters:**

- `config`: Base configuration
- `profile_name`: One of `balanced`, `throughput`, `latency`

**Returns:**

- Modified configuration

**Example:**

```python
from MinerUExperiment.batch_processor import BatchProcessorConfig
from MinerUExperiment.performance_config import apply_profile_to_config

config = BatchProcessorConfig(...)
config = apply_profile_to_config(config, "throughput")
```

---

## 8. Troubleshooting

### Common Issues

#### GPU Not Detected

**Symptoms:**

- Error: `CUDA not available`
- Processing falls back to CPU (very slow)

**Solutions:**

```bash
# Check CUDA installation
python -c "import torch; print(torch.cuda.is_available())"
python -c "import torch; print(torch.cuda.get_device_name(0))"

# Verify NVIDIA driver
nvidia-smi

# Force specific GPU
export CUDA_VISIBLE_DEVICES=0

# Reinstall CUDA-enabled PyTorch
pip install --force-reinstall torch --index-url https://download.pytorch.org/whl/cu128
```

#### Out of Memory (GPU)

**Symptoms:**

- Error: `CUDA out of memory`
- Processing hangs or crashes

**Solutions:**

```bash
# Reduce GPU memory utilization
python scripts/process_batch.py \
  --env MINERU_VLLM_GPU_MEMORY_UTILIZATION=0.85

# Use latency profile (lower GPU usage)
python scripts/process_batch.py --profile latency

# Process PDFs sequentially
python scripts/process_batch.py --workers 1
```

#### Out of Memory (System RAM)

**Symptoms:**

- System becomes unresponsive
- `MemoryError` or kernel OOM killer

**Solutions:**

```bash
# Reduce worker count
python scripts/process_batch.py --workers 8

# Lower memory threshold
python scripts/process_batch.py --mem-threshold 0.75

# Use latency profile
python scripts/process_batch.py --profile latency --workers 6
```

#### LibGL Error

**Symptoms:**

- Error: `libGL.so.1: cannot open shared object file`

**Solution:**

```bash
sudo apt install -y libgl1-mesa-glx libglib2.0-0
```

#### Missing CJK Fonts

**Symptoms:**

- Chinese/Japanese/Korean text appears as boxes or missing

**Solution:**

```bash
sudo apt install -y fonts-noto-core fonts-noto-cjk
fc-cache -fv
```

#### Stale Lock Files

**Symptoms:**

- PDFs not processing even though no workers active
- `.pdf.lock` files remain after crash

**Solution:**

```bash
# Remove all lock files
find PDFsToProcess/ -name '*.pdf.lock' -delete
find PDFsToProcess/ -name '*.pdf.done' -delete

# Or let the system auto-clean (waits 30 minutes)
python scripts/process_batch.py  # Automatically cleans stale locks
```

#### Processing Hangs

**Symptoms:**

- Workers start but no progress
- Log shows: `Waiting for available PDF...`

**Solutions:**

```bash
# Check for stale locks
find PDFsToProcess/ -name '*.pdf.lock' -ls

# Verify PDFs exist
ls -lh PDFsToProcess/*.pdf

# Check worker logs for errors
tail -f logs/batch_processor.log  # If logging enabled
```

#### Poor Performance

**Symptoms:**

- Processing slower than expected
- GPU utilization < 70%

**Diagnostics:**

```bash
# Monitor GPU
nvidia-smi dmon -s pucvmet

# Check CPU usage
htop

# Enable benchmark mode
python scripts/process_batch.py --benchmark
```

**Solutions:**

- Increase worker count: `--workers 14`
- Use throughput profile: `--profile throughput`
- Verify GPU isn't thermal throttling: `nvidia-smi -q -d TEMPERATURE`
- Check for I/O bottlenecks: use SSD for input/output

---

## 9. Advanced Topics

### Custom Performance Profiles

Create custom profiles in Python:

```python
from MinerUExperiment.performance_config import PerformanceProfile, WorkerConfig, VLLMConfig

custom_profile = PerformanceProfile(
    name="custom_ultra",
    description="Ultra-aggressive settings",
    workers=WorkerConfig(
        worker_count=16,
        worker_memory_limit_gb=11.0,
        enable_cpu_affinity=True,
        worker_niceness=5,
    ),
    vllm=VLLMConfig(
        gpu_memory_utilization=0.98,
        tensor_parallel_size=1,
        max_model_len=16384,
        dtype="bfloat16",
    ),
)

# Apply to config
from MinerUExperiment.batch_processor import BatchProcessorConfig
config = BatchProcessorConfig(...)
config.profile = "custom_ultra"
config.workers = custom_profile.workers.worker_count
config.gpu_memory_utilization = custom_profile.vllm.gpu_memory_utilization
```

### CPU Affinity

Pin workers to specific CPU cores for better cache locality:

```python
from MinerUExperiment.batch_processor import BatchProcessorConfig

config = BatchProcessorConfig(
    ...,
    enable_cpu_affinity=True,
    cpu_affinity_plan={
        0: [0, 1],      # Worker 0 → cores 0-1
        1: [2, 3],      # Worker 1 → cores 2-3
        2: [4, 5],      # Worker 2 → cores 4-5
        # ... etc
    }
)
```

### Custom MinerU Configuration

Override MinerU settings via environment variables:

```python
from MinerUExperiment.batch_processor import BatchProcessorConfig

config = BatchProcessorConfig(
    ...,
    env_overrides={
        "MINERU_VLLM_GPU_MEMORY_UTILIZATION": "0.95",
        "MINERU_VLLM_BLOCK_SIZE": "32",
        "MINERU_VLLM_MAX_MODEL_LEN": "32768",
        "OMP_NUM_THREADS": "2",
        "MKL_NUM_THREADS": "2",
    }
)
```

### Metrics Collection

Collect detailed metrics programmatically:

```python
from MinerUExperiment.metrics import MetricsCollector

# Initialize collector
metrics = MetricsCollector()

# Start monitoring
metrics.start()

# ... process PDFs ...

# Stop and get summary
summary = metrics.stop()

print(f"GPU utilization: {summary['gpu_util_avg']:.1f}%")
print(f"Throughput: {summary['throughput_pdfs_per_hour']:.1f} PDFs/hour")
print(f"Memory peak: {summary['memory_peak_gb']:.1f}GB")

# Export to JSON
metrics.export_to_json(Path("metrics.json"))
```

### Multi-GPU Processing

For systems with multiple GPUs:

```bash
# Use GPUs 0 and 1
export CUDA_VISIBLE_DEVICES=0,1

python scripts/process_batch.py \
  --workers 28 \
  --env MINERU_VLLM_TENSOR_PARALLEL_SIZE=2
```

### Integration with RAG Systems

The structured Markdown is optimized for RAG pipelines:

```python
from pathlib import Path
from langchain.text_splitter import MarkdownHeaderTextSplitter

# Load structured Markdown
md_path = Path("MDFilesCreated/document/document.md")
content = md_path.read_text()

# Split by headers for RAG chunks
splitter = MarkdownHeaderTextSplitter(
    headers_to_split_on=[
        ("#", "title"),
        ("##", "section"),
        ("###", "subsection"),
    ]
)

chunks = splitter.split_text(content)
print(f"Created {len(chunks)} chunks for RAG")
```

### Automated Quality Assurance

Validate output quality:

```python
from pathlib import Path
from MinerUExperiment.validation import validate_markdown_structure

# Validate output
md_path = Path("MDFilesCreated/document/document.md")
report = validate_markdown_structure(md_path)

if report.has_errors:
    print(f"Quality issues: {report.errors}")
else:
    print(f"Quality: OK (score: {report.quality_score:.2f})")
```

---

## 10. Support

### Documentation

- **README**: [README.md](README.md) - Project overview
- **Quick Start**: [QUICKSTART.md](QUICKSTART.md) - Fast setup guide
- **Implementation Plan**: [IMPLEMENTATION_PLAN.md](IMPLEMENTATION_PLAN.md) - Development roadmap
- **Architecture**: [ARCHITECTURE.txt](ARCHITECTURE.txt) - System design
- **OpenSpec**: `openspec/` directory - Detailed specifications

### Getting Help

**Issues and Bugs:**

- GitHub Issues: <https://github.com/paul-heyse/MinerUExperiment/issues>
- Include: Python version, GPU model, error messages, logs

**Feature Requests:**

- Open a GitHub issue with `[Feature Request]` prefix
- Describe use case and expected behavior

**Community:**

- MinerU GitHub: <https://github.com/opendatalab/MinerU>
- vLLM GitHub: <https://github.com/vllm-project/vllm>

### Logging

Enable detailed logging:

```python
import logging

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('batch_processor.log'),
        logging.StreamHandler()
    ]
)
```

Or via CLI:

```bash
python scripts/process_batch.py --verbose
```

### Performance Expectations

With RTX 5090 (24GB), 192GB RAM, AMD 9950x (16 cores):

| Profile    | Workers | PDFs/hour | GPU Util | CPU Util | Memory  | Per-PDF Avg |
|------------|---------|-----------|----------|----------|---------|-------------|
| Latency    | 6       | 20-30     | 85%      | 40-50%   | 60-80GB | 1-2 min     |
| Balanced   | 12      | 30-40     | 90%      | 70-80%   | 120-140GB | 1.5-3 min |
| Throughput | 14      | 40-50     | 95%      | 80-90%   | 140-160GB | 1-2 min   |

*Note: Actual performance varies significantly based on PDF complexity (pages, images, tables, equations).*

### License

This project uses MinerU which is licensed under **AGPL-3.0**. Any derivative works must also be open-sourced under AGPL-3.0 or a compatible license.

See [LICENSE](LICENSE) for details.

### Acknowledgments

- **MinerU Team** (opendatalab) - Core PDF extraction framework
- **vLLM Team** - High-performance LLM inference engine
- **HuggingFace** - Model hosting and distribution

---

## Appendix: Command Reference

### Batch Processing

```bash
# Basic
python scripts/process_batch.py

# With options
python scripts/process_batch.py \
  --input-dir PDFsToProcess \
  --output-dir MDFilesCreated \
  --workers 14 \
  --profile throughput \
  --benchmark

# Help
python scripts/process_batch.py --help
```

### Post-Processing

```bash
# Basic
python scripts/postprocess_markdown.py MDFilesCreated

# Verbose
python scripts/postprocess_markdown.py MDFilesCreated --verbose

# Help
python scripts/postprocess_markdown.py --help
```

### Validation

```bash
# Validate setup
python scripts/validate_mineru_setup.py PDFsToProcess/sample.pdf

# Validate with custom output
python scripts/validate_mineru_setup.py \
  PDFsToProcess/sample.pdf \
  --output-dir /tmp/test_output
```

### OpenSpec

```bash
# List changes
openspec list

# Show change details
openspec show add-batch-processing-system

# Validate changes
openspec validate --strict

# View specs
openspec list --specs
```

---

**Version**: 1.0
**Last Updated**: October 2025
**Author**: Paul Heyse / MinerUExperiment Team
