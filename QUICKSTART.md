# Quick Start Guide - MinerU with vLLM

This guide provides quick commands to get started after proposals are approved.

## Prerequisites Check

```bash
# Verify Python version (3.10-3.13)
python --version

# Check CUDA availability
nvidia-smi

# Verify GPU (should show RTX 5090)
nvidia-smi --query-gpu=name,memory.total --format=csv

# Check system resources
lscpu | grep -E "^CPU\(s\)|Model name"
free -h
```

## Installation (Change 1)

```bash
# Install system dependencies
sudo apt update
sudo apt install -y libgl1-mesa-glx fonts-noto-core fonts-noto-cjk
fc-cache -fv

# Install MinerU with vLLM
pip install --upgrade pip
pip install uv
uv pip install -U "mineru[all]"

# Download models
export MINERU_MODEL_SOURCE=huggingface  # or modelscope
mineru-models-download

# Verify installation
mineru --version
which mineru
```

## Test Single PDF (Change 1)

```bash
# Create test directories
mkdir -p PDFsToProcess MDFilesCreated

# Test with a single PDF (replace with your PDF)
export CUDA_VISIBLE_DEVICES=0
mineru -p PDFsToProcess/test.pdf -o MDFilesCreated -b vlm-vllm-engine

# Check outputs
ls -lh MDFilesCreated/
```

## Batch Processing (Change 2)

```bash
# Process all PDFs in PDFsToProcess with 14 workers
python scripts/process_batch.py --workers 14

# Custom directories
python scripts/process_batch.py \
  --workers 14 \
  --input-dir /path/to/pdfs \
  --output-dir /path/to/output

# View progress
# The script will show live progress updates
```

## Enhanced Markdown Generation (Change 3)

```bash
# Post-process all outputs to create .structured.md files
python scripts/postprocess_markdown.py MDFilesCreated

# Post-process specific directory
python scripts/postprocess_markdown.py /path/to/outputs
```

## Performance Profiles (Change 4)

```bash
# Throughput profile (max speed, 14 workers)
python scripts/process_batch.py --profile throughput --workers 14

# Balanced profile (default, safe)
python scripts/process_batch.py --profile balanced --workers 12

# Latency profile (fast per-PDF)
python scripts/process_batch.py --profile latency --workers 6

# Benchmark mode
python scripts/process_batch.py --benchmark --workers 14
```

## Monitor Performance

```bash
# In another terminal while processing:

# Monitor GPU utilization
watch -n 1 nvidia-smi

# Monitor CPU and memory
htop

# Monitor detailed GPU stats
nvidia-smi dmon -s pucvmet

# Check performance report (after completion)
cat MDFilesCreated/performance_report.json | jq
```

## Expected Performance

With your hardware (RTX 5090, 192GB RAM, AMD 9950x):

- **Throughput Profile**: ~20-50 PDFs/hour (depends on PDF complexity)
- **GPU Utilization**: 90-95%
- **CPU Utilization**: 80-90% (with 14 workers)
- **Memory Usage**: 100-150GB (out of 192GB)
- **Per-PDF Processing**: 1-5 minutes (average)

## Troubleshooting

### GPU not detected

```bash
# Check CUDA
python -c "import torch; print(torch.cuda.is_available())"
python -c "import torch; print(torch.cuda.get_device_name(0))"

# Force GPU device
export CUDA_VISIBLE_DEVICES=0
```

### Out of memory

```bash
# Reduce workers
python scripts/process_batch.py --workers 8

# Use balanced profile
python scripts/process_batch.py --profile balanced
```

### LibGL error

```bash
sudo apt install -y libgl1-mesa-glx
```

### Missing CJK text

```bash
sudo apt install -y fonts-noto-core fonts-noto-cjk
fc-cache -fv
```

## Verify Outputs

```bash
# Check Markdown files
ls -lh MDFilesCreated/*.md

# Check structured Markdown
ls -lh MDFilesCreated/*.structured.md

# Check intermediate JSON files
ls -lh MDFilesCreated/*_content_list.json
ls -lh MDFilesCreated/*_middle.json

# View a sample structured Markdown
head -50 MDFilesCreated/sample.structured.md
```

## Development Workflow

1. **Add PDFs**: Place PDFs in `PDFsToProcess/`
2. **Run Batch**: `python scripts/process_batch.py --workers 14`
3. **Check Outputs**: Review `MDFilesCreated/` for .structured.md files
4. **Review Quality**: Check section hierarchies, tables, equations
5. **Iterate**: Adjust performance profile if needed

## Configuration Files

- `~/mineru.json` - MinerU configuration (model paths, backend settings)
- `src/MinerUExperiment/performance_config.py` - Performance profiles
- `openspec/project.md` - Project conventions

## Getting Help

```bash
# View CLI help
python scripts/process_batch.py --help
python scripts/postprocess_markdown.py --help

# View OpenSpec proposals
openspec list
openspec show add-mineru-vllm-integration
openspec show add-batch-processing-system
openspec show add-enhanced-markdown-generation
openspec show add-performance-optimization

# View detailed specs
cat openspec/changes/add-mineru-vllm-integration/specs/mineru-integration/spec.md
```

## Next Steps

1. Review proposals: `openspec list`
2. Approve proposals (confirm you're ready to implement)
3. Implement Change 1: MinerU integration
4. Test single PDF processing
5. Implement Change 2: Batch processing
6. Test with 5-10 PDFs
7. Implement Change 3: Enhanced Markdown
8. Verify .structured.md quality
9. Implement Change 4: Performance optimization
10. Benchmark and tune settings

Refer to `IMPLEMENTATION_PLAN.md` for comprehensive details.
