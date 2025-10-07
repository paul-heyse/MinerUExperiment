# OpenSpec Change Proposals Summary

## Overview

This document summarizes the comprehensive change proposals created for implementing MinerU with vLLM for high-performance PDF-to-Markdown batch processing.

## System Specifications

- **GPU**: RTX 5090 (24GB VRAM)
- **CPU**: AMD 9950x (16 cores)
- **RAM**: 192GB
- **OS**: Linux (Ubuntu 22.04+)

## Change Proposals Created

### 1. add-mineru-vllm-integration

**Status**: Validated ✓
**Purpose**: Core MinerU integration with vLLM acceleration

**Key Features**:

- MinerU installation with vLLM extension (`mineru[all]`)
- GPU configuration and warmup routine
- MinerU2.5-2509-1.2B model download and configuration
- Configuration management via `~/mineru.json`
- Force GPU usage for all processing
- Single-PDF processing capability

**New Modules**:

- `src/MinerUExperiment/mineru_config.py` - configuration management
- `src/MinerUExperiment/gpu_utils.py` - GPU warmup and utilities

**Spec**: `openspec/changes/add-mineru-vllm-integration/specs/mineru-integration/spec.md`

---

### 2. add-batch-processing-system

**Status**: Validated ✓
**Purpose**: Parallel batch processing with worker coordination

**Key Features**:

- Batch orchestrator for PDFsToProcess directory
- Multi-worker parallel processing (12-14 workers recommended)
- File-based worker coordination (lock files)
- Progress tracking and status reporting
- Error recovery and retry logic (max 3 retries)
- Graceful shutdown handling
- Output to MDFilesCreated directory
- Resource monitoring (CPU, memory)

**New Modules**:

- `src/MinerUExperiment/batch_processor.py` - orchestration and worker management
- `src/MinerUExperiment/worker_coordinator.py` - coordination logic
- `scripts/process_batch.py` - CLI entry point

**CLI Example**:

```bash
python scripts/process_batch.py --workers 14 --input-dir PDFsToProcess --output-dir MDFilesCreated
```

**Spec**: `openspec/changes/add-batch-processing-system/specs/batch-processing/spec.md`

---

### 3. add-enhanced-markdown-generation

**Status**: Validated ✓
**Purpose**: Post-process content_list.json into structured, RAG-ready Markdown

**Key Features**:

- Rebuild Markdown from `content_list.json` with clear section hierarchies
- Map `text_level` (1-6) to Markdown headings (#, ##, ###, etc.)
- Preserve tables as HTML blocks
- Preserve equations as LaTeX ($$...$$)
- Inline image captions
- Normalize excessive blank lines
- Generate `.structured.md` files
- Integrate with batch workflow
- Standalone CLI for post-processing existing outputs

**New Modules**:

- `src/MinerUExperiment/markdown_builder.py` - content_list.json → Markdown transformation
- `scripts/postprocess_markdown.py` - standalone post-processing utility

**CLI Example**:

```bash
python scripts/postprocess_markdown.py MDFilesCreated
```

**Spec**: `openspec/changes/add-enhanced-markdown-generation/specs/markdown-generation/spec.md`

---

### 4. add-performance-optimization

**Status**: Validated ✓
**Purpose**: Maximize performance on high-end hardware

**Key Features**:

- vLLM engine tuning for RTX 5090 (95% GPU memory utilization)
- Optimal worker count for AMD 9950x (12-14 workers)
- CPU affinity and thread pinning
- Memory optimization for 192GB RAM (10-12GB per worker)
- Three performance profiles: throughput, balanced, latency
- Performance metrics collection (GPU utilization, throughput, etc.)
- I/O optimization (large buffers, async operations)
- Benchmarking support

**New Modules**:

- `src/MinerUExperiment/performance_config.py` - performance tuning profiles
- `src/MinerUExperiment/metrics.py` - metrics collection

**Performance Profiles**:

- **Throughput**: 14 workers, 95% GPU memory, aggressive batching
- **Balanced**: 12 workers, 90% GPU memory (default)
- **Latency**: 4-6 workers, optimized for per-PDF speed

**CLI Example**:

```bash
python scripts/process_batch.py --profile throughput --workers 14
```

**Spec**: `openspec/changes/add-performance-optimization/specs/performance-optimization/spec.md`

---

## Implementation Order (Recommended)

1. **add-mineru-vllm-integration** (foundation)
   - Install dependencies
   - Configure GPU and model
   - Test single-PDF processing

2. **add-batch-processing-system** (parallelization)
   - Implement worker coordination
   - Test with multiple PDFs
   - Verify no duplicate processing

3. **add-enhanced-markdown-generation** (quality)
   - Implement content_list.json parser
   - Test structured Markdown output
   - Integrate with batch processor

4. **add-performance-optimization** (efficiency)
   - Apply vLLM tuning
   - Configure performance profiles
   - Benchmark and validate

## Dependencies Between Changes

```
add-mineru-vllm-integration (base)
    ↓
    ├── add-batch-processing-system (requires mineru-integration)
    │       ↓
    │       └── add-enhanced-markdown-generation (integrates with batch-processing)
    │
    └── add-performance-optimization (optimizes mineru-integration and batch-processing)
```

## Validation Status

All change proposals have been validated with `openspec validate --strict`:

```bash
✓ add-mineru-vllm-integration is valid
✓ add-batch-processing-system is valid
✓ add-enhanced-markdown-generation is valid
✓ add-performance-optimization is valid
```

## Next Steps

1. **Review Proposals**: Review each proposal in detail:
   - `openspec show add-mineru-vllm-integration`
   - `openspec show add-batch-processing-system`
   - `openspec show add-enhanced-markdown-generation`
   - `openspec show add-performance-optimization`

2. **Approve Proposals**: Once reviewed, provide approval to begin implementation

3. **Implementation**: Implement changes sequentially following tasks.md in each change directory

4. **Testing**: Test each capability with sample PDFs before moving to next change

5. **Archiving**: After deployment, archive changes using `openspec archive [change-id]`

## Directory Structure

```
MinerUExperiment/
├── PDFsToProcess/          # Input directory for PDFs
├── MDFilesCreated/         # Output directory for Markdown files
├── openspec/
│   ├── project.md          # Project conventions
│   └── changes/
│       ├── add-mineru-vllm-integration/
│       │   ├── proposal.md
│       │   ├── tasks.md
│       │   └── specs/mineru-integration/spec.md
│       ├── add-batch-processing-system/
│       │   ├── proposal.md
│       │   ├── tasks.md
│       │   └── specs/batch-processing/spec.md
│       ├── add-enhanced-markdown-generation/
│       │   ├── proposal.md
│       │   ├── tasks.md
│       │   └── specs/markdown-generation/spec.md
│       └── add-performance-optimization/
│           ├── proposal.md
│           ├── tasks.md
│           └── specs/performance-optimization/spec.md
└── src/
    └── MinerUExperiment/
        └── __init__.py
```

## Expected Capabilities After Implementation

Once all changes are implemented, the system will:

1. ✓ Install and configure MinerU with vLLM
2. ✓ Warm up GPU for optimal performance
3. ✓ Process all PDFs in PDFsToProcess directory
4. ✓ Use 12-14 parallel workers coordinated to prevent duplicate work
5. ✓ Generate high-quality Markdown with clear section hierarchies
6. ✓ Preserve tables (HTML), equations (LaTeX), and images with captions
7. ✓ Output all files to MDFilesCreated directory
8. ✓ Maximize utilization of RTX 5090, 192GB RAM, and AMD 9950x
9. ✓ Collect and report performance metrics
10. ✓ Support multiple performance profiles for different use cases

## Questions or Modifications?

If you need to modify any proposals before implementation:

- Review specific specs: `openspec show [change-id]`
- Review spec differences: `openspec diff [change-id]`
- Edit proposal files directly and re-validate

For questions about the implementation approach, refer to:

- `openspec/AGENTS.md` for workflow guidance
- `Guidance.md` for MinerU-specific technical details
