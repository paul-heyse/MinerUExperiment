# Project Context

## Purpose

MinerUExperiment is a high-performance PDF-to-Markdown conversion system leveraging MinerU with vLLM acceleration to transform complex PDFs into structured, RAG-ready Markdown with clear section hierarchies.

## Tech Stack

- **Python 3.10+** - Core language
- **MinerU 2.5+** - PDF extraction and VLM processing
- **vLLM** - GPU-accelerated model inference (MinerU2.5-2509-1.2B)
- **PyTorch** - Deep learning framework
- **CUDA 12.8+** - GPU acceleration

## Project Conventions

### Code Style

- Python: Follow PEP 8 with type hints
- Use pathlib.Path for file operations
- Async/await for I/O-bound operations where beneficial
- Descriptive variable names (e.g., `pdf_path`, `worker_id`, `content_blocks`)

### Architecture Patterns

- Single-responsibility modules
- Configuration via environment variables and JSON config files
- Graceful error handling with logging
- Worker coordination via file-based locking or queue system
- Post-processing pipeline pattern for Markdown enhancement

### Testing Strategy

- Unit tests for transformation logic
- Integration tests with sample PDFs
- Performance benchmarks for batch processing
- Validation of output structure and content

### Git Workflow

- Feature branches for new capabilities
- Descriptive commit messages
- Test before commit

## Domain Context

- **PDF Processing**: Complex documents with tables, equations, images, multi-column layouts
- **VLM (Vision-Language Model)**: Understands document structure and layout
- **text_level**: MinerU's heading hierarchy indicator (1=H1, 2=H2, etc.)
- **content_list.json**: Structured, reading-order content blocks from MinerU
- **RAG (Retrieval-Augmented Generation)**: Requires clear chunk boundaries via headings

## Important Constraints

- AGPL-3.0 license (MinerU)
- GPU memory limits (24GB on RTX 5090)
- System has ample resources: 192GB RAM, 16-core AMD 9950x
- Linux environment (Ubuntu 22.04+)

## External Dependencies

- MinerU with vLLM extension: `mineru[all]`
- HuggingFace model hub for opendatalab/MinerU2.5-2509-1.2B
- CUDA drivers supporting CUDA 12.8+
- System packages: libgl1-mesa-glx, fonts-noto-cjk
