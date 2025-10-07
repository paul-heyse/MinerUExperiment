# Proposal: Enhanced Markdown Generation with Clear Section Hierarchies

## Why

MinerU's default Markdown output doesn't always provide optimal section hierarchies for RAG chunking. By post-processing the `content_list.json` file (which contains structured content blocks with `text_level` indicators), we can rebuild Markdown with crystal-clear section boundaries, proper heading levels, preserved tables (HTML), equations (LaTeX), and inline image captions. This produces "RAG-ready" Markdown ideal for semantic chunking.

## What Changes

- Implement post-processing module to rebuild Markdown from `content_list.json`
- Map `text_level` (1-6) to Markdown heading levels (#, ##, ###, etc.)
- Preserve tables as HTML blocks for structural fidelity
- Preserve equations as LaTeX ($$...$$)
- Inline image captions with image references
- Normalize excessive blank lines
- Generate `.structured.md` files alongside original outputs
- Integrate post-processing into batch workflow

## Impact

- Affected specs: **markdown-generation** (new capability)
- Affected code:
  - New module: `src/MinerUExperiment/markdown_builder.py` - content_list.json → Markdown transformation
  - Modified: `src/MinerUExperiment/batch_processor.py` - integrate post-processing step
  - New script: `scripts/postprocess_markdown.py` - standalone post-processing utility
- Dependencies: Requires **mineru-integration** capability; integrates with **batch-processing**
