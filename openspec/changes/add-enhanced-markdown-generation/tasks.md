# Implementation Tasks

## 1. Markdown Builder Module

- [x] 1.1 Create `src/MinerUExperiment/markdown_builder.py`
- [x] 1.2 Implement function to load content_list.json
- [x] 1.3 Implement blocks_to_markdown() converter
- [x] 1.4 Add text_level to heading mapper (1→#, 2→##, etc.)
- [x] 1.5 Add table block handler (preserve HTML)
- [x] 1.6 Add equation block handler (LaTeX with $$ wrapper)
- [x] 1.7 Add image block handler (Markdown image + caption)
- [x] 1.8 Add other content type handlers (code, list, etc.)
- [x] 1.9 Implement blank line normalization (max 2 consecutive)

## 2. Content Type Handling

- [x] 2.1 Handle "text" blocks with text_level for headings
- [x] 2.2 Handle "text" blocks without text_level as paragraphs
- [x] 2.3 Handle "table" blocks (extract HTML from html or content field)
- [x] 2.4 Handle "equation" blocks (extract LaTeX from content or latex field)
- [x] 2.5 Handle "image" blocks (extract img_path and img_caption)
- [x] 2.6 Handle "image_caption" blocks (if separate from image)
- [x] 2.7 Handle "code" and "algorithm" blocks
- [x] 2.8 Handle "list" blocks
- [x] 2.9 Gracefully handle unknown block types

## 3. Output File Generation

- [x] 3.1 Generate .structured.md filename from content_list.json path
- [x] 3.2 Write UTF-8 encoded Markdown file
- [x] 3.3 Preserve original Markdown for comparison
- [x] 3.4 Log successful generation with file paths

## 4. Standalone CLI Script

- [x] 4.1 Create `scripts/postprocess_markdown.py`
- [x] 4.2 Accept directory path as argument
- [x] 4.3 Recursively find all *_content_list.json files
- [x] 4.4 Process each file and generate .structured.md
- [x] 4.5 Display progress and summary

## 5. Integration with Batch Processing

- [x] 5.1 Modify batch_processor.py to call markdown post-processing
- [x] 5.2 Run post-processing after MinerU completes for each PDF
- [x] 5.3 Ensure .structured.md files are saved to MDFilesCreated
- [x] 5.4 Handle post-processing errors without failing entire batch

## 6. Testing

- [x] 6.1 Test with sample content_list.json files
- [x] 6.2 Verify heading hierarchy is correct
- [x] 6.3 Verify tables are preserved as HTML
- [x] 6.4 Verify equations are properly wrapped in $$
- [x] 6.5 Verify images and captions are inline
- [x] 6.6 Compare with original Markdown for quality
- [x] 6.7 Test with edge cases (no headings, all tables, etc.)
