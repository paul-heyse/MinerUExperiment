Below is a **from‑scratch, end‑to‑end recipe** to run **MinerU with vLLM** and convert a **folder of PDFs** into **high‑quality, sectioned Markdown** that’s ideal for chunking in RAG/LLM systems. I’ve included system prerequisites, environment setup, the two common deployment patterns (embedded vLLM engine vs. HTTP server), and robust post‑processing code that converts MinerU’s structured JSON into “RAG‑ready” Markdown with clear, hierarchical sections.

---

## 0) What you’ll build

* **Input:** a folder of PDFs (`/data/pdfs`)
* **Engine:** MinerU **VLM** backend accelerated by **vLLM** running the **MinerU2.5-2509-1.2B** model
* **Output:** per‑PDF Markdown files with clear `#`/`##`/`###` headings, preserved tables (HTML), equations (LaTeX), images (links + captions), plus intermediate JSON for auditing and optional chunking.

MinerU is designed exactly for PDF→Markdown/JSON with structure (titles, paragraphs, lists, equations, tables, images, captions, etc.), and its VLM backend can be accelerated via vLLM for 20–30× speedups vs. a plain Transformers loop. ([opendatalab.github.io][1])

---

## 1) Hardware & OS preflight

**Recommended (for vLLM acceleration):**

* **OS:** Linux (Ubuntu 22.04+). vLLM also works in Windows **via WSL2**; macOS is fine for the non‑VLM “pipeline” backend but **not** for CUDA‑accelerated vLLM. ([GitHub][2])
* **GPU:** NVIDIA with **compute capability ≥ 7.0** (e.g., T4/V100/RTX20xx/A100/H100/L4) and **≥ 8 GB VRAM**. ([ploomber.io][3])
* **Drivers/CUDA:** vLLM supports CUDA 11.8+; newer hardware (e.g., Blackwell) requires **CUDA ≥ 12.8**. If you use MinerU’s Docker image path below, target **driver supporting CUDA ≥ 12.8**. ([VLLM Documentation][4])
* **Python:** **3.10–3.13** (MinerU supports this range). ([GitHub][2])
* **RAM:** 16 GB+ (32 GB recommended). ([GitHub][2])

> **Note on Windows:** For the **vlm‑vllm** backend MinerU lists Windows “via WSL2”, while the CPU‑capable **pipeline** backend supports Windows natively. ([GitHub][2])

---

## 2) System packages you should install (Ubuntu/Debian)

These eliminate the most common runtime issues:

```bash
sudo apt update
# OpenGL headless lib (fixes: ImportError: libGL.so.1)
sudo apt install -y libgl1-mesa-glx
# CJK fonts to prevent text dropouts when rendering (pypdfium2)
sudo apt install -y fonts-noto-core fonts-noto-cjk
fc-cache -fv
```

MinerU’s FAQ calls out both **libGL** on Ubuntu/WSL2 and missing CJK fonts when using **pypdfium2** as the renderer. ([opendatalab.github.io][5])

---

## 3) Install MinerU + vLLM

MinerU ships “extras” that include vLLM integration. The project recommends `uv` (a fast pip). Either:

```bash
# 3a) Install uv and MinerU with the vLLM-enabled extra
pip install --upgrade pip
pip install uv
uv pip install -U "mineru[all]"
```

> `mineru[all]` == `mineru[core,vllm]`. Use this on **Linux** (or WSL2) for vLLM acceleration; `mineru[core]` excludes vLLM and works on Linux/Windows/macOS. ([opendatalab.github.io][6])

Or install from source with the same extras:

```bash
git clone https://github.com/opendatalab/MinerU.git
cd MinerU
uv pip install -e .[all]
```

(“Local deployment” guidance with supported OS/GPU/Python ranges is in the repo README.) ([GitHub][2])

---

## 4) Model download & source selection

MinerU uses Hugging Face by default; you can switch via env var:

```bash
# Optional: switch model hub (mainland China often prefers ModelScope)
export MINERU_MODEL_SOURCE=modelscope
```

You can also pre‑download all models interactively:

```bash
mineru-models-download
```

The tool writes model paths into `~/mineru.json` automatically after downloads. ([opendatalab.github.io][7])

---

## 5) Two ways to run MinerU with vLLM

### Option A — **Embedded vLLM engine inside MinerU** (simplest)

This runs vLLM inside the MinerU process. If you have multiple GPUs, you can pass through vLLM parameters such as `--data-parallel-size`.

```bash
# Use only GPU 0 (optional)
CUDA_VISIBLE_DEVICES=0 \
mineru \
  -p /data/pdfs \
  -o /data/out \
  -b vlm-vllm-engine \
  --data-parallel-size 1
```

MinerU forwards **official vLLM flags** to its commands (`mineru`, `mineru-gradio`, `mineru-api`, `mineru-vllm-server`). For multi‑GPU speedups: `--data-parallel-size N`. To select GPUs, use `CUDA_VISIBLE_DEVICES`. ([opendatalab.github.io][8])

### Option B — **Decoupled vLLM HTTP server** + MinerU HTTP client (scales well)

1. Serve the **MinerU2.5-2509-1.2B** model directly via vLLM:

```bash
# vLLM ≥ 0.10.1 (preferred): add MinerU logits processor to support no_repeat_ngram_size
vllm serve opendatalab/MinerU2.5-2509-1.2B \
  --host 127.0.0.1 --port 8000 \
  --logits-processors mineru_vl_utils:MinerULogitsProcessor
```

(If you’re on vLLM < 0.10.1, omit the logits processor.) ([GitHub][9])

2. Point MinerU at that server:

```bash
mineru \
  -p /data/pdfs \
  -o /data/out \
  -b vlm-http-client \
  -u http://127.0.0.1:8000
```

> The `mineru-vl-utils` package provides the logits processor and also a programmatic client if you need one. ([GitHub][9])

---

## 6) Docker (optional, but avoids many CUDA/env headaches)

MinerU provides a Docker path built on `vllm/vllm-openai`. The docs recommend updating the base to **v0.10.2** if you need better support on Turing‑era GPUs, and your **host driver must support CUDA ≥ 12.8**. ([opendatalab.github.io][10])

```bash
# Build
wget https://gcore.jsdelivr.net/gh/opendatalab/MinerU@master/docker/global/Dockerfile
docker build -t mineru-vllm:latest -f Dockerfile .

# Run with GPU access and ports for vLLM (30000), Gradio (7860), and API (8000)
docker run --gpus all --shm-size 32g --ipc=host \
  -p 30000:30000 -p 7860:7860 -p 8000:8000 \
  -v /data:/data \
  -it mineru-vllm:latest /bin/bash
```

You can also `docker compose` profiles to start `vllm-server`, the web API or Gradio UI directly. ([opendatalab.github.io][10])

---

## 7) Running the conversion (folder → structured outputs)

MinerU’s CLI accepts either a **file** or a **directory** for `-p`. Use the VLM backend with vLLM acceleration:

```bash
# Embedded vLLM engine (Option A)
mineru -p /data/pdfs -o /data/out -b vlm-vllm-engine

# Or if you’re using the separate vLLM server (Option B)
mineru -p /data/pdfs -o /data/out -b vlm-http-client -u http://127.0.0.1:8000
```

Key flags (from `mineru --help`):

* `-p/--path`: input file or directory
* `-o/--output`: output directory
* `-b/--backend`: `pipeline | vlm-transformers | vlm-vllm-engine | vlm-http-client`
* `-s/--start`, `-e/--end`: page range
* `-f/--formula`, `-t/--table`: enable/disable equation/table parsing

You can also set env vars like `MINERU_MODEL_SOURCE` and control GPUs via `CUDA_VISIBLE_DEVICES`. ([opendatalab.github.io][11])

**What MinerU writes:** in addition to a Markdown file, it writes rich structured files, notably **`*_content_list.json`** (flat, reading‑order content blocks with `text_level` for headings) and **`*_middle.json`** (detailed layout tree). These are ideal for post‑processing or QA. ([opendatalab.github.io][12])

* In `content_list.json`, **`text_level`** is set as:

  * `1` → H1, `2` → H2, … (no `text_level` or `0` means body text). ([opendatalab.github.io][12])
* VLM output also labels block types (title, text, image, image_caption, table, equation, code, etc.). ([opendatalab.github.io][12])

---

## 8) (Optional but recommended) Improve heading hierarchy via LLM‑aided titles

MinerU supports an **LLM‑assisted “title hierarchy”** step (OpenAI‑protocol compatible). Enable it in your user config `~/mineru.json`:

* Set `llm-aided-config.enable = true`
* Provide `base_url`, `api_key`, and a strong instruction‑tuned model name

MinerU’s docs describe the `llm-aided-config` block and note it defaults to an Alibaba Cloud Bailian model if enabled. You can point it to your own OpenAI‑compatible endpoint (including a **local vLLM server** running a text‑only instruct model) to strengthen heading classification. ([opendatalab.github.io][13])

> If you don’t enable this, MinerU still produces `text_level` from the VLM; LLM‑aided refinement just makes headings more consistent on messy PDFs.

---

## 9) Post‑process into “RAG‑ready Markdown” with crystal‑clear sections

MinerU’s default Markdown is good, but for **consistent chunk boundaries** you often want to rebuild Markdown from `content_list.json`, mapping `text_level` → `#`/`##`/`###` headings, keeping tables as HTML, and inlining image captions.

Create `postprocess_mineru_content.py` and run it after MinerU:

```python
#!/usr/bin/env python3
"""
Rebuilds clean, hierarchical Markdown from MinerU *_content_list.json.
- Maps text_level -> # headings
- Preserves tables (HTML) and equations (LaTeX as-is)
- Inlines image captions
- Writes <docname>.structured.md next to original outputs
"""

import json, sys, re
from pathlib import Path

def blocks_to_markdown(blocks):
    lines = []
    def heading_prefix(level: int) -> str:
        level = max(1, min(level, 6))
        return "#" * level

    for b in blocks:
        t = b.get("type")
        if t == "text":
            txt = (b.get("text") or "").strip()
            lvl = b.get("text_level", 0)
            if lvl and txt:
                lines.append(f"{heading_prefix(lvl)} {txt}\n")
            elif txt:
                lines.append(f"{txt}\n")
        elif t == "table":
            html = (b.get("html") or b.get("content") or "").strip()
            # Leave table as HTML block for faithful structure
            if html:
                lines.append(html + "\n")
        elif t == "equation":
            # MinerU provides LaTeX; wrap in display math for clarity
            latex = (b.get("content") or b.get("latex") or "").strip()
            if latex:
                # Ensure not double-wrapped
                if not (latex.startswith("$$") and latex.endswith("$$")):
                    latex = f"$$\n{latex}\n$$"
                lines.append(latex + "\n")
        elif t == "image":
            # Save image path + optional caption
            path = b.get("img_path") or ""
            cap_list = b.get("img_caption") or []
            caption = " ".join([c.strip() for c in cap_list if isinstance(c, str)]).strip()
            if path:
                lines.append(f"![{caption}]({path})\n" if caption else f"![]({path})\n")
                if caption:
                    lines.append(f"*{caption}*\n")
        else:
            # Other recognized types (code, algorithm, list, ref_text...) if present
            content = b.get("content")
            if content:
                lines.append(f"{content}\n")

    # Normalize excessive blank lines
    md = "\n".join(lines)
    md = re.sub(r"\n{3,}", "\n\n", md).strip() + "\n"
    return md

def process_content_list(content_json_path: Path):
    with open(content_json_path, "r", encoding="utf-8") as f:
        blocks = json.load(f)
    md = blocks_to_markdown(blocks)
    out_path = content_json_path.with_suffix("").with_name(
        content_json_path.name.replace("_content_list", "")
    )
    out_path = out_path.with_suffix(".structured.md")
    out_path.write_text(md, encoding="utf-8")
    print(f"Wrote {out_path}")

def main(root: Path):
    jsons = list(root.glob("**/*_content_list.json"))
    if not jsons:
        print(f"No *_content_list.json found under {root}")
        return
    for j in jsons:
        process_content_list(j)

if __name__ == "__main__":
    base = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(".")
    main(base)
```

* Run it like:

  ```bash
  python postprocess_mineru_content.py /data/out
  ```

* This creates `*.structured.md` files with **consistent heading levels** driven by `text_level`, which MinerU documents explicitly for chunking‑friendly structure. ([opendatalab.github.io][12])

---

## 10) (Optional) Chunking logic (heading‑aware)

If you want ready‑to‑index chunks (JSONL), split your `*.structured.md` on headings and cap chunk sizes (e.g., ~1–2k tokens). A simple heuristic is to split on `^#{1,3}\s` boundaries and then re‑merge small sections. (MinerU’s own `content_list.json` also carries **page numbers** and **bbox**—useful to retain page provenance metadata.) ([opendatalab.github.io][12])

---

## 11) Quality control artifacts

MinerU writes visual QA files:

* `*_layout.pdf` shows **detected blocks** with reading order numbers.
* `*_spans.pdf` (pipeline backend) shows **span types** per page.

These are invaluable to debug mis‑ordered columns, missed spans, etc. ([opendatalab.github.io][12])

---

## 12) Troubleshooting & tips

* **WSL2 + “libGL.so.1 not found”** → `sudo apt install libgl1-mesa-glx`. ([opendatalab.github.io][5])
* **CJK text missing** → install Noto fonts (`fonts-noto-core fonts-noto-cjk`). ([opendatalab.github.io][5])
* **Selecting GPUs** → `CUDA_VISIBLE_DEVICES=0,1` before your command; MinerU passes this through for both embedded vLLM and server modes. ([opendatalab.github.io][8])
* **Throughput on multi‑GPU** → add `--data-parallel-size N` to MinerU’s vLLM commands (engine or server). ([opendatalab.github.io][8])
* **Model choice** → **opendatalab/MinerU2.5-2509-1.2B** is the current, small yet SOTA MinerU VLM that MinerU recommends, and it’s optimized to run fast with **vLLM async engine**. ([Hugging Face][14])
* **Licensing** → MinerU is **AGPL‑3.0**; ensure that aligns with your usage scenario. ([GitHub][2])

---

## 13) Minimal “one‑shot” scripts you can reuse

### A) Convert a whole folder with embedded vLLM

```bash
#!/usr/bin/env bash
set -euo pipefail
IN=${1:-/data/pdfs}
OUT=${2:-/data/out}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

# Hugging Face by default; for mainland China:
# export MINERU_MODEL_SOURCE=modelscope

mineru -p "$IN" -o "$OUT" -b vlm-vllm-engine --data-parallel-size 1
python postprocess_mineru_content.py "$OUT"
```

### B) Run a vLLM server + point MinerU at it

```bash
# Terminal 1: start vLLM (≥ 0.10.1 preferred)
vllm serve opendatalab/MinerU2.5-2509-1.2B \
  --host 0.0.0.0 --port 8000 \
  --logits-processors mineru_vl_utils:MinerULogitsProcessor

# Terminal 2: run MinerU as a client of that server
mineru -p /data/pdfs -o /data/out -b vlm-http-client -u http://127.0.0.1:8000
python postprocess_mineru_content.py /data/out
```

(These commands reflect MinerU’s Quick Usage, CLI help, and vLLM server guidance in `mineru-vl-utils`.) ([opendatalab.github.io][13])

---

## 14) Why this yields “chunkable” Markdown

* MinerU’s **VLM** recognizes *titles/headings* and emits **`text_level`** you can trust. We turn those into `#`/`##`/`###` so chunkers can split **on clear section boundaries**. ([opendatalab.github.io][12])
* **Tables → HTML** and **equations → LaTeX** preserve structure and semantics (lossless conversion) which greatly helps retrieval and display in UIs. MinerU natively converts tables to HTML and formulas to LaTeX. ([opendatalab.github.io][1])
* You keep **images and captions**, but they won’t pollute text embeddings. Use captions in metadata as needed. (Both caption and bbox/page come from the JSON.) ([opendatalab.github.io][12])

---

### Sources / docs referenced

* **MinerU docs:** overview, quick usage & backends, CLI, advanced vLLM params, model source switching, output formats (including `content_list.json` & `text_level`), FAQ (libGL & fonts). ([opendatalab.github.io][1])
* **MinerU GitHub README:** supported OS/GPU/Python ranges; vLLM acceleration path; MinerU2.5 notes. ([GitHub][2])
* **mineru‑vl‑utils:** vLLM serving command & logits processor; programmatic usage; supported backends. ([GitHub][9])
* **vLLM docs:** GPU/CUDA requirements; OpenAI‑compatible server usage. ([VLLM Documentation][4])
* **Docker path:** base image and driver/CUDA notes; compose profiles. ([opendatalab.github.io][10])

---

If you tell me your target OS/GPU (and whether you prefer Docker), I can tailor the commands (e.g., `--data-parallel-size`, GPU selection, and chunking thresholds) to your machine.

[1]: https://opendatalab.github.io/MinerU/ "MinerU - MinerU"
[2]: https://github.com/opendatalab/MinerU "GitHub - opendatalab/MinerU: Transforms complex documents like PDFs into LLM-ready markdown/JSON for your Agentic workflows."
[3]: https://ploomber.io/blog/vllm-deploy/?utm_source=chatgpt.com "Deploying vLLM: a Step-by-Step Guide"
[4]: https://docs.vllm.ai/en/stable/getting_started/installation/gpu.html?utm_source=chatgpt.com "GPU - vLLM"
[5]: https://opendatalab.github.io/MinerU/faq/ "FAQ - MinerU"
[6]: https://opendatalab.github.io/MinerU/quick_start/extension_modules/ "Extension Modules - MinerU"
[7]: https://opendatalab.github.io/MinerU/usage/model_source/ "Model Source - MinerU"
[8]: https://opendatalab.github.io/MinerU/usage/advanced_cli_parameters/ "Advanced CLI Parameters - MinerU"
[9]: https://github.com/opendatalab/mineru-vl-utils "GitHub - opendatalab/mineru-vl-utils: A Python package for interacting with the MinerU Vision-Language Model."
[10]: https://opendatalab.github.io/MinerU/quick_start/docker_deployment/ "Docker Deployment - MinerU"
[11]: https://opendatalab.github.io/MinerU/usage/cli_tools/ "CLI Tools - MinerU"
[12]: https://opendatalab.github.io/MinerU/reference/output_files/ "Output File Format - MinerU"
[13]: https://opendatalab.github.io/MinerU/usage/quick_usage/ "Quick Usage - MinerU"
[14]: https://huggingface.co/opendatalab/MinerU2.5-2509-1.2B?utm_source=chatgpt.com "opendatalab/MinerU2.5-2509-1.2B"
