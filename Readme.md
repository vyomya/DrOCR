# Doctor Prescription OCR Pipeline

A multimodal OCR pipeline for transcribing handwritten and printed **medical prescriptions** — patient and prescriber information, drug entries, dosages, abbreviations, and instructions — using **Qwen2.5-VL-7B-Instruct** with few-shot prompting, benchmarked against three zero-shot baselines (**PaddleOCR**, **Tesseract**, **Nougat**) across an entire folder of prescription images using Word Error Rate (WER).

This is the proof-of-concept accompanying the written report and presentation for the Cotiviti Intern Assessment.

| Deliverable | Link |
|---|---|
| 📄 Written Report (Word) | [Add your link here] |
| 📊 Slide Presentation (PowerPoint) | [Add your link here] |
| 🎥 Video Demonstration | [Add your link here] |

---

## Why prescription OCR

Handwritten prescriptions carry a documented error rate of 35.7%, compared to 2.5% for electronic prescriptions — a fourteen-fold difference — and illegible handwriting is linked to roughly 7,000 deaths annually in the United States alone. This pipeline demonstrates how a vision-language model, properly prompted for the clinical domain, can extract structured, verifiable data from a handwritten prescription image: drug names, dosages, routes, frequencies, and instructions — the exact fields a pharmacy verification or claims-adjudication system needs.

---

## Methods compared

### 1. Zero-shot baselines

- **Tesseract** — general-purpose OCR with an Otsu-threshold binarization pass. No domain awareness; returns plain text.
- **PaddleOCR** — general-purpose multilingual OCR. No domain awareness; returns plain text.
- **Nougat** (`facebook/nougat-base`, ICLR 2024) — a transformer trained specifically on academic PDFs, included for its native LaTeX math output (relevant to dosage formulae and concentration calculations). Its preprocessing is reimplemented manually in this pipeline (crop margin → thumbnail → pad → rescale → normalize) to avoid a known `huggingface_hub` strict-validation bug that otherwise raises `StrictDataclassFieldValidationError` on Nougat's saved config.

### 2. Few-shot Qwen2.5-VL (this pipeline's primary method)

`run_ocr()` prompts Qwen2.5-VL-7B-Instruct with a prescription-specific system prompt (`SYSTEM_PROMPT_TEXT`) covering:

- **Patient & prescriber info** — name, age, date, registration number, prescriber name/designation/clinic, extracted on separate lines.
- **Structured drug entries** — drug name → dosage/strength → route → frequency/duration → special instructions, in that exact order, preserving Rx numbering.
- **Medical abbreviations** — OD, BD, TDS, QID, PRN, SOS, IM, IV, SC, PO, NPO, etc. preserved verbatim, never expanded.
- **Dosage notation** — inline LaTeX (`$...$` / `$$...$$`) for calculations and concentrations; degrees always as `$^\circ$`.
- **Symbols and arrows** — →, ↑, ↓, ± preserved exactly as written.
- **Diagrams** — anatomical sketches or injection-site markers described using a `[FIGURE: ...]` tag.
- **Tables** — lab values or vitals rendered as markdown tables, values preserved exactly (no rounding, inference, or correction).
- **General fidelity** — misspellings, crossed-out text (`~~crossed~~`), and corrections (`[corrected: original → new]`) preserved; illegible content marked `[illegible]`; no markdown code fences in output.

A one-shot example (`FEW_SHOT_ASSISTANT_OUTPUT`) — a fully transcribed sample prescription pad — anchors this exact output format more reliably than instructions alone.

A post-processing pass (`clean_latex_delimiters()`) normalizes any stray `\[...\]` / `\(...\)` delimiters or markdown code fences the model may still produce, regardless of prompt compliance.

---

## Installation

### 1. Create and activate a Python environment

```bash
conda create -n ocr python=3.10
conda activate ocr
```

### 2. Install Python dependencies

```bash
pip install -r requirements.txt
```

### 3. Install Tesseract (system binary)

`pytesseract` is a Python wrapper — it requires the actual Tesseract engine installed separately. No-root install via conda-forge:

```bash
conda install -c conda-forge tesseract
```

Verify:

```bash
tesseract --version
```

### 4. Install Nougat's optional dependencies

```bash
pip install nltk python-Levenshtein --break-system-packages
python3 -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab')"
```

### 5. Set your Hugging Face cache directory

Qwen2.5-VL-7B-Instruct (~16GB) and Nougat (~1.3GB) download on first run. Edit `CACHE_DIR` near the top of the pipeline script to control where they're cached:

```python
CACHE_DIR = "/path/to/your/hf_cache"
```

---

## Required input layout

```
project/
├── example.jpg              # one-shot example prescription image
├── test_data/                # folder of prescription images to OCR
│   ├── 001.jpg
│   ├── 002.jpg
│   └── ...
├── ground_truth/             # matching ground-truth transcription per image
│   ├── 001.txt               # filename stem must match the image stem
│   ├── 002.txt
│   └── ...
└── ocr_pipeline.py
```

Images without a matching ground-truth file are skipped automatically (and logged) rather than causing the run to fail.

---

## How to run

```bash
python ocr_pipeline.py
```

This loads all four engines once, then iterates over every image in `test_data/`, running all four engines on each and writing results progressively. Console output is one line per engine per image:

```
Found 12 images in test_data
Loading Qwen2.5-VL ...
Loading Nougat ...
Loading PaddleOCR ...

── 001.jpg ────────────────────────────────────────
  Qwen       WER:  0%  (0s)
  PaddleOCR  WER:  0%  (0s)
  Tesseract  WER:  0%  (0s)
  Nougat     WER:  0%  (0s)
...
==============================================================
  Done. 12 images processed.
  Results  -> outputs/{qwen,paddleocr,tesseract,nougat}/
  Summary  -> outputs/wer_summary.txt
==============================================================
```

---

## Output layout

```
outputs/
├── qwen/
│   ├── 001.txt          # Qwen2.5-VL transcription per image
│   ├── 002.txt
│   └── ...
├── paddleocr/
│   ├── 001.txt
│   └── ...
├── tesseract/
│   ├── 001.txt
│   └── ...
├── nougat/
│   ├── 001.txt
│   └── ...
└── wer_summary.txt       # single combined report — no per-image breakdown
```

### Understanding `wer_summary.txt`

The summary reports only the aggregate numbers across the entire batch — no individual diffs or per-word error listings:

```
==============================================================
  OCR BATCH WER SUMMARY REPORT
  Images evaluated : 12
==============================================================
Metric                              Qwen2.5-VL         PaddleOCR         Tesseract            Nougat
----------------------------------------------------------------------------------------------------
Avg WER (%)                               27.8             29.15             59.55             80.87
Corpus WER (%)                           34.48             30.54             60.59             86.21
Total substitutions                         26                38                98                37
Total deletions                             27                19                23               136
Total insertions                            17                 5                 2                 2
Total hits                                 150               146                82                30
Total ref words                            203               203               203               203
```

| Metric | Meaning |
|---|---|
| **Avg WER (%)** | Mean of per-image WER scores — treats every image equally regardless of length. |
| **Corpus WER (%)** | `(substitutions + deletions + insertions) / total reference words`, computed across the whole batch — more stable when image lengths vary, since it weights by word count rather than by image count. |
| **Total substitutions** | Sum across all images — a ground-truth word replaced by an incorrect predicted word (e.g. a misread drug name). |
| **Total deletions** | Sum across all images — a ground-truth word missing entirely from the prediction (e.g. a skipped instruction line). |
| **Total insertions** | Sum across all images — extra words in the prediction not present in ground truth (e.g. hallucinated text). |
| **Total hits** | Sum across all images — words that matched exactly. |
| **Total ref words** | Total ground-truth word count across the whole batch — the denominator for corpus WER. |

> **Note:** WER normalization strips `[FIGURE: ...]`, `[illegible]`, and `[Not Specified: ...]` tags before scoring, so these formatting tokens don't penalize or inflate any engine's score. Since only Qwen produces these tags by design, this keeps the comparison against the plain-text baselines (PaddleOCR, Tesseract) fair.

---

## Notes

- Qwen2.5-VL-7B-Instruct requires a GPU with ≥16GB VRAM recommended for `float16` inference.
- All four models load once at the start of the run and are reused across every image — this matters most for PaddleOCR and Nougat, which are otherwise expensive to reinitialize per image.
- The prompt, few-shot example, and prescription field schema can be edited directly in `SYSTEM_PROMPT_TEXT` and `FEW_SHOT_ASSISTANT_OUTPUT` near the top of `ocr_pipeline.py`.
- If PaddleOCR raises `NotImplementedError: ConvertPirAttribute2RuntimeAttribute`, this is a known PaddlePaddle 3.3.x regression — `enable_mkldnn=False` (already set in this pipeline) works around it.
- If Nougat returns blank output on a given image, check the console diagnostic for the pixel tensor stats — Nougat is trained on printed academic PDFs and may produce empty or short output on heavily cursive handwriting that's far outside its training distribution; this is expected behavior for that baseline, not a bug.
