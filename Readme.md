# Handwritten Scientific Notes OCR Pipeline

This project transcribes handwritten and printed scientific/technical notes: including text, LaTeX-formatted mathematics, hand-drawn chemical structures, reaction diagrams, and tables using **Qwen2.5-VL-7B-Instruct**, and benchmarks it against two zero-shot OCR baselines (**PaddleOCR**, **Tesseract**) using Word Error Rate (WER).

[Video demonstration](https://drive.google.com/file/d/1eDDqdOAr2k1J8abAa733XRwpZ1afwvnL/view?usp=sharing)

---

## Methods

This pipeline runs three OCR approaches on the same input image so their outputs can be directly compared.

### 1. Zero-shot baselines: PaddleOCR and Tesseract

`ocr_paddleocr()` and `ocr_tesseract()` run the target image through general-purpose OCR engines. They return plain, unstructured text. There is no awareness of LaTeX math, figure/diagram content, or table structure. These exist purely as a reference point: they show what "off-the-shelf OCR" achieves on the same handwriting, so the improvement from prompting Qwen can be measured rather than assumed.

### 2. Zero-shot Qwen2.5-VL

Qwen2.5-VL can be run directly on an image with a simple instruction (e.g. "transcribe this image") and no examples. This gives a reasonable baseline transcription, but in practice it is inconsistent about formatting: it may skip LaTeX delimiters for equations, miss degree symbols, or describe diagrams inconsistently from one run to the next.

### 3. Few-shot Qwen2.5-VL (used in this pipeline)

`run_ocr()` is the core method used here. It prompts Qwen2.5-VL with:
- A detailed system prompt (`SYSTEM_PROMPT_TEXT`) specifying exact rules for line breaks, LaTeX math delimiters (`$...$` / `$$...$$`), arrows/reaction notation, figure/diagram description format, and table fidelity.
- One **few-shot example**: an example image paired with a hand-written "ideal" transcription (`FEW_SHOT_ASSISTANT_OUTPUT`), demonstrating the exact `[FIGURE: ...]` tag format expected for hand-drawn chemical structures and reactions.
- The real target image, with an instruction to transcribe it the same way.

The few-shot example anchors the model's output format far more reliably than instructions alone. Particularly for the `[FIGURE: ...]` tagging behavior, which is difficult to specify purely in words. 

Note: The qwen after specific instructions cannot produce inline latex formulaes using '\$' delimiters.
Therefore after generation, a regex cleanup pass (`clean_latex_delimiters()`) normalizes any stray `\[...\]`/`\(...\)` delimiters or markdown code fences the model may still produce, guaranteeing consistent `$...$`/`$$...$$` formatting in the final output regardless of model drift.

A fourth, optional pass (`run_critique()`) sends the Qwen transcription back to the model along with the original rules, asking it to identify which rules were not followed — useful for spotting systematic formatting issues without manually re-reading every output. 
There is scope for future improvements here, where we can make this more aggresive in order to improve the output.

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

`pytesseract` is a Python wrapper — it requires the actual Tesseract OCR engine to be installed separately. The easiest no-root install (works on shared/cluster environments) is via conda-forge:

```bash
conda install -c conda-forge tesseract
```

Verify the install:

```bash
tesseract --version
```

### 4. Set your Hugging Face cache directory (optional)

Qwen2.5-VL-7B-Instruct (~16GB) downloads on first run. To control where it's cached (e.g. on a shared cluster with limited home-directory quota), edit `CACHE_DIR` near the top of `ocr_pipeline.py`:

```python
CACHE_DIR = "/path/to/your/hf_cache"
```

---

## Required input files

Place these in the same directory as `ocr_pipeline.py` (or update the paths at the top of the script):

| File | Purpose |
|---|---|
| `example.jpg` | The few-shot example image (shown to the model alongside its ideal transcription) |
| `test.jpg` | The target image you want transcribed |
| `ground_truth.txt` | Ground truth transcription of `test.jpg`, used to compute WER |

---

## How to run

```bash
python ocr_pipeline.py
```

This runs all three OCR engines on `test.jpg`, scores each against the ground truth, and runs the critique pass on the Qwen output. Console output is a short summary; full results are written to the `outputs/` directory.

```
Qwen result        -> outputs/result.txt (WER: 9.05%)
PaddleOCR result   -> outputs/result_paddleocr.txt (WER: 77.89%)
Tesseract result   -> outputs/result_tesseract.txt (WER: 96.98%)
Critique report    -> outputs/critique_report.txt
```

---

## Output files (`outputs/`)

| File | Contents |
|---|---|
| `result.txt` | Few-shot Qwen2.5-VL transcription — the main pipeline output, with LaTeX math and `[FIGURE: ...]` diagram descriptions |
| `extracted_report.json` | Json format for Goal, condition, procedure and the results. |
| `wer_report.txt` | WER score for the Qwen transcription vs. ground truth, plus a breakdown of substitutions, deletions, insertions, and hits, and a list of every misaligned word/phrase |
| `result_paddleocr.txt` | Zero-shot PaddleOCR transcription (plain text, no LaTeX/figure awareness) |
| `wer_report_paddleocr.txt` | Same WER breakdown, scored against the PaddleOCR output |
| `result_tesseract.txt` | Zero-shot Tesseract transcription (plain text, no LaTeX/figure awareness) |
| `wer_report_tesseract.txt` | Same WER breakdown, scored against the Tesseract output |
| `critique_report.txt` | A second LLM pass listing which of the six prompt rules the Qwen transcription may have failed to follow |

### Understanding the WER report

Word Error Rate (WER) is computed as `(substitutions + deletions + insertions) / total ground-truth words`. Lower is better; 0% means a perfect match after normalization (lowercased, whitespace-collapsed).

- **Substitutions** — a ground-truth word was replaced with a different predicted word (e.g. misread character).
- **Deletions** — a ground-truth word is missing entirely from the prediction (e.g. skipped line or symbol).
- **Insertions** — the prediction contains extra words not present in the ground truth (e.g. hallucinated text).
- **Hits** — words that matched exactly.

The `=== ERRORS ===` section beneath the summary lists every non-matching chunk side-by-side (`GT` vs `PRED`), which is the fastest way to spot a recurring failure pattern (e.g. consistently dropped degree symbols, or a specific Greek letter being misread) worth fixing in the prompt.

> **Note:** because the Qwen output includes LaTeX delimiters and `[FIGURE: ...]` tags, its WER is only meaningful if your ground-truth file (`ground_truth.txt`) uses the same formatting conventions.

---

## Notes and observations

- The results show some issues, with one degree sign but predicts all the signs except for that.
- The results also only omits one out of two scratched out numbers.
- Qwen2.5-VL-7B-Instruct requires a GPU with sufficient VRAM (recommended: ≥16GB) for `float16` inference at the default `max_new_tokens=1024`.
- The few-shot prompt format and rules can be edited directly in `SYSTEM_PROMPT_TEXT` and `FEW_SHOT_ASSISTANT_OUTPUT` near the top of `ocr_pipeline.py`.


---

## Future Possible Research work

- Finetuning method, Using Qwen or other synthetic methods to create ground truths on specific chemistry or science Lab reports in order to get create a Dataset for Lora finetuning a smaller VLM model specifially for the required format of the output. This will reduce the compute requirements and make the model more faster.

