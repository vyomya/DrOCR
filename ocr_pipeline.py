"""
Batch OCR pipeline — runs all four engines over a folder of images.

Directory layout expected:
    test_data/
        001.jpg
        002.jpg
        ...
    ground_truth/
        001.txt          # ground truth matching each image by stem
        002.txt
        ...
    example.jpg          # few-shot example for Qwen

Output layout produced:
    outputs/
        qwen/
            001.txt
            002.txt
            ...
        paddleocr/
            001.txt
            ...
        tesseract/
            001.txt
            ...
        nougat/
            001.txt
            ...
        wer_summary.txt  # single combined report — averages across all images
"""

import os
CACHE_DIR = "/fs/nexus-scratch/vyomwal5/anaconda3/envs/whisper/hf_cache"
os.environ["HF_HOME"] = CACHE_DIR

import warnings
warnings.filterwarnings("ignore")

import re
import time
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from jiwer import process_words


# --------------------------------------------------------------------------
# Config — edit these paths to match your setup
# --------------------------------------------------------------------------

MODEL_ID       = "Qwen/Qwen2.5-VL-7B-Instruct"
NOUGAT_ID      = "facebook/nougat-base"
MAX_NEW_TOKENS = 1024
IMAGE_EXTS     = {".jpg", ".jpeg", ".png", ".tiff", ".bmp"}

INPUT_DIR          = Path(".")
IMAGE_DIR          = INPUT_DIR / "test_data"       # folder of input images
GROUND_TRUTH_DIR   = INPUT_DIR / "ground_truth"    # matching .txt per image stem
EXAMPLE_IMAGE_PATH = INPUT_DIR / "example.jpg"     # few-shot example for Qwen
OUTPUT_DIR         = Path("outputs")

# Per-engine output subfolders
ENGINE_DIRS = {
    "qwen":      OUTPUT_DIR / "qwen",
    "paddleocr": OUTPUT_DIR / "paddleocr",
    "tesseract": OUTPUT_DIR / "tesseract",
    "nougat":    OUTPUT_DIR / "nougat",
}
WER_SUMMARY_PATH = OUTPUT_DIR / "wer_summary.txt"


# --------------------------------------------------------------------------
# Prompt
# --------------------------------------------------------------------------

SYSTEM_PROMPT_TEXT = """You are a clinical OCR system specialized in transcribing handwritten and printed medical prescriptions with full accuracy and structural fidelity. Your output will be used for downstream pharmacy verification, claims processing, and patient safety checks — errors in drug names, dosages, or instructions are critical. Transcribe the prescription image following these rules:

1. Patient and prescriber information: Extract and preserve on separate lines:
   - Patient name, age, date, patient ID or registration number if present
   - Prescriber name, designation (Dr./MD/etc.), clinic or hospital name, contact if present
   - Preserve original formatting and abbreviations exactly as written.

2. Drug entries: For each drug prescribed, extract in this exact order on separate lines:
   - Drug name (preserve brand name or generic exactly as written, including spelling errors — mark clearly with [illegible] if unreadable)
   - Dosage and strength (e.g. 500mg, 0.5mg/kg) — transcribe exactly as written
   - Route of administration (oral, IV, topical, etc.) if specified
   - Frequency and duration (e.g. BID x 5 days, TDS for 1 week, OD at night)
   - Special instructions (e.g. "take after food", "avoid sunlight", "with plenty of water")
   If multiple drugs are listed (Rx 1, Rx 2, etc.), preserve the numbering and order exactly.

3. Dosage and medical notation: Transcribe all dosage calculations, concentrations, and units.
  
4. Abbreviations: Preserve all medical abbreviations exactly as written (Rx, OD, BD, TDS, QID, PRN, SOS, IM, IV, SC, PO, NPO, etc.). Do not expand abbreviations unless they appear expanded in the source.

5. Symbols and arrows: Preserve all symbols exactly:
   → for directional instructions, ↑ for increase/high, ↓ for decrease/low, ± for approximate.

6. Diagrams and non-text content: If the prescription contains a diagram, describe it:
   [FIGURE: <detailed description>]

7. Tables: If lab values, vitals, or drug charts appear in tabular form, render as a markdown table preserving all values exactly as written. Do NOT correct, round, or infer any cell value.

8. General fidelity: Transcribe exactly what is written, including apparent misspellings, crossed-out text (mark as ~~crossed~~), and correction overwriting (mark as [corrected: original -> new]). If a word or symbol is genuinely illegible, mark as [illegible]. Never wrap output in markdown code fences."""

FEW_SHOT_ASSISTANT_OUTPUT = """CITY GENERAL HOSPITAL
Department of Internal Medicine • 123 Medical Center Drive, Springfield
Tel: (555) 234-5678 • Fax: (555) 234-5679 • www.citygeneralhospital.org

Dr. Arjun Kumar, MD                                               Date:
MBBS, MD (Internal Medicine)                                12 / 06 / 2026
Reg. No: MCI/2014/56789                                   OPD No: 10234

Name:    John Smith                     Age: 45 yrs             Sex: M
Address: 42 Oak Street, Springfield

Rx  Amoxicillin       500 mg      caps.
    Sig: 1 cap TDS x 7 days — take after food
    Disp: 21 capsules

Rx  Paracetamol       650 mg      tabs.
    Sig: 1 tab SOS — if fever > $38^\\circ C$
    Max 4 doses/day — Disp: 10 tablets

Rx  ORS Sachet
    Sig: 1 sachet in 200 mL water — QID
    Continue till diarrhoea stops — Disp: 20 sachets

ADVICE:
1. Plenty of oral fluids — minimum 2-3 litres/day
2. Avoid spicy / oily food during course of antibiotics
3. Complete full antibiotic course — do not stop early
4. Return immediately if breathlessness / rash develops

FOLLOW-UP:       Review after 7 days or earlier if no improvement

INVESTIGATIONS:  CBC, CRP — fasting blood sugar

                                              DR. ARJUN KUMAR MD
                                              CITY GENERAL HOSPITAL
                                              Signature & Stamp
                                              [FIGURE: Handwritten signature]
                                              ______________________________
                                              Dr. Arjun Kumar, MD

Refills:  None

This prescription is valid for 30 days from the date of issue • For emergencies call 911"""
ASSISTANT_MARKER = "above.\nassistant"


# --------------------------------------------------------------------------
# WER dataclass — accumulates counts across images
# --------------------------------------------------------------------------

@dataclass
class WerAccumulator:
    """Accumulates per-image WER stats; call summary() for the combined report."""
    name: str
    wers:          list = field(default_factory=list)
    substitutions: list = field(default_factory=list)
    deletions:     list = field(default_factory=list)
    insertions:    list = field(default_factory=list)
    hits:          list = field(default_factory=list)
    n_images:      int  = 0

    def add(self, wer_result):
        self.wers.append(wer_result.wer)
        self.substitutions.append(wer_result.substitutions)
        self.deletions.append(wer_result.deletions)
        self.insertions.append(wer_result.insertions)
        self.hits.append(wer_result.hits)
        self.n_images += 1

    def summary(self) -> dict:
        if self.n_images == 0:
            return {}
        total_sub = sum(self.substitutions)
        total_del = sum(self.deletions)
        total_ins = sum(self.insertions)
        total_hit = sum(self.hits)
        total_ref = total_sub + total_del + total_hit   # total reference words
        corpus_wer = (total_sub + total_del + total_ins) / max(total_ref, 1)
        return {
            "engine":               self.name,
            "images_processed":     self.n_images,
            "avg_wer_pct":          round(sum(self.wers) / self.n_images * 100, 2),
            "corpus_wer_pct":       round(corpus_wer * 100, 2),
            "total_substitutions":  total_sub,
            "total_deletions":      total_del,
            "total_insertions":     total_ins,
            "total_hits":           total_hit,
            "total_ref_words":      total_ref,
        }


@dataclass
class WerResult:
    wer: float
    substitutions: int
    deletions: int
    insertions: int
    hits: int


# --------------------------------------------------------------------------
# Model loading
# --------------------------------------------------------------------------

def load_model():
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_ID, torch_dtype=torch.float16, device_map="auto"
    )
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    return model, processor


def run_chat(model, processor, messages, images=None, max_new_tokens=MAX_NEW_TOKENS):
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    kwargs = dict(text=[text], return_tensors="pt")
    if images:
        kwargs["images"] = images
    inputs = processor(**kwargs).to(model.device)
    output = model.generate(**inputs, max_new_tokens=max_new_tokens)
    return processor.decode(output[0], skip_special_tokens=True)


# --------------------------------------------------------------------------
# LaTeX cleanup
# --------------------------------------------------------------------------

def clean_latex_delimiters(text: str) -> str:
    text = re.sub(r"```(?:latex)?\s*", "", text)
    text = re.sub(r"\\\[\s*",  "$$", text)
    text = re.sub(r"\s*\\\]", "$$", text)
    text = re.sub(r"\\\(\s*",  "$",  text)
    text = re.sub(r"\s*\\\)", "$",  text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


# --------------------------------------------------------------------------
# Engine 1: Qwen2.5-VL — few-shot, prescription-aware
# --------------------------------------------------------------------------

def run_ocr(model, processor, example_image_path, target_image_path) -> str:
    example_image = Image.open(example_image_path)
    target_image  = Image.open(target_image_path)

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": example_image},
                {"type": "text",  "text":  SYSTEM_PROMPT_TEXT},
            ],
        },
        {
            "role": "assistant",
            "content": [{"type": "text", "text": FEW_SHOT_ASSISTANT_OUTPUT}],
        },
        {
            "role": "user",
            "content": [
                {"type": "image", "image": target_image},
                {"type": "text",  "text":  "Now transcribe this medical prescription following the same rules and format demonstrated above."},
            ],
        },
    ]

    raw = run_chat(model, processor, messages, images=[example_image, target_image])
    if ASSISTANT_MARKER in raw:
        _, transcription = raw.split(ASSISTANT_MARKER, 1)
    else:
        transcription = raw
    return clean_latex_delimiters(transcription)


# --------------------------------------------------------------------------
# Engine 2: PaddleOCR — zero-shot
# --------------------------------------------------------------------------

def ocr_paddleocr(target_image_path, engine) -> str:
    result = engine.predict(str(target_image_path))
    lines = result[0]["rec_texts"] if result and "rec_texts" in result[0] else []
    return "\n".join(lines).strip()


# --------------------------------------------------------------------------
# Engine 3: Tesseract — zero-shot with Otsu binarization
# --------------------------------------------------------------------------

def ocr_tesseract(target_image_path) -> str:
    import pytesseract
    img = Image.open(target_image_path).convert("L")
    arr = np.array(img)
    _, binary = cv2.threshold(arr, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    text = pytesseract.image_to_string(
        Image.fromarray(binary), config="--oem 3 --psm 6 -l eng --dpi 300"
    )
    return text.strip()


# --------------------------------------------------------------------------
# Engine 4: Nougat — zero-shot, manual preprocessing
# --------------------------------------------------------------------------

NOUGAT_SIZE = (896, 672)
NOUGAT_MEAN = [0.485, 0.456, 0.406]
NOUGAT_STD  = [0.229, 0.224, 0.225]


def _nougat_preprocess(img: Image.Image) -> torch.Tensor:
    gray = np.array(img.convert("L"))
    non_white = np.where(gray < 250)
    if non_white[0].size > 0:
        top, bottom = non_white[0].min(), non_white[0].max()
        left, right = non_white[1].min(), non_white[1].max()
        img = img.crop((left, top, right + 1, bottom + 1))
    target_w, target_h = NOUGAT_SIZE[1], NOUGAT_SIZE[0]
    img.thumbnail((target_w, target_h), Image.Resampling.BILINEAR)
    padded = Image.new("RGB", (target_w, target_h), (255, 255, 255))
    padded.paste(img, (0, 0))
    arr  = np.array(padded, dtype=np.float32) / 255.0
    mean = np.array(NOUGAT_MEAN, dtype=np.float32)
    std  = np.array(NOUGAT_STD,  dtype=np.float32)
    arr  = (arr - mean) / std
    return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)


def ocr_nougat(target_image_path, nougat_processor, nougat_model, device) -> str:
    img          = Image.open(target_image_path).convert("RGB")
    pixel_values = _nougat_preprocess(img).to(device)

    with torch.no_grad():
        outputs = nougat_model.generate(
            pixel_values,
            min_length=1,
            max_new_tokens=MAX_NEW_TOKENS,
            early_stopping=True,
            num_beams=1,
            no_repeat_ngram_size=3,
        )

    raw = nougat_processor.batch_decode(outputs, skip_special_tokens=True)[0]
    if not raw.strip():
        return "[Nougat: no output generated]"
    if len(raw.strip()) < 50:
        return raw.strip()
    result = nougat_processor.post_process_generation(raw, fix_markdown=True)
    return result.strip() if result.strip() else raw.strip()


# --------------------------------------------------------------------------
# WER helpers
# --------------------------------------------------------------------------

def normalize_text(text: str) -> str:
    text = re.sub(r"\[illegible\]", "", text)
    text = re.sub(r"\[Not Specified:[^\]]*\]", "", text)
    text = re.sub(r"\[FIGURE:[^\]]*\]", "", text)
    text = text.lower()
    text = text.replace("\n", " ").replace("\t", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def compute_wer(ground_truth_raw: str, prediction_raw: str) -> WerResult:
    gt   = normalize_text(ground_truth_raw)
    pred = normalize_text(prediction_raw)
    r    = process_words(gt, pred)
    return WerResult(
        wer=r.wer,
        substitutions=r.substitutions,
        deletions=r.deletions,
        insertions=r.insertions,
        hits=r.hits,
    )


# --------------------------------------------------------------------------
# Summary report writer
# --------------------------------------------------------------------------

def write_summary(accumulators: list, image_names: list, path: Path):
    lines = [
        "=" * 62,
        "  OCR BATCH WER SUMMARY REPORT",
        f"  Images evaluated : {len(image_names)}",
        "=" * 62,
        "",
    ]

    # List images processed
    lines.append("Images processed:")
    for name in image_names:
        lines.append(f"  {name}")
    lines.append("")

    # Column-aligned table header
    col = 18
    lines.append(
        f"{'Metric':<28}"
        + "".join(f"{a.name:>{col}}" for a in accumulators)
    )
    lines.append("-" * (28 + col * len(accumulators)))

    summaries = [a.summary() for a in accumulators]

    # Rows
    rows = [
        ("Avg WER (%)",         "avg_wer_pct"),
        ("Corpus WER (%)",      "corpus_wer_pct"),
        ("Total substitutions", "total_substitutions"),
        ("Total deletions",     "total_deletions"),
        ("Total insertions",    "total_insertions"),
        ("Total hits",          "total_hits"),
        ("Total ref words",     "total_ref_words"),
    ]

    for label, key in rows:
        row = f"{label:<28}"
        for s in summaries:
            val = s.get(key, "-")
            row += f"{str(val):>{col}}"
        lines.append(row)

    lines.append("")
    lines.append("Notes:")
    lines.append("  Avg WER    = mean of per-image WER scores")
    lines.append("  Corpus WER = (sub+del+ins) / total_ref_words across all images")
    lines.append("             = more stable metric when image lengths vary")

    path.write_text("\n".join(lines), encoding="utf-8")


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def main():
    # Create output subdirs
    OUTPUT_DIR.mkdir(exist_ok=True)
    for d in ENGINE_DIRS.values():
        d.mkdir(parents=True, exist_ok=True)

    # Collect images and their ground truth files
    images = sorted([
        p for p in IMAGE_DIR.iterdir()
        if p.suffix.lower() in IMAGE_EXTS
    ])
    if not images:
        print(f"No images found in {IMAGE_DIR}")
        return
    print(f"Found {len(images)} images in {IMAGE_DIR}")

    # Load heavy models once — reused across all images
    print("Loading Qwen2.5-VL ...")
    qwen_model, qwen_processor = load_model()

    print("Loading Nougat ...")
    from transformers import NougatProcessor, VisionEncoderDecoderModel
    nougat_processor = NougatProcessor.from_pretrained(NOUGAT_ID)
    nougat_model     = VisionEncoderDecoderModel.from_pretrained(NOUGAT_ID)
    nougat_device    = "cuda" if torch.cuda.is_available() else "cpu"
    nougat_model.to(nougat_device)
    nougat_model.eval()

    print("Loading PaddleOCR ...")
    from paddleocr import PaddleOCR
    paddle_engine = PaddleOCR(use_angle_cls=True, lang="en", enable_mkldnn=False)

    # Accumulators — one per engine
    acc = {
        "qwen":      WerAccumulator("Qwen2.5-VL"),
        "paddleocr": WerAccumulator("PaddleOCR"),
        "tesseract": WerAccumulator("Tesseract"),
        "nougat":    WerAccumulator("Nougat"),
    }

    image_names = []

    for img_path in images:
        stem = img_path.stem
        gt_path = GROUND_TRUTH_DIR / f"{stem}.txt"

        if not gt_path.exists():
            print(f"  [SKIP] no ground truth for {img_path.name}")
            continue

        gt_raw = gt_path.read_text(encoding="utf-8")
        image_names.append(img_path.name)
        print(f"\n── {img_path.name} {'─' * (50 - len(img_path.name))}")

        # ── Qwen ──────────────────────────────────────────────────────────
        t = time.time()
        pred = run_ocr(qwen_model, qwen_processor, EXAMPLE_IMAGE_PATH, img_path)
        (ENGINE_DIRS["qwen"] / f"{stem}.txt").write_text(pred, encoding="utf-8")
        wer = compute_wer(gt_raw, pred)
        acc["qwen"].add(wer)
        print(f"  Qwen       WER: {wer.wer*100:6.2f}%  ({time.time()-t:.1f}s)")

        # ── PaddleOCR ─────────────────────────────────────────────────────
        t = time.time()
        pred = ocr_paddleocr(img_path, paddle_engine)
        (ENGINE_DIRS["paddleocr"] / f"{stem}.txt").write_text(pred, encoding="utf-8")
        wer = compute_wer(gt_raw, pred)
        acc["paddleocr"].add(wer)
        print(f"  PaddleOCR  WER: {wer.wer*100:6.2f}%  ({time.time()-t:.1f}s)")

        # ── Tesseract ─────────────────────────────────────────────────────
        t = time.time()
        pred = ocr_tesseract(img_path)
        (ENGINE_DIRS["tesseract"] / f"{stem}.txt").write_text(pred, encoding="utf-8")
        wer = compute_wer(gt_raw, pred)
        acc["tesseract"].add(wer)
        print(f"  Tesseract  WER: {wer.wer*100:6.2f}%  ({time.time()-t:.1f}s)")

        # ── Nougat ────────────────────────────────────────────────────────
        t = time.time()
        pred = ocr_nougat(img_path, nougat_processor, nougat_model, nougat_device)
        (ENGINE_DIRS["nougat"] / f"{stem}.txt").write_text(pred, encoding="utf-8")
        wer = compute_wer(gt_raw, pred)
        acc["nougat"].add(wer)
        print(f"  Nougat     WER: {wer.wer*100:6.2f}%  ({time.time()-t:.1f}s)")

    # Write combined summary
    write_summary(list(acc.values()), image_names, WER_SUMMARY_PATH)

    print(f"\n{'=' * 62}")
    print(f"  Done. {len(image_names)} images processed.")
    print(f"  Results  -> {OUTPUT_DIR}/{{qwen,paddleocr,tesseract,nougat}}/")
    print(f"  Summary  -> {WER_SUMMARY_PATH}")
    print(f"{'=' * 62}")


if __name__ == "__main__":
    main()