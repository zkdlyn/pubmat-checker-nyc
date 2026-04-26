"""
checker.py — Pubmat validation functions for NYC post compliance checks.
"""

from unittest import result

import cv2
import numpy as np
import difflib
import re
from doctr.models import ocr_predictor
from doctr.io import DocumentFile
from PIL import Image
import tempfile
import os
from spellchecker import SpellChecker


# ── Lazy-load docTR model ─────────────────────────────────────────────────────
_doctr_model = None

def get_doctr_model():
    global _doctr_model
    if _doctr_model is None:
        _doctr_model = ocr_predictor(pretrained=True)
    return _doctr_model

# ── Shared helpers ────────────────────────────────────────────────────────────
def _run_doctr(image_bgr):
    """Converts a BGR numpy array to a temp JPEG, runs docTR, returns the result."""
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    image_pil = Image.fromarray(image_rgb)
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
        tmp_path = tmp.name
        image_pil.save(tmp_path)
    try:
        doc = DocumentFile.from_images([tmp_path])
        result = get_doctr_model()(doc)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
    return result


def _extract_ocr_data(doctr_result):
    """
    Flattens a docTR result into parallel lists:
    - words: list of recognized word strings
    - confidences: list of OCR confidence scores (0-1)
    - boxes: list of bounding boxes as (x0, y0, x1, y1) relative coordinates (0-1)
    """
    words, confidences, boxes = [], [], []
    for page in doctr_result.pages:
        for block in page.blocks:
            for line in block.lines:
                for word in line.words:
                    if word.confidence <0.5:
                        continue # skip low confidence detections 
                    words.append(word.value)
                    confidences.append(word.confidence)
                    (x0, y0), (x1, y1) = word.geometry
                    boxes.append((x0, y0, x1, y1))
    return words, confidences, boxes

def _make_result(passed: bool, label_ok: str, label_fail: str, 
                 remark_ok: str="OK", remark_fail: str="Issue found", 
                 details: dict = None, level:str = "error") -> dict:
    return{
        "pass": passed,
        "label": label_ok if passed else label_fail,
        "remark": remark_ok if passed else remark_fail,
        "details": details or {},

    }

SPELL_WORD_LISTS = [
    "word_list/tagalog_word_list.txt",   
    "word_list/custom_list.txt",  
    "word_list/filipino_word_list.txt",
    "word_list/hiligaynon_word_list.txt",
    "word_list/ilocano_word_list.txt",
    "word_list/cebuano_word_list.txt"

]

def load_spell_checker(wordlist_paths: list[str]) -> SpellChecker:
    """
    Initialize SpellChecker and load Filipino/Tagalog + custom word lists.
    """
    # Use language=None if you're ONLY using custom dictionaries,
    # or "en" if you still want English base + your additions
    spell = SpellChecker(language="en")

    for path in wordlist_paths:
        spell.word_frequency.load_text_file(path)

    return spell
    

# ── Rule config per post type ─────────────────────────────────────────────────

POST_TYPE_RULES = {
    "news": {
        "requires_watermark": "error",
        "requires_template": "error",
        "readability_threshold": 0.70,
        "requires_spell_check": "warning",
    },
    "quotes": {
        "requires_template": "error",
        "readability_threshold": 0.70,
        "requires_spell_check": "warning",
    },
    "advisory": {
        "requires_template": T"error",
        "readability_threshold": 0.70,
        "requires_sgd": "error",
        "requires_spell_check": "warning"
    },
    "resolution": {
        "requires_template": "error",
        "readability_threshold": 0.70,
        "requires_sgd": "error",
        "requires_spell_check": "warning"
    },
    "opportunity": {
        "requires_watermark": "error",
        "requires_template": "error",
        "readability_threshold": 0.70,
        "requires_spell_check": "warning",
    },
    "photo": {
        "requires_template": "error",
        "check_photo_quality": "warning"
    },
    "holiday": {
        "requires_watermark": "error",
        "requires_template": "error",
        "readability_threshold": 0.50,
        "requires_spell_check": "warning",
    },
    "other": {
        "requires_watermark": "error",
        "readability_threshold": 0.50,
        "requires_spell_check": "warning",
    },
}

WATERMARK_HANDLES = [
    "nyc.gov.ph",
    "nationalyouthcommission",
    "@nycpilipinas",
]

FUZZY_THRESHOLD = 0.75


# ── Check functions ───────────────────────────────────────────────────────────

def check_watermark(image, precomputed_words=None, precomputed_boxes=None) -> dict:
    """
    Fuzzy-matches  watermark handles against the bottom 15% of the image.
    Accepts pre-computed OCR data (filtered to y0 >= 0.85) to avoid a second
    docTR call when called from generate_report(). Falls back to cropping and
    running OCR directly if none supplied.
    """
    img_h, img_w = image.shape[:2]
    crop_y = int(img_h * 0.85)

    if precomputed_words is not None:
        words = precomputed_words
        raw_boxes = precomputed_boxes or []
        boxes_abs = [
            (int(x0 * img_w), int(y0 * img_h), 
             int(x1 * img_w), int(y1 * img_h))
            for (x0, y0, x1, y1) in raw_boxes
        ]
    else:
        crop = image[crop_y:img_h, 0:img_w]
        doctr_result = _run_doctr(crop)
        words, _, raw_boxes = _extract_ocr_data(doctr_result)
        boxes_abs = [
            (int(x0 * img_w), crop_y + int(y0 * crop.shape[0]),
             int(x1 * img_w), crop_y + int(y1 * crop.shape[0]))
            for (x0, y0, x1, y1) in raw_boxes
        ]

    words_lower = [w.lower() for w in words]
    full_text = " ".join(words_lower)
    handle_details = {}
    missing = []

    for handle in WATERMARK_HANDLES:
        clean = handle.replace("@", "").replace(".", "").lower()
        best = max(
            difflib.SequenceMatcher(None, clean, full_text.replace(" ", "")).ratio(),
            max(
                (difflib.SequenceMatcher(None, clean, w.replace(" ", "")).ratio()
                 for w in words_lower),
                default=0.0,
            ),
        )
        found = best >= FUZZY_THRESHOLD
        handle_details[handle] = {"found": found, "score": round(best, 3)}
        if not found:
            missing.append(handle)

    passed = len(missing) == 0
    result= _make_result(
        passed=passed,
        label_ok="Watermark OK",
        label_fail="Watermark missing or incorrect",
        remark_fail= f"Missing: {', '.join(missing)}",
        level ="error",
    )
    result["missing"] = missing
    return result, boxes_abs

def _mask_regions(image: np.ndarray, logo_boxes: list) -> np.ndarray:
    """
    Returns a copy of the image with logo regions and the bottom
    watermark strip filled with white (255,255,255), so docTR
    does not OCR text from those areas.
    """
    masked = image.copy()
    h, w   = image.shape[:2]

    # blank each detected logo bounding box
    for (x0, y0, x1, y1) in logo_boxes:
        # add a small padding so edge text is also excluded
        pad = 10
        x0 = max(0, x0 - pad)
        y0 = max(0, y0 - pad)
        x1 = min(w, x1 + pad)
        y1 = min(h, y1 + pad)
        masked[y0:y1, x0:x1] = 255

    return masked

def check_readability(confidences:list, threshold: float) -> dict:
    """
    OCR-confidence-based readability check.
    """
    if not confidences:
        return _make_result(
            passed=False,
            label_ok="",
            label_fail="No readable text_found",
            remark_fail="No text detected",
        )

    score = round(sum(confidences) / len(confidences),3)
    passed = score >= threshold 
    if score >= threshold:
        label = "Readable"
    elif score >= threshold * 0.5:
        label = "Moderately readable"
    else:
        label = "Low readability"
    return _make_result(
        passed=passed,
        label_ok=label,
        label_fail=label,
        remark_ok=f"Average OCR confidence: {score}",
        remark_fail=f"Average OCR confidence: {score} below threshold {threshold}",
        details={"average_confidence": score, "num_words": len(confidences)},
        level="error",
    )


def check_pubmat_quality(image) -> dict:
    """Universal image quality check: resolution, blur, contrast. Runs for all post types."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    h, w = image.shape[:2]
    issues = []

    if w < 1080 or h < 1080:
        issues.append(f"Low resolution ({w}x{h}). Minimum is 1080x1080 px.")
    if cv2.Laplacian(gray, cv2.CV_64F).var() < 50:
        issues.append("Image appears blurry.")
    if np.std(gray) < 30:
        issues.append("Low contrast.")

    return _make_result(
        len(issues) == 0,
        label_ok="Pubmat quality OK",
        label_fail="Pubmat quality issues",
        remark_fail=" | ".join(issues),
        details={"Resolution": f"{w}x{h}", 
                 "Blur Metric": round(cv2.Laplacian(gray, cv2.CV_64F).var(), 1), 
                 "Contrast Metric": round(np.std(gray), 1)},
    )

def check_sgd(ocr_words: list) ->dict:
    """
    Checks that 'SGD' appears as a whole word in the OCR output.
    Applied to: advisory, resolution.
    """
    found = bool(re.search(r"\bsgd\b"," ".join(ocr_words).lower()))
    return _make_result(
        passed=found,
        label_ok="SGD present",
        label_fail="SGD MISSING",
        remark_fail="Use SGD for resolutions/advisories",
        level ="error",
    )


def check_photo_quality(image) ->dict:
    """
    Photo-specific quality checks: resolution, subject centering, brightness,
    and colour saturation.
    Applied to: photo.


    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    h, w = image.shape[:2]
    min_w, min_h = 1080, 1080
    issues = []
    details = {}

    # Resolution
    details["resolution"] = f"{w}x{h} (required at least {min_w}x{min_h})"
    if w < min_w or h < min_h:
        issues.append(f"Image is {w}x{h}, minimum is {min_w}x{min_h}")

    # Brightness
    mean_brightness = float(np.mean(gray))
    details["brightness"] = round(mean_brightness, 1)
    if mean_brightness < 60:
        issues.append("Image appears dark")

    # Colour saturation
    h, w = image.shape[:2]

    end_y = int(h * 0.80)
    # crop out colored template 
    cropped = image[0:end_y, 0:w]
    hsv = cv2.cvtColor(cropped, cv2.COLOR_BGR2HSV)
    s = hsv[:, :, 1]

    mean_sat = float(np.mean(s))
    std_sat = float(np.std(s))

    details["mean_saturation"] = round(mean_sat, 1)
    if mean_sat < 30 and std_sat < 20:
        issues.append("Image appears grayscale or desaturated, use a colorized photo")
    return _make_result(
        passed=len(issues) == 0,
        label_ok="Photo quality OK",
        label_fail="Photo quality issues",
        remark_fail=" | ".join(issues),
        details=details,
        # level ="warning",
    )

def check_logo_order(detected:dict, collaborators:list=None) -> dict:
    """
    Validates left-to-right order: NYC leftmost, BP rightmost,
    SK/YORP (if required) in between.
    Runs for all types.
    """
    collaborators = [c.lower() for c in collaborators]

    if detected.get("nyc") is None or detected.get("bp") is None:
        return _make_result(
            passed=False,
            label_ok="",
            label_fail="Cannot check logo order",
            remark_fail="Missing NYC or BP logo prevents order validation",
            level="error"
        )

    def _center_x(entry):
        xyxy = entry["box"].xyxy[0].cpu().numpy().astype(int)
        return (xyxy[0] + xyxy[2]) / 2

    relevant = ["nyc", "bp"] + [c for c in collaborators if c in ("sk", "yorp")]
    positions = {
        name: _center_x(detected[name])
        for name in relevant
        if detected.get(name) is not None
    }

    order = [name for name, _ in sorted(positions.items(), key=lambda x: x[1])]
    issues = []

    if order[0] != "nyc":
        issues.append("NYC should be leftmost")
    if order[-1] != "bp":
        issues.append("BP should be rightmost")

    for name in ("sk", "yorp"):
        if name in collaborators:
            if name not in positions:
                issues.append(f"{name.upper()} is required but not detected")
            else:
                if positions[name] <= positions["nyc"]:
                    issues.append(f"{name.upper()} should be to the right of NYC")
                if positions[name] >= positions["bp"]:
                    issues.append(f"{name.upper()} should be to the left of BP")

    detected_order = " → ".join(n.upper() for n in order)
    return _make_result(
        passed=len(issues) == 0,
        label_ok="Logo order OK",
        label_fail="Logo order issues",
        remark_fail=" | ".join(issues),
        details={
            "order": detected_order,
            "positions": {k: round(v, 1) for k, v in positions.items()},
        },
        level ="error",
    )


def logo_report(image, model, conf_threshold:float=0.8, collaborators:list=None):
    """
    Run YOLO detection and generate a report dict for each logo, plus an annotated image.
    NYC and BP are always required.
    SK and YORP are only checked when explicitly listed in collaborators.
    """

    collaborators = [c.lower() if isinstance(c, str) else c for c in collaborators]

    detected = {"nyc": None, "bp": None, "sk": None, "yorp": None}
    results = model(image)
    for r in results:
        for box in r.boxes:
            conf = float(box.conf[0])
            if conf < conf_threshold:
                continue
            cls = int(box.cls[0])
            label = model.names[cls].lower()
            parts = label.split("_") if "_" in label else [label, "unknown"]
            logo_name, status = parts[0], parts[-1]
            if logo_name in detected:
                if detected[logo_name] is None or conf > detected[logo_name]["conf"]:
                    detected[logo_name] = {"status": status, "conf": conf, "box": box}

    report = []
    annotated = image.copy()
    for logo in ["nyc", "bp"] + [c for c in collaborators if c in ("sk", "yorp")]:
        entry = detected[logo]
        if entry is None:
            report.append({
                "logo": logo.upper(), 
                "detected": False, 
                "confidence": None,
                "status": "Missing", 
                "pass": False,
                "remark": "Add logo",
                "level": "error",
            })
        else:
            is_correct = entry["status"] == "correct"
            report.append({
                "logo": logo.upper(), 
                "detected": True, 
                "confidence": round(entry["conf"], 3),
                "status": entry["status"].capitalize(), 
                "pass":       is_correct,
                "remark": "OK" if is_correct else f"Incorrect version detected",
                "level": "error",
            })
            
            xyxy = entry["box"].xyxy[0].cpu().numpy().astype(int)
            color = (0, 255, 0) if is_correct else (0, 0, 255)
            
            cv2.rectangle(annotated, (xyxy[0], xyxy[1]), (xyxy[2], xyxy[3]), color, 2)
            cv2.putText(annotated, logo.upper(), (xyxy[0], xyxy[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    
    all_pass = all(r["pass"] for r in report)
    return {
        "pass":    all_pass,
        "level":   "error",
        "label":   "All logos OK" if all_pass else "Logo issues found",
        "remark":  "OK" if all_pass else f"{sum(1 for r in report if not r['pass'])} logo(s) failed",
        "details": {"logos": report},   # individual results still accessible here
    }, detected, annotated


def _get_logo_boxes_abs(detected: dict, img_shape: tuple) -> list:
    """Converts YOLO xyxy boxes to absolute int coords for masking."""
    boxes = []
    for entry in detected.values():
        if entry is None:
            continue
        xyxy = entry["box"].xyxy[0].cpu().numpy().astype(int)
        boxes.append((int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])))
    return boxes


def check_spelling(text: str, spell: SpellChecker) -> list[dict]:
    """
    Check text and return a list of misspelled words with their positions.

    """

    words = spell.split_words(text)
    misspelled = spell.unknown(words)
    errors = []
    search_start = 0

    for word in words:
        if word.lower() in misspelled:
            # Find the word's position in the original text
            idx = text.lower().find(word.lower(), search_start)
            if idx != -1:
                errors.append({
                    "word": word,
                    "start": idx,
                    "end": idx + len(word),
                    "suggestions": spell.candidates(word)
                })
                search_start = idx + len(word)

    return errors

# spell= load_spell_checker(SPELL_WORD_LISTS)

def check_spelling_on_image(
    image: np.ndarray,
    ocr_words: list,
    ocr_boxes: list,
    spell: SpellChecker | None = None,
) -> tuple[np.ndarray, dict]:
    """
    Underlines misspelled word.
    """
    if spell is None:
        spell = load_spell_checker(SPELL_WORD_LISTS)
 
    h_img, w_img = image.shape[:2]
    annotated = image.copy()
 
    misspelled_set = spell.unknown(
        [re.sub(r"[^a-zA-Z]", "", w) for w in ocr_words]
    )
    misspelled_set = {w.lower() for w in misspelled_set if w}
 
    found_errors = []
    for word, (x0_r, y0_r, x1_r, y1_r) in zip(ocr_words, ocr_boxes):
        clean_word = re.sub(r"[^a-zA-Z]", "", word).lower()
        if clean_word not in misspelled_set:
            continue
 
        # Convert relative → absolute pixel coordinates
        x0 = int(x0_r * w_img)
        x1 = int(x1_r * w_img)
        y1 = int(y1_r * h_img)
 
        # Draw a 2-px red underline just below the word's bottom edge
        underline_y = min(y1 + 2, h_img - 1)
        cv2.line(annotated, (x0, underline_y), (x1, underline_y), (0, 0, 255), 2)
 
        found_errors.append({
            "word": word,
            "suggestions": list(spell.candidates(clean_word) or []),
            "box_abs": (x0, int(y0_r * h_img), x1, y1),
        })
 
    passed = len(found_errors) == 0
    result = _make_result(
        passed=passed,
        label_ok="No spelling errors",
        label_fail=f"{len(found_errors)} spelling error(s) found",
        remark_fail=", ".join(e["word"] for e in found_errors),
        details={"errors": found_errors},
        level="warning",
    )
    return annotated, result

# ── Master report generator ───────────────────────────────────────────────────
def generate_report(image, logo_model, post_type: str, collaborators:list=None) -> tuple:
    """
    Runs all checks and returns a unified audit report + annotated image.
    docTR OCR is run exactly once and shared across all checks.

    Which checks run is determined entirely by POST_TYPE_RULES for the given
    post_type.

    Returns:
        audit (dict)    - full structured report
        img   (ndarray) - annotated image with bounding boxes
    """
    
    rules = POST_TYPE_RULES.get(post_type.lower(), {})

    if image is None or image.size == 0:
        raise ValueError("Image could not be decoded or is empty.")
    img = image.copy()
    h_img = img.shape[0]
    # --- build audit dict incrementally ---
    audit = {
        "post_type":      post_type,
        "overall":        None,         # filled in after all checks run
    }
    # Logo detection
    logo_result, detected, img_annotated = logo_report(
        img, 
        model=logo_model,
        conf_threshold=0.8,
        collaborators=collaborators or [],
    )
    audit["logos"] = logo_result  
    # Masked image for OCR 
    logo_boxes_abs = _get_logo_boxes_abs(detected, img.shape)
    masked_image   = _mask_regions(img_annotated, logo_boxes_abs)

    # OCR on masked image: logos and watermark strip are blank 
    ocr_words, ocr_confidences, ocr_boxes = _extract_ocr_data(_run_doctr(masked_image))
    # filter out bottom strip (watermark) for readability and spell check
    content_confidences = [c for c, b in zip(ocr_confidences, ocr_boxes) if b[1] < 0.85]
    content_words       = [w for w, b in zip(ocr_words, ocr_boxes) if b[1] < 0.85]
    content_boxes       = [b for b in ocr_boxes if b[1] < 0.85]


    # Logo Order Check
    logo_order = check_logo_order(detected, collaborators=collaborators or [])
    audit["logo_order"] = logo_order

    # PubMat Quality Check
    pubmat_quality = check_pubmat_quality(img)
    audit["pubmat_quality"] = pubmat_quality

    # Conditional checks based on post type rules:
    
    # Watermark check

    if rules.get("requires_watermark"):
        bottom_pairs = [(w, b) for w, b in zip(ocr_words, ocr_boxes) if b[1] >= 0.85]
        watermark_result, watermark_boxes_abs = check_watermark(
            image,
            precomputed_words=[p[0] for p in bottom_pairs],
            precomputed_boxes=[p[1] for p in bottom_pairs],
        )
        audit["watermark"] = watermark_result
        for (x0, y0, x1, y1) in watermark_boxes_abs:
            cv2.rectangle(img_annotated, (x0, y0), (x1, y1), (255, 200, 0), 1)
        label_text = "Watermark OK" if watermark_result["pass"] \
            else f"Watermark MISSING: {', '.join(watermark_result['missing'])}"
        
        color = (0, 255, 0) if watermark_result["pass"] else (255, 0, 0)
        cv2.putText(img_annotated, label_text, (10, h_img - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

        
    # Readability check
    if "readability_threshold" in rules:
        threshold = rules["readability_threshold"]
        readability = check_readability(content_confidences, threshold=threshold)
        audit["readability"] = readability
    
    # Spelling check
    if rules.get("requires_spell_check"):
        img_annotated, spell_result = check_spelling_on_image(
        img_annotated, content_words, content_boxes)
        audit["spelling"] = spell_result

    # SGD check
    if rules.get("requires_sgd"):
        audit["sgd"] = check_sgd(ocr_words)

    # Photo 
    if rules.get("check_photo_quality"):
        audit["photo_quality"] = check_photo_quality(
            img)


    # 8. Overall pass/fail

    SKIP_KEYS  = {"post_type", "overall", "logos"}
    overall_pass = all(
    v["pass"] or v.get("level") != "error"
    for k, v in audit.items()
    if k not in SKIP_KEYS and isinstance(v, dict) and "pass" in v
)
    audit["overall"] = "PASS" if overall_pass else "FAIL"
 
    return audit, img_annotated

