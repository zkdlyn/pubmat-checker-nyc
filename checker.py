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
                    words.append(word.value)
                    confidences.append(word.confidence)
                    (x0, y0), (x1, y1) = word.geometry
                    boxes.append((x0, y0, x1, y1))
    return words, confidences, boxes

def _make_result(passed: bool, label_ok: str, label_fail: str, 
                 remark_ok: str="OK", remark_fail: str="Issue found", 
                 details: dict = None) -> dict:
    return{
        "pass": passed,
        "label": label_ok if passed else label_fail,
        "remark": remark_ok if passed else remark_fail,
        "details": details or {},
    }
    

# ── Rule config per post type ─────────────────────────────────────────────────

POST_TYPE_RULES = {
    "news": {
        "requires_watermark": True,
        "requires_template": True,
        "readability_threshold": 0.70,
    },
    "quotes": {
        "requires_template": True,
        "readability_threshold": 0.70,
    },
    "advisory": {
        "requires_template": True,
        "readability_threshold": 0.70,
        "requires_sgd": True,
    },
    "resolution": {
        "requires_template": True,
        "readability_threshold": 0.70,
        "requires_sgd": True,
    },
    "opportunity": {
        "requires_watermark": True,
        "requires_template": True,
        "readability_threshold": 0.70,
    },
    "photo": {
        "requires_template": True,
        "check_photo_quality": True
    },
    "holiday": {
        "requires_watermark": True,
        "requires_template": False,
        "readability_threshold": 0.50,
    },
    "other": {
        "requires_watermark": True,
        "requires_template": False,
        "readability_threshold": 0.50,
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
    return _make_result(
        passed=passed,
        label_ok="Watermark OK",
        label_fail="Watermark missing or incorrect",
        remark_fail= f"Missing: {', '.join(missing)}"
    )
    result["missing"] = missing
    result["boxes"] = boxes_abs
    return result


def check_readability(image, confidences:list, boxes: list, 
                    threshold: float =0.65) -> dict:
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
        details={"Resolution": f"{w}x{h}", "Blur Metric": round(cv2.Laplacian(gray, cv2.CV_64F).var(), 1), "Contrast Metric": round(np.std(gray), 1)},
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
    details["resolution"] = f"{w}x{h} (required {min_w}x{min_h})"
    if w < min_w or h < min_h:
        issues.append(f"Image is {w}x{h}, minimum is {min_w}x{min_h}")

    # Brightness
    mean_brightness = float(np.mean(gray))
    details["brightness"] = round(mean_brightness, 1)
    if mean_brightness < 60:
        issues.append("Image appears dark")

    # Colour saturation
    mean_saturation = float(np.mean(cv2.cvtColor(image, cv2.COLOR_BGR2HSV)[:, :, 1]))
    details["mean_saturation"] = round(mean_saturation, 1)
    if mean_saturation < 20:
        issues.append("Image appears grayscale or desaturated — use a colorized photo")

    passed = len(issues) == 0
    return {
        "pass": passed,
        "label": "Photo quality OK" if passed else "Photo quality issues",
        "details": details,
        "remark": "OK" if passed else " | ".join(issues),
    }


def check_logo_order(detected, collaborators=None):
    """
    Validates left-to-right order: NYC leftmost, BP rightmost,
    SK/YORP (if required) in between.
    """
    if collaborators is None:
        collaborators = []
    collaborators = [c.lower() for c in collaborators]

    if detected.get("nyc") is None or detected.get("bp") is None:
        return {
            "pass": False,
            "order_valid": False,
            "label": "Cannot check order",
            "remark": "NYC or BP not detected",
            "details": {},
        }

    def get_center_x(entry):
        xyxy = entry["box"].xyxy[0].cpu().numpy().astype(int)
        return (xyxy[0] + xyxy[2]) / 2

    relevant = ["nyc", "bp"] + [c for c in collaborators if c in ("sk", "yorp")]
    positions = {
        name: get_center_x(detected[name])
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

    passed = len(issues) == 0
    detected_order = " → ".join(n.upper() for n in order)
    return {
        "pass": passed,
        "order_valid": passed,
        "label": "Logo order OK" if passed else "Logo order issues",
        "detected_order": detected_order,
        "remark": "OK" if passed else " | ".join(issues),
        "details": {
            "order": detected_order,
            "positions": {k: round(v, 1) for k, v in positions.items()},
        },
    }


def logo_report(results, model, img, conf_threshold=0.8, collaborators=None):
    """
    NYC and BP are always required.
    SK and YORP are only checked when explicitly listed in collaborators.
    """
    if collaborators is None:
        collaborators = []
    collaborators = [c.lower() if isinstance(c, str) else c for c in collaborators]

    detected = {"nyc": None, "bp": None, "sk": None, "yorp": None}

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
    for logo in ["nyc", "bp"] + [c for c in collaborators if c in ("sk", "yorp")]:
        entry = detected[logo]
        if entry is None:
            report.append({
                "Logo": logo.upper(), "Detected": "No", "Confidence": "-",
                "Status": "Missing", "Remark": "Add Logo", "_compliant": False,
            })
        else:
            is_correct = entry["status"] == "correct"
            report.append({
                "Logo": logo.upper(), "Detected": "Yes",
                "Confidence": round(entry["conf"], 3),
                "Status": entry["status"].capitalize(),
                "Remark": "Logo is correct" if is_correct else "Revise logo",
                "_compliant": is_correct,
            })
            xyxy = entry["box"].xyxy[0].cpu().numpy().astype(int)
            color = (0, 255, 0) if is_correct else (0, 0, 255)
            cv2.rectangle(img, (xyxy[0], xyxy[1]), (xyxy[2], xyxy[3]), color, 2)
            cv2.putText(img, logo.upper(), (xyxy[0], xyxy[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    return report, detected, img


# ── Master report generator ───────────────────────────────────────────────────
def generate_report(image, logo_model, post_type, collaborators=None):
    """
    Runs all checks and returns a unified audit report + annotated image.
    docTR OCR is run exactly once and shared across all checks.

    Which checks run is determined entirely by POST_TYPE_RULES for the given
    post_type — no logic is hardcoded here beyond reading those flags.

    Returns:
        audit (dict)    — full structured report
        img   (ndarray) — annotated image with bounding boxes
    """
    img = image.copy()

    if img is None or img.size == 0:
        raise ValueError("Image could not be decoded or is empty.")

    rules = POST_TYPE_RULES.get(post_type.lower(), {})
    h_img = image.shape[0]

    # 1. OCR — run once, reused by readability, watermark, and SGD
    ocr_words, ocr_confidences, ocr_boxes = _extract_ocr_data(_run_doctr(image))

    # 2. Logo detection + annotation

    logo_rep, detected, img = logo_report( img, model=logo_model,
        conf_threshold=0.8,
        collaborators=collaborators or [],
    )

    # 3. Logo order (all post types)
    order_result = check_logo_order(detected, collaborators=collaborators or [])
    
    # 4. Pubmat quality (all post types)
    pubmat_quality = check_pubmat_quality(image)

    # build audit dict incrementally 
    audit = {
        "post_type":      post_type,
        "overall":        None,         # filled in after all checks run
        "logos":          [{k: v for k, v in r.items() if not k.startswith("_")} for r in logo_rep],
        "logo_order":     order_result,
        "pubmat_quality": pubmat_quality,
    }

    # .5 Watermark (conditional)

    if rules.get("requires_watermark"):
        bottom_pairs = [(w, b) for w, b in zip(ocr_words, ocr_boxes) if b[1] >= 0.85]
        wm_result = check_watermark(
            image,
            precomputed_words=[p[0] for p in bottom_pairs],
            precomputed_boxes=[p[1] for p in bottom_pairs],
        )
        for (x0, y0, x1, y1) in wm_result["boxes"]:
            cv2.rectangle(img, (x0, y0), (x1, y1), (255, 200, 0), 1)
        label_text = "Watermark OK" if wm_result["watermark_present"] \
            else f"Watermark MISSING: {', '.join(wm_result['missing'])}"
        color = (0, 255, 0) if wm_result["watermark_present"] else (255, 0, 0)
        cv2.putText(img, label_text, (10, h_img - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
        audit["watermark"] = wm_result
        
    # 6. Readability score 
    if "readability_threshold" in rules:
        threshold = rules["readability_threshold"]
        readability = check_readability(image, ocr_confidences, ocr_boxes)
        score = readability["score"]
        audit["readability"] = {
            **readability,
            "threshold": threshold,
            "pass":      score >= threshold,
            "remark":    "OK" if score >= threshold else f"Score {score} below threshold {threshold}",
        }

    # 7. 

    if rules.get("requires_sgd"):
        audit["sgd"] = check_sgd(ocr_words)

    if rules.get("check_photo_quality"):
        audit["photo_quality"] = check_photo_quality(
            image, min_resolution=rules.get("min_resolution", (1080, 1080))
        )


    # 8. Overall pass/fail
    ALWAYS_REQUIRED = {"post_type", "logos", "overall"}
    logo_issues = any(not r["_compliant"] for r in logo_rep)
    conditional_checks ={
        k: v for k, v in audit.items() if k not in ALWAYS_REQUIRED and isinstance(v, dict) and "pass" in v
    }
    overall_pass = (
        not logo_issues
        and all(v["pass"] for v in conditional_checks.values())

    )

    audit["overall"] = "PASS" if overall_pass else "FAIL"

    return audit, img

