import cv2
import numpy as np
import difflib
from doctr.models import ocr_predictor
from doctr.io import DocumentFile
from PIL import Image
import tempfile
import os

# LOAD DOCTR MODEL ONCE
_doctr_model = None

def get_doctr_model():
    global _doctr_model
    if _doctr_model is None:
        _doctr_model = ocr_predictor(pretrained=True)
    return _doctr_model


# ── Shared helper: run docTR OCR on a numpy BGR image ────────────────────────
def _run_doctr(image_bgr):
    """
    Converts a BGR numpy array to a temp JPEG, runs docTR, returns the result.
    Cleans up the temp file in all cases.
    """
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


# rule config per post type
POST_TYPE_RULES = {
    "news": {
        "requires_watermark": True,
        "requires_template": True,
        "readability_threshold": 0.65,
    },
    "quotes": {
        "requires_template": True,
        "readability_threshold": 0.60,
    },
    "advisory": {
        "requires_template": True,
        "readability_threshold": 0.70,
        # "requires_full_page": True,
        "requires_sgd": True,
    },
    "resolution": {
        "requires_template": True,
        "readability_threshold": 0.70,
        # "requires_full_page": True,
        "requires_sgd": True,
    },
    "hiring": {
        "requires_watermark": True,
        "requires_template": True,
        "readability_threshold": 0.65,
        # "strict_logo_order": True,
    },
    "photo": {
        "requires_template": True,
        "readability_threshold": 0.50,
        "check_photo_quality": True,
        "min_resolution": (1080, 1080),
    },
    "holiday": {
        "requires_watermark": True,
        "requires_template": False,
        "readability_threshold": 0.50,
        # "strict_logo_order": False,
    },
    "other": {
        "requires_watermark": True,
        "requires_template": False,
        "readability_threshold": 0.50,
        # "strict_logo_order": True,
    }
}

WATERMARK_HANDLES = [
    "nyc.gov.ph",
    "nationalyouthcommission",
    "@nycpilipinas",
]

# Fuzzy match threshold — 0.0 to 1.0, lower = more lenient
WATERMARK_FUZZY_THRESHOLD = 0.75


# ── Watermark check using OCR ─────────────────────────────────────────────────
def check_watermark(image):
    """
    Crops the bottom 15% of the image, runs docTR OCR on it,
    and fuzzy-matches against known watermark handles.
    Returns detection results and the bounding boxes found.
    """
    img_h, img_w = image.shape[:2]
    crop_y = int(img_h * 0.85)
    crop = image[crop_y:img_h, 0:img_w]

    result = _run_doctr(crop)

    found_handles = {handle: False for handle in WATERMARK_HANDLES}
    raw_words = []
    boxes_abs = []

    for page in result.pages:
        for block in page.blocks:
            for line in block.lines:
                for word in line.words:
                    raw_words.append(word.value.lower())
                    # docTR boxes are relative (0-1); convert to absolute image coords
                    (x0, y0), (x1, y1) = word.geometry
                    abs_x0 = int(x0 * img_w)
                    abs_y0 = crop_y + int(y0 * crop.shape[0])
                    abs_x1 = int(x1 * img_w)
                    abs_y1 = crop_y + int(y1 * crop.shape[0])
                    boxes_abs.append((abs_x0, abs_y0, abs_x1, abs_y1))

    full_text = " ".join(raw_words)
    match_results = {}
    for handle in WATERMARK_HANDLES:
        handle_clean = handle.lower().replace("@", "")
        ratio = difflib.SequenceMatcher(None, handle_clean, full_text).ratio()
        best_word_ratio = max(
            (difflib.SequenceMatcher(None, handle_clean, w).ratio() for w in raw_words),
            default=0.0
        )
        best = max(ratio, best_word_ratio)
        match_results[handle] = {
            "found": best >= WATERMARK_FUZZY_THRESHOLD,
            "score": round(best, 3)
        }

    all_present = all(v["found"] for v in match_results.values())
    missing = [h for h, v in match_results.items() if not v["found"]]

    return {
        "watermark_present": all_present,
        "handles": match_results,
        "missing": missing,
        "remark": "Watermark OK" if all_present else f"Missing: {', '.join(missing)}",
        "boxes": boxes_abs,
    }

# ── Readability check (docTR) ─────────────────────────────────────────────────
def check_readability(image, threshold=0.65):
    """
    Uses docTR for OCR confidence + CV metrics for blur/contrast/size.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    remarks = []

    result = _run_doctr(image)

    confidences = []
    word_heights_rel = []

    for page in result.pages:
        for block in page.blocks:
            for line in block.lines:
                for word in line.words:
                    confidences.append(word.confidence)
                    (x0, y0), (x1, y1) = word.geometry
                    word_heights_rel.append(y1 - y0)

    if not confidences:
        return {
            "Readability Status": "No readable text found",
            "Remarks": "No text detected",
            "Score": 0.0
        }

    avg_conf = sum(confidences) / len(confidences)
    ocr_score = avg_conf  # docTR confidence is already 0-1
    if ocr_score < 0.5:
        remarks.append("Low OCR confidence")

    # Text size: relative height * image height → pixel height
    img_h = image.shape[0]
    if word_heights_rel:
        avg_height_px = (sum(word_heights_rel) / len(word_heights_rel)) * img_h
        if avg_height_px < 12:
            size_score = 0.4
            remarks.append("Text too small")
        elif avg_height_px < 20:
            size_score = 0.7
        else:
            size_score = 1.0
    else:
        avg_height_px = 0
        size_score = 0.0
        remarks.append("No word geometry data")

    # Contrast check
    global_contrast = np.std(gray)
    contrast_score = min(global_contrast / 60, 1.0)
    if contrast_score < 0.5:
        remarks.append("Low contrast")

    # Blur check
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    if laplacian_var < 50:
        blur_score = 0.3
        remarks.append("Blurry image")
    elif laplacian_var < 100:
        blur_score = 0.6
        remarks.append("Slightly blurry")
    else:
        blur_score = 1.0

    final_score = (
        0.4 * ocr_score +
        0.2 * size_score +
        0.2 * contrast_score +
        0.2 * blur_score
    )

    if final_score < 0.5:
        label = "Low readability"
    elif final_score < threshold:
        label = "Moderate readability"
    else:
        label = "Readable"

    return {
        "Readability Status": label,
        "Remarks": ", ".join(remarks) if remarks else "Good readability",
        "Score": round(final_score, 3)
    }


# ── Pubmat quality check (all post types) ────────────────────────────────────
def check_pubmat_quality(image):
    """
    Universal image quality checks applied to every post type.
    Detects pixelation, blur, low resolution, and low contrast.

    Pixelation detection: compares the Laplacian edge response before and after
    a mild blur. A clean high-res image loses significant edge energy when blurred.
    A pixelated/blocky image already has coarse, low-frequency edges that survive
    blurring — so the ratio stays high, indicating pixelation.

    Returns a dict with pass/fail, individual sub-check details, and remarks.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    h, w = image.shape[:2]
    issues = []
    details = {}

    # --- Resolution floor (applies to all post types) ---
    # 720x720 is the minimum acceptable pubmat size.
    MIN_W, MIN_H = 720, 720
    res_ok = w >= MIN_W and h >= MIN_H
    details["resolution"] = f"{w}x{h}"
    if not res_ok:
        issues.append(f"Resolution {w}x{h} is below minimum {MIN_W}x{MIN_H}")

    # --- Blur check ---
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    details["laplacian_var"] = round(laplacian_var, 2)
    if laplacian_var < 50:
        issues.append("Image is blurry")
    elif laplacian_var < 100:
        issues.append("Image is slightly blurry")

    # --- Pixelation check ---
    # Sharp edges in a pixelated image are blocky and survive blurring more than
    # natural edges. We compare edge energy before and after a Gaussian blur:
    # a high survival ratio (blurred energy / original energy) signals pixelation.
    lap_orig = cv2.Laplacian(gray, cv2.CV_64F).var()
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    lap_blurred = cv2.Laplacian(blurred, cv2.CV_64F).var()
    # Avoid division by zero
    if lap_orig > 0:
        pixelation_ratio = lap_blurred / lap_orig
    else:
        pixelation_ratio = 1.0
    details["pixelation_ratio"] = round(pixelation_ratio, 3)
    # A ratio above 0.5 means blurring barely reduced edge energy — blocky/pixelated.
    # Threshold tuned conservatively to avoid false positives on busy graphics.
    if pixelation_ratio > 0.5 and laplacian_var > 50:
        issues.append("Image appears pixelated or upscaled")

    # --- Contrast check ---
    contrast = float(np.std(gray))
    details["contrast_std"] = round(contrast, 2)
    if contrast < 20:
        issues.append("Very low contrast — image may appear washed out or too dark")
    elif contrast < 35:
        issues.append("Low contrast")

    overall_pass = len(issues) == 0
    return {
        "pass": overall_pass,
        "details": details,
        "remark": "OK" if overall_pass else " | ".join(issues),
    }


# ── Logo order check ──────────────────────────────────────────────────────────
def check_logo_order(detected: dict, collaborators: list = None):
    """
    Validates the spatial left-to-right order of detected logos.
    Rule: NYC must be leftmost, BP must be rightmost.
    SK and YORP (if present) must be between NYC and BP.

    `detected` is the dict from logo_report: { "nyc": {..., "box": box}, ... }
    """
    present = {
        name: entry for name, entry in detected.items()
        if entry is not None
    }

    if collaborators is None:
        collaborators = []
 
    required = {"nyc", "bp"} | {c for c in collaborators if c in ("sk", "yorp")}
    
    # Only consider logos that are both required and detected
    present = {
        name: entry for name, entry in detected.items()
        if entry is not None and name in required
    }


    if "nyc" not in present or "bp" not in present:
        return {
            "order_valid": False,
            "detected_order": "N/A — NYC or BP not detected",
            "remark": "Cannot check order — NYC or BP not detected",
            "details": {}
        }

    def get_center_x(entry):
        xyxy = entry["box"].xyxy[0].cpu().numpy().astype(int)
        return (xyxy[0] + xyxy[2]) / 2

    positions = {name: get_center_x(entry) for name, entry in present.items()}
    sorted_logos = sorted(positions.items(), key=lambda x: x[1])
    order = [name for name, _ in sorted_logos]

    issues = []

    if order[0] != "nyc":
        issues.append(f"NYC should be first but found '{order[0]}' leftmost")
    if order[-1] != "bp":
        issues.append(f"Bagong Pilipinas should be last but found '{order[-1]}' rightmost")

    collab_logos = {"sk", "yorp"}
    for name in collab_logos:
        if name in positions:
            if positions[name] <= positions["nyc"]:
                issues.append(f"{name.upper()} is to the left of or overlapping NYC")
            if positions[name] >= positions["bp"]:
                issues.append(f"{name.upper()} is to the right of or overlapping Bagong Pilipinas")

    return {
        "order_valid": len(issues) == 0,
        "detected_order": " → ".join(n.upper() for n in order),
        "remark": "Logo order correct" if not issues else " | ".join(issues),
        "details": {n: round(x, 1) for n, x in positions.items()}
    }


# ── Per-type post rules check ─────────────────────────────────────────────────
def check_post_type_rules(post_type: str, image, readability_result: dict,
                          detected: dict = None, wm_result: dict = None,
                          order_result: dict = None):
    """
    Applies post-type-specific checks beyond logos and watermark.
    """
    rules = POST_TYPE_RULES.get(post_type.lower())
    if rules is None:
        return {
            "post_type": post_type,
            "status": "Unknown post type",
            "checks": {}
        }

    checks = {}
    h, w = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # --- Readability threshold ---
    threshold = rules.get("readability_threshold", 0.65)
    score = readability_result.get("Score", 0)
    checks["readability"] = {
        "pass": score >= threshold,
        "score": score,
        "threshold": threshold,
        "remark": "OK" if score >= threshold else f"Score {score} below threshold {threshold}"
    }

    # --- Logo order check (reuse pre-computed result, avoid double call) ---
    if rules.get("strict_logo_order") and detected:
        result = order_result if order_result is not None else check_logo_order(detected)
        checks["logo_order"] = {
            "pass": result["order_valid"],
            "remark": result["remark"]
        }

    # --- Watermark check (reuse pre-computed result, avoid double OCR) ---
    if rules.get("requires_watermark"):
        result = wm_result if wm_result is not None else check_watermark(image)
        checks["watermark"] = {
            "pass": result["watermark_present"],
            "remark": result["remark"]
        }

    # --- Photo: quality checks (resolution, centering, lighting, color) ---
    if rules.get("check_photo_quality"):
        min_w, min_h = rules.get("min_resolution", (1080, 1080))
        photo_issues = []
        photo_details = {}

        # Resolution
        res_pass = w >= min_w and h >= min_h
        photo_details["resolution"] = f"{w}x{h} (required {min_w}x{min_h})"
        if not res_pass:
            photo_issues.append(f"Image is {w}x{h}, minimum is {min_w}x{min_h}")

        # Subject centering
        edges = cv2.Canny(gray, 50, 150)
        moments = cv2.moments(edges)
        if moments["m00"] > 0:
            cx = moments["m10"] / moments["m00"]
            cy = moments["m01"] / moments["m00"]
            offset_x = abs(cx - w / 2) / (w / 2)
            offset_y = abs(cy - h / 2) / (h / 2)
            centered = offset_x < 0.25 and offset_y < 0.25
            photo_details["centering"] = f"offset_x={round(offset_x, 3)}, offset_y={round(offset_y, 3)}"
        else:
            centered = False
            photo_details["centering"] = "Could not compute"
        if not centered:
            photo_issues.append("Subject may be off-center — review composition")

        # Lighting
        mean_brightness = float(np.mean(gray))
        too_dark = mean_brightness < 60
        photo_details["brightness"] = round(mean_brightness, 1)
        if too_dark:
            photo_issues.append("Image appears dark")

        # Color check — flags grayscale/desaturated images
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        mean_saturation = float(np.mean(hsv[:, :, 1]))
        is_colorized = mean_saturation >= 20
        photo_details["mean_saturation"] = round(mean_saturation, 1)
        if not is_colorized:
            photo_issues.append("Image appears grayscale or desaturated — use a colorized photo")

        checks["photo_quality"] = {
            "pass": len(photo_issues) == 0,
            "details": photo_details,
            "remark": "OK" if not photo_issues else " | ".join(photo_issues),
        }
    # --- Advisory / Resolution: SGD text check ---
    if rules.get("requires_sgd"):
        # Reuse the docTR result from readability if possible — run fresh only here
        doctr_result = _run_doctr(image)
        extracted_text = ""
        for page in doctr_result.pages:
            for block in page.blocks:
                for line in block.lines:
                    for word in line.words:
                        extracted_text += word.value + " "

        sgd_found = "sgd" in extracted_text.lower()
        checks["SGD"] = {
            "pass": sgd_found,
            "remark": "SGD Present" if sgd_found else "Use SGD for resolutions/advisories"
        }

    overall_pass = all(v["pass"] for v in checks.values())
    return {
        "post_type": post_type,
        "status": "Pass" if overall_pass else "Fail",
        "checks": checks
    }


# ── Logo detection + annotation ───────────────────────────────────────────────
def logo_report(results, model, img, conf_threshold=0.25, collaborators=None):
    """
    NYC and BP are always required.
    SK and YORP are only checked and included in the report when explicitly
    listed in collaborators (e.g. ["sk"], ["yorp"], ["sk", "yorp"]).
    If not in collaborators, SK/YORP are silently ignored regardless of
    whether the model detects them.
    """
    if collaborators is None:
        collaborators = []

    collaborators = [c.lower() if isinstance(c, str) else c for c in collaborators]

    # Always track all four so check_logo_order has position data if needed,
    # but only report/evaluate SK and YORP when they are required.
    detected = {"nyc": None, "bp": None, "sk": None, "yorp": None}

    for r in results:
        for box in r.boxes:
            conf = float(box.conf[0])
            if conf < conf_threshold:
                continue
            cls = int(box.cls[0])
            label = model.names[cls].lower()
            if "_" in label:
                parts = label.split("_")
                logo_name = parts[0]
                status = parts[-1]
            else:
                logo_name = label
                status = "unknown"

            if logo_name in detected:
                if detected[logo_name] is None or conf > detected[logo_name]["conf"]:
                    detected[logo_name] = {"status": status, "conf": conf, "box": box}

    # NYC and BP are always required; SK/YORP only when in collaborators.
    logos_to_report = ["nyc", "bp"] + [c for c in collaborators if c in ("sk", "yorp")]

    report = []
    for logo in logos_to_report:
        entry = detected[logo]

        if entry is None:
            report.append({
                "Logo": logo.upper(),
                "Detected": "No",
                "Confidence": "-",
                "Status": "Missing",
                "Remark": "Add Logo",
                "_compliant": False,
            })
        else:
            status = entry["status"]
            conf = round(entry["conf"], 3)
            is_correct = status == "correct"
            report.append({
                "Logo": logo.upper(),
                "Detected": "Yes",
                "Confidence": conf,
                "Status": status.capitalize(),
                "Remark": "Logo is correct" if is_correct else "Revise logo",
                "_compliant": is_correct,
            })
            xyxy = entry["box"].xyxy[0].cpu().numpy().astype(int)
            color = (0, 255, 0) if is_correct else (0, 0, 255)
            cv2.rectangle(img, (xyxy[0], xyxy[1]), (xyxy[2], xyxy[3]), color, 2)
            cv2.putText(img, f"{logo.upper()}",
                        (xyxy[0], xyxy[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    return report, detected, img


# ── Master report generator ───────────────────────────────────────────────────
def generate_report(yolo_results, model, image, post_type,
                    collaborators=None, conf_threshold=0.25):
    """
    Runs all checks and returns a unified audit report + annotated image.

    Returns:
        audit (dict)    — full structured report
        img   (ndarray) — annotated image with logo + watermark boxes
    """
    img = image.copy()

    # Guard against undecodable images
    if img is None or img.size == 0:
        raise ValueError("Image could not be decoded or is empty.")

    # 1. Logo detection + bounding box annotation
    logo_rep, detected, img = logo_report(
        yolo_results, model, img,
        conf_threshold=conf_threshold,
        collaborators=collaborators or []
    )

    # 2. Logo order — only evaluates required logos (NYC, BP + collaborators)
    order_result = check_logo_order(detected, collaborators=collaborators or [])

    # 3. Watermark (only if required by post type)
    rules = POST_TYPE_RULES.get(post_type.lower(), {})
    requires_watermark = rules.get("requires_watermark", False)

    if requires_watermark:
        wm_result = check_watermark(img)
        for (x0, y0, x1, y1) in wm_result["boxes"]:
            cv2.rectangle(img, (x0, y0), (x1, y1), (255, 200, 0), 1)
        h_img = image.shape[0]
        if wm_result["watermark_present"]:
            cv2.putText(img, "Watermark OK",
                        (10, h_img - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        else:
            cv2.putText(img, f"Watermark MISSING: {', '.join(wm_result['missing'])}",
                        (10, h_img - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 0, 0), 2)
    else:
        wm_result = None

    # 4. Pubmat quality — universal check for all post types
    pubmat_quality = check_pubmat_quality(image)

    # 5. Readability (docTR)
    readability = check_readability(image)

    # 6. Post-type-specific rules — pass pre-computed results to avoid redundant calls
    type_rules = check_post_type_rules(
        post_type, image, readability,
        detected=detected,
        wm_result=wm_result,
        order_result=order_result
    )

    # 7. Overall pass/fail — use the explicit _compliant boolean field
    logo_issues = any(not r["_compliant"] for r in logo_rep)

    overall_pass = (
        not logo_issues
        and pubmat_quality["pass"]
        and type_rules["status"] == "Pass"
    )

    # Strip internal field before returning to UI
    logos_display = [{k: v for k, v in r.items() if not k.startswith("_")} for r in logo_rep]

    audit = {
        "post_type": post_type,
        "overall": "PASS" if overall_pass else "FAIL",
        "logos": logos_display,
        "logo_order": order_result,
        "pubmat_quality": pubmat_quality,
        "readability": readability,
        "type_checks": type_rules,
    }

    if wm_result is not None:
        audit["watermark"] = wm_result

    return audit, img
