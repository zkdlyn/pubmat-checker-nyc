import cv2
import numpy as np
import difflib
from PIL import Image
import tempfile
import os

# LOAD DOCTR MODEL ONCE
_doctr_model = None
DOCTR_AVAILABLE = False

try:
    from doctr.models import ocr_predictor
    from doctr.io import DocumentFile
    DOCTR_AVAILABLE = True
except ImportError:
    print("Warning: doctr not available. OCR features will be limited.")
    ocr_predictor = None
    DocumentFile = None

def get_doctr_model():
    global _doctr_model
    if not DOCTR_AVAILABLE:
        raise ImportError("doctr module not available on this deployment")
    if _doctr_model is None:
        _doctr_model = ocr_predictor(pretrained=True)
    return _doctr_model


# rule config per post type
POST_TYPE_RULES = {
    "news": {
        "requires_watermark": True,
        "requires_template": True,
        "readability_threshold": 0.65,
        "strict_logo_order": True,
    },
    "quotes": {
        "requires_watermark": True,
        "requires_template": True,
        "readability_threshold": 0.60,
        "strict_logo_order": True,
    },
    "advisory": {
        "requires_template": True,
        "readability_threshold": 0.70,
        "strict_logo_order": True,
        "requires_full_page": True,       
        "requires_sgd": True,        
    },
    "resolution": {
        "requires_template": True,
        "readability_threshold": 0.70,
        "strict_logo_order": True,
        "requires_full_page": True,
        "requires_sgd": True,
    },
    "hiring": {
        "requires_watermark": True,
        "requires_template": True,
        "readability_threshold": 0.65,
        "strict_logo_order": True,
    },
    "photo": {
        "requires_watermark": False,
        "requires_template": True,
        "readability_threshold": 0.50,    
        "strict_logo_order": True,
        "requires_high_resolution": True,
        "min_resolution": (1080, 1080),   
        "check_centering": True,
        "check_lighting": True,
    },
    "holiday": {
        "requires_watermark": True,
        "requires_template": False,       
        "readability_threshold": 0.50,
        "strict_logo_order": False,
    },
    "other": {
        "requires_watermark": True,
        "requires_template": False,
        "readability_threshold": 0.50,
        "strict_logo_order": True,
    }
}

WATERMARK_HANDLES = [
    "nyc.gov.ph",
    "nationalyouthcommission",
    "@nycpilipinas",
]

# Fuzzy match threshold — 0.0 to 1.0, lower = more lenient
WATERMARK_FUZZY_THRESHOLD = 0.75


# watermark check using OCR
def check_watermark(image):
    """
    Crops the bottom 15% of the image, runs docTR OCR on it,
    and fuzzy-matches against known watermark handles.
    Returns detection results and the bounding boxes found.
    """
    if not DOCTR_AVAILABLE:
        return {"found": False, "reason": "OCR module not available on this deployment"}
    h, w = image.shape[:2]
    crop_y = int(h * 0.85)
    crop = image[crop_y:h, 0:w]

    # Convert numpy array to PIL Image and save to temp file for docTR
    crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    crop_pil = Image.fromarray(crop_rgb)
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
        tmp_path = tmp.name
        crop_pil.save(tmp_path)
    
    try:
        doc = DocumentFile.from_images([tmp_path])
        result = get_doctr_model()(doc)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

    found_handles = {h: False for h in WATERMARK_HANDLES}
    raw_words = []
    boxes_abs = []  

    for page in result.pages:
        for block in page.blocks:
            for line in block.lines:
                line_text = " ".join(w.value for w in line.words).lower().strip()
                for word in line.words:
                    raw_words.append(word.value.lower())
                    # docTR boxes are relative (0-1); convert to absolute crop coords
                    (x0, y0), (x1, y1) = word.geometry
                    abs_x0 = int(x0 * w)
                    abs_y0 = crop_y + int(y0 * crop.shape[0])
                    abs_x1 = int(x1 * w)
                    abs_y1 = crop_y + int(y1 * crop.shape[0])
                    boxes_abs.append((abs_x0, abs_y0, abs_x1, abs_y1))

    # Fuzzy match each handle against all detected words and line combinations
    full_text = " ".join(raw_words)
    match_results = {}
    for handle in WATERMARK_HANDLES:
        handle_clean = handle.lower().replace("@", "")
        ratio = difflib.SequenceMatcher(None, handle_clean, full_text).ratio()
        # Also check individual word matches
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


# ── Readability check (docTR) ────────────────────────────────────────────────
def check_readability(image, threshold=0.65):
    """
    Uses docTR for OCR confidence + existing CV metrics for blur/contrast/size.
    """
    if not DOCTR_AVAILABLE:
        return {"readability_score": 0.5, "status": "WARN", "reason": "OCR module not available - using fallback"}
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    remarks = []

    # --- docTR OCR confidence ---
    # Convert numpy array to PIL Image and save to temp file for docTR
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
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

    confidences = []
    word_heights_rel = []  # relative heights from docTR geometry

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
    else:
        avg_height_px = 0
        size_score = 0.0
        remarks.append("No word geometry data")
        # Skip to final calculation if no height data
        final_score = (
            0.4 * ocr_score +
            0.2 * 0.0 +
            0.2 * (np.std(gray) / 60) +
            0.2 * max(0, min(cv2.Laplacian(gray, cv2.CV_64F).var() / 100, 1.0))
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
    if avg_height_px < 12:
        size_score = 0.4
        remarks.append("Text too small")
    elif avg_height_px < 20:
        size_score = 0.7
    else:
        size_score = 1.0

    # Contrast Check
    global_contrast = np.std(gray)
    contrast_score = min(global_contrast / 60, 1.0)
    if contrast_score < 0.5:
        remarks.append("Low contrast")

    # Blur
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


# ── Logo order check ─────────────────────────────────────────────────────────
def check_logo_order(detected: dict):
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

    if "nyc" not in present or "bp" not in present:
        return {
            "order_valid": False,
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
        issues.append(f"BP should be last but found '{order[-1]}' rightmost")

    # Collaborators must be strictly between NYC and BP
    collab_logos = {"sk", "yorp"}
    for name in collab_logos:
        if name in positions:
            if positions[name] <= positions["nyc"]:
                issues.append(f"{name.upper()} is to the left of or overlapping NYC")
            if positions[name] >= positions["bp"]:
                issues.append(f"{name.upper()} is to the right of or overlapping BP")

    return {
        "order_valid": len(issues) == 0,
        "detected_order": " → ".join(n.upper() for n in order),
        "remark": "Logo order correct" if not issues else " | ".join(issues),
        "details": {n: round(x, 1) for n, x in positions.items()}
    }


# ── Per-type post rules check ─────────────────────────────────────────────────
def check_post_type_rules(post_type: str, image, readability_result: dict, detected: dict = None):
    """
    Applies post-type-specific checks beyond logos and watermark.
    """
    if not DOCTR_AVAILABLE:
        return {"post_type": post_type, "status": "OCR unavailable", "checks_performed": []}
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

    # --- Logo order check (only if required) ---
    if rules.get("strict_logo_order") and detected:
        order_result = check_logo_order(detected)
        checks["logo_order"] = {
            "pass": order_result["order_valid"],
            "remark": order_result["remark"]
        }

    # --- Watermark check (only if required) ---
    if rules.get("requires_watermark"):
        wm_result = check_watermark(image)
        checks["watermark"] = {
            "pass": wm_result["watermark_present"],
            "remark": wm_result["remark"]
        }

    # --- Photo: resolution check ---
    if rules.get("requires_high_resolution"):
        min_w, min_h = rules.get("min_resolution", (1080, 1080))
        res_pass = w >= min_w and h >= min_h
        checks["resolution"] = {
            "pass": res_pass,
            "actual": f"{w}x{h}",
            "required": f"{min_w}x{min_h}",
            "remark": "OK" if res_pass else f"Image is {w}x{h}, minimum is {min_w}x{min_h}"
        }

    # --- Photo: subject centering (using saliency / center-of-mass of edges) ---
    if rules.get("check_centering"):
        edges = cv2.Canny(gray, 50, 150)
        moments = cv2.moments(edges)
        if moments["m00"] > 0:
            cx = moments["m10"] / moments["m00"]
            cy = moments["m01"] / moments["m00"]
            offset_x = abs(cx - w / 2) / (w / 2)  # 0 = centered, 1 = at edge
            offset_y = abs(cy - h / 2) / (h / 2)
            centered = offset_x < 0.25 and offset_y < 0.25
        else:
            centered = False
            offset_x = offset_y = None
        checks["centering"] = {
            "pass": centered,
            "offset_x": round(offset_x, 3) if offset_x is not None else "N/A",
            "offset_y": round(offset_y, 3) if offset_y is not None else "N/A",
            "remark": "Subject appears centered" if centered else "Subject may be off-center — review composition"
        }

    # --- Photo: lighting (brightness check) ---
    if rules.get("check_lighting"):
        mean_brightness = float(np.mean(gray))
        # Flag very dark images — likely no flash used in dark venue
        too_dark = mean_brightness < 60
        checks["lighting"] = {
            "pass": not too_dark,
            "mean_brightness": round(mean_brightness, 1),
            "remark": "OK" if not too_dark else "Image appears dark"
        }

    # --- Advisory / Resolution: full-page check ---
    # Heuristic: a full-page doc should have significant text coverage across height
    if rules.get("requires_full_page"):
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        # Rough proxy: high sharpness + good text confidence suggests a clear full-page scan
        is_full_page = laplacian_var > 80 and readability_result.get("Score", 0) >= 0.65
        checks["full_page"] = {
            "pass": is_full_page,
            "remark": "Full page appears present" if is_full_page else "Ensure full advisory/resolution page is shown"
        }

    # --- Advisory / Resolution: SGD text check (check if "SGD" is present in document) ---
    if rules.get("requires_sgd"):
        # Extract text using docTR and check if "SGD" is present
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
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
        
        # Extract all text from document
        extracted_text = ""
        for page in result.pages:
            for block in page.blocks:
                for line in block.lines:
                    for word in line.words:
                        extracted_text += word.value + " "
        
        # Check if "SGD" text is present
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


# ── Updated logo_report (now also returns raw detected dict) ─────────────────
def logo_report(results, model, img, conf_threshold=0.25, collaborators=None):
    """
    collaborators: list of logo names that are expected (e.g. ["sk"], ["yorp"], ["sk","yorp"])
                   Pass None or [] if no collaborators.
    """
    if collaborators is None:
        collaborators = []
    
    # Ensure collaborators are lowercase
    collaborators = [c.lower() if isinstance(c, str) else c for c in collaborators]

    detected = {"nyc": None, "bp": None, "sk": None, "yorp": None}
    report = []

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

    required_logos = ["nyc", "bp"] + collaborators

    for logo in ["nyc", "bp", "sk", "yorp"]:
        entry = detected[logo]
        is_required = logo in required_logos

        if entry is None:
            remark = "Add Logo" if is_required else "Optional — not detected"
            report.append({
                "Logo": logo.upper(),
                "Detected": "No",
                "Confidence": "-",
                "Status": "Missing",
                "Remark": remark
            })
        else:
            status = entry["status"]
            conf = round(entry["conf"], 3)
            remark = "Logo is correct" if status == "correct" else "Revise logo"
            report.append({
                "Logo": logo.upper(),
                "Detected": "Yes",
                "Confidence": conf,
                "Status": status.capitalize(),
                "Remark": remark
            })
            xyxy = entry["box"].xyxy[0].cpu().numpy().astype(int)
            color = (0, 255, 0) if status == "correct" else (0, 0, 255)
            cv2.rectangle(img, (xyxy[0], xyxy[1]), (xyxy[2], xyxy[3]), color, 2)
            cv2.putText(img, f"{logo.upper()}",
                        (xyxy[0], xyxy[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    return report, detected, img


# ── Master report generator 
def generate_report(yolo_results, model, image, post_type,
                    collaborators=None, conf_threshold=0.25):
    """
    Runs all checks and returns a unified audit report + annotated image.

    Returns:
        audit (dict)  — full structured report
        img   (ndarray) — annotated image with logo + watermark boxes
    """
    img = image.copy()

    # 1. Logo detection + bounding box annotation
    logo_rep, detected, img = logo_report(
        yolo_results, model, img,
        conf_threshold=conf_threshold,
        collaborators=collaborators or []
    )

    # 2. Logo order
    order_result = check_logo_order(detected)

    # 3. Watermark (only if required by post type)
    rules = POST_TYPE_RULES.get(post_type.lower(), {})
    requires_watermark = rules.get("requires_watermark", False)
    
    if requires_watermark:
        wm_result = check_watermark(img)
        # Draw watermark boxes in cyan
        for (x0, y0, x1, y1) in wm_result["boxes"]:
            cv2.rectangle(img, (x0, y0), (x1, y1), (255, 200, 0), 1)
        if wm_result["watermark_present"]:
            h = image.shape[0]
            cv2.putText(img, "Watermark OK",
                        (10, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        else:
            h = image.shape[0]
            cv2.putText(img, f"Watermark MISSING: {', '.join(wm_result['missing'])}",
                        (10, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 0, 0), 2)
    else:
        wm_result = None

    # 4. Readability (docTR)
    readability = check_readability(image)

    # 5. Post-type-specific rules
    type_rules = check_post_type_rules(post_type, image, readability, detected)

    # 6. Overall pass/fail
    logo_issues = any(
        r["Remark"] in ("Add Logo", "Revise logo") for r in logo_rep
    )
    
    overall_pass = (
        not logo_issues
        and type_rules["status"] == "Pass"
    )

    audit = {
        "post_type": post_type,
        "overall": "PASS" if overall_pass else "FAIL",
        "logos": logo_rep,
        "logo_order": order_result,
        "readability": readability,
        "type_checks": type_rules,
    }
    
    # Only include watermark if it was checked
    if wm_result is not None:
        audit["watermark"] = wm_result

    return audit, img