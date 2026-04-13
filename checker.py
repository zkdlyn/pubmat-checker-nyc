import cv2
import numpy as np
import difflib
import easyocr

# LOAD EASYOCR READER ONCE
_ocr_reader = None

def get_ocr_reader():
    global _ocr_reader
    if _ocr_reader is None:
        _ocr_reader = easyocr.Reader(["en"], gpu=False)
    return _ocr_reader


# ── Shared helper: run EasyOCR on a numpy BGR image ──────────────────────────
def _run_ocr(image_bgr):
    """
    Converts BGR to RGB and runs EasyOCR.
    Returns a list of (bbox, text, confidence) tuples where:
      bbox = [[x0,y0],[x1,y0],[x1,y1],[x0,y1]]  (four corner points)
    """
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    return get_ocr_reader().readtext(image_rgb)


# ── EasyOCR bbox helpers ──────────────────────────────────────────────────────
def _bbox_to_xyxy(bbox):
    """Convert EasyOCR 4-point bbox to (x0, y0, x1, y1)."""
    xs = [p[0] for p in bbox]
    ys = [p[1] for p in bbox]
    return int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))

def _bbox_center_x(bbox):
    x0, _, x1, _ = _bbox_to_xyxy(bbox)
    return (x0 + x1) / 2

def _bbox_height(bbox):
    _, y0, _, y1 = _bbox_to_xyxy(bbox)
    return y1 - y0


# rule config per post type
# Note: logo order (NYC leftmost, BP rightmost) is enforced globally for all post types
#       in generate_report — no per-type flag needed.
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
        "requires_sgd": True,
    },
    "resolution": {
        "requires_template": True,
        "readability_threshold": 0.70,
        "requires_sgd": True,
    },
    "hiring": {
        "requires_watermark": True,
        "requires_template": True,
        "readability_threshold": 0.65,
    },
    "photo": {
        "requires_template": True,
        "readability_threshold": 0.50,
        "requires_high_resolution": True,
        "min_resolution": (1080, 1080),
        "check_centering": True,
        "check_lighting": True,
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
    Crops the bottom 15% of the image, runs EasyOCR on it, and checks that
    all watermark handles are present AND appear together in the correct order
    (nyc.gov.ph → nationalyouthcommission → @nycpilipinas) left-to-right.

    Each handle is matched word-by-word using fuzzy scoring only.
    Returns detection results and the bounding boxes found.
    """
    img_h, img_w = image.shape[:2]
    crop_y = int(img_h * 0.85)
    crop = image[crop_y:img_h, 0:img_w]

    ocr_results = _run_ocr(crop)

    # Collect words with their absolute horizontal center position
    raw_words = []   # list of (word_str, center_x_abs)
    boxes_abs = []

    for (bbox, text, _conf) in ocr_results:
        x0, y0, x1, y1 = _bbox_to_xyxy(bbox)
        abs_x0 = x0
        abs_y0 = crop_y + y0
        abs_x1 = x1
        abs_y1 = crop_y + y1
        center_x = (abs_x0 + abs_x1) / 2
        raw_words.append((text.lower(), center_x))
        boxes_abs.append((abs_x0, abs_y0, abs_x1, abs_y1))

    # For each handle, find the best-matching word and record its position
    match_results = {}
    handle_positions = {}   # handle → center_x of best matched word (or None)

    for handle in WATERMARK_HANDLES:
        handle_clean = handle.lower().replace("@", "")
        best_score = 0.0
        best_cx = None
        for w_str, cx in raw_words:
            score = difflib.SequenceMatcher(None, handle_clean, w_str).ratio()
            if score > best_score:
                best_score = score
                best_cx = cx
        found = best_score >= WATERMARK_FUZZY_THRESHOLD
        match_results[handle] = {"found": found, "score": round(best_score, 3)}
        handle_positions[handle] = best_cx if found else None

    all_present = all(v["found"] for v in match_results.values())
    missing = [h for h, v in match_results.items() if not v["found"]]

    # Order check: handles must appear left-to-right in WATERMARK_HANDLES sequence
    order_valid = False
    order_remark = "Order not checked — one or more handles missing"
    if all_present:
        positions = [handle_positions[h] for h in WATERMARK_HANDLES]
        order_valid = all(positions[i] < positions[i + 1] for i in range(len(positions) - 1))
        if order_valid:
            order_remark = "Handles present and in correct order"
        else:
            order_remark = (
                "Handles found but order is incorrect — "
                "expected: " + " → ".join(WATERMARK_HANDLES)
            )

    watermark_ok = all_present and order_valid

    return {
        "watermark_present": watermark_ok,
        "handles": match_results,
        "missing": missing,
        "order_valid": order_valid,
        "remark": "Watermark OK" if watermark_ok else (
            f"Missing: {', '.join(missing)}" if missing else order_remark
        ),
        "boxes": boxes_abs,
    }


# ── Readability check (EasyOCR) ───────────────────────────────────────────────
def check_readability(image, threshold=0.65, ocr_result=None):
    """
    Uses EasyOCR confidence + CV metrics for blur/contrast/size.
    Accepts an optional pre-computed ocr_result to avoid a redundant OCR run.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    remarks = []

    if ocr_result is None:
        ocr_result = _run_ocr(image)

    img_h = image.shape[0]
    confidences = []
    word_heights_px = []

    for (bbox, _text, conf) in ocr_result:
        confidences.append(conf)
        word_heights_px.append(_bbox_height(bbox))

    if not confidences:
        return {
            "Readability Status": "No readable text found",
            "Remarks": "No text detected",
            "Score": 0.0
        }

    avg_conf = sum(confidences) / len(confidences)
    if avg_conf < 0.5:
        remarks.append("Low OCR confidence")

    # Text size check (pixel height directly from bbox)
    if word_heights_px:
        avg_height_px = sum(word_heights_px) / len(word_heights_px)
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
        0.4 * avg_conf +
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


# ── Logo order check ──────────────────────────────────────────────────────────
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
                          order_result: dict = None, ocr_result=None):
    """
    Applies post-type-specific checks beyond logos and watermark.
    Accepts pre-computed ocr_result to avoid redundant OCR calls.
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

    # gray is only computed if a check actually needs it
    _gray = None
    def get_gray():
        nonlocal _gray
        if _gray is None:
            _gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return _gray

    # --- Readability threshold ---
    threshold = rules.get("readability_threshold", 0.65)
    score = readability_result.get("Score", 0)
    checks["readability"] = {
        "pass": score >= threshold,
        "score": score,
        "threshold": threshold,
        "remark": "OK" if score >= threshold else f"Score {score} below threshold {threshold}"
    }

    # --- Watermark check (reuse pre-computed result, avoid double OCR) ---
    if rules.get("requires_watermark"):
        result = wm_result if wm_result is not None else check_watermark(image)
        checks["watermark"] = {
            "pass": result["watermark_present"],
            "remark": result["remark"]
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

    # --- Photo: subject centering ---
    if rules.get("check_centering"):
        edges = cv2.Canny(get_gray(), 50, 150)
        moments = cv2.moments(edges)
        if moments["m00"] > 0:
            cx = moments["m10"] / moments["m00"]
            cy = moments["m01"] / moments["m00"]
            offset_x = abs(cx - w / 2) / (w / 2)
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
        mean_brightness = float(np.mean(get_gray()))
        too_dark = mean_brightness < 60
        checks["lighting"] = {
            "pass": not too_dark,
            "mean_brightness": round(mean_brightness, 1),
            "remark": "OK" if not too_dark else "Image appears dark"
        }

    # --- Advisory / Resolution: SGD text check ---
    # Reuses the ocr_result already computed in generate_report — no extra OCR call.
    if rules.get("requires_sgd"):
        ocr = ocr_result if ocr_result is not None else _run_ocr(image)
        extracted_text = " ".join(text for (_bbox, text, _conf) in ocr)
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
    collaborators: list of logo names that are expected (e.g. ["sk"], ["yorp"], ["sk","yorp"])
                   Pass None or [] if no collaborators.
    """
    if collaborators is None:
        collaborators = []

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
                "Remark": remark,
                "_compliant": not is_required,  # optional logos don't count as failures
            })
        else:
            status = entry["status"]
            conf = round(entry["conf"], 3)
            is_correct = status == "correct"
            remark = "Logo is correct" if is_correct else "Revise logo"
            report.append({
                "Logo": logo.upper(),
                "Detected": "Yes",
                "Confidence": conf,
                "Status": status.capitalize(),
                "Remark": remark,
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
    EasyOCR on the full image is run exactly once and shared across
    readability and SGD checks.

    Returns:
        audit (dict)    — full structured report
        img   (ndarray) — annotated image with logo + watermark boxes
    """
    img = image.copy()

    # Guard against undecodable images
    if img is None or img.size == 0:
        raise ValueError("Image could not be decoded or is empty.")

    # 1. Run EasyOCR once for the full image — shared by readability + SGD
    ocr_result = _run_ocr(image)

    # 2. Logo detection + bounding box annotation
    logo_rep, detected, img = logo_report(
        yolo_results, model, img,
        conf_threshold=conf_threshold,
        collaborators=collaborators or []
    )

    # 3. Logo order (computed once, reused in type_checks)
    order_result = check_logo_order(detected)

    # 4. Watermark (only if required by post type)
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

    # 5. Readability — reuses the OCR result, no second OCR call
    readability = check_readability(image, ocr_result=ocr_result)

    # 6. Post-type-specific rules — reuses OCR result (SGD) and pre-computed results
    type_rules = check_post_type_rules(
        post_type, image, readability,
        detected=detected,
        wm_result=wm_result,
        order_result=order_result,
        ocr_result=ocr_result,
    )

    # 7. Overall pass/fail
    logo_issues = any(not r["_compliant"] for r in logo_rep)
    overall_pass = (
        not logo_issues
        and order_result["order_valid"]
        and type_rules["status"] == "Pass"
    )

    # Strip internal fields before returning to UI
    logos_display = [{k: v for k, v in r.items() if not k.startswith("_")} for r in logo_rep]

    audit = {
        "post_type": post_type,
        "overall": "PASS" if overall_pass else "FAIL",
        "logos": logos_display,
        "logo_order": order_result,
        "readability": readability,
        "type_checks": type_rules,
    }

    if wm_result is not None:
        audit["watermark"] = wm_result

    return audit, img
