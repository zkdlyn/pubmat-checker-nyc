import os
import sys
import warnings

# Set YOLO config directory to writable location BEFORE importing YOLO
os.environ['YOLO_CONFIG_DIR'] = '/tmp/Ultralytics'
os.environ['YOLO_VERBOSE'] = 'false'

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

import streamlit as st

st.set_page_config(layout="wide", page_title="PubMat Checker", page_icon=":mag_right:")

import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO
from streamlit_option_menu import option_menu
from checker import generate_report, get_doctr_model, POST_TYPE_RULES


# ── Model loading ─────────────────────────────────────────────────────────────
@st.cache_resource
def load_doctr():
    return get_doctr_model()

@st.cache_resource
def load_yolo():
    try:
        return YOLO("best_v3.pt")
    except Exception as e:
        st.error(f"Failed to load YOLO model: {e}")
        return None

load_doctr()
model = load_yolo()


# ── UI helpers ────────────────────────────────────────────────────────────────
def _pass_badge(passed: bool) -> str:
    return "✅" if passed else "❌"

def _render_check_row(label: str, passed: bool, remark: str):
    """Render a single check line with icon, label, and remark."""
    icon = "✅" if passed else "❌"
    if passed:
        st.success(f"{icon} **{label}** — {remark}")
    else:
        st.error(f"{icon} **{label}** — {remark}")


# ── Page header ───────────────────────────────────────────────────────────────
st.markdown("## Publication Material Compliance Checker")

col1, col2 = st.columns(2)
with col1:
    post_type = st.selectbox(
        "Select Post Type",
        ["News", "Quotes", "Advisory", "Resolution", "Hiring", "Photo", "Holiday", "Other"],
        index=0
    )

with col2:
    collaborators = st.multiselect(
        "Collaborators",
        ["SK", "YORP"],
        default=[]
    )

# Rules summary expander
with st.expander("ℹ️ What does this post type check?", expanded=False):
    rules = POST_TYPE_RULES.get(post_type.lower(), {})
    st.markdown(f"✅ Correct logos present (NYC, BP{', ' + ', '.join(collaborators) if collaborators else ''})")
    st.markdown("✅ Correct logo order")
    st.markdown("✅ Pubmat image quality (resolution, blur, pixelation, contrast)")
    if rules.get("requires_watermark"):
        st.markdown("✅ Watermark present")
    if rules.get("requires_template"):
        st.markdown("✅ Template correctly used")
    if rules.get("check_photo_quality"):
        min_res = rules.get("min_resolution", (1080, 1080))
        st.markdown(f"✅ Photo quality (min {min_res[0]}×{min_res[1]}, centering, lighting, color)")
    if rules.get("requires_sgd"):
        st.markdown("✅ SGD signature required")

collaborators_lower = [c.lower() for c in collaborators]

# ── Upload controls ───────────────────────────────────────────────────────────
if "key" not in st.session_state:
    st.session_state.key = 0

def clear():
    st.session_state.key += 1

uploaded_files = st.file_uploader(
    "Upload Images",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=True,
    key=f"uploader_{st.session_state.key}"
)

if uploaded_files:
    st.success(f"{len(uploaded_files)} file(s) uploaded!")
    col_preview = st.columns(5)
    for i, uploaded_file in enumerate(uploaded_files):
        with col_preview[i % 5]:
            st.image(uploaded_file)

col_submit, col_clear = st.columns(2)
with col_submit:
    submit = st.button("Submit", type="primary")
with col_clear:
    st.button("Clear", on_click=clear)


# ── Processing ────────────────────────────────────────────────────────────────
if uploaded_files and submit:
    if model is None:
        st.error("YOLO model is not loaded. Cannot process images.")
    else:
        all_results = []
        processed = []

        progress = st.progress(0, text="Processing images...")

        for i, uploaded_file in enumerate(uploaded_files):
            progress.progress(i / len(uploaded_files), text=f"Processing {uploaded_file.name}…")

            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            img = cv2.imdecode(file_bytes, 1)

            if img is None:
                st.warning(f"⚠️ Could not decode **{uploaded_file.name}** — skipping. "
                           "Make sure it is a valid JPG or PNG.")
                continue

            try:
                results = model(img)
                audit, annotated_img = generate_report(
                    results,
                    model,
                    img,
                    post_type=post_type,
                    collaborators=collaborators_lower
                )
            except Exception as e:
                st.error(f"Error processing **{uploaded_file.name}**: {e}")
                continue

            all_results.append({
                "Image": uploaded_file.name,
                "Post Type": post_type,
                "Status": audit["overall"]
            })
            processed.append({
                "name": uploaded_file.name,
                "img": img,
                "annotated_img": annotated_img,
                "audit": audit
            })

        progress.progress(1.0, text="Done!")

        # ── Summary table ─────────────────────────────────────────────────────
        if all_results:
            st.markdown("---")
            st.subheader("📊 Summary")

            df_summary = pd.DataFrame(all_results)
            pass_count = (df_summary["Status"] == "PASS").sum()
            fail_count = (df_summary["Status"] == "FAIL").sum()

            m1, m2, m3 = st.columns(3)
            m1.metric("Total", len(df_summary))
            m2.metric("✅ Pass", pass_count)
            m3.metric("❌ Fail", fail_count)

            st.dataframe(df_summary, hide_index=True, use_container_width=True)

            # ── Detailed results ──────────────────────────────────────────────
            st.markdown("---")
            st.subheader("📋 Detailed Results")

            for item in processed:
                audit = item["audit"]

                with st.expander(
                    f"{_pass_badge(audit['overall'] == 'PASS')}  {item['name']}",
                    expanded=True
                ):
                    col_img, col_result = st.columns(2)

                    # ── Left: annotated image ─────────────────────────────────
                    with col_img:
                        st.image(
                            cv2.cvtColor(item["annotated_img"], cv2.COLOR_BGR2RGB),
                            caption="Annotated",
                            use_container_width=True
                        )

                    # ── Right: all checks ─────────────────────────────────────
                    with col_result:

                        # 1. Overall verdict (top for quick scan)
                        if audit["overall"] == "PASS":
                            st.success("✅ **PASS** — Material is compliant")
                        else:
                            st.error("❌ **FAIL** — Material has compliance issues")


                        # 2. Logo analysis table
                        st.markdown("**Logo Analysis**")
                        logo_df = pd.DataFrame(audit["logos"]).reset_index(drop=True)
                        # Color Status column for quick reading
                        st.dataframe(logo_df, hide_index=True, use_container_width=True)

                        # 3. Logo order
                        st.markdown("**Logo Order**")
                        order = audit["logo_order"]
                        _render_check_row(
                            order["detected_order"] if order["detected_order"] != "N/A — NYC or BP not detected" else "Logo Order",
                            order["order_valid"],
                            order["remark"]
                        )

                        # 4. Pubmat quality (universal)
                        st.markdown("**Pubmat Quality**")
                        pq = audit["pubmat_quality"]
                        _render_check_row("Image Quality", pq["pass"], pq["remark"])
                        with st.expander("Quality details", expanded=False):
                            details = pq["details"]
                            st.caption(f"Resolution: {details.get('resolution', '—')}")
                            st.caption(f"Laplacian variance (sharpness): {details.get('laplacian_var', '—')}")
                            st.caption(f"Pixelation ratio: {details.get('pixelation_ratio', '—')}")
                            st.caption(f"Contrast (std dev): {details.get('contrast_std', '—')}")

                        # 5. Readability
                        st.markdown("**Readability**")
                        readability = audit["readability"]
                        status = readability["Readability Status"]
                        score = readability["Score"]
                        passed = status == "Readable"
                        _render_check_row(
                            f"Readability ({status})",
                            passed,
                            f"Score: {score} — {readability['Remarks']}"
                        )

                        # 6. Watermark (conditional)
                        if "watermark" in audit:
                            st.markdown("**Watermark**")
                            wm = audit["watermark"]
                            _render_check_row("Watermark", wm["watermark_present"], wm["remark"])
                            # Show per-handle match scores
                            with st.expander("Handle details", expanded=False):
                                for handle, result in wm["handles"].items():
                                    icon = "✅" if result["found"] else "❌"
                                    st.caption(f"{icon} {handle} — score: {result['score']}")
                            st.divider()

                        # 7. Post-type-specific checks
                        type_checks = audit["type_checks"]["checks"]
                        
                        if type_checks:
                            st.markdown("**Post-Type Checks**")
                            for check_name, check_result in type_checks.items():
                                # Skip readability and watermark — already shown above
                                if check_name in ("readability", "watermark"):
                                    continue

                                label = check_name.replace("_", " ").title()

                                # photo_quality has sub-details worth expanding
                                if check_name == "photo_quality":
                                    _render_check_row(label, check_result["pass"], check_result["remark"])
                                    with st.expander("Photo quality details", expanded=False):
                                        for k, v in check_result.get("details", {}).items():
                                            st.caption(f"{k.replace('_', ' ').title()}: {v}")
                                else:
                                    _render_check_row(label, check_result["pass"], check_result.get("remark", ""))
