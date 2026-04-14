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


# cache docTR model via Streamlit so it doesn't reload on every rerun
@st.cache_resource
def load_doctr():
    return get_doctr_model()

@st.cache_resource
def load_yolo():
    try:
        return YOLO("best_v2.pt")
    except Exception as e:
        st.error(f"Failed to load YOLO model: {e}")
        return None

# Pre-load both models at startup
load_doctr()
model = load_yolo()

st.markdown("## Publication Material Compliance Checker")

# Show what rules apply for the selected post type
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

# Show rules summary for selected post type
with st.expander("ℹ️ What does this post type check?", expanded=False):
    rules = POST_TYPE_RULES.get(post_type.lower(), {})
    rule_lines = []
    rule_lines.append("✅ Correct logos present")
    rule_lines.append("✅ Correct logo order")
    if rules.get("requires_watermark"):
        rule_lines.append("✅ Watermark present")
    if rules.get("requires_template"):
        rule_lines.append("✅ Template correctly used")
    if rules.get("requires_high_resolution"):
        min_res = rules.get("min_resolution", (1080, 1080))
        rule_lines.append(f"✅ Minimum resolution: {min_res[0]}×{min_res[1]}")
    if rules.get("requires_full_page"):
        rule_lines.append("✅ Full page document required")
    if rules.get("requires_sgd"):
        rule_lines.append("✅ SGD signature required")
    # if rules.get("requires_re")
    for line in rule_lines:
        st.markdown(line)

collaborators_lower = [c.lower() for c in collaborators]

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

if uploaded_files and submit:
    if model is None:
        st.error("YOLO model is not loaded. Cannot process images.")
    else:
        all_results = []
        processed = []

        progress = st.progress(0, text="Processing images...")

        for i, uploaded_file in enumerate(uploaded_files):
            progress.progress((i) / len(uploaded_files), text=f"Processing {uploaded_file.name}…")

            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            img = cv2.imdecode(file_bytes, 1)

            # guard against undecodable images
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

            overall_status = audit["overall"]

            all_results.append({
                "Image": uploaded_file.name,
                "Post Type": post_type,
                "Status": overall_status
            })

            processed.append({
                "name": uploaded_file.name,
                "img": img,
                "annotated_img": annotated_img,
                "audit": audit
            })

        progress.progress(1.0, text="Done!")

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

            st.markdown("---")
            st.subheader("📋 Detailed Results")

            for item in processed:
                audit = item["audit"]
                with st.expander(f"Results for {item['name']}", expanded=True):

                    col_img, col_result = st.columns(2)
                    with col_img:
                        st.image(cv2.cvtColor(item["annotated_img"], cv2.COLOR_BGR2RGB),
                                    caption="Annotated", use_container_width=True)
                    with col_result:
                        # Logo Analysis
                        st.markdown("**Logo Analysis**")
                        st.table(pd.DataFrame(audit["logos"]).reset_index(drop=True))

                        # Logo Order
                        st.markdown("**Logo Order Check**")
                        order = audit["logo_order"]
                        if order["order_valid"]:
                            st.success(f"✅ {order['detected_order']}")
                        else:
                            st.warning(f"⚠️ {order['detected_order']}")
                        st.caption(order["remark"])

                        # Readability
                        st.markdown("**Readability Analysis**")
                        readability = audit["readability"]
                        st.metric("Readability Score", readability["Score"])

                        status = readability["Readability Status"]
                        if status == "Readable":
                            st.success(status)
                        elif status == "Moderate readability":
                            st.warning(status)
                        elif status == "No readable text found":
                            st.error(status)
                        else:
                            st.info(status)

                        st.caption(readability["Remarks"])

                        # Post Type-Specific Checks
                        type_rules = audit["type_checks"]
                        checks = type_rules["checks"]
                        for check_name, check_result in checks.items():
                            check_display = check_name.replace("_", " ").title()
                            st.markdown(f"**{check_display}**")
                            if check_result["pass"]:
                                st.success(f"✅ {check_result.get('remark', 'Pass')}")
                            else:
                                st.error(f"❌ {check_result.get('remark', 'Fail')}")

                        # Overall Verdict
                        st.markdown("** Overall Verdict **")
                        if audit["overall"] == "PASS":
                            st.success("✅ PASS — Material is compliant")
                        else:
                            st.error("❌ FAIL — Material has compliance issues")
