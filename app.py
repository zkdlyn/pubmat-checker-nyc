import streamlit as st
import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO
import os
import sys

from streamlit_option_menu import option_menu

selected = option_menu(
    menu_title=None,
    options=["Home", "Caption Verifier", "Publication Material Checker"],
    icons=["house", "pencil", "image"],
    orientation="horizontal",
)

st.set_page_config(layout="wide", page_title="SMARTech Auditing Tool", page_icon=":mag_right:")
from checker import generate_report


if selected == "Home":
    st.title("Welcome to SMARTech")
    st.markdown("""
        This application is designed to help you verify the compliance of your social media captions and publication materials with our guidelines. 
        You can use the **Caption Verifier** to check if your captions meet the required standards, and the **Publication Material Checker** to ensure that your images contain the necessary logos and readable text.
        
        Please select an option from the menu above to get started. \n\n
    """)
    st.markdown("""
    #### ⚠️ Reminder
    This tool is intended to assist users in checking the compliance of their publication materials. However, it may not catch all issues or nuances related to compliance. Users are encouraged to review the results carefully and use their judgment when making final decisions about their materials. 

    """)

if selected == "Caption Verifier":
    # Code for Caption Verifier
    pass
elif selected == "Publication Material Checker":
    # Load model
    @st.cache_resource
    def load_model():
        return YOLO("best.pt")
    if st.button("Load Model"):
        model = load_model()
        st.success("Model loaded successfully!")

    st.markdown("## Publication Material Compliance Checker")

    # Configure post type and collaborators
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
    
    # Convert collaborators to lowercase for checker
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
        st.success("Uploaded!")
    
    st.button("Clear", on_click=clear)


    if uploaded_files:
        all_results = []
        processed = []

        for uploaded_file in uploaded_files:
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            img = cv2.imdecode(file_bytes, 1)

            results = model(img)
            
            # Use generate_report for comprehensive checking
            audit, annotated_img = generate_report(
                results, 
                model, 
                img, 
                post_type=post_type,
                collaborators=collaborators_lower
            )

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

        st.markdown("---")
        st.subheader("📊 Summary")
        st.dataframe(pd.DataFrame(all_results),hide_index=True, use_container_width=True)

        st.markdown("---")
        st.subheader("📋 Detailed Results")

        for item in processed:
            audit = item["audit"]
            with st.expander(f"Results for {item['name']}", expanded=True):
                col1, col2 = st.columns(2)

                with col1:
                    st.image(cv2.cvtColor(item["img"], cv2.COLOR_BGR2RGB), caption="Original")
                with col2:
                    st.image(cv2.cvtColor(item["annotated_img"], cv2.COLOR_BGR2RGB), caption="Annotated")

                # Logo Analysis
                st.markdown("### Logo Analysis")
                st.table(pd.DataFrame(audit["logos"]).reset_index(drop=True))
                
                # Logo Order
                st.markdown("### Logo Order Check")
                order = audit["logo_order"]
                if order["order_valid"]:
                    st.success(f"✅ {order['detected_order']}")
                else:
                    st.warning(f"⚠️ {order['detected_order']}")
                st.caption(order["remark"])

                # Watermark (only if checked)
                if "watermark" in audit:
                    st.markdown("### Watermark Check")
                    watermark = audit["watermark"]
                    if watermark["watermark_present"]:
                        st.success(watermark["remark"])
                    else:
                        st.error(f"{watermark['remark']}")

                # Readability
                st.markdown("### Readability Analysis")
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
                    st.markdown(f"### {check_display}")
                    if check_result["pass"]:
                        st.success(f"{check_result.get('remark', 'Pass')}")
                    else:
                        st.error(f"{check_result.get('remark', 'Fail')}")

                # Overall Verdict
                st.markdown("### 🧾 Overall Verdict")
                if audit["overall"] == "PASS":
                    st.success("✅ PASS - Material is compliant")
                else:
                    st.error("❌ FAIL - Material has compliance issues")


