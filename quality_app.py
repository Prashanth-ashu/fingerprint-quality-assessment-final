"""
quality_app.py

Streamlit Application

Assignment 4
Contactless Fingerprint Quality Assessment

"""

import cv2
import numpy as np
import streamlit as st

from quality_assessment import (
    quality_gate,
)

# ---------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------

st.set_page_config(

    page_title="Fingerprint Quality Assessment",

    page_icon="🖐",

    layout="wide"

)

# ---------------------------------------------------
# TITLE
# ---------------------------------------------------

st.title("📱 Contactless Fingerprint Quality Assessment")

st.markdown(
"""
Upload a fingerprint image captured from your mobile phone.

The system evaluates:

- Blur
- Brightness
- Glare
- ROI Completeness
- Ridge Clarity

and generates a **Composite Quality Score (0–100).**
"""
)

# ---------------------------------------------------
# SIDEBAR
# ---------------------------------------------------

st.sidebar.title("Quality Settings")

blur_threshold = st.sidebar.slider(

    "Blur Threshold",

    min_value=5.0,

    max_value=50.0,

    value=10.0,

    step=1.0

)

brightness_min = st.sidebar.slider(

    "Minimum Brightness",

    min_value=0,

    max_value=120,

    value=50

)

brightness_max = st.sidebar.slider(

    "Maximum Brightness",

    min_value=150,

    max_value=255,

    value=210

)

glare_threshold = st.sidebar.slider(

    "Maximum Glare Fraction",

    min_value=0.01,

    max_value=0.20,

    value=0.05,

    step=0.01

)

roi_threshold = st.sidebar.slider(

    "Minimum ROI",

    min_value=0.05,

    max_value=0.50,

    value=0.15,

    step=0.01

)

ridge_threshold = st.sidebar.slider(

    "Minimum Ridge Score",

    min_value=5.0,

    max_value=40.0,

    value=15.0,

    step=1.0

)

# ---------------------------------------------------
# FILE UPLOADER
# ---------------------------------------------------

uploaded_file = st.file_uploader(

    "Upload Fingerprint Image",

    type=["jpg", "jpeg", "png"]

)

# ---------------------------------------------------
# IF IMAGE EXISTS
# ---------------------------------------------------

if uploaded_file is not None:

    file_bytes = np.asarray(

        bytearray(uploaded_file.read()),

        dtype=np.uint8

    )

    image_bgr = cv2.imdecode(

        file_bytes,

        cv2.IMREAD_COLOR

    )

    if image_bgr is None:

        st.error("Unable to read image.")

        st.stop()

    result = quality_gate(image_bgr)


        # ---------------------------------------------------
    # LAYOUT
    # ---------------------------------------------------

    col1, col2 = st.columns([1, 1])

    # ---------------------------------------------------
    # LEFT COLUMN
    # IMAGE PREVIEW
    # ---------------------------------------------------

    with col1:

        st.subheader("Uploaded Image")

        rgb_image = cv2.cvtColor(
            image_bgr,
            cv2.COLOR_BGR2RGB
        )

        st.image(
            rgb_image,
            use_container_width=True
        )

    # ---------------------------------------------------
    # RIGHT COLUMN
    # RESULTS
    # ---------------------------------------------------

    with col2:

        st.subheader("Quality Assessment")

        score = result["composite_score"]

        if result["passed"]:

            st.success(
                f"✅ PASSED\n\nComposite Score : {score}/100"
            )

        else:

            st.error(
                f"❌ REJECTED\n\nComposite Score : {score}/100"
            )

        # --------------------------------------------
        # SCORE BAR
        # --------------------------------------------

        st.progress(
            min(int(score), 100)
        )

        # --------------------------------------------
        # GUIDANCE
        # --------------------------------------------

        st.info(
            f"**Guidance:** {result['guidance']}"
        )

        st.divider()

        st.subheader("Quality Metrics")

        # --------------------------------------------
        # BLUR
        # --------------------------------------------

        blur = result["blur"]

        if blur["is_blurry"]:

            st.error(
                f"Blur : ❌ FAIL ({blur['blur_score']})"
            )

        else:

            st.success(
                f"Blur : ✅ PASS ({blur['blur_score']})"
            )

        # --------------------------------------------
        # BRIGHTNESS
        # --------------------------------------------

        brightness = result["brightness"]

        if brightness["too_dark"]:

            st.error(
                f"Brightness : ❌ TOO DARK ({brightness['brightness']})"
            )

        elif brightness["too_bright"]:

            st.error(
                f"Brightness : ❌ TOO BRIGHT ({brightness['brightness']})"
            )

        else:

            st.success(
                f"Brightness : ✅ PASS ({brightness['brightness']})"
            )

        # --------------------------------------------
        # GLARE
        # --------------------------------------------

        glare = result["glare"]

        if glare["has_glare"]:

            st.error(
                f"Glare : ❌ FAIL ({glare['glare_fraction']})"
            )

        else:

            st.success(
                f"Glare : ✅ PASS ({glare['glare_fraction']})"
            )


                    # --------------------------------------------
        # ROI RESULT
        # --------------------------------------------

        roi = result["roi"]

        if roi["roi_complete"]:

            st.success(
                f"ROI Completeness : ✅ PASS ({roi['roi_fraction']:.4f})"
            )

        else:

            st.error(
                f"ROI Completeness : ❌ FAIL ({roi['roi_fraction']:.4f})"
            )

        # --------------------------------------------
        # RIDGE RESULT
        # --------------------------------------------

        ridge = result["ridge"]

        if ridge["ridges_clear"]:

            st.success(
                f"Ridge Clarity : ✅ PASS ({ridge['ridge_score']})"
            )

        else:

            st.error(
                f"Ridge Clarity : ❌ FAIL ({ridge['ridge_score']})"
            )

    # ---------------------------------------------------
    # METRICS TABLE
    # ---------------------------------------------------

    st.divider()

    st.subheader("Detailed Quality Report")

    report = {
        "Metric": [
            "Blur",
            "Brightness",
            "Glare",
            "ROI",
            "Ridge",
            "Composite Score",
            "Status"
        ],

        "Value": [
            blur["blur_score"],
            brightness["brightness"],
            glare["glare_fraction"],
            roi["roi_fraction"],
            ridge["ridge_score"],
            score,
            "PASS" if result["passed"] else "FAIL"
        ]
    }

    st.table(report)

    # ---------------------------------------------------
    # RAW JSON OUTPUT
    # ---------------------------------------------------

    with st.expander("View Raw Output"):

        st.json(result)

# ---------------------------------------------------
# NO IMAGE
# ---------------------------------------------------

else:

    st.info(
        "Upload a fingerprint image to begin quality assessment."
    )

# ---------------------------------------------------
# FOOTER
# ---------------------------------------------------

st.markdown("---")

st.caption(
    "Assignment 4 • Contactless Fingerprint Quality Assessment Pipeline"
)