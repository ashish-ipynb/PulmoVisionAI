"""
=========================================================
PulmoVision AI
Prediction
=========================================================
"""

from pathlib import Path
import tempfile

import streamlit as st

from utils.inference import predict_image
from utils.image_analysis import analyze_image


st.set_page_config(
    page_title="AI Assessment",
    page_icon="🧠",
    layout="wide"
)

st.title("🧠 Pulmonary Nodule Prediction")

st.write(
    """
Upload a pulmonary nodule ROI image for automated classification.
The uploaded image is analyzed using the trained DenseNet121 model,
and additional image statistics are computed for interpretation.
"""
)

st.divider()

uploaded_file = st.file_uploader(

    "Upload ROI Image",

    type=["png", "jpg", "jpeg"]

)

if uploaded_file is not None:

    with tempfile.NamedTemporaryFile(
        delete=False,
        suffix=Path(uploaded_file.name).suffix
    ) as tmp:

        tmp.write(uploaded_file.getbuffer())

        image_path = tmp.name

    image, prediction, heatmap, overlay = predict_image(image_path)

    analysis = analyze_image(image_path)
    
    # Store prediction data for Grad-CAM page
    st.session_state["uploaded_image"] = image
    st.session_state["image_path"] = image_path
    st.session_state["prediction"] = prediction
    st.session_state["analysis"] = analysis
    st.session_state["heatmap"] = heatmap
    st.session_state["overlay"] = overlay


    left, right = st.columns([1, 1])

    with left:

        st.subheader("Uploaded Image")

        st.image(
            image,
            use_container_width=True
        )

    with right:

        st.subheader("Prediction")

        if prediction["class"] == 1:

            st.error("🔴 Malignant-like")

        else:

            st.success("🟢 Benign-like")

        c1, c2 = st.columns(2)

        with c1:
                    st.metric(
            "Benign Probability",
            f"{prediction['benign_probability']*100:.1f}%"
        )

        st.metric(
            "Malignant Probability",
            f"{prediction['malignant_probability']*100:.1f}%"
        )
        with c2:

            st.metric(
            "Model Confidence",
            prediction["confidence"]
        )

            st.metric(
            "Decision Threshold",
            f"{prediction['threshold']*100:.0f}%"
        )
        st.divider()

        if prediction["class"] == 1:

            st.warning(
                            f"""
                The model predicts this pulmonary nodule as 
                **malignant-like imaging characteristics**.

                Estimated malignancy probability:
                **{prediction['malignant_probability']*100:.1f}%**

                This result is intended for research purposes and should
                not be considered a clinical diagnosis.
                """
        )

        else:

            st.info(
                            f"""
                The model predicts this pulmonary nodule as 
                **benign-like  imaging characteristics**.

                Estimated benign probability:
                **{prediction['benign_probability']*100:.1f}%**

                Estimated malignancy probability:
                **{prediction['malignant_probability']*100:.1f}%**
                """
        )

    st.divider()

    tab1, tab2, tab3 = st.tabs(

        [
            "📊 Statistics",
            "🧬 Texture",
            "📐 Morphology"
        ]

    )

    with tab1:

        col1, col2 = st.columns(2)

        with col1:

            st.metric(
                "Width",
                analysis["Width"]
            )

            st.metric(
                "Height",
                analysis["Height"]
            )

            st.metric(
                "Brightness",
                f"{analysis['Brightness']:.2f}"
            )

        with col2:

            st.metric(
                "Mean Intensity",
                f"{analysis['MeanIntensity']:.2f}"
            )

            st.metric(
                "Contrast",
                f"{analysis['Contrast']:.2f}"
            )

            st.metric(
                "Std Intensity",
                f"{analysis['StdIntensity']:.2f}"
            )
            st.divider()
    
    if st.button("🔥 View Grad-CAM"):
        st.switch_page("pages/4_GradCAM.py")

    with tab2:

        c1, c2 = st.columns(2)

        with c1:

            st.metric(
                "GLCM Contrast",
                f"{analysis['GLCM_Contrast']:.3f}"
            )

            st.metric(
                "GLCM Correlation",
                f"{analysis['GLCM_Correlation']:.4f}"
            )

        with c2:

            st.metric(
                "Energy",
                f"{analysis['GLCM_Energy']:.4f}"
            )

            st.metric(
                "Homogeneity",
                f"{analysis['GLCM_Homogeneity']:.4f}"
            )

    with tab3:

        c1, c2 = st.columns(2)

        with c1:

            st.metric(
                "Area",
                f"{analysis['Area']:.1f}"
            )

            st.metric(
                "Perimeter",
                f"{analysis['Perimeter']:.1f}"
            )

        with c2:

            st.metric(
                "Bounding Width",
                analysis["BoundingWidth"]
            )

            st.metric(
                "Bounding Height",
                analysis["BoundingHeight"]
            )

            st.metric(
                "Circularity",
                f"{analysis['Circularity']:.3f}"
            )

    Path(image_path).unlink(missing_ok=True)