import streamlit as st

st.set_page_config(
    page_title="Grad-CAM",
    page_icon="🔥",
    layout="wide"
)

st.title("🔥 Grad-CAM Visualization")

if "overlay" not in st.session_state:
    st.warning(
        "Please make a prediction first from the Prediction page."
    )
    st.stop()

prediction = st.session_state["prediction"]

st.subheader("Prediction Summary")

c1, c2, c3 = st.columns(3)

c1.metric("Prediction", prediction["label"])
c2.metric("Probability", f"{prediction['probability']:.4f}")
c3.metric("Confidence", prediction["confidence"])

st.divider()

col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("Original ROI")
    st.image(
        st.session_state["uploaded_image"],
        use_container_width=True
    )

with col2:
    st.subheader("Grad-CAM Heatmap")
    st.image(
        st.session_state["heatmap"],
        use_container_width=True
    )

with col3:
    st.subheader("Overlay")
    st.image(
        st.session_state["overlay"],
        use_container_width=True
    )

st.divider()

st.subheader("Interpretation")

st.info(
"""
### Color Guide

🔴 High contribution

🟠 Moderate contribution

🟡 Low contribution

🔵 Minimal contribution

The highlighted regions indicate the areas that most influenced the DenseNet121 prediction.
"""
)

st.success(
"""
Grad-CAM provides a visual explanation of the model's decision by highlighting
the regions within the pulmonary nodule that contributed most strongly to the
predicted class. This improves model interpretability but should not be
considered a standalone clinical diagnostic tool.
"""
)