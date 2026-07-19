import streamlit as st

st.set_page_config(
    page_title="PulmoVision AI",
    page_icon="🫁",
    layout="wide"
)

st.title("🫁 PulmoVision AI")

st.success("Welcome!")

st.write(
    """
    Use the sidebar to navigate through the application.

    - 🏠 Home
    - 🧠 Prediction
    - 🔥 GradCAM
    - 📊 Model Comparison
    - 📈 Performance
    - 🫁 Dataset
    - ℹ️ About
    """
)