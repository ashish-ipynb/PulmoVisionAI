"""
=========================================================
PulmoVision AI
Main Application
=========================================================
"""

import streamlit as st


# -------------------------------------------------------
# Page Configuration
# -------------------------------------------------------

st.set_page_config(
    page_title="PulmoVision AI",
    page_icon="🫁",
    layout="wide",
    initial_sidebar_state="expanded"
)


# -------------------------------------------------------
# Custom CSS
# -------------------------------------------------------

st.markdown(
    """
    <style>

    .main-title{
        font-size:40px;
        font-weight:bold;
        color:#0E7490;
    }

    .subtitle{
        font-size:18px;
        color:#555555;
    }

    .metric-card{
        padding:15px;
        border-radius:12px;
        background-color:#F5F7FA;
        border:1px solid #E2E8F0;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# -------------------------------------------------------
# Home Screen
# -------------------------------------------------------

st.markdown(
    '<p class="main-title">🫁 PulmoVision AI</p>',
    unsafe_allow_html=True
)

st.markdown(
    """
PulmoVision AI is a deep learning–based medical image analysis platform
developed for pulmonary nodule classification using CT-derived ROI images.

The application uses a **DenseNet121** model trained on the **LIDC-IDRI**
dataset and provides predictions along with image analysis and model
performance insights.

Use the navigation panel on the left to explore the available modules.
"""
)

st.divider()

col1, col2, col3 = st.columns(3)

with col1:

    st.metric(
        "Model",
        "DenseNet121"
    )

with col2:

    st.metric(
        "ROC-AUC",
        "0.9897"
    )

with col3:

    st.metric(
        "Accuracy",
        "95.93%"
    )

st.divider()

st.subheader("Available Modules")

st.markdown(
"""
- 🧠 Prediction
- 🔥 Grad-CAM Visualization
- 📊 Model Comparison
- 📈 Performance Dashboard
- 🫁 Dataset Explorer
- ℹ️ About
"""
)

st.info(
    "Select a module from the sidebar to begin."
)