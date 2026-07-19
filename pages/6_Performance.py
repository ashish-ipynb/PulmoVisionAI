import streamlit as st

st.set_page_config(
    page_title="Performance Dashboard",
    page_icon="📈",
    layout="wide"
)

st.title("📈 Model Performance Dashboard")

st.markdown(
"""
This page summarizes the performance of the deployed DenseNet121 model
on the consensus-labelled LIDC-IDRI dataset.
"""
)

# -----------------------------
# Metrics
# -----------------------------

st.subheader("Overall Performance")

c1, c2, c3 = st.columns(3)

c1.metric("Accuracy", "95.93%")
c2.metric("ROC-AUC", "0.9897")
c3.metric("Balanced Accuracy", "95.94%")

c4, c5, c6 = st.columns(3)

c4.metric("Precision", "93.41%")
c5.metric("Recall", "96.02%")
c6.metric("F1 Score", "94.70%")

st.divider()

# -----------------------------
# ROC + PR
# -----------------------------

left, right = st.columns(2)

with left:
    st.subheader("ROC Curve")
    st.image("figures/roc_curve.png", use_container_width=True)

with right:
    st.subheader("Precision–Recall Curve")
    st.image("figures/pr_curve.png", use_container_width=True)

st.divider()

# -----------------------------
# Confusion Matrix
# -----------------------------

st.subheader("Confusion Matrix")
st.image("figures/confusion_matrix.png", use_container_width=True)

st.divider()

# -----------------------------
# Overall Metrics Plot
# -----------------------------

st.subheader("Performance Metrics")
st.image("figures/overall_metrics.png", use_container_width=True)

st.divider()

st.success(
"""
The DenseNet121 model demonstrated excellent discrimination between
benign and malignant pulmonary nodules, achieving a ROC-AUC of 0.9897
with high precision and recall, indicating strong generalization
performance on the consensus-labelled LIDC-IDRI dataset.
"""
)