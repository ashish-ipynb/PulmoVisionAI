import streamlit as st

st.set_page_config(
    page_title="Dataset Explorer",
    page_icon="🫁",
    layout="wide"
)

st.title("🫁 LIDC-IDRI Dataset Explorer")

st.markdown("""
The model was trained using the **LIDC-IDRI (Lung Image Database Consortium and Image Database Resource Initiative)** dataset, one of the most widely used public datasets for pulmonary nodule analysis.
""")

st.subheader("📊 Dataset Statistics")

c1, c2, c3 = st.columns(3)

c1.metric("Patients", "1,010")
c2.metric("CT Series", "883")
c3.metric("DICOM Images", "244,527")

c4, c5, c6 = st.columns(3)

c4.metric("XML Files", "1,318")
c5.metric("Radiologist Annotations", "7,032")
c6.metric("Physical Nodules", "2,735")

st.divider()

st.subheader("Final Consensus Dataset")

a, b, c = st.columns(3)

a.metric("Final Nodules", "616")
b.metric("Benign", "397")
c.metric("Malignant", "219")

st.divider()

st.subheader("Dataset Preparation Workflow")

st.markdown("""LIDC-IDRI Dataset

↓

CT DICOM Images

↓

XML Annotation Parsing

↓

Nodule Clustering

↓

Consensus Label Generation

↓

2.5D ROI Extraction

↓

224 × 224 ROI Images

↓

Deep Learning Model""")

st.divider()

st.subheader("XML Annotation Files")

st.info("""
Each CT scan is accompanied by XML annotation files created by up to four experienced thoracic radiologists.

These annotations contain:

• Nodule boundaries

• Malignancy ratings (1–5)

• Subtlety

• Texture

• Sphericity

• Margin

• Calcification

• Internal structure

• Lobulation

• Spiculation
""")

st.subheader("Consensus Labeling")

st.success("""
Multiple radiologists may assign different malignancy scores to the same pulmonary nodule.

A consensus score was calculated by averaging the available malignancy ratings.

Final labels:

• Mean score < 3 → Benign

• Mean score > 3 → Malignant

• Mean score = 3 → Excluded to reduce ambiguity
""")

st.divider()

st.subheader("Dataset Quality")

st.write("""
✔ Publicly available

✔ Multi-institutional

✔ Expert annotations

✔ Multiple radiologist opinions

✔ Consensus labels

✔ Patient-level separation

✔ Suitable for deep learning research
""")

st.divider()

st.subheader("Example Pulmonary Nodules")

left, right = st.columns(2)

with left:
    st.image(
        "sample_images/benign.png",
        caption="Benign Pulmonary Nodule",
        use_container_width=True
    )

with right:
    st.image(
        "sample_images/malignant.png",
        caption="Malignant Pulmonary Nodule",
        use_container_width=True
    )