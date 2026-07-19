import streamlit as st

st.set_page_config(
    page_title="About",
    page_icon="ℹ️",
    layout="wide"
)

st.title("ℹ️ About PulmoVision AI")

st.markdown("""
**PulmoVision AI** is a deep learning-based web application developed for the classification of pulmonary nodules as **benign** or **malignant** using CT scan images from the **LIDC-IDRI** dataset.

The application was developed as part of an M.Sc. Bioinformatics research project and demonstrates the use of artificial intelligence for computer-aided diagnosis of lung cancer.
""")

st.divider()

st.subheader("📌 Project Information")

col1, col2 = st.columns(2)

with col1:
    st.metric("Project", "PulmoVision AI")
    st.metric("Version", "1.0")
    st.metric("Dataset", "LIDC-IDRI")
    st.metric("Framework", "PyTorch")

with col2:
    st.metric("Model", "DenseNet121")
    st.metric("Input Size", "224 × 224")
    st.metric("Frontend", "Streamlit")
    st.metric("Programming Language", "Python")
    
    st.divider()

st.subheader("🛠️ Technologies Used")

st.markdown("""
- Python
- PyTorch
- Streamlit
- OpenCV
- NumPy
- Pandas
- Matplotlib
- scikit-learn
- Pillow
- LIDC-IDRI Dataset
""")

st.divider()

st.subheader("⚙️ Project Workflow")

st.markdown("""
1. CT scan acquisition

2. XML annotation parsing

3. Consensus label generation

4. 2.5D ROI extraction

5. Image preprocessing

6. DenseNet121 inference

7. Benign/Malignant prediction

8. Grad-CAM visualization
""")

st.divider()

st.subheader("👨‍💻 Developer")

st.markdown("""
**Ashish Kumar**

M.Sc. Bioinformatics

Amity Institute of Biotechnology

Amity University, Noida
""")

st.divider()

st.subheader("🎯 Research Objective")

st.info("""
The objective of this project is to develop an interpretable deep learning
framework capable of assisting pulmonary nodule classification using CT
images. The system combines automated prediction with Grad-CAM
visualization to improve transparency and support research in
computer-aided diagnosis.
""")

st.divider()

st.subheader("⚠️ Disclaimer")

st.warning("""
This application is intended solely for research, educational and
demonstration purposes.

It is **not** a certified medical device and must not be used as a
substitute for professional clinical diagnosis or treatment.
""")

st.divider()

st.caption(
    "PulmoVision AI • Version 1.0 • © 2026 Ashish Kumar • "
    "M.Sc. Bioinformatics Thesis Project"
)