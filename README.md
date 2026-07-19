# 🫁 PulmoVision AI

An AI-powered web application for pulmonary nodule classification from CT scan images using deep learning and explainable AI.

🌐 **Live Demo:** https://pulmovisionai.streamlit.app/Prediction

📂 **GitHub Repository:** https://github.com/ashish-ipynb/PulmoVisionAI

---

## 📖 Overview

PulmoVision AI is a deep learning–based web platform developed to assist in the classification of pulmonary nodules as **Benign-like** or **Malignant-like** using CT scan images.

The application utilizes a fine-tuned **DenseNet121** model trained on the **LIDC-IDRI** dataset and incorporates **Grad-CAM** visualizations to improve interpretability by highlighting image regions that influence the model's predictions.

This project was developed as part of an M.Sc. Bioinformatics research thesis.

---

## ✨ Features

- 🫁 Pulmonary nodule classification
- 🤖 DenseNet121 deep learning model
- 🔥 Grad-CAM explainability
- 📊 Performance visualization
- 📈 ROC, PR Curve and Confusion Matrix
- 📂 Dataset information
- 📱 Interactive Streamlit interface
- ⚡ Fast inference

---

## 🖥️ Application Pages

### 🏠 Home
Project overview and workflow.

### 📊 Model Comparison
Comparison of different CNN architectures evaluated during experimentation.

### 🔍 Prediction
Upload a pulmonary nodule CT image and receive:

- Benign-like / Malignant-like prediction
- Class probabilities
- Model confidence
- Clinical interpretation

### 🔥 Grad-CAM
Visual explanation highlighting the image regions used by the model.

### 📂 Dataset
Overview of the LIDC-IDRI dataset and preprocessing workflow.

### 📈 Performance
Model evaluation metrics including:

- ROC Curve
- Precision–Recall Curve
- Confusion Matrix
- Overall Performance Metrics

### ℹ️ About
Project information and acknowledgements.

---

# 🧠 Model Information

| Property | Value |
|----------|-------|
| Model | DenseNet121 |
| Framework | PyTorch |
| Input Size | 224 × 224 |
| Image Type | 2.5D CT ROI |
| Task | Binary Classification |
| Explainability | Grad-CAM |

---

# 📊 Dataset

**Dataset:** LIDC-IDRI

The Lung Image Database Consortium Image Collection (LIDC-IDRI) contains thoracic CT scans with pulmonary nodule annotations provided by multiple experienced radiologists.

The project uses consensus-labelled pulmonary nodules for supervised deep learning.

---

# 📈 Model Performance

Evaluation metrics include:

- Accuracy
- ROC-AUC
- Precision
- Recall
- F1 Score
- Specificity
- Sensitivity
- Balanced Accuracy
- Matthews Correlation Coefficient (MCC)

Performance visualizations are available directly within the application.

---

# 🔥 Explainable AI

PulmoVision AI integrates **Gradient-weighted Class Activation Mapping (Grad-CAM)** to improve model interpretability.

Grad-CAM highlights the image regions that contribute most to the model's prediction, providing an intuitive visual explanation for each inference.

---

# 🚀 Installation

Clone the repository

```bash
git clone https://github.com/ashish-ipynb/PumoVisionAI.git
```

Move into the project directory

```bash
cd PumoVisionAI
```

Install dependencies

```bash
pip install -r requirements.txt
```

Run the application

```bash
streamlit run app.py
```

---

# 🛠 Technologies Used

- Python
- PyTorch
- Streamlit
- OpenCV
- NumPy
- Pandas
- Matplotlib
- Plotly
- scikit-learn
- Grad-CAM
- ReportLab

---

# 📁 Project Structure

```
PulmoVision_AI/
│
├── app.py
├── models/
├── pages/
├── utils/
├── figures/
├── sample_images/
├── requirements.txt
└── README.md
```

---

# ⚠️ Disclaimer

This application is intended **solely for research and educational purposes**.

It is **not** a medical device and should **not** be used as a substitute for professional medical diagnosis or clinical decision-making.

---

# 👨‍💻 Author

**Ashish Kumar**

M.Sc. Bioinformatics

Amity University, Noida

---

# ⭐ If you found this project useful

Please consider giving the repository a ⭐ on GitHub.
