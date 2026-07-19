import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="Model Comparison",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Deep Learning Model Comparison")

comparison = pd.DataFrame({

    "Model":[
        "DenseNet121",
        "EfficientNet-B0",
        "ShuffleNetV2",
        "MobileNetV3",
        "RegNetY-400MF"
    ],

    "Accuracy":[
        0.9593,
        0.9274,
        0.9597,
        0.9355,
        0.9194
    ],

    "ROC-AUC":[
        0.9897,
        0.9741,
        0.9818,
        0.9801,
        0.9790
    ]

})

st.dataframe(comparison, use_container_width=True)

st.success("""
DenseNet121 was selected as the deployment model because it
achieved the best overall validation performance and
demonstrated robust learning during cross-validation.
""")