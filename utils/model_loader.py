"""
=========================================================
PulmoVision AI
Model Loader
=========================================================
"""

from pathlib import Path

import json

import torch
import torch.nn as nn
from torchvision import models


# -------------------------------------------------------
# Project Paths
# -------------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]

MODEL_DIR = ROOT / "models"

MODEL_PATH = MODEL_DIR / "densenet121_best.pth"

INFO_PATH = MODEL_DIR / "model_info.json"


# -------------------------------------------------------
# Device
# -------------------------------------------------------

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# -------------------------------------------------------
# Build DenseNet121
# -------------------------------------------------------

def build_model():

    model = models.densenet121(weights=None)

    in_features = model.classifier.in_features

    model.classifier = nn.Linear(
        in_features,
        1
    )

    return model


# -------------------------------------------------------
# Load Model
# -------------------------------------------------------

def load_model():

    model = build_model()

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=DEVICE
    )

    model.load_state_dict(checkpoint)

    model.to(DEVICE)

    model.eval()

    return model


# -------------------------------------------------------
# Load Metadata
# -------------------------------------------------------

def load_metadata():

    with open(INFO_PATH, "r") as f:

        metadata = json.load(f)

    return metadata


# -------------------------------------------------------
# Test
# -------------------------------------------------------

if __name__ == "__main__":

    model = load_model()

    metadata = load_metadata()

    print("=" * 60)

    print("Model Loaded Successfully")

    print(metadata)

    print("=" * 60)