"""
=========================================================
PulmoVision AI
Inference Engine
=========================================================
"""
import torch
try:
    from utils.model_loader import (
        load_model,
        load_metadata,
        DEVICE
    )

    from utils.preprocessing import prepare_image
    from utils.gradcam_utils import GradCAMGenerator

except Exception as e:
    print("Import error:", e)
    raise

# -------------------------------------------------------
# Load Model (only once)
# -------------------------------------------------------

MODEL = load_model()
METADATA = load_metadata()

THRESHOLD = METADATA["Threshold"]

# Initialize Grad-CAM
GRADCAM = GradCAMGenerator(MODEL)


# -------------------------------------------------------
# Confidence Interpretation
# -------------------------------------------------------

def get_confidence(probability, predicted_class):

    confidence = (
        probability
        if predicted_class == 1
        else 1 - probability
    )

    if confidence >= 0.95:
        return "Very High"

    elif confidence >= 0.80:
        return "High"

    elif confidence >= 0.60:
        return "Moderate"

    else:
        return "Low"

# -------------------------------------------------------
# Prediction
# -------------------------------------------------------

@torch.no_grad()
def predict(tensor):

    tensor = tensor.to(DEVICE)

    logits = MODEL(tensor)

    probability = torch.sigmoid(logits).item()

    predicted_class = int(probability >= THRESHOLD)

    label = (
        "Malignant"
        if predicted_class == 1
        else "Benign"
    )

    confidence = get_confidence(probability,predicted_class)

    return {

        "label": label,

        "class": predicted_class,

        "probability": probability,
        
        "malignant_probability": probability,

        "benign_probability": 1 - probability,

        "threshold": THRESHOLD,

        "confidence": confidence,

        "model": METADATA["Model"],

        "roc_auc": METADATA["ROC_AUC"],

        "accuracy": METADATA["Accuracy"],

        "balanced_accuracy": METADATA["BalancedAccuracy"]

    }


# -------------------------------------------------------
# Complete Pipeline
# -------------------------------------------------------

def predict_image(image_path):

    image, tensor = prepare_image(image_path)

    result = predict(tensor)
    
    heatmap, overlay = GRADCAM.generate(
        image=image,
        tensor=tensor,
    )

    return image, result, heatmap, overlay


# -------------------------------------------------------
# Test
# -------------------------------------------------------

if __name__ == "__main__":

    IMAGE_PATH = input("Enter image path: ")

    image, result, heatmap, overlay = predict_image(IMAGE_PATH)

    print("=" * 60)

    print(f"Prediction   : {result['label']}")
    print(f"Probability  : {result['probability']:.4f}")
    print(f"Threshold    : {result['threshold']:.2f}")
    print(f"Confidence   : {result['confidence']}")
    
    heatmap.save("heatmap_test.png")
    overlay.save("overlay_test.png")

    print("Heatmap saved as heatmap_test.png")
    print("Overlay saved as overlay_test.png")

    print("=" * 60)
    
    