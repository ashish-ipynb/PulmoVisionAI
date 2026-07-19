"""
=========================================================
PulmoVision AI
Image Preprocessing
=========================================================
"""

from PIL import Image
import torchvision.transforms as transforms
import torch

# -------------------------------------------------------
# Image Configuration
# -------------------------------------------------------

IMAGE_SIZE = 224

MEAN = [0.485, 0.456, 0.406]

STD = [0.229, 0.224, 0.225]

# -------------------------------------------------------
# Transformation Pipeline
# -------------------------------------------------------

transform = transforms.Compose([

    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=MEAN,
        std=STD
    )

])

# -------------------------------------------------------
# Load Image
# -------------------------------------------------------

def load_image(image_file):

    image = Image.open(image_file).convert("RGB")

    return image

# -------------------------------------------------------
# Preprocess Image
# -------------------------------------------------------

def preprocess_image(image):

    tensor = transform(image)

    tensor = tensor.unsqueeze(0)

    return tensor

# -------------------------------------------------------
# Complete Pipeline
# -------------------------------------------------------

def prepare_image(image_file):

    image = load_image(image_file)

    tensor = preprocess_image(image)

    return image, tensor

# -------------------------------------------------------
# Test
# -------------------------------------------------------

if __name__ == "__main__":

    TEST_IMAGE = input("Enter image path: ")

    image, tensor = prepare_image(TEST_IMAGE)

    print("=" * 60)

    print("Original Size :", image.size)

    print("Tensor Shape  :", tensor.shape)

    print("Tensor Type   :", tensor.dtype)

    print("=" * 60)