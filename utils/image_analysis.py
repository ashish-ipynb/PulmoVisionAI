"""
=========================================================
PulmoVision AI
Image Analysis Utilities
=========================================================
"""

from pathlib import Path

import cv2
import numpy as np

from PIL import Image

from skimage.feature import graycomatrix, graycoprops


# -------------------------------------------------------
# Load Image
# -------------------------------------------------------

def load_image(image_path):

    image = Image.open(image_path).convert("RGB")

    return image


# -------------------------------------------------------
# Basic Statistics
# -------------------------------------------------------

def basic_statistics(image):

    image_np = np.array(image)

    gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)

    stats = {

        "Width": image.width,

        "Height": image.height,

        "Channels": image_np.shape[2],

        "MeanIntensity": float(np.mean(gray)),

        "StdIntensity": float(np.std(gray)),

        "MinIntensity": int(np.min(gray)),

        "MaxIntensity": int(np.max(gray))

    }

    return stats


# -------------------------------------------------------
# Contrast & Brightness
# -------------------------------------------------------

def image_quality(image):

    image_np = np.array(image)

    gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)

    quality = {

        "Brightness": float(np.mean(gray)),

        "Contrast": float(np.std(gray))

    }

    return quality


# -------------------------------------------------------
# Texture Features (GLCM)
# -------------------------------------------------------

def texture_features(image):

    image_np = np.array(image)

    gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)

    glcm = graycomatrix(

        gray,

        distances=[1],

        angles=[0],

        levels=256,

        symmetric=True,

        normed=True

    )

    texture = {

        "GLCM_Contrast": float(graycoprops(glcm, "contrast")[0,0]),

        "GLCM_Correlation": float(graycoprops(glcm, "correlation")[0,0]),

        "GLCM_Energy": float(graycoprops(glcm, "energy")[0,0]),

        "GLCM_Homogeneity": float(graycoprops(glcm, "homogeneity")[0,0])

    }

    return texture


# -------------------------------------------------------
# Morphology
# -------------------------------------------------------

def morphology(image):

    image_np = np.array(image)

    gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)

    _, thresh = cv2.threshold(

        gray,

        0,

        255,

        cv2.THRESH_BINARY + cv2.THRESH_OTSU

    )

    contours, _ = cv2.findContours(

        thresh,

        cv2.RETR_EXTERNAL,

        cv2.CHAIN_APPROX_SIMPLE

    )

    if len(contours) == 0:

        return {}

    contour = max(contours, key=cv2.contourArea)

    area = cv2.contourArea(contour)

    perimeter = cv2.arcLength(contour, True)

    x, y, w, h = cv2.boundingRect(contour)

    circularity = 0

    if perimeter > 0:

        circularity = (4 * np.pi * area) / (perimeter ** 2)

    return {

        "Area": float(area),

        "Perimeter": float(perimeter),

        "BoundingWidth": int(w),

        "BoundingHeight": int(h),

        "Circularity": float(circularity)

    }


# -------------------------------------------------------
# Complete Analysis
# -------------------------------------------------------

def analyze_image(image_path):

    image = load_image(image_path)

    report = {}

    report.update(basic_statistics(image))

    report.update(image_quality(image))

    report.update(texture_features(image))

    report.update(morphology(image))

    return report


# -------------------------------------------------------
# Test
# -------------------------------------------------------

if __name__ == "__main__":

    IMAGE_PATH = input("Enter image path: ")

    report = analyze_image(IMAGE_PATH)

    print("=" * 60)

    for key, value in report.items():

        print(f"{key:20s}: {value}")

    print("=" * 60)