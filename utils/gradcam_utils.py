"""
=========================================================
PulmoVision AI
Grad-CAM Utilities
=========================================================
"""

import numpy as np
import cv2
from PIL import Image

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget


class GradCAMGenerator:

    def __init__(self, model):
        self.model = model

        # DenseNet121 final convolutional layer
        self.target_layers = [model.features]

        self.cam = GradCAM(
            model=self.model,
            target_layers=self.target_layers
        )

    def generate(self, image, tensor, class_idx=None):
        """
        image  : PIL Image
        tensor : torch tensor (1,3,224,224)
        """

        rgb = np.array(image).astype(np.float32) / 255.0

        # Binary classifier (single sigmoid output)
        targets = None

        grayscale_cam = self.cam(
            input_tensor=tensor,
            targets=targets
        )[0]

        overlay = show_cam_on_image(
            rgb,
            grayscale_cam,
            use_rgb=True
        )

        heatmap = np.uint8(255 * grayscale_cam)

        heatmap = cv2.applyColorMap(
            heatmap,
            cv2.COLORMAP_JET
        )

        heatmap = cv2.cvtColor(
            heatmap,
            cv2.COLOR_BGR2RGB
        )

        return (
            Image.fromarray(heatmap),
            Image.fromarray(overlay)
        )