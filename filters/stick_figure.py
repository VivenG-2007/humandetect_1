"""
Stick Figure Filter — clear stylized figure on a black background.
"""
import cv2
import numpy as np
from skeleton_renderer import SkeletonRenderer

# Shared renderer instance to maintain state if any
_renderer = SkeletonRenderer(
    line_color=(0, 255, 255),    # Yellow
    joint_color=(220, 220, 220), # Grey
    torso_color=(40, 160, 40),   # Green
    line_thickness=6,
    joint_radius=5,
)

def apply(canvas: np.ndarray, pose, **kwargs) -> np.ndarray:
    """
    Draws the stylized stick figure on the provided canvas.
    Expects a black canvas from the main window.
    """
    h, w = canvas.shape[:2]
    return _renderer.render(pose, (h, w), canvas)
