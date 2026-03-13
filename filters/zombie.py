"""
Zombie Filter — undead aesthetic with green flesh and missing limb effects.
"""
import cv2
import numpy as np
import random
import time
from pose_detector import PoseResult
from skeleton_renderer import SkeletonRenderer

_renderer = SkeletonRenderer(
    line_color=(50, 100, 50),     # Dark Olive
    joint_color=(100, 120, 100),  # Grey-green
    torso_color=(40, 80, 40),     # Rot-green
    line_thickness=8,
)

VIS_THRESHOLD = 0.4

def apply(canvas: np.ndarray, pose: PoseResult, **kwargs) -> np.ndarray:
    h, w = canvas.shape[:2]
    t = time.time()
    
    if not pose.detected:
        return canvas

    lm = pose.landmarks
    vis = pose.visibility

    # Zombie Body: Pale/Greenish
    # We use a custom render but occasionally "glitch" or hide parts
    original_vis = vis.copy()
    
    # 1. "Missing Limb" effect - randomly hide some connections
    # Every few seconds, hide an arm segment
    if (int(t * 2) % 4) == 0:
        if 15 in vis: vis[15] = 0 # Hide left hand
        if 13 in vis: vis[13] = 0.1
        
    # 2. Draw standard skeleton but with zombie colors
    _renderer.render(pose, (h, w), canvas)
    
    # 3. Add "Goo" particles (Red/Dark Green)
    if random.random() < 0.3:
        # Spawn near joints
        joint_idx = random.choice([11, 12, 13, 14, 23, 24])
        if joint_idx in lm and vis.get(joint_idx, 0) > VIS_THRESHOLD:
            px, py = lm[joint_idx]
            for _ in range(5):
                color = random.choice([(0, 0, 120), (20, 60, 20)]) # Blood or Slime
                cv2.circle(canvas, (px + random.randint(-10, 10), py + random.randint(-10, 10)), random.randint(2, 4), color, -1)

    # 4. Vignette / Dark Fog
    overlay = np.zeros_like(canvas)
    cv2.circle(overlay, (w//2, h//2), int(w * 0.8), (20, 30, 20), -1)
    overlay = cv2.GaussianBlur(overlay, (99, 99), 0)
    canvas[:] = cv2.addWeighted(canvas, 1.0, overlay, 0.3, 0)

    # Restore visibility for next frame if shared
    pose.visibility = original_vis
    
    return canvas
