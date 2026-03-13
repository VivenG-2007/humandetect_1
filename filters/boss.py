"""
Boss Mode Filter — giant red glowing outline and intimidating aura.
"""
import cv2
import numpy as np
import random
import time
from pose_detector import PoseResult, CONNECTIONS
from skeleton_renderer import SkeletonRenderer

_renderer = SkeletonRenderer(
    line_color=(0, 0, 255),       # Pure Red
    joint_color=(100, 100, 100),  # Grey
    torso_color=(50, 0, 100),     # Dark Purple-Red
)

VIS_THRESHOLD = 0.4

def apply(canvas: np.ndarray, pose: PoseResult, **kwargs) -> np.ndarray:
    h, w = canvas.shape[:2]
    t = time.time()
    
    if not pose.detected:
        return canvas

    lm = pose.landmarks
    vis = pose.visibility

    # 1. Create Giant Red Glow Silhouette
    mask = np.zeros((h, w), dtype=np.uint8)
    for (a, b) in CONNECTIONS:
        if a in lm and b in lm and vis.get(a, 0) > VIS_THRESHOLD and vis.get(b, 0) > VIS_THRESHOLD:
            cv2.line(mask, lm[a], lm[b], 255, 45, cv2.LINE_AA) # Thicker for "Giant" feel
            
    # Red Glow Layer
    glow_layer = np.zeros_like(canvas)
    glow_layer[mask > 0] = (0, 0, 255) # BGR: Red
    
    # Blur for glow effect
    glow_small = cv2.resize(glow_layer, (w//4, h//4))
    glow_small = cv2.GaussianBlur(glow_small, (15, 15), 0)
    glow = cv2.resize(glow_small, (w, h))
    
    # 2. Pulsing Extra Outline
    pulse = (np.sin(t * 8) + 1) / 2
    cv2.addWeighted(canvas, 1.0, glow, 0.4 + 0.3 * pulse, 0, canvas)
    
    # 3. Intimidating Ground Shadow / Red Floor
    # (Optional: just a dark red gradient at bottom)
    
    # 4. Render the stick figure on top
    _renderer.render(pose, (h, w), canvas)
    
    # 5. Red particles "Boss Aura"
    for _ in range(5):
        if random.random() < 0.2:
            joint = random.choice(list(lm.values()))
            cv2.circle(canvas, (joint[0] + random.randint(-20, 20), joint[1] + random.randint(-20, 20)), 1, (50, 50, 255), -1)

    return canvas
