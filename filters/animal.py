"""
Animal Morph Filter — Replaces the human body with a robust wolf-like form.
Improved robustness for partial detections. Stick figure is hidden.
"""
import cv2
import numpy as np
import random
import time
from pose_detector import PoseResult, CONNECTIONS

VIS_THRESHOLD = 0.35

# Wolf-like color palette
FUR_BASE = (60, 60, 60)       # Dark Gray
FUR_HIGHLIGHT = (100, 100, 100) # Lighter Gray
EYE_GLOW = (0, 255, 255)      # Yellow
MUZZLE_COLOR = (40, 40, 40)   # Near black

def apply(canvas: np.ndarray, pose: PoseResult, **kwargs) -> np.ndarray:
    if not pose.detected:
        return canvas

    h, w = canvas.shape[:2]
    lm = pose.landmarks
    vis = pose.visibility

    def get_pt(idx): return np.array(lm[idx]) if idx in lm and vis.get(idx, 0) > VIS_THRESHOLD else None

    # 1. DRAW BODY SEGMENTS (LOWER TO UPPER)
    # Hips Area
    hip_l, hip_r = get_pt(23), get_pt(24)
    if hip_l is not None and hip_r is not None:
        hip_mid = (hip_l + hip_r) // 2
        cv2.ellipse(canvas, tuple(hip_mid), (int(np.linalg.norm(hip_l-hip_r)*0.7), 40), 0, 0, 360, FUR_BASE, -1)

    # Torso/Chest
    sh_l, sh_r = get_pt(11), get_pt(12)
    if sh_l is not None and sh_r is not None:
        sh_mid = (sh_l + sh_r) // 2
        cv2.ellipse(canvas, tuple(sh_mid), (int(np.linalg.norm(sh_l-sh_r)*0.75), 50), 0, 0, 360, FUR_BASE, -1)
        
        # Connect shoulders to hips to fill the middle
        if hip_l is not None and hip_r is not None:
            torso_pts = np.array([sh_l, sh_r, hip_r, hip_l], np.int32)
            cv2.fillPoly(canvas, [torso_pts], FUR_BASE)

    # 2. DRAW LIMBS (THICK FURRY ARMS/LEGS)
    # Define limb thickness maps
    thickness_map = {
        (11, 13): 35, (13, 15): 25, # Arms
        (12, 14): 35, (14, 16): 25,
        (23, 25): 40, (25, 27): 30, # Legs
        (24, 26): 40, (26, 28): 30
    }
    
    for (a, b) in CONNECTIONS:
        pt_a, pt_b = get_pt(a), get_pt(b)
        if pt_a is not None and pt_b is not None:
            if a == 0 or b == 0: continue # Skip face-to-shoulder connections
            
            thickness = thickness_map.get((a, b), 20)
            # Outer fur
            cv2.line(canvas, tuple(pt_a), tuple(pt_b), FUR_BASE, thickness, cv2.LINE_AA)
            # Inner highlight
            cv2.line(canvas, tuple(pt_a), tuple(pt_b), FUR_HIGHLIGHT, thickness-12, cv2.LINE_AA)

    # 3. WOLF HEAD
    nose = get_pt(0)
    if nose is not None:
        nx, ny = nose
        head_radius = 45
        
        # Determine head orientation (roughly)
        eye_l, eye_r = get_pt(2), get_pt(5)
        
        # Head Base
        cv2.circle(canvas, (nx, ny-10), head_radius, FUR_BASE, -1, cv2.LINE_AA)
        
        # Pointy Ears
        ear_h = 50
        ear_w = 20
        # Left Ear
        pts_el = np.array([[nx-35, ny-20], [nx-45, ny-ear_h-20], [nx-15, ny-30]], np.int32)
        cv2.fillPoly(canvas, [pts_el], FUR_BASE)
        # Right Ear
        pts_er = np.array([[nx+35, ny-20], [nx+45, ny-ear_h-20], [nx+15, ny-30]], np.int32)
        cv2.fillPoly(canvas, [pts_er], FUR_BASE)

        # Muzzle (Wolf snout)
        cv2.ellipse(canvas, (nx, ny+5), (25, 20), 0, 0, 360, MUZZLE_COLOR, -1, cv2.LINE_AA)
        cv2.circle(canvas, (nx, ny+20), 6, (10, 10, 10), -1) # Nose tip

        # Glowing Eyes
        if eye_l is not None:
            cv2.circle(canvas, tuple(eye_l), 5, EYE_GLOW, -1, cv2.LINE_AA)
        if eye_r is not None:
            cv2.circle(canvas, tuple(eye_r), 5, EYE_GLOW, -1, cv2.LINE_AA)

    return canvas
