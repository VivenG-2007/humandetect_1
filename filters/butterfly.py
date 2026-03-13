"""
Butterfly Filter — Replaces the human body with a magical, glowing butterfly.
The face remains a high-detail point cloud, but the body has large, flapping wings.
Stick figure is hidden.
"""
import cv2
import numpy as np
import random
import time
from pose_detector import PoseResult, CONNECTIONS

VIS_THRESHOLD = 0.35

def apply(canvas: np.ndarray, pose: PoseResult, **kwargs) -> np.ndarray:
    if not pose.detected:
        return canvas

    h, w = canvas.shape[:2]
    lm = pose.landmarks
    vis = pose.visibility
    t_val = time.time()
    original_frame = kwargs.get('original_frame')

    def get_pt(idx): return np.array(lm[idx]) if idx in lm and vis.get(idx, 0) > VIS_THRESHOLD else None

    # 1. WING DRAWING LOGIC
    # Attached to shoulders (11, 12)
    sh_l, sh_r = get_pt(11), get_pt(12)
    hip_l, hip_r = get_pt(23), get_pt(24)
    
    if sh_l is not None and sh_r is not None:
        mid_sh = (sh_l + sh_r) // 2
        
        # Wing Flap Animation
        flap = np.sin(t_val * 6) * 0.4 + 0.6 # Cycles between small and large
        
        # Colors: Iridescent Blue/Purple
        color_top = (255, 100, 200) # Purple (BGR)
        color_bot = (255, 200, 50)  # Cyan (BGR)
        
        # Draw Wings (Top and Bottom lobes)
        # Left Wing
        def draw_lobe(center, scale_x, scale_y, angle, col):
            # Procedural wing lobe
            lobe_pts = []
            for a in range(0, 360, 10):
                rad = np.deg2rad(a)
                # Heart-like shape for lobes
                r = (1 + np.sin(rad)) * 80 * flap
                lx = center[0] + np.cos(rad + angle) * r * scale_x
                ly = center[1] + np.sin(rad + angle) * r * scale_y
                lobe_pts.append([lx, ly])
            cv2.fillPoly(canvas, [np.array(lobe_pts, np.int32)], col)
            # Add glowing edge
            cv2.polylines(canvas, [np.array(lobe_pts, np.int32)], True, (255, 255, 255), 2, cv2.LINE_AA)

        # Draw left wing lobes
        draw_lobe(sh_l, 1.2, 1.5, np.pi/1.2, color_top)
        draw_lobe(sh_l + [0, 40], 0.8, 1.2, np.pi/1.5, color_bot)
        
        # Draw right wing lobes
        draw_lobe(sh_r, 1.2, 1.5, -np.pi/1.2, color_top)
        draw_lobe(sh_r + [0, 40], 0.8, 1.2, -np.pi/1.5, color_bot)

    # 2. BUTTERFLY BODY
    # Slender body along the spine
    if sh_l is not None and sh_r is not None and hip_l is not None and hip_r is not None:
        mid_sh = (sh_l + sh_r) // 2
        mid_hip = (hip_l + hip_r) // 2
        # Abdomen
        cv2.line(canvas, tuple(mid_sh), tuple(mid_hip), (40, 40, 40), 12, cv2.LINE_AA)
        cv2.line(canvas, tuple(mid_sh), tuple(mid_hip), (80, 80, 80), 4, cv2.LINE_AA)

    # 3. FACE CAPTURE (RAW FEED ON TOP)
    if all(idx in lm and vis.get(idx, 0) > VIS_THRESHOLD for idx in [0, 1, 2]):
        nx, ny = lm[0]
        # Create a circle mask for the face
        face_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.circle(face_mask, (nx, ny-10), 65, 255, -1)
        face_mask = cv2.GaussianBlur(face_mask, (25, 25), 0)
        
        if original_frame is not None:
            # Blend raw face onto canvas
            mask_3ch = face_mask.astype(float) / 255.0
            mask_3ch = np.repeat(mask_3ch[:, :, np.newaxis], 3, axis=2)
            canvas[:] = (canvas * (1 - mask_3ch) + original_frame * mask_3ch).astype(np.uint8)
            
        # Add magical sparkling antennae on top of the raw face
        mid_eyes = (np.array(lm[1]) + np.array(lm[2])) / 2
        for side in [-1, 1]:
            ant_end = mid_eyes + np.array([side * 60, -100 + np.sin(t_val*4)*20])
            cv2.line(canvas, (int(mid_eyes[0]), int(mid_eyes[1]-20)), (int(ant_end[0]), int(ant_end[1])), (200, 255, 255), 2, cv2.LINE_AA)
            cv2.circle(canvas, (int(ant_end[0]), int(ant_end[1])), 5, (200, 255, 255), -1)

    return canvas
