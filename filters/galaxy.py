"""
Galaxy Filter — the body is composed of shimmering stars and cosmic nebulae.
Enhanced with 'constellation' lines connecting the major stars and a glowing core.
Stick figure hidden.
"""
import cv2
import numpy as np
import random
import time
from pose_detector import PoseResult, CONNECTIONS

VIS_THRESHOLD = 0.25

def apply(canvas: np.ndarray, pose: PoseResult, **kwargs) -> np.ndarray:
    h, w = canvas.shape[:2]
    t = time.time()
    
    if pose.detected:
        lm = pose.landmarks
        vis = pose.visibility
        
        # 1. Subtle Nebula Glow
        # Create a soft mask of the body to apply a purple/blue nebula tint
        mask = np.zeros((h, w), dtype=np.uint8)
        for (a, b) in CONNECTIONS:
            if a in lm and b in lm and vis.get(a, 0) > VIS_THRESHOLD and vis.get(b, 0) > VIS_THRESHOLD:
                cv2.line(mask, lm[a], lm[b], 255, 40, cv2.LINE_AA)
        
        # Downsample for faster blurring
        mask_small = cv2.resize(mask, (w//4, h//4))
        soft_small = cv2.GaussianBlur(mask_small, (15, 15), 0)
        soft_mask = cv2.resize(soft_small, (w, h))
        nebula_color = (120, 20, 80) # BGR: Deep purple
        alpha = soft_mask.astype(float) / 255.0 * 0.3
        for c in range(3):
            canvas[:, :, c] = (canvas[:, :, c] * (1 - alpha) + nebula_color[c] * alpha).astype(np.uint8)

        # 2. Constellation Lines
        # Draw thin, faint lines between major joints
        constellation_color = (255, 200, 150) # Pale cyan
        for (a, b) in CONNECTIONS:
            if a in lm and b in lm and vis.get(a, 0) > VIS_THRESHOLD and vis.get(b, 0) > VIS_THRESHOLD:
                pulse = (np.sin(t * 3 + a) + 1) / 2
                cv_color = tuple(int(c * (0.1 + 0.2 * pulse)) for c in constellation_color)
                cv2.line(canvas, lm[a], lm[b], cv_color, 1, cv2.LINE_AA)

        # 3. Shimmering Stars
        for (a, b) in CONNECTIONS:
            if a not in lm or b not in lm: continue
            if vis.get(a, 0) < VIS_THRESHOLD or vis.get(b, 0) < VIS_THRESHOLD: continue
            
            pt1 = np.array(lm[a])
            pt2 = np.array(lm[b])
            dist = np.linalg.norm(pt2 - pt1)
            
            num_stars = int(dist / 6)
            for i in range(num_stars):
                pos_t = i / num_stars
                pos = pt1 * pos_t + pt2 * (1 - pos_t)
                pos += np.random.uniform(-6, 6, 2)
                
                size = random.uniform(1, 2.0)
                # Twinkle based on position and time
                twinkle = (np.sin(t * 4 + pos[0] * 0.05 + pos[1] * 0.05) + 1) / 2
                star_color = (255, 255, 255) if random.random() > 0.3 else (255, 220, 200)
                c = tuple(int(ch * (0.4 + 0.6 * twinkle)) for ch in star_color)
                
                cv2.circle(canvas, (int(pos[0]), int(pos[1])), int(size), c, -1, cv2.LINE_AA)
                
                # Occasional flare
                if random.random() < 0.005:
                    cv2.circle(canvas, (int(pos[0]), int(pos[1])), int(size*2), (255, 255, 255), -1, cv2.LINE_AA)

        # 4. Joint "Galaxies"
        # Large clusters at joints
        for idx in [11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28, 0]:
            if idx in lm and vis.get(idx, 0) > VIS_THRESHOLD:
                pt = lm[idx]
                # Glowing center for joint
                pulse = (np.sin(t * 5 + idx) + 1) / 2
                cv2.circle(canvas, pt, int(5 + 2 * pulse), (255, 255, 200), -1, cv2.LINE_AA)
                cv2.circle(canvas, pt, int(12 + 5 * pulse), (200, 100, 50), 1, cv2.LINE_AA)

    return canvas
