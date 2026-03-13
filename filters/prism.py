"""
Prism Filter — the body acts as a faceted glass prism.
Refracts and distorts the camera feed. Stick figure is hidden.
Optimized for performance.
"""
import cv2
import numpy as np
import time
from pose_detector import PoseResult, CONNECTIONS

VIS_THRESHOLD = 0.4

def apply(canvas: np.ndarray, pose: PoseResult, **kwargs) -> np.ndarray:
    original_frame = kwargs.get('original_frame')
    if original_frame is None:
        return canvas
        
    h, w = canvas.shape[:2]
    
    # 1. Background: very dark, blurred room
    bg_small = cv2.resize(original_frame, (w//4, h//4))
    bg_blur = cv2.GaussianBlur(bg_small, (11, 11), 0)
    canvas[:] = cv2.resize(bg_blur, (w, h))
    cv2.addWeighted(canvas, 0.2, np.zeros_like(canvas), 0.8, 0, dst=canvas)

    if pose.detected:
        lm = pose.landmarks
        vis = pose.visibility
        
        # 2. Create Body Mask
        mask = np.zeros((h, w), dtype=np.uint8)
        for (a, b) in CONNECTIONS:
            if a not in lm or b not in lm: continue
            if vis.get(a, 0) < VIS_THRESHOLD or vis.get(b, 0) < VIS_THRESHOLD: continue
            cv2.line(mask, lm[a], lm[b], 255, 45, cv2.LINE_AA)
        
        # 3. Create Refraction Effect (Optimized)
        refraction = original_frame.copy()
        facet_w = 80
        for x in range(0, w, facet_w):
            shift = int(np.sin(x/100.0 + time.time()*2) * 15)
            roi = refraction[:, x:min(x+facet_w, w)]
            refraction[:, x:min(x+facet_w, w)] = np.roll(roi, shift, axis=0)

        # 4. Color fringe
        b, g, r = cv2.split(refraction)
        b = np.roll(b, 4, axis=1)
        refraction = cv2.merge([b, r, g])

        # 5. Apply with optimized blurring
        mask_small = cv2.resize(mask, (w//4, h//4))
        mask_small = cv2.GaussianBlur(mask_small, (7, 7), 0)
        float_mask = cv2.resize(mask_small, (w, h)).astype(float) / 255.0
        float_mask = np.repeat(float_mask[:, :, np.newaxis], 3, axis=2)
        
        prism_body = (refraction * float_mask).astype(np.uint8)
        canvas[:] = cv2.add(canvas, (prism_body * 0.9).astype(np.uint8))
        
        # Crystal outline
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(canvas, contours, -1, (255, 255, 255), 1, cv2.LINE_AA)

    return canvas
