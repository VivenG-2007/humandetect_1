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
_temporal_mask = None

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
        
        # 2. Create Solid Body Mask from Segmentation 
        # (This gives the entire body silhouette instead of just skeleton lines)
        if getattr(pose, 'segmentation_mask', None) is not None:
            mask_float = pose.segmentation_mask
            if mask_float.shape[:2] != (h, w):
                mask_float = cv2.resize(mask_float, (w, h), interpolation=cv2.INTER_LINEAR)
            
            # Temporal Smoothing to stop edge flickering
            global _temporal_mask
            if '_temporal_mask' not in globals() or _temporal_mask is None or _temporal_mask.shape != mask_float.shape:
                _temporal_mask = mask_float.copy()
            else:
                cv2.addWeighted(_temporal_mask, 0.8, mask_float, 0.2, 0, dst=_temporal_mask)
            # FAST BLUR: Temporal Smoothing without crushing FPS
            small_mask = cv2.resize(_temporal_mask, (0, 0), fx=0.5, fy=0.5, interpolation=cv2.INTER_LINEAR)
            small_mask = cv2.GaussianBlur(small_mask, (7, 7), 0)
            smooth_float = cv2.resize(small_mask, (w, h), interpolation=cv2.INTER_LINEAR)
            
            mask = (smooth_float > 0.35).astype(np.uint8) * 255
            # Draw skeletal hands explicitly so they don't get lost, including new fingertips
            for w_idx, i_k, i_t, p_k, p_t, t_k, t_t in [(15, 19, 35, 17, 33, 21, 37), (16, 20, 36, 18, 34, 22, 38)]:
                if w_idx in lm and vis.get(w_idx, 0) > 0.15:
                    cv2.circle(mask, lm[w_idx], 22, 255, -1, cv2.LINE_AA)
                    if i_k in lm and i_t in lm and vis.get(i_t, 0) > 0.15:
                        cv2.line(mask, lm[w_idx], lm[i_k], 255, 30, cv2.LINE_AA)
                        cv2.line(mask, lm[i_k], lm[i_t], 255, 25, cv2.LINE_AA)
                        cv2.circle(mask, lm[i_t], 15, 255, -1, cv2.LINE_AA)
                    if p_k in lm and p_t in lm and vis.get(p_t, 0) > 0.15:
                        cv2.line(mask, lm[w_idx], lm[p_k], 255, 30, cv2.LINE_AA)
                        cv2.line(mask, lm[p_k], lm[p_t], 255, 20, cv2.LINE_AA)
                    if t_k in lm and t_t in lm and vis.get(t_t, 0) > 0.15:
                        cv2.line(mask, lm[w_idx], lm[t_k], 255, 30, cv2.LINE_AA)
                        cv2.line(mask, lm[t_k], lm[t_t], 255, 25, cv2.LINE_AA)
                        
            kernel = np.ones((5, 5), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        else:
            # Fallback to thick skeleton
            mask = np.zeros((h, w), dtype=np.uint8)
            for (a, b) in CONNECTIONS:
                if a not in lm or b not in lm: continue
                if vis.get(a, 0) < VIS_THRESHOLD or vis.get(b, 0) < VIS_THRESHOLD: continue
                cv2.line(mask, lm[a], lm[b], 255, 45, cv2.LINE_AA)
                cv2.circle(mask, lm[a], 22, 255, -1, cv2.LINE_AA)
                cv2.circle(mask, lm[b], 22, 255, -1, cv2.LINE_AA)
        
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
