"""
Hologram Skeleton Filter
The human body is rendered as translucent, glowing cyber-skin with full neon skeletal internals projected over it.
"""
import cv2
import numpy as np
import random
import time
from pose_detector import PoseResult, CONNECTIONS

_temporal_mask = None

def blur_glow(img, iterations=1):
    small = cv2.resize(img, (0, 0), fx=0.5, fy=0.5, interpolation=cv2.INTER_LINEAR)
    blur = small.copy()
    for _ in range(iterations):
        blur = cv2.GaussianBlur(blur, (9, 9), 0)
    blur_up = cv2.resize(blur, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_LINEAR)
    return cv2.addWeighted(img, 1.0, blur_up, 0.8, 0)

def apply(canvas: np.ndarray, pose: PoseResult, **kwargs) -> np.ndarray:
    if not pose.detected:
        return canvas

    h, w = canvas.shape[:2]
    original = kwargs.get('original_frame')
    
    if original is not None:
        gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
        bg = cv2.merge([(gray*0.3).astype(np.uint8), (gray*0.2).astype(np.uint8), (gray*0.1).astype(np.uint8)])
    else:
        bg = np.zeros((h, w, 3), dtype=np.uint8)
        
    canvas[:] = bg
    t = time.time()

    mask_float = None
    if getattr(pose, 'segmentation_mask', None) is not None:
        mask_raw = pose.segmentation_mask
        if mask_raw.shape[:2] != (h, w):
            mask_raw = cv2.resize(mask_raw, (w, h), interpolation=cv2.INTER_LINEAR)
            
        global _temporal_mask
        if '_temporal_mask' not in globals() or _temporal_mask is None or _temporal_mask.shape != mask_raw.shape:
            _temporal_mask = mask_raw.copy()
        else:
            cv2.addWeighted(_temporal_mask, 0.85, mask_raw, 0.15, 0, dst=_temporal_mask)
            
        # Fast Blur
        small_mask = cv2.resize(_temporal_mask, (0, 0), fx=0.5, fy=0.5, interpolation=cv2.INTER_LINEAR)
        small_mask = cv2.GaussianBlur(small_mask, (7, 7), 0)
        mask_float = cv2.resize(small_mask, (w, h), interpolation=cv2.INTER_LINEAR)

    hologram_layer = np.zeros((h, w, 3), dtype=np.uint8)
    if mask_float is not None:
        scanline_y = int((t % 1.5) / 1.5 * h)
        
        hologram_layer[:, :, 0] = (mask_float * 100).astype(np.uint8)
        hologram_layer[:, :, 1] = (mask_float * 200).astype(np.uint8)
        hologram_layer[:, :, 2] = (mask_float * 100).astype(np.uint8)
        
        cv2.line(hologram_layer, (0, scanline_y), (w, scanline_y), (255, 255, 255), 4)
        cv2.line(hologram_layer, (0, scanline_y-5), (w, scanline_y-5), (255, 255, 0), 2)
        
        # Fast Glow
        small_glow = cv2.resize(hologram_layer, (0, 0), fx=0.5, fy=0.5, interpolation=cv2.INTER_LINEAR)
        small_glow = cv2.GaussianBlur(small_glow, (9, 9), 0)
        scan_glow = cv2.resize(small_glow, (w, h), interpolation=cv2.INTER_LINEAR)
        
        hologram_layer = cv2.addWeighted(hologram_layer, 1.0, scan_glow, 0.5, 0)
        
        # Fast int blending
        mask_3ch = cv2.merge([mask_float, mask_float, mask_float])
        m_int = (mask_3ch * 179).astype(np.uint16) # ~0.7 scale
        inv_m = 256 - m_int
        canvas[:] = ((hologram_layer.astype(np.uint16) * m_int + canvas.astype(np.uint16) * inv_m) >> 8).astype(np.uint8)

    skel_layer = np.zeros((h, w, 3), dtype=np.uint8)
    lm = pose.landmarks
    vis = pose.visibility
    
    for a, b in CONNECTIONS:
        if a in lm and b in lm and vis.get(a, 0) > 0.3 and vis.get(b, 0) > 0.3:
            pt1, pt2 = lm[a], lm[b]
            cv2.line(skel_layer, pt1, pt2, (255, 255, 255), 3, cv2.LINE_AA)
            cv2.line(skel_layer, pt1, pt2, (255, 50, 200), 8, cv2.LINE_AA)
            cv2.line(skel_layer, pt1, pt2, (200, 0, 150), 16, cv2.LINE_AA) # Slashed thickness heavily
            
    for idx, pt in lm.items():
        if vis.get(idx, 0) > 0.3:
            cv2.circle(skel_layer, pt, 6, (255, 255, 255), -1)
            cv2.circle(skel_layer, pt, 12, (0, 255, 255), 2, cv2.LINE_AA)

    skel_layer = blur_glow(skel_layer, 1) # Only 1 iteration
    canvas[:] = cv2.add(canvas, skel_layer)

    return canvas
