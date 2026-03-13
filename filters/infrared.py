"""
Infrared Filter — simulates a classic FLIR thermal rainbow camera.
Backgrounds are dark blue/cold. Human bodies map across cyan, green, yellow, and red based on simulated core heat.
"""
import cv2
import numpy as np
from pose_detector import PoseResult, CONNECTIONS

_temporal_mask = None

def apply(canvas: np.ndarray, pose: PoseResult, **kwargs) -> np.ndarray:
    original = kwargs.get('original_frame')
    if original is None:
        return canvas

    h, w = canvas.shape[:2]

    # Convert original to grayscale
    gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
    
    # 1. Background Heatmap (Very Cold: Dark Blue -> Cyan)
    # Scale grayscale so max is ~60 (which is dark blue in JET)
    bg_heat = (gray * 0.2).astype(np.uint8)

    if pose.detected and getattr(pose, 'segmentation_mask', None) is not None:
        mask_float = pose.segmentation_mask
        if mask_float.shape[:2] != (h, w):
            mask_float = cv2.resize(mask_float, (w, h), interpolation=cv2.INTER_LINEAR)
        
        global _temporal_mask
        if '_temporal_mask' not in globals() or _temporal_mask is None or _temporal_mask.shape != mask_float.shape:
            _temporal_mask = mask_float.copy()
        else:
            cv2.addWeighted(_temporal_mask, 0.85, mask_float, 0.15, 0, dst=_temporal_mask)
            
        # FAST BLUR: Downscale -> Blur -> Upscale (4x faster)
        small_mask = cv2.resize(_temporal_mask, (0, 0), fx=0.5, fy=0.5, interpolation=cv2.INTER_LINEAR)
        small_mask = cv2.GaussianBlur(small_mask, (7, 7), 0)
        smooth_float = cv2.resize(small_mask, (w, h), interpolation=cv2.INTER_LINEAR)
        
        binary_mask = (smooth_float > 0.4).astype(np.uint8) * 255
        kernel = np.ones((3, 3), np.uint8) # reduced kernel size
        binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)
        
        fg_heat = cv2.add((gray * 0.3).astype(np.uint8), 80)
        
        # Optimize Distance Transform
        small_binary = cv2.resize(binary_mask, (0, 0), fx=0.5, fy=0.5, interpolation=cv2.INTER_NEAREST)
        dist_transform = cv2.distanceTransform(small_binary, cv2.DIST_L2, 3) # smaller mask param
        cv2.normalize(dist_transform, dist_transform, 0, 1.0, cv2.NORM_MINMAX)
        core_heat_small = (dist_transform * 100).astype(np.uint8)
        core_heat = cv2.resize(core_heat_small, (w, h), interpolation=cv2.INTER_LINEAR)
        
        fg_heat = cv2.add(fg_heat, core_heat)
        
        lm = pose.landmarks
        vis = pose.visibility
        heat_blobs = np.zeros((h//2, w//2), dtype=np.uint8) # Draw blobs downscaled!
        
        if 0 in lm and vis.get(0, 0) > 0.3:
            cv2.circle(heat_blobs, (lm[0][0]//2, lm[0][1]//2), 12, 80, -1, cv2.LINE_AA) 
        for hw in [15, 16]:
            if hw in lm and vis.get(hw, 0) > 0.2:
                cv2.circle(heat_blobs, (lm[hw][0]//2, lm[hw][1]//2), 12, 80, -1, cv2.LINE_AA)
                
        heat_blobs = cv2.GaussianBlur(heat_blobs, (21, 21), 0)
        heat_blobs_up = cv2.resize(heat_blobs, (w, h), interpolation=cv2.INTER_LINEAR)
        fg_heat = cv2.add(fg_heat, heat_blobs_up)

        thermal_bgr = cv2.applyColorMap(fg_heat, cv2.COLORMAP_JET)
        
        white_mask = (fg_heat > 240).astype(np.uint8) * 255
        white_mask_3ch = cv2.merge([white_mask, white_mask, white_mask])
        thermal_bgr = cv2.addWeighted(thermal_bgr, 1.0, white_mask_3ch, 0.5, 0)

        bg_b = cv2.add((gray * 0.4).astype(np.uint8), 40)
        bg_g = (gray * 0.15).astype(np.uint8)
        bg_r = np.zeros_like(gray)
        bg_custom = cv2.merge([bg_b, bg_g, bg_r])
        bg_custom[::2, :] = (bg_custom[::2, :] * 0.6).astype(np.uint8)

        # FAST Alpha Blend using INT16 to avoid float matrix overhead
        mask_3ch = cv2.merge([smooth_float, smooth_float, smooth_float])
        mask_int = (mask_3ch * 256).astype(np.uint16)
        inv_mask_int = 256 - mask_int
        
        canvas[:] = ((thermal_bgr.astype(np.uint16) * mask_int + bg_custom.astype(np.uint16) * inv_mask_int) >> 8).astype(np.uint8)
        
    else:
        # Fallback if no human is visible: entire room is cold tactical blue
        bg_b = cv2.add((gray * 0.4).astype(np.uint8), 40)
        bg_g = (gray * 0.15).astype(np.uint8)
        bg_r = np.zeros_like(gray)
        bg_custom = cv2.merge([bg_b, bg_g, bg_r])
        bg_custom[::2, :] = (bg_custom[::2, :] * 0.6).astype(np.uint8)
        canvas[:] = bg_custom

    return canvas
