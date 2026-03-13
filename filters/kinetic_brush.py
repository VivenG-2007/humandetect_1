"""
Kinetic Brushstrokes Filter - Pro Smooth Edition
Every movement generates an artistic, trailing paint stroke with sub-pixel interpolation.
"""
import cv2
import numpy as np
from collections import deque
import time
from pose_detector import PoseResult

# Config
_TRACKED_POINTS = [0, 35, 36] # Head, Left Index Tip, Right Index Tip
_trajectories   = {}
_smoothed_lms   = {}
_MAX_HISTORY    = 30 
_SMOOTH_FACTOR  = 0.75 # EMA factor (0.0 to 1.0). Higher = more responsive, lower = steadier.

def get_color(idx):
    # Vibrant artistic palette
    colors = [
        (255, 180, 50),   # Cyan-ish
        (50, 255, 255),   # Yellow
        (220, 50, 255),   # Pink/Magenta
        (100, 100, 255),  # Soft Blue
        (50, 255, 150),   # Mint
        (50, 150, 255),   # Orange
        (255, 100, 100)   # Purple
    ]
    return colors[idx % len(colors)]

def apply(canvas: np.ndarray, pose: PoseResult, **kwargs) -> np.ndarray:
    global _trajectories, _smoothed_lms
    
    h, w = canvas.shape[:2]
    
    # 1. Background: Muted grayscale
    original = kwargs.get('original_frame')
    if original is not None:
        gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
        # Fast muted background
        canvas[:] = (gray // 6)[:, :, np.newaxis]
    else:
        canvas[:] = (12, 12, 15)

    if not pose.detected:
        for idx in _trajectories:
            if _trajectories[idx]:
                _trajectories[idx].popleft()
    else:
        lm = pose.landmarks
        vis = pose.visibility
        
        for idx in _TRACKED_POINTS:
            if idx not in lm or vis.get(idx, 0) < 0.35:
                if idx in _trajectories and _trajectories[idx]:
                    _trajectories[idx].popleft()
                continue
            
            # EMA Smoothing for jitter reduction
            curr_pt = np.array(lm[idx], dtype=np.float32)
            if idx not in _smoothed_lms:
                _smoothed_lms[idx] = curr_pt
            else:
                _smoothed_lms[idx] = _smoothed_lms[idx] * (1.0 - _SMOOTH_FACTOR) + curr_pt * _SMOOTH_FACTOR
            
            if idx not in _trajectories:
                _trajectories[idx] = deque(maxlen=_MAX_HISTORY)
            
            _trajectories[idx].append(_smoothed_lms[idx].copy())

    # 2. Draw Tapered, Interpolated Strokes
    sw, sh = w // 2, h // 2
    paint_layer = np.zeros((sh, sw, 3), dtype=np.uint8)
    
    for i, idx in enumerate(_TRACKED_POINTS):
        pts = list(_trajectories.get(idx, []))
        if len(pts) < 2:
            continue
            
        color = get_color(i)
        num_pts = len(pts)
        
        for j in range(num_pts - 1):
            p1 = pts[j]
            p2 = pts[j+1]
            
            # Calculate tapering thickness for this segment
            # j=0 is tail, j=num_pts-2 is head
            prog1 = j / (num_pts - 1)
            prog2 = (j + 1) / (num_pts - 1)
            
            thick1 = int(24 * (prog1 ** 1.8)) + 1
            thick2 = int(24 * (prog2 ** 1.8)) + 1
            
            # Sub-segment interpolation to ensure curved smoothness
            dist = np.linalg.norm(p2 - p1)
            # 1 step per 3 pixels of movement at half-res
            num_steps = max(1, int(dist / 6)) 
            
            for s in range(num_steps):
                f1 = s / num_steps
                f2 = (s + 1) / num_steps
                
                # Interpolate position
                seg_p1 = (p1 * (1 - f1) + p2 * f1) / 2
                seg_p2 = (p1 * (1 - f2) + p2 * f2) / 2
                
                # Interpolate thickness
                seg_thick = int(thick1 * (1 - f1) + thick2 * f1)
                
                sp1 = (int(seg_p1[0]), int(seg_p1[1]))
                sp2 = (int(seg_p2[0]), int(seg_p2[1]))
                
                cv2.line(paint_layer, sp1, sp2, color, seg_thick, cv2.LINE_AA)
                if s == num_steps - 1: # Cap the joint
                    cv2.circle(paint_layer, sp2, seg_thick // 2, color, -1, cv2.LINE_AA)

    # 3. Post-Process 'Bleed'
    paint_blur = cv2.GaussianBlur(paint_layer, (3, 3), 0)
    smoothed_paint = cv2.resize(paint_blur, (w, h), interpolation=cv2.INTER_LINEAR)
    
    # Additive Blend
    cv2.add(canvas, smoothed_paint, dst=canvas)

    return canvas
