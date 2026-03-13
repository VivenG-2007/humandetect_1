"""
Kinetic Brushstrokes Filter
Every movement generates animated, trailing paint strokes following the trajectory of major cinematic limbs.
"""
import cv2
import numpy as np
from collections import deque
from pose_detector import PoseResult

# Track trajectories for massive structural nodes: Hands, feet, head
_TRACKED_POINTS = [0, 15, 16, 27, 28]
_trajectories = {}
_MAX_HISTORY = 35 # Length of the paint smear

def get_color(idx):
    colors = [
        (255, 200, 50),   # Cyan for head
        (50, 255, 255),   # Yellow for Right Hand
        (255, 50, 255),   # Magenta for Left Hand
        (50, 50, 255),    # Red for Left Foot
        (255, 100, 100)   # Blue/Purple for Right Foot
    ]
    return colors[idx % len(colors)]

def apply(canvas: np.ndarray, pose: PoseResult, **kwargs) -> np.ndarray:
    global _trajectories
    
    # 1. Dark artistic background
    original = kwargs.get('original_frame')
    if original is not None:
        gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
        canvas[:] = cv2.merge([(gray*0.2).astype(np.uint8)]*3)
    else:
        canvas[:] = (20, 20, 20)

    if not pose.detected:
        for idx in _trajectories:
            if len(_trajectories[idx]) > 0:
                _trajectories[idx].popleft()
    else:
        lm = pose.landmarks
        vis = pose.visibility
        
        for idx in _TRACKED_POINTS:
            if idx not in _trajectories:
                _trajectories[idx] = deque(maxlen=_MAX_HISTORY)
            
            if idx in lm and vis.get(idx, 0) > 0.4:
                _trajectories[idx].append(lm[idx])
            else:
                if len(_trajectories[idx]) > 0:
                    _trajectories[idx].popleft()

    # 2. Draw thick, tapering Splines / Paint strokes
    paint_layer = np.zeros_like(canvas)
    
    for i, idx in enumerate(_TRACKED_POINTS):
        points = list(_trajectories.get(idx, []))
        if len(points) < 2:
            continue
            
        color = get_color(i)
        
        # Tapering filled circles logic mimics oil paint brush physics
        num_points = len(points)
        for j in range(num_points - 1):
            pt1 = points[j]
            pt2 = points[j+1]
            
            # Head of the array (later index) is max thickness
            progress = (j + 1) / num_points
            thickness = int(45 * progress) + 3
            
            # Stepwise interpolation filling gaps between frames
            dist = int(np.hypot(pt2[0]-pt1[0], pt2[1]-pt1[1]))
            steps = max(dist // 2, 1)
            for s in range(steps):
                t = s / steps
                ix = int(pt1[0] + (pt2[0] - pt1[0]) * t)
                iy = int(pt1[1] + (pt2[1] - pt1[1]) * t)
                cv2.circle(paint_layer, (ix, iy), thickness//2, color, -1, cv2.LINE_AA)

    # FAST BLUR: Downscale to proxy resolution to evaluate heavy brush liquefaction logic
    small_paint = cv2.resize(paint_layer, (0, 0), fx=0.5, fy=0.5, interpolation=cv2.INTER_LINEAR)
    small_blur = cv2.GaussianBlur(small_paint, (7, 7), 0)
    smoothed_paint = cv2.resize(small_blur, (canvas.shape[1], canvas.shape[0]), interpolation=cv2.INTER_LINEAR)
    
    # Additive Blend for neon acrylic intensity
    canvas[:] = cv2.addWeighted(canvas, 1.0, smoothed_paint, 1.0, 0)

    return canvas
