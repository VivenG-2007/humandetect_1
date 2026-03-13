"""
Action Filter — Detects actions like spinning and punching to trigger specific effects.
- Spinning: Spawns sparkles
- Punching: Trigger lightning arcs
"""
import cv2
import numpy as np
import random
import time
from pose_detector import PoseResult
from skeleton_renderer import SkeletonRenderer
from utils.particle_system import ParticleSystem

# Initialize components
_renderer = SkeletonRenderer()
_sparks = ParticleSystem(max_particles=400)

# Constants
VIS_THRESHOLD = 0.4
PUNCH_SPEED_THRESHOLD = 25  # px/frame
SPIN_THRESHOLD = 0.8        # relative x-distance between shoulders

# Persistent state
_prev_lms = {}
_action_state = {
    "is_punching": False,
    "last_punch_time": 0,
    "spin_count": 0,
    "facing_forward": True
}

def _sparkle_color():
    return (random.randint(200, 255), random.randint(200, 255), random.randint(200, 255))

def _lightning_arc(img, pt1, pt2, color, segments=5):
    pts = [pt1]
    for i in range(1, segments):
        t = i / segments
        mx = int(pt1[0] + (pt2[0] - pt1[0]) * t + random.randint(-15, 15))
        my = int(pt1[1] + (pt2[1] - pt1[1]) * t + random.randint(-15, 15))
        pts.append((mx, my))
    pts.append(pt2)
    for i in range(len(pts) - 1):
        cv2.line(img, pts[i], pts[i+1], color, 2, cv2.LINE_AA)
        cv2.line(img, pts[i], pts[i+1], (255, 255, 255), 1, cv2.LINE_AA)

def apply(canvas: np.ndarray, pose: PoseResult, **kwargs) -> np.ndarray:
    global _prev_lms, _action_state
    h, w = canvas.shape[:2]
    t = time.time()
    
    _sparks.update()
    
    if pose.detected:
        lm = pose.landmarks
        vis = pose.visibility
        
        # 1. DETECT PUNCHING (Wrist speed)
        punch_detected = False
        for wrist_idx in [15, 16]: # Left, Right Wrist
            if wrist_idx in lm and wrist_idx in _prev_lms:
                if vis.get(wrist_idx, 0) > VIS_THRESHOLD:
                    v = np.linalg.norm(np.array(lm[wrist_idx]) - np.array(_prev_lms[wrist_idx]))
                    if v > PUNCH_SPEED_THRESHOLD:
                        punch_detected = True
                        # Trigger lightning from wrist to random nearby point
                        for _ in range(3):
                            angle = random.uniform(0, 2*np.pi)
                            dist = random.uniform(50, 150)
                            target = (int(lm[wrist_idx][0] + np.cos(angle)*dist), int(lm[wrist_idx][1] + np.sin(angle)*dist))
                            _lightning_arc(canvas, lm[wrist_idx], target, (255, 200, 50))
        
        # 2. DETECT SPINNING (Shoulder overlap/reversal)
        if 11 in lm and 12 in lm and vis.get(11,0) > 0.5 and vis.get(12,0) > 0.5:
            shoulder_dist = lm[12][0] - lm[11][0] # Right X - Left X (Positive if forward)
            # If shoulders get very close horizontally, person is likely sideways/spinning
            if abs(shoulder_dist) < (w * 0.05):
                # Spawn sparkles around the center of mass
                cx = (lm[11][0] + lm[12][0]) // 2
                cy = (lm[11][1] + lm[12][1]) // 2
                _sparks.spawn(cx, cy, count=10, color_fn=_sparkle_color, size_range=(2, 4), speed_scale=2.0)
        
        # Update prev state
        _prev_lms = {idx: pt for idx, pt in lm.items()}
        
    _sparks.draw(canvas)
    _renderer.render(pose, (h, w), canvas)
    
    # Overlay Action feedback
    return canvas
