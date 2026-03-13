"""
Lightning Filter — electric arcs drawn along the skeleton's bones.
Includes multi-stage gesture toggle system.
"""
import cv2
import numpy as np
import random
import time
from pose_detector import PoseResult, CONNECTIONS

VIS_THRESHOLD = 0.25

_effect_mode = 0  # 0: Thunder, 1: Axe, 2: Flower, 3: Fire
_last_toggle_time = 0
_fire_particles = []

def _lightning_bolt(img, pt1, pt2, color, segments=8, jitter=12, thickness=2):
    pts = [pt1]
    for i in range(1, segments):
        t = i / segments
        mx = int(pt1[0] + (pt2[0] - pt1[0]) * t + random.randint(-jitter, jitter))
        my = int(pt1[1] + (pt2[1] - pt1[1]) * t + random.randint(-jitter, jitter))
        pts.append((mx, my))
    pts.append(pt2)
    for i in range(len(pts) - 1):
        cv2.line(img, pts[i], pts[i + 1], tuple(c // 4 for c in color), thickness + 6, cv2.LINE_AA)
        cv2.line(img, pts[i], pts[i + 1], color, thickness, cv2.LINE_AA)

def apply(canvas: np.ndarray, pose: PoseResult, **kwargs) -> np.ndarray:
    global _effect_mode, _last_toggle_time, _fire_particles

    # Update fire particles
    _fire_particles = [[x, y - random.uniform(2, 6), max(0, r - 0.5)]
                       for x, y, r in _fire_particles if r > 0.5]

    if not pose.detected:
        return canvas

    h, w = canvas.shape[:2]
    lm = pose.landmarks
    vis = pose.visibility

    # Detect Fist to toggle mode
    current_time = time.time()
    if (current_time - _last_toggle_time) > 1.0:
        fist_detected = False
        if 11 in lm and 12 in lm:
            shoulder_width = np.hypot(lm[11][0] - lm[12][0], lm[11][1] - lm[12][1])
            # Dramatically increased the maximum distance so even slight curls trigger the weapon swap safely.
            fist_thresh = max(50.0, shoulder_width * 0.45)
            
            for w_idx, i_idx, p_idx in [(15, 19, 17), (16, 20, 18)]: # Wrist, Index, Pinky
                if w_idx in lm and vis.get(w_idx, 0) > 0.15: # Highly forgiving visibility constraint
                    if i_idx in lm and p_idx in lm:
                        d_i = np.hypot(lm[w_idx][0]-lm[i_idx][0], lm[w_idx][1]-lm[i_idx][1])
                        d_p = np.hypot(lm[w_idx][0]-lm[p_idx][0], lm[w_idx][1]-lm[p_idx][1])
                        
                        if d_i < fist_thresh and d_p < fist_thresh:
                            # Wrist must be safely inside the active camera boundary to reject edge glitches
                            if 40 < lm[w_idx][0] < w - 40 and 40 < lm[w_idx][1] < h - 60:
                                fist_detected = True
        
        if fist_detected:
            _effect_mode = (_effect_mode + 1) % 4
            _last_toggle_time = current_time

    # Decide hand color based on mode
    hand_color = (255, 220, 100) if _effect_mode == 0 else \
                 (100, 200, 255) if _effect_mode == 1 else \
                 (200, 100, 255) if _effect_mode == 2 else \
                 (50, 150, 255)

    for idx in [15, 16]: 
        if idx in lm and vis.get(idx, 0) > VIS_THRESHOLD:
            pt = lm[idx]
            
            # Core Hand (Fingers actively removed per request)
            cv2.circle(canvas, pt, 10, hand_color, -1, cv2.LINE_AA)
            cv2.circle(canvas, pt, 16, (255, 255, 255), 2, cv2.LINE_AA)

            if _effect_mode == 0:
                for _ in range(random.randint(3, 5)):
                    angle = random.uniform(0, 2 * np.pi)
                    dist = random.uniform(40, 120)
                    target = (int(pt[0] + np.cos(angle)*dist), int(pt[1] + np.sin(angle)*dist))
                    lc = random.choice([(255,255,255), (255,220,100), (255,150,50)])
                    _lightning_bolt(canvas, pt, target, lc, segments=6, jitter=15, thickness=2)
            
            elif _effect_mode == 1:
                p1 = (pt[0], pt[1] + 80)
                p2 = (pt[0], pt[1] - 40)
                cv2.line(canvas, p1, p2, (100, 150, 200), 10, cv2.LINE_AA)
                blade_pts = np.array([[pt[0], pt[1] - 20], [pt[0] + 60, pt[1] - 50], [pt[0] + 75, pt[1]], [pt[0] + 50, pt[1] + 40], [pt[0], pt[1] + 20]], dtype=np.int32)
                if idx == 16: blade_pts[:, 0] = pt[0] - (blade_pts[:, 0] - pt[0])
                cv2.fillPoly(canvas, [blade_pts], (255, 100, 50))
                cv2.polylines(canvas, [blade_pts], True, (255, 255, 255), 3, cv2.LINE_AA)

            elif _effect_mode == 2:
                spin = time.time() * 2
                for i in range(6):
                    angle = spin + (i * 2 * np.pi / 6)
                    px = int(pt[0] + np.cos(angle) * 35)
                    py = int(pt[1] + np.sin(angle) * 35)
                    cv2.circle(canvas, (px, py), 18, (255, 100, 220), -1, cv2.LINE_AA)
                    cv2.circle(canvas, (px, py), 18, (255, 255, 255), 2, cv2.LINE_AA)
                cv2.circle(canvas, pt, 20, (50, 255, 255), -1, cv2.LINE_AA)

            elif _effect_mode == 3:
                for _ in range(4):
                    _fire_particles.append([pt[0] + random.uniform(-20, 20), pt[1] + random.uniform(-15, 15), random.uniform(10, 30)])

    if _effect_mode == 3:
        for x, y, r in _fire_particles:
            color = (random.randint(0, 50), random.randint(100, 150), random.randint(200, 255))
            cv2.circle(canvas, (int(x), int(y)), int(r), color, -1, cv2.LINE_AA)
            cv2.circle(canvas, (int(x), int(y)), int(r*0.6), (150, 220, 255), -1, cv2.LINE_AA)

    return canvas
