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
_was_fist_any = False
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
    global _effect_mode, _was_fist_any, _last_toggle_time, _fire_particles

    # Update fire particles (Scale-aware physics)
    # We use a base scale of 100 for normalization
    drift = max(1.0, (kwargs.get('scale_ref', 100) / 100.0) * 3.0)
    shrink = max(0.1, (kwargs.get('scale_ref', 100) / 100.0) * 0.4)
    
    _fire_particles = [[x, y - random.uniform(drift*0.5, drift*1.5), max(0, r - shrink)]
                       for x, y, r in _fire_particles if r > shrink]

    if not pose.detected:
        return canvas

    h, w = canvas.shape[:2]
    lm = pose.landmarks
    vis = pose.visibility

    # 1. Scale/Distance Calibration
    scale_ref = 100.0
    if 11 in lm and 12 in lm:
        scale_ref = max(30.0, np.hypot(lm[11][0] - lm[12][0], lm[11][1] - lm[12][1]))
    
    # Detect Fist to toggle mode (Scale-aware threshold)
    current_time = time.time()
    fist_detected = False
    is_currently_fist = False
    
    # 2. Check both hands for a closed fist (Index 35/36, Middle 40/41, Pinky 33/34)
    # Using a ratio of scale_ref (~45-50% of shoulder width) is far more robust than fixed pixels.
    fist_thresh = scale_ref * 0.45 

    for w_idx, i_tip, m_tip, p_tip in [(15, 35, 40, 33), (16, 36, 41, 34)]:
        if all(idx in lm for idx in [w_idx, i_tip, m_tip, p_tip]):
            if vis.get(w_idx, 0) > 0.25:
                # Calculate distances from tips to wrist
                d_i = np.hypot(lm[w_idx][0]-lm[i_tip][0], lm[w_idx][1]-lm[i_tip][1])
                d_m = np.hypot(lm[w_idx][0]-lm[m_tip][0], lm[w_idx][1]-lm[m_tip][1])
                d_p = np.hypot(lm[w_idx][0]-lm[p_tip][0], lm[w_idx][1]-lm[p_tip][1])
                
                if d_i < fist_thresh and d_m < fist_thresh and d_p < fist_thresh:
                    is_currently_fist = True
                    break
    
    # Trigger transition on "Down" stroke (Open -> Closed)
    if is_currently_fist and not _was_fist_any and (current_time - _last_toggle_time > 0.35):
        fist_detected = True
        _last_toggle_time = current_time
    
    _was_fist_any = is_currently_fist
    
    if fist_detected:
        _effect_mode = (_effect_mode + 1) % 4

    # Decide hand color based on mode
    hand_color = (255, 220, 100) if _effect_mode == 0 else \
                 (100, 200, 255) if _effect_mode == 1 else \
                 (200, 100, 255) if _effect_mode == 2 else \
                 (50, 150, 255)

    for side, wrist_idx, index_tip in [("left", 15, 35), ("right", 16, 36)]:
        if wrist_idx in lm and vis.get(wrist_idx, 0) > VIS_THRESHOLD:
            # Refine hand position: Move 35% from wrist toward index tip for a 'palm' center
            w_pt = np.array(lm[wrist_idx])
            if index_tip in lm:
                t_pt = np.array(lm[index_tip])
                pt = tuple((w_pt * 0.65 + t_pt * 0.35).astype(int))
            else:
                pt = tuple(w_pt.astype(int))
            
            # Core Hand Glow (Proportional)
            cg1 = int(scale_ref * 0.10)
            cg2 = int(scale_ref * 0.15)
            cv2.circle(canvas, pt, cg1, hand_color, -1, cv2.LINE_AA)
            cv2.circle(canvas, pt, cg2, (255, 255, 255), 2, cv2.LINE_AA)

            if _effect_mode == 0:
                for _ in range(random.randint(3, 5)):
                    angle = random.uniform(0, 2 * np.pi)
                    dist = random.uniform(scale_ref * 0.3, scale_ref * 0.7)
                    target = (int(pt[0] + np.cos(angle)*dist), int(pt[1] + np.sin(angle)*dist))
                    lc = random.choice([(255,255,255), (255,220,100), (255,150,50)])
                    bolt_jitter = int(scale_ref * 0.10)
                    _lightning_bolt(canvas, pt, target, lc, segments=6, jitter=bolt_jitter, thickness=2)
            
            elif _effect_mode == 1:
                # Scaled Axe (Compact)
                h1, h2 = int(scale_ref * 0.5), int(scale_ref * 0.25)
                p1 = (pt[0], pt[1] + h1)
                p2 = (pt[0], pt[1] - h2)
                cv2.line(canvas, p1, p2, (100, 150, 200), max(3, int(scale_ref*0.06)), cv2.LINE_AA)
                
                # Scaled Blade (Compact)
                bw, bh = int(scale_ref * 0.4), int(scale_ref * 0.3)
                blade_pts = np.array([
                    [pt[0], pt[1] - int(bh*0.4)], 
                    [pt[0] + bw, pt[1] - int(bh)], 
                    [pt[0] + int(bw*1.2), pt[1]], 
                    [pt[0] + bw, pt[1] + int(bh*0.8)], 
                    [pt[0], pt[1] + int(bh*0.4)]
                ], dtype=np.int32)
                
                if wrist_idx == 16: blade_pts[:, 0] = pt[0] - (blade_pts[:, 0] - pt[0])
                cv2.fillPoly(canvas, [blade_pts], (255, 100, 50))
                cv2.polylines(canvas, [blade_pts], True, (255, 255, 255), 3, cv2.LINE_AA)

            elif _effect_mode == 2:
                # Scaled Flower (Compact)
                spin = time.time() * 2
                orb_r = int(scale_ref * 0.24)
                pet_r = int(scale_ref * 0.12)
                for i in range(6):
                    angle = spin + (i * 2 * np.pi / 6)
                    px = int(pt[0] + np.cos(angle) * orb_r)
                    py = int(pt[1] + np.sin(angle) * orb_r)
                    cv2.circle(canvas, (px, py), pet_r, (255, 100, 220), -1, cv2.LINE_AA)
                    cv2.circle(canvas, (px, py), pet_r, (255, 255, 255), 2, cv2.LINE_AA)
                cv2.circle(canvas, pt, int(orb_r * 0.6), (50, 255, 255), -1, cv2.LINE_AA)

            elif _effect_mode == 3:
                # Scaled Fire (Compact)
                f_rad = scale_ref * 0.12
                for _ in range(4):
                    _fire_particles.append([pt[0] + random.uniform(-f_rad, f_rad), 
                                          pt[1] + random.uniform(-f_rad*0.75, f_rad*0.75), 
                                          random.uniform(scale_ref*0.05, scale_ref*0.15)])

    if _effect_mode == 3:
        for x, y, r in _fire_particles:
            color = (random.randint(0, 50), random.randint(100, 150), random.randint(200, 255))
            cv2.circle(canvas, (int(x), int(y)), int(r), color, -1, cv2.LINE_AA)
            cv2.circle(canvas, (int(x), int(y)), int(r*0.6), (150, 220, 255), -1, cv2.LINE_AA)

    return canvas
