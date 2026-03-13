"""
Extreme Filters - A comprehensive collection of high-energy interactive filters:
1. Plasma Mode - Flowing energy pulses through bones.
2. Energy Shield - Circular shield when hands meet.
3. Black Hole - Swirling gravity particles around the torso.
4. Supernova - Explosion of stars when jumping.
5. Action FX (Merged):
   - Sparkles when spinning.
   - Lightning arcs when punching.
"""
import cv2
import numpy as np
import random
import time
from pose_detector import PoseResult, CONNECTIONS
from skeleton_renderer import SkeletonRenderer
from utils.particle_system import ParticleSystem

# Initialize components
_renderer = SkeletonRenderer()
_star_system = ParticleSystem(max_particles=600)
_gravity_system = ParticleSystem(max_particles=400)
_sparks = ParticleSystem(max_particles=300)

# Persistent state
_prev_y = 0
_prev_lms = {}
_last_jump_time = 0

# Thresholds
_JUMP_VEL_THRESH = -15
_PUNCH_SPEED_THRESH = 25
_VIS_THRESH = 0.4

def _star_color():
    return (random.randint(200, 255), random.randint(230, 255), random.randint(230, 255))

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
    global _prev_y, _prev_lms, _last_jump_time
    h, w = canvas.shape[:2]
    t = time.time()
    
    # Update particle systems
    _star_system.update()
    _gravity_system.update()
    _sparks.update()

    if pose.detected:
        lm = pose.landmarks
        vis = pose.visibility
        
        # --- 1. PLASMA MODE (Flowing Energy) ---
        for (a, b) in CONNECTIONS:
            if a in lm and b in lm and vis.get(a,0) > _VIS_THRESH and vis.get(b,0) > _VIS_THRESH:
                p1, p2 = np.array(lm[a]), np.array(lm[b])
                cv2.line(canvas, tuple(p1), tuple(p2), (100, 50, 0), 2, cv2.LINE_AA)
                pulse_t = (t * 3) % 1.0
                pulse_pt = (p1 * (1 - pulse_t) + p2 * pulse_t).astype(int)
                cv2.circle(canvas, tuple(pulse_pt), 4, (255, 180, 100), -1, cv2.LINE_AA)
                cv2.circle(canvas, tuple(pulse_pt), 8, (255, 100, 50), 1, cv2.LINE_AA)

        # --- 2. ENERGY SHIELD (Hands Together) ---
        if 15 in lm and 16 in lm and vis.get(15,0) > 0.5 and vis.get(16,0) > 0.5:
            dist = np.linalg.norm(np.array(lm[15]) - np.array(lm[16]))
            if dist < 65:
                center = ((lm[15][0] + lm[16][0]) // 2, (lm[15][1] + lm[16][1]) // 2)
                radius = int(85 + np.sin(t * 15) * 8)
                overlay = np.zeros_like(canvas)
                cv2.circle(overlay, center, radius, (255, 200, 0), 3, cv2.LINE_AA)
                cv2.circle(overlay, center, radius + 10, (255, 100, 0), 1, cv2.LINE_AA)
                
                # Optimized blur: Downsample -> Blur -> Upsample
                overlay_small = cv2.resize(overlay, (w // 4, h // 4))
                glow_small = cv2.GaussianBlur(overlay_small, (11, 11), 0)
                glow = cv2.resize(glow_small, (w, h))
                
                canvas[:] = cv2.add(canvas, glow)
                canvas[:] = cv2.add(canvas, overlay)

        # --- 3. BLACK HOLE AURA (Swirling Gravity) ---
        if 23 in lm and 24 in lm and 11 in lm:
            center = ((lm[23][0] + lm[24][0]) // 2, (lm[11][1] + lm[23][1]) // 2)
            for _ in range(2):
                angle = random.uniform(0, 2*np.pi)
                r = random.uniform(40, 180)
                px = int(center[0] + np.cos(angle) * r)
                py = int(center[1] + np.sin(angle) * r)
                vec = np.array(center) - np.array([px, py])
                norm = np.linalg.norm(vec)
                if norm > 0:
                    v_in = vec / norm * 2.5
                    v_tan = np.array([-v_in[1], v_in[0]]) * 2
                    _gravity_system.spawn(px, py, count=1, color_fn=lambda: (random.randint(40, 80), 0, random.randint(40, 80)), speed_scale=1.0)
                    _gravity_system.particles[-1].velocity = (v_in[0] + v_tan[0], v_in[1] + v_tan[1])

        # --- 4. SUPERNOVA BURST (Jumping) ---
        curr_hip_y = (lm[23][1] + lm[24][1]) // 2
        vel_y = curr_hip_y - _prev_y
        if vel_y < _JUMP_VEL_THRESH and (t - _last_jump_time) > 1.0:
            _last_jump_time = t
            _star_system.spawn(lm[23][0], lm[23][1], count=40, color_fn=_star_color, speed_scale=5.0)
            _star_system.spawn(lm[24][0], lm[24][1], count=40, color_fn=_star_color, speed_scale=5.0)
        _prev_y = curr_hip_y

        # --- 5. ACTION FX: Punching & Spinning ---
        # Punching (Lightning)
        for wrist_idx in [15, 16]:
            if wrist_idx in lm and wrist_idx in _prev_lms:
                if vis.get(wrist_idx, 0) > _VIS_THRESH:
                    dv = np.linalg.norm(np.array(lm[wrist_idx]) - np.array(_prev_lms[wrist_idx]))
                    if dv > _PUNCH_SPEED_THRESH:
                        for _ in range(2):
                            angle = random.uniform(0, 2*np.pi)
                            dist = random.uniform(40, 120)
                            target = (int(lm[wrist_idx][0] + np.cos(angle)*dist), int(lm[wrist_idx][1] + np.sin(angle)*dist))
                            _lightning_arc(canvas, lm[wrist_idx], target, (255, 200, 50))

        # Spinning (Sparkles)
        if 11 in lm and 12 in lm and vis.get(11,0) > 0.5 and vis.get(12,0) > 0.5:
            shoulder_width = abs(lm[12][0] - lm[11][0])
            if shoulder_width < (w * 0.06): # Shoulders very close = spinning profile
                cx = (lm[11][0] + lm[12][0]) // 2
                cy = (lm[11][1] + lm[12][1]) // 2
                _sparks.spawn(cx, cy, count=8, color_fn=_sparkle_color, size_range=(2, 4), speed_scale=2.0)

        # Update persistent landmark state
        _prev_lms = {idx: pt for idx, pt in lm.items()}

    # Draw all systems
    _star_system.draw(canvas)
    _gravity_system.draw(canvas)
    _sparks.draw(canvas)
    
    # Final stick figure render
    _renderer.render(pose, (h, w), canvas)
    
    return canvas
