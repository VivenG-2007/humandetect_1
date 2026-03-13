"""
Aura Filter — Shimmering energy field.
Optimized with low-res blur proxy and vectorized particle updates.
"""
import cv2
import numpy as np
import time
import random

class SmokeParticle:
    def __init__(self, x, y, color, has_core=False):
        self.x, self.y = float(x), float(y)
        self.size = random.uniform(10, 22)
        self.color = color 
        self.vx, self.vy = random.uniform(-0.6, 0.6), random.uniform(-1.2, 0.3)
        self.alpha = random.uniform(0.5, 0.8)
        self.lifetime = random.uniform(1.0, 2.0)
        self.start_time = time.time()
        self.has_core = has_core

    def update(self, t):
        elapsed = t - self.start_time
        self.x += self.vx
        self.y += self.vy
        self.size += 0.4 
        self.alpha *= 0.95
        return elapsed < self.lifetime

_particles = []
_prev_landmarks = {}
_MOVE_THRESHOLD = 6.0
VIS_THRESHOLD = 0.4
_prev_mask = None

def apply(canvas: np.ndarray, pose, **kwargs) -> np.ndarray:
    global _particles, _prev_landmarks, _prev_mask
    h, w = canvas.shape[:2]
    t = time.time()

    # == 1. Hand Smoke Trailing ==
    _particles = [p for p in _particles if p.update(t)]
    
    if pose.detected:
        lm, vis = pose.landmarks, pose.visibility
        # Track all fingers for high-density trails
        HAND_INDICES = list(range(15, 23)) + list(range(33, 39)) + [40, 41, 42, 43]
        
        for idx in HAND_INDICES:
            if idx in lm and vis.get(idx, 0) > VIS_THRESHOLD:
                pt = lm[idx]
                if idx in _prev_landmarks:
                    dist = np.hypot(pt[0] - _prev_landmarks[idx][0], pt[1] - _prev_landmarks[idx][1])
                    if dist > _MOVE_THRESHOLD:
                        for _ in range(int(dist / 8) + 1):
                            t_lerp = random.random()
                            sx = pt[0] * t_lerp + _prev_landmarks[idx][0] * (1 - t_lerp)
                            sy = pt[1] * t_lerp + _prev_landmarks[idx][1] * (1 - t_lerp)
                            _particles.append(SmokeParticle(sx, sy, (255, 180, 50), random.random() < 0.4))
        
        _prev_landmarks = {i: lm[i] for i in HAND_INDICES if i in lm and vis.get(i,0) > VIS_THRESHOLD}

    # == 2. Body Aura (Segmentation) ==
    if pose.detected and pose.segmentation_mask is not None:
        mask = pose.segmentation_mask.astype(np.float32)
        if _prev_mask is None or _prev_mask.shape != mask.shape:
            _prev_mask = mask
        else:
            _prev_mask = cv2.addWeighted(mask, 0.4, _prev_mask, 0.6, 0)
            mask = _prev_mask

        pulse = (np.sin(t * 2) + 1) / 2
        c1, c2 = np.array([255, 180, 50]), np.array([255, 50, 200])
        current_color = (c1 * pulse + c2 * (1 - pulse)).astype(np.uint8)
        
        # Optimized Dual-Glow Blur
        mask_s = cv2.resize(mask, (w // 4, h // 4))
        ig_s = cv2.GaussianBlur(mask_s, (5, 5), 0)
        og_s = cv2.GaussianBlur(mask_s, (15, 15), 0)
        
        # Add outer aura
        oa = (cv2.resize(og_s, (w, h)) * 0.4)[:, :, np.newaxis]
        cv2.add(canvas, (oa * current_color).astype(np.uint8), dst=canvas)
        
        # Add inner core
        ia = (cv2.resize(ig_s, (w, h)) * 0.7)[:, :, np.newaxis]
        cv2.add(canvas, (ia * np.array([255, 255, 100])).astype(np.uint8), dst=canvas)

    # == 3. Draw Smoke Particles (Layered) ==
    if _particles:
        s_layer = np.zeros_like(canvas)
        c_layer = np.zeros_like(canvas)
        for p in _particles:
            cv2.circle(s_layer, (int(p.x), int(p.y)), int(p.size), p.color, -1)
            if p.has_core:
                cv2.circle(c_layer, (int(p.x), int(p.y)), int(p.size * 0.4), (200, 50, 255), -1)
        
        # Fast blur for smoke
        for layer, strength, ksize in [(s_layer, 0.6, 9), (c_layer, 0.4, 7)]:
            ls = cv2.resize(layer, (w//4, h//4))
            lb = cv2.GaussianBlur(ls, (ksize, ksize), 0)
            cv2.addWeighted(canvas, 1.0, cv2.resize(lb, (w, h)), strength, 0, dst=canvas)

    return canvas
