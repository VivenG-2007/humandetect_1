"""
Aura Filter — Shimmering energy field that wraps around the entire human body,
with neon smoke trailing from the hands on movement.
"""
import cv2
import numpy as np
import time
import random

class SmokeParticle:
    def __init__(self, x, y, color, has_core=False):
        self.x = float(x)
        self.y = float(y)
        self.size = random.uniform(10, 22)
        self.color = color 
        self.vx = random.uniform(-0.6, 0.6)
        self.vy = random.uniform(-1.2, 0.3)
        self.alpha = random.uniform(0.5, 0.8)
        self.lifetime = random.uniform(1.0, 2.0)
        self.start_time = time.time()
        self.has_core = has_core

    def update(self):
        elapsed = time.time() - self.start_time
        self.x += self.vx
        self.y += self.vy
        self.size += 0.4 # Smoke spreads
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
    _particles = [p for p in _particles if p.update()]
    
    if pose.detected:
        lm = pose.landmarks
        vis = pose.visibility
        
        blue_smoke = (255, 180, 50) # BGR Cyan/Blue
        # Just track wrists/hands
        HAND_INDICES = [15, 16, 17, 18, 19, 20, 21, 22]
        
        for idx in HAND_INDICES:
            if idx not in lm or vis.get(idx, 0) < VIS_THRESHOLD:
                continue
            pt = lm[idx]
            if idx in _prev_landmarks:
                prev_pt = _prev_landmarks[idx]
                dist = np.linalg.norm(np.array(pt) - np.array(prev_pt))
                
                if dist > _MOVE_THRESHOLD:
                    num = int(dist / 6) + 1
                    for _ in range(num):
                        t_lerp = random.random()
                        spawn_x = pt[0] * t_lerp + prev_pt[0] * (1 - t_lerp)
                        spawn_y = pt[1] * t_lerp + prev_pt[1] * (1 - t_lerp)
                        _particles.append(SmokeParticle(
                            spawn_x + random.uniform(-6, 6),
                            spawn_y + random.uniform(-6, 6),
                            blue_smoke,
                            has_core=random.random() < 0.4
                        ))
        
        _prev_landmarks = {idx: pt for idx, pt in lm.items() if vis.get(idx, 0) > VIS_THRESHOLD}

    # == 2. Body Aura (Segmentation Mask) ==
    if pose.detected and pose.segmentation_mask is not None:
        mask = pose.segmentation_mask.astype(np.float32)
        
        # Temporal mask smoothing to eliminate glitches
        if _prev_mask is None or _prev_mask.shape != mask.shape:
            _prev_mask = mask.copy()
        else:
            mask = cv2.addWeighted(mask, 0.4, _prev_mask, 0.6, 0)
            _prev_mask = mask.copy()
        
        pulse = (np.sin(t * 2) + 1) / 2
        color1 = np.array([255, 180, 50]) # Sky Blue
        color2 = np.array([255, 50, 200]) # Vivid Purple
        current_color = (color1 * pulse + color2 * (1 - pulse)).astype(np.uint8)
        
        mask_small = cv2.resize(mask, (w // 4, h // 4))
        inner_glow_small = cv2.GaussianBlur(mask_small, (5, 5), 0)
        outer_glow_small = cv2.GaussianBlur(mask_small, (15, 15), 0)
        
        inner_glow = cv2.resize(inner_glow_small, (w, h))
        outer_glow = cv2.resize(outer_glow_small, (w, h))

        outer_alpha = outer_glow * 0.4
        for c in range(3):
            canvas[:, :, c] = np.clip(canvas[:, :, c] + outer_alpha * current_color[c], 0, 255)
        
        inner_alpha = inner_glow * 0.7
        core_color = np.array([255, 255, 100]) # Cyan core
        for c in range(3):
            canvas[:, :, c] = np.clip(canvas[:, :, c] + inner_alpha * core_color[c], 0, 255)

        if int(t * 15) % 2 == 0:
            grid = 15
            for y in range(0, h, grid):
                for x in range(0, w, grid):
                    if mask[y, x] > 0.6 and random.random() < 0.05:
                        spark_h = random.randint(10, 30)
                        cv2.line(canvas, (x, y), (x, y - spark_h), (255, 255, 255), 1, cv2.LINE_AA)
                        cv2.circle(canvas, (x, y - spark_h), 2, (255, 255, 255), -1, cv2.LINE_AA)

    # == 3. Draw Smoke Particles on top ==
    if _particles:
        smoke_layer = np.zeros_like(canvas)
        core_layer = np.zeros_like(canvas)
        
        for p in _particles:
            cv2.circle(smoke_layer, (int(p.x), int(p.y)), int(p.size), p.color, -1)
            if p.has_core:
                cv2.circle(core_layer, (int(p.x), int(p.y)), int(p.size * 0.4), (200, 50, 255), -1)
        
        smoke_small = cv2.resize(smoke_layer, (w//4, h//4))
        smoke_small = cv2.GaussianBlur(smoke_small, (9, 9), 0)
        smoke_layer = cv2.resize(smoke_small, (w, h))

        core_small = cv2.resize(core_layer, (w//4, h//4))
        core_small = cv2.GaussianBlur(core_small, (7, 7), 0)
        core_layer = cv2.resize(core_small, (w, h))
        
        canvas[:] = cv2.addWeighted(canvas, 1.0, smoke_layer, 0.6, 0)
        canvas[:] = cv2.addWeighted(canvas, 1.0, core_layer, 0.4, 0)
        
        for p in _particles[:10]:
            if p.alpha > 0.6:
                cv2.circle(canvas, (int(p.x), int(p.y)), int(p.size/6), (255, 255, 255), -1)

    return canvas
