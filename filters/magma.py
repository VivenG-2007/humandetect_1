"""
Magma Flow Filter — the body is composed of flowing lava and glowing cracks.
Embers rise from moving limbs. Stick figure is hidden.
"""
import cv2
import numpy as np
import random
import time
from pose_detector import PoseResult, CONNECTIONS

class Ember:
    def __init__(self, x, y):
        self.x = float(x)
        self.y = float(y)
        self.size = random.uniform(2, 4)
        self.vx = random.uniform(-1, 1)
        self.vy = random.uniform(-2, -4)
        self.lifetime = random.uniform(0.5, 1.0)
        self.start_time = time.time()
        # Bright orange to red
        self.color = (random.randint(0, 50), random.randint(100, 200), 255) # BGR: orange-ish

    def update(self):
        elapsed = time.time() - self.start_time
        self.x += self.vx
        self.y += self.vy
        self.vx *= 0.98
        return elapsed < self.lifetime

_embers = []
_prev_landmarks = {}
VIS_THRESHOLD = 0.25
_magma_base = None

def apply(canvas: np.ndarray, pose: PoseResult, **kwargs) -> np.ndarray:
    global _embers, _prev_landmarks, _magma_base
    h, w = canvas.shape[:2]
    t = time.time()
    
    # 1. Update Embers
    _embers = [e for e in _embers if e.update()]
    
    if pose.detected:
        lm = pose.landmarks
        vis = pose.visibility
        
        # 2. Create Body Mask for Magma
        mask = np.zeros((h, w), dtype=np.uint8)
        for (a, b) in CONNECTIONS:
            if a not in lm or b not in lm: continue
            if vis.get(a, 0) < VIS_THRESHOLD or vis.get(b, 0) < VIS_THRESHOLD: continue
            cv2.line(mask, lm[a], lm[b], 255, 35, cv2.LINE_AA)
        
        # 3. Create Magma Texture Cache dynamically
        if _magma_base is None or _magma_base.shape[:2] != (h, w):
            _magma_base = np.full((h, w, 3), (20, 40, 100), dtype=np.uint8) # Deep dark red/orange
            
        # Extract a fresh copy from cache immediately
        magma_tex = _magma_base.copy()
        
        # Highlight "cracks"
        for i in range(5):
            # Oscillating lines/blobs
            offset = int(t * 50 + i * 100) % w
            cv2.line(magma_tex, (offset, 0), (offset - 200, h), (0, 150, 255), 2, cv2.LINE_AA)
            cv2.line(magma_tex, (w - offset, 0), (w - offset + 200, h), (0, 100, 200), 1, cv2.LINE_AA)

        # Optimized Blur
        mask_small = cv2.resize(mask, (w//4, h//4))
        mask_small = cv2.GaussianBlur(mask_small, (9, 9), 0)
        float_mask = cv2.resize(mask_small, (w, h)).astype(float) / 255.0
        float_mask = np.repeat(float_mask[:, :, np.newaxis], 3, axis=2)
        
        # Add to canvas
        magma_final = (magma_tex * float_mask).astype(np.uint8)
        canvas[:] = cv2.add(canvas, magma_final)
        
        # 4. Spawning Embers on Movement
        effector_indices = [15, 16, 27, 28, 0] # Hands, Feet, Head
        for idx in effector_indices:
            if idx in lm and vis.get(idx, 0) > VIS_THRESHOLD:
                pt = lm[idx]
                if idx in _prev_landmarks:
                    prev_pt = _prev_landmarks[idx]
                    dist = np.linalg.norm(np.array(pt) - np.array(prev_pt))
                    if dist > 6.0:
                        for _ in range(int(dist/5)):
                            _embers.append(Ember(pt[0] + random.uniform(-10, 10), pt[1]))
                            
        _prev_landmarks = {idx: pt for idx, pt in lm.items() if vis.get(idx, 0) > VIS_THRESHOLD}

    # 5. Draw Embers
    for e in _embers:
        cv2.circle(canvas, (int(e.x), int(e.y)), int(e.size), e.color, -1)
        # Glow
        cv2.circle(canvas, (int(e.x), int(e.y)), int(e.size*2), (e.color[0], e.color[1], e.color[2]), 1)

    return canvas
