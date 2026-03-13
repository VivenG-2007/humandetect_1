"""
Bubble Filter — floating bubbles spawn from moving body parts.
Stick figure is hidden.
"""
import cv2
import numpy as np
import random
import time
from pose_detector import PoseResult, CONNECTIONS
from skeleton_renderer import SkeletonRenderer

_renderer = SkeletonRenderer()

class Bubble:
    def __init__(self, x, y, size, color, speed_y, speed_x):
        self.x = float(x)
        self.y = float(y)
        self.size = size
        self.max_size = size + random.randint(2, 5)
        self.color = color
        self.speed_y = speed_y
        self.speed_x = speed_x
        self.wobble_speed = random.uniform(0.05, 0.1)
        self.wobble_amplitude = random.uniform(1, 3)
        self.start_time = time.time()
        self.lifetime = random.uniform(0.5, 1.2) # Keeping the fast disappear request
        self.alpha = 1.0

    def update(self):
        elapsed = time.time() - self.start_time
        self.y -= self.speed_y
        self.x += self.speed_x + np.sin(elapsed * self.wobble_speed * 10) * self.wobble_amplitude * 0.1
        
        if elapsed > self.lifetime * 0.7:
            self.alpha = max(0, 1.0 - (elapsed - self.lifetime * 0.7) / (self.lifetime * 0.3))
            
        return elapsed < self.lifetime

    def draw(self, canvas):
        h, w = canvas.shape[:2]
        if 0 <= self.x < w and 0 <= self.y < h:
            color = tuple(int(c * self.alpha) for c in self.color)
            cv2.circle(canvas, (int(self.x), int(self.y)), int(self.size), color, 1, cv2.LINE_AA)
            highlight_pos = (int(self.x - self.size * 0.3), int(self.y - self.size * 0.3))
            cv2.circle(canvas, highlight_pos, max(1, int(self.size * 0.2)), (255, 255, 255), -1, cv2.LINE_AA)

_bubbles = []
_prev_landmarks = {}
_MOVE_THRESHOLD = 5.0
VIS_THRESHOLD = 0.4

def apply(canvas: np.ndarray, pose: PoseResult, **kwargs) -> np.ndarray:
    global _bubbles, _prev_landmarks
    
    # Update and filter bubbles
    _bubbles = [b for b in _bubbles if b.update()]
    
    if pose.detected:
        lm = pose.landmarks
        vis = pose.visibility
        
        # Specific indices for: Hands (15, 16), Hips (23, 24), Feet (27, 28, 29, 30, 31, 32)
        EFFECTOR_INDICES = [15, 16, 23, 24, 27, 28, 29, 30, 31, 32]
        
        # Check movement for spawning bubbles at joint positions
        for idx in EFFECTOR_INDICES:
            if idx not in lm or vis.get(idx, 0) < VIS_THRESHOLD:
                continue
                
            if idx in _prev_landmarks:
                pt = lm[idx]
                prev_pt = _prev_landmarks[idx]
                dist = np.sqrt((pt[0] - prev_pt[0])**2 + (pt[1] - prev_pt[1])**2)
                
                if dist > _MOVE_THRESHOLD:
                    # Spawn bubbles based on movement speed
                    num_to_spawn = int(dist / 10) + 1
                    for _ in range(num_to_spawn):
                        if random.random() < 0.3:
                            t = random.random()
                            spawn_x = pt[0] * t + prev_pt[0] * (1 - t)
                            spawn_y = pt[1] * t + prev_pt[1] * (1 - t)
                            
                            size = random.randint(3, 8)
                            color = (255, 200, 150) if random.random() > 0.5 else (255, 255, 200)
                            
                            _bubbles.append(Bubble(
                                x=spawn_x + random.uniform(-5, 5),
                                y=spawn_y + random.uniform(-5, 5),
                                size=size,
                                color=color,
                                speed_y=random.uniform(1.0, 3.0),
                                speed_x=random.uniform(-0.5, 0.5)
                            ))
                            
        # Update previous landmarks
        _prev_landmarks = {idx: pt for idx, pt in lm.items() if vis.get(idx, 0) > VIS_THRESHOLD}

    # Draw all bubbles
    for b in _bubbles:
        b.draw(canvas)
        
    # Draw Stick Figure
    if pose.detected:
        _renderer.render(pose, (canvas.shape[0], canvas.shape[1]), canvas)

    return canvas
