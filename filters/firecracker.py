"""
Firecracker Filter — sparkling particles blazing thickly across the continuous boundary outline.
"""
import cv2
import numpy as np
import random
from pose_detector import PoseResult, CONNECTIONS
from utils.particle_system import ParticleSystem

# Dramatically increased particle cap to support a perfectly dense outline mapping
_system = ParticleSystem(max_particles=4500)

def _fire_color():
    return random.choice([
        (0, random.randint(100, 200), 255),   # orange
        (0, 200, 255),                          # yellow-orange
        (0, 50, 255),                           # red
        (50, 220, 255),                         # bright yellow
    ])

def apply(canvas: np.ndarray, pose: PoseResult, **kwargs) -> np.ndarray:
    _system.update()

    if pose.detected:
        h, w = canvas.shape[:2]
        spawned = False
        
        # 1. Primary Strategy: Spawn heavily on true visual outer boundary (contours)
        if getattr(pose, 'segmentation_mask', None) is not None:
            mask_float = pose.segmentation_mask
            if mask_float.shape[:2] != (h, w):
                mask_float = cv2.resize(mask_float, (w, h), interpolation=cv2.INTER_LINEAR)
            
            mask_bin = (mask_float > 0.4).astype(np.uint8) * 255
            contours, _ = cv2.findContours(mask_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            
            edge_points = []
            if contours:
                edge_points = np.vstack(contours).squeeze(1)
                    
            if len(edge_points) > 50:
                # High Density Spawning: Sample a massive chunk of the boundary every frame
                num_spawns = min(400, len(edge_points) // 10) # Increased density for optimized engine
                sampled = edge_points[np.random.choice(len(edge_points), size=num_spawns, replace=False)]
                
                # Apply spatial jitter vectorized (+- 6 pixels wide)
                jittered_pts = sampled + np.random.randint(-6, 7, sampled.shape)
                
                _system.spawn_batch(
                    jittered_pts,
                    count_per_point=1,
                    color_fn=_fire_color,
                    size_range=(1, 3),
                    lifetime_range=(12, 25),
                    speed_scale=0.8
                )
                
                # Occasional slight cluster at an edge string to make it pop organically
                if random.random() < 0.20:
                    pt = random.choice(edge_points)
                    _system.spawn(
                        pt[0], pt[1],
                        count=5, # Burst size
                        color_fn=_fire_color,
                        size_range=(2, 5),
                        lifetime_range=(20, 35),
                        speed_scale=2.0,
                    )
                spawned = True
                
        # 2. Fallback Strategy: Center skeletal interpolation (massive thick spawning)
        if not spawned:
            lm = pose.landmarks
            vis = pose.visibility
            valid_lines = []
            for a, b in CONNECTIONS:
                if a in lm and b in lm and vis.get(a, 0) > 0.3 and vis.get(b, 0) > 0.3:
                    valid_lines.append((lm[a], lm[b]))
                    
            if valid_lines:
                for _ in range(35): # High intensity fallback
                    pt1, pt2 = random.choice(valid_lines)
                    t = random.random()
                    x = int(pt1[0] + (pt2[0] - pt1[0]) * t) + random.randint(-10, 10)
                    y = int(pt1[1] + (pt2[1] - pt1[1]) * t) + random.randint(-10, 10)
                    
                    _system.spawn(
                        x, y,
                        count=1,
                        color_fn=_fire_color,
                        size_range=(1, 3),
                        lifetime_range=(15, 30),
                        speed_scale=0.7,
                    )

    # Blur the raw camera so the brilliant sparking thick outline pops visually
    if kwargs.get('original_frame') is not None:
        pass # Optional: Could dim the original frame
        
    _system.draw(canvas)
    return canvas
