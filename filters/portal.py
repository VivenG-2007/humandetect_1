"""
Portal Filter — Doctor Strange style circular spark portal around the body.
"""
import cv2
import numpy as np
import random
import time
from pose_detector import PoseResult, CONNECTIONS
from utils.particle_system import ParticleSystem

_spark_system = ParticleSystem(max_particles=800)

def _spark_color():
    # Bright orange/gold sparks
    return (random.randint(0, 100), random.randint(150, 255), 255) # BGR: Golden Orange

def apply(canvas: np.ndarray, pose: PoseResult, **kwargs) -> np.ndarray:
    h, w = canvas.shape[:2]
    t = time.time()
    
    _spark_system.update()

    if pose.detected:
        lm = pose.landmarks
        # Center of the portal is the center of the torso
        if 11 in lm and 12 in lm and 23 in lm and 24 in lm:
            cx = (lm[11][0] + lm[12][0] + lm[23][0] + lm[24][0]) // 4
            cy = (lm[11][1] + lm[12][1] + lm[23][1] + lm[24][1]) // 4
            
            radius = 180 # Large circular portal
            
            # Spawn sparks in a circular pattern with rotation
            for _ in range(12):
                angle = random.uniform(0, 2 * np.pi)
                # Jitter the radius to create "thickness"
                r = radius + random.uniform(-15, 15)
                px = int(cx + np.cos(angle) * r)
                py = int(cy + np.sin(angle) * r)
                
                _spark_system.spawn(px, py, count=1, color_fn=_spark_color, size_range=(1, 3), lifetime_range=(10, 30), speed_scale=2.5)
                # Give tangential velocity for swirl
                perp_v = np.array([-np.sin(angle), np.cos(angle)]) * random.uniform(4, 8)
                _spark_system.particles[-1].velocity = (perp_v[0], perp_v[1])

            # Draw a faint glowing ring
            glow_mask = np.zeros_like(canvas)
            cv2.circle(glow_mask, (cx, cy), radius, (0, 100, 255), 4, cv2.LINE_AA)
            
            # Optimized blur: Downsample -> Blur -> Upsample
            glow_small = cv2.resize(glow_mask, (w // 4, h // 4))
            glow_small = cv2.GaussianBlur(glow_small, (9, 9), 0)
            glow_mask = cv2.resize(glow_small, (w, h))
            
            cv2.addWeighted(canvas, 1.0, glow_mask, 0.6, 0, dst=canvas)

        # Extract the true human outline using the segmentation mask
        if getattr(pose, 'segmentation_mask', None) is not None:
            mask_float = pose.segmentation_mask
            if mask_float.shape[:2] != (h, w):
                mask_float = cv2.resize(mask_float, (w, h), interpolation=cv2.INTER_LINEAR)
            # Binary threshold and smooth to keep the contour elegant
            mask = (mask_float > 0.5).astype(np.uint8) * 255
            mask = cv2.GaussianBlur(mask, (5, 5), 0)
            _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
        else:
            # Fallback skeleton blob if segmentation isn't supported by the model
            vis = pose.visibility
            mask = np.zeros((h, w), dtype=np.uint8)
            for (a, b) in CONNECTIONS:
                if a in lm and b in lm and vis.get(a, 0) > 0.25 and vis.get(b, 0) > 0.25:
                    cv2.line(mask, lm[a], lm[b], 255, 36, cv2.LINE_AA)
                    cv2.circle(mask, lm[a], 18, 255, -1, cv2.LINE_AA)
                    cv2.circle(mask, lm[b], 18, 255, -1, cv2.LINE_AA)
            if 0 in lm and vis.get(0, 0) > 0.25:
                top_y = int(lm[0][1] - 60)
                cv2.line(mask, lm[0], (lm[0][0], top_y), 255, 50, cv2.LINE_AA)
                cv2.circle(mask, (lm[0][0], top_y), 25, 255, -1, cv2.LINE_AA)
            mask = cv2.GaussianBlur(mask, (31, 31), 0)
            _, mask = cv2.threshold(mask, 120, 255, cv2.THRESH_BINARY)
        
        # Find the smooth continuous outer boundary
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Draw the glowing outline
        cv2.drawContours(canvas, contours, -1, (200, 240, 255), 2, cv2.LINE_AA)

    _spark_system.draw(canvas)
    
    return canvas
