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
_temporal_mask = None

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
                
                # Give tangential velocity for swirl
                perp_v = np.array([-np.sin(angle), np.cos(angle)]) * random.uniform(4, 8)
                _spark_system.spawn(px, py, count=1, color_fn=_spark_color, size_range=(1, 3), 
                                    lifetime_range=(10, 30), velocity=(perp_v[0], perp_v[1]))

            # Draw a faint glowing ring
            glow_mask = np.zeros_like(canvas)
            cv2.circle(glow_mask, (cx, cy), radius, (0, 100, 255), 4, cv2.LINE_AA)
            
            # Optimized blur: Downsample -> Blur -> Upsample
            glow_small = cv2.resize(glow_mask, (w // 4, h // 4))
            glow_small = cv2.GaussianBlur(glow_small, (9, 9), 0)
            glow_mask = cv2.resize(glow_small, (w, h))
            
            cv2.addWeighted(canvas, 1.0, glow_mask, 0.6, 0, dst=canvas)

        # Extract the true human silhouette using the segmentation mask
        if getattr(pose, 'segmentation_mask', None) is not None:
            mask_float = pose.segmentation_mask
            if mask_float.shape[:2] != (h, w):
                mask_float = cv2.resize(mask_float, (w, h), interpolation=cv2.INTER_LINEAR)
            
            # Segmentation mask from PoseResult is now temporally smoothed globally.
            small_mask = cv2.resize(mask_float, (0, 0), fx=0.25, fy=0.25, interpolation=cv2.INTER_LINEAR)
            small_mask = cv2.GaussianBlur(small_mask, (11, 11), 0)
            smooth_float = cv2.resize(small_mask, (w, h), interpolation=cv2.INTER_LINEAR)
            
            # Threshold into binary
            mask = (smooth_float > 0.4).astype(np.uint8) * 255
            
            # Force hand tracking out to newly mapped physical fingertips (All 5 fingers)
            vis = pose.visibility
            # Hand index layout: wrist, index_t, pinky_t, thumb_t, middle_t, ring_t
            for w_idx, i_t, p_t, t_t, m_t, r_t in [(15, 35, 33, 37, 40, 42), (16, 36, 34, 38, 41, 43)]:
                if w_idx in lm and vis.get(w_idx, 0) > 0.15:
                    # Draw a solid core for the palm
                    cv2.circle(mask, lm[w_idx], 15, 255, -1, cv2.LINE_AA)
                    
                    # Connect all available fingers to the wrist
                    for f_tip in [i_t, p_t, t_t, m_t, r_t]:
                        if f_tip in lm and vis.get(f_tip, 0) > 0.15:
                            # Use a moderate thickness that doesn't blob but stays connected
                            cv2.line(mask, lm[w_idx], lm[f_tip], 255, 15, cv2.LINE_AA)
                            cv2.circle(mask, lm[f_tip], 8, 255, -1, cv2.LINE_AA)
            
            # Use CLOSE (fills gaps) instead of OPEN (which deletes fingers)
            kernel = np.ones((5, 5), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(canvas, contours, -1, (200, 240, 255), 2, cv2.LINE_AA)
            
        else:
            # Fallback to metaball skeleton
            vis = pose.visibility
            mask = np.zeros((h, w), dtype=np.uint8)
            for (a, b) in CONNECTIONS:
                if a in lm and b in lm and vis.get(a, 0) > 0.25 and vis.get(b, 0) > 0.25:
                    cv2.line(mask, lm[a], lm[b], 255, 36, cv2.LINE_AA)
                    cv2.circle(mask, lm[a], 18, 255, -1, cv2.LINE_AA)
                    cv2.circle(mask, lm[b], 18, 255, -1, cv2.LINE_AA)
                    
            # Add full 5-finger hands to fallback blob
            for w_idx, i_t, p_t, t_t, m_t, r_t in [(15, 35, 33, 37, 40, 42), (16, 36, 34, 38, 41, 43)]:
                if w_idx in lm and vis.get(w_idx, 0) > 0.15:
                    cv2.circle(mask, lm[w_idx], 15, 255, -1, cv2.LINE_AA)
                    for f_tip in [i_t, p_t, t_t, m_t, r_t]:
                        if f_tip in lm and vis.get(f_tip, 0) > 0.15:
                            cv2.line(mask, lm[w_idx], lm[f_tip], 255, 15, cv2.LINE_AA)
                            cv2.circle(mask, lm[f_tip], 8, 255, -1, cv2.LINE_AA)

            if 0 in lm and vis.get(0, 0) > 0.25:
                top_y = int(lm[0][1] - 60)
                cv2.line(mask, lm[0], (lm[0][0], top_y), 255, 50, cv2.LINE_AA)
                cv2.circle(mask, (lm[0][0], top_y), 25, 255, -1, cv2.LINE_AA)
                
            mask = cv2.GaussianBlur(mask, (15, 15), 0)
            _, mask = cv2.threshold(mask, 140, 255, cv2.THRESH_BINARY)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(canvas, contours, -1, (200, 240, 255), 2, cv2.LINE_AA)

    _spark_system.draw(canvas)
    
    return canvas
