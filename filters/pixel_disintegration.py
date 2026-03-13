"""
Pixel Disintegration Filter
Parts of the body fragment into digital pixels or shards reacting to aggressive limb motion.
"""
import cv2
import numpy as np
import random
from pose_detector import PoseResult

_temporal_mask = None
_fragments = []

class DisintegratedPixel:
    def __init__(self, x, y, color, speed_x, speed_y):
        self.x = float(x)
        self.y = float(y)
        self.color = color
        self.vx = speed_x
        self.vy = speed_y
        self.life = 255 # Opacity

def apply(canvas: np.ndarray, pose: PoseResult, **kwargs) -> np.ndarray:
    global _temporal_mask, _fragments
    
    h, w = canvas.shape[:2]
    original = kwargs.get('original_frame')
    if original is None:
        return canvas

    canvas[:] = original

    if pose.detected and getattr(pose, 'segmentation_mask', None) is not None:
        mask_raw = pose.segmentation_mask
        if mask_raw.shape[:2] != (h, w):
            mask_raw = cv2.resize(mask_raw, (w, h), interpolation=cv2.INTER_LINEAR)
            
        if '_temporal_mask' not in globals() or _temporal_mask is None or _temporal_mask.shape != mask_raw.shape:
            _temporal_mask = mask_raw.copy()
            motion_diff = np.zeros_like(mask_raw)
        else:
            motion_diff = np.abs(mask_raw - _temporal_mask)
            cv2.addWeighted(_temporal_mask, 0.7, mask_raw, 0.3, 0, dst=_temporal_mask)
            
        smooth_float = cv2.GaussianBlur(_temporal_mask, (11, 11), 0)
        
        # 1. Identify highly dynamic silhouette edges (motion delta matrix)
        motion_thresh = (motion_diff > 0.25).astype(np.uint8) * 255
        
        # 2. Rip visual data directly from the camera at moving edges
        points = np.column_stack(np.where(motion_thresh > 0)) # returns [y, x]
        
        sample_rate = 60 # 1 fragment extracted per N pixels of motion
        if len(points) > 0:
            sampled_indices = np.random.choice(len(points), size=max(1, len(points)//sample_rate), replace=False)
            for idx in sampled_indices:
                py, px = points[idx]
                if 0 <= py < h and 0 <= px < w:
                    color = [int(x) for x in original[py, px]]
                    vx = random.uniform(-15, 15)
                    vy = random.uniform(-25, 5) # Drift slightly upwards
                    _fragments.append(DisintegratedPixel(px, py, color, vx, vy))

        # Blackout the physical body edge space where shards were ripped from
        canvas[motion_thresh > 0] = (20, 20, 30)

        # 3. Simulate and render all fragmented block units
        new_fragments = []
        for frag in _fragments:
            frag.x += frag.vx
            frag.y += frag.vy
            frag.life -= 12 # Lifespan fade
            if frag.life > 0:
                # Size inversely correlates to velocity and life
                sz = int(14 * (frag.life / 255))
                pt1 = (int(frag.x - sz), int(frag.y - sz))
                pt2 = (int(frag.x + sz), int(frag.y + sz))
                
                final_col = frag.color
                # 15% chance to glitch into a purely digital color
                if random.random() < 0.15:
                    final_col = (255, 0, 255) if random.random() < 0.5 else (0, 255, 255)
                    
                cv2.rectangle(canvas, pt1, pt2, final_col, -1, cv2.LINE_4)
                new_fragments.append(frag)
                
        # Limit shard cap
        _fragments = new_fragments[-800:]
            
    return canvas
