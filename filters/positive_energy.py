"""
Positive Energy Filter
- Emits a glowing aura/blast of positive energy from the person's outline 
  when they jump or turn around.
"""
import cv2
import numpy as np
import random
import time
from pose_detector import PoseResult, CONNECTIONS

# --- State ---
_prev_y = 0
_last_jump_time = 0.0
_prev_facing = None
_last_turn_time = 0.0

# --- Positive Energy Particles ---
_PARTICLES = np.zeros((0, 2), dtype=np.float32)
_VELOCITIES = np.zeros((0, 2), dtype=np.float32)
_LIVES = np.zeros(0, dtype=np.float32)
_MAX_LIVES = np.zeros(0, dtype=np.float32)
_COLORS = np.zeros((0, 3), dtype=np.float32)
_SIZES = np.zeros(0, dtype=np.int32)

def _is_facing_camera(lm, vis):
    # Make facing detection very forgiving 
    return (vis.get(0, 0) > 0.30 and
            vis.get(11, 0) > 0.30 and
            vis.get(12, 0) > 0.30)

def _trigger_positive_burst(lm, vis, mask_img, h, w):
    global _PARTICLES, _VELOCITIES, _LIVES, _MAX_LIVES, _COLORS, _SIZES
    
    count = 200 # number of particles in a burst
    
    if len(_PARTICLES) > 1000:
        return # Prevent overload
        
    pts = []
    # Try to spawn from the segmentation mask outline
    if mask_img is not None:
        # mask is usually float32 [0.0, 1.0] with same shape as (h, w)
        mask_uint8 = (mask_img * 255).astype(np.uint8)
        mask_uint8 = cv2.resize(mask_uint8, (w, h))
        contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) > 0:
            # Flatten contour points
            cnt_pts = np.vstack(contours).squeeze(1)
            if len(cnt_pts) > count:
                # Randomly sample from outline
                indices = np.random.choice(len(cnt_pts), count, replace=False)
                for idx in indices:
                    pts.append(cnt_pts[idx])
            else:
                pts = cnt_pts.tolist()
                count = len(pts)
                
    # Fallback to joints / connections if mask fails
    if len(pts) < 10:
        for _ in range(count):
            # random connection
            a, b = random.choice(CONNECTIONS)
            if a in lm and b in lm and vis.get(a, 0) > 0.2 and vis.get(b, 0) > 0.2:
                pt1 = np.array(lm[a])
                pt2 = np.array(lm[b])
                t = random.random()
                p = pt1 * t + pt2 * (1 - t)
                pts.append(p.tolist())
                
    if len(pts) == 0:
        return
        
    new_pos = np.array(pts, dtype=np.float32)
    count = len(new_pos)
    
    # Calculate outward velocity from center of mass
    cx = np.mean(new_pos[:, 0])
    cy = np.mean(new_pos[:, 1])
    
    dx = new_pos[:, 0] - cx
    dy = new_pos[:, 1] - cy
    dist = np.sqrt(dx**2 + dy**2) + 1.0
    
    # Push outward from center
    speed = np.random.uniform(5.0, 12.0, count)
    vx = (dx / dist) * speed
    vy = (dy / dist) * speed
    vy -= 2.0 # Slight upward lift for positive energy
    
    new_vel = np.column_stack((vx, vy)).astype(np.float32)
    
    new_lives = np.random.uniform(0.8, 2.0, count).astype(np.float32)
    new_max_lives = new_lives.copy()
    
    new_sizes = np.random.randint(3, 9, count)
    
    new_colors = np.zeros((count, 3), dtype=np.float32)
    for i in range(count):
        # Golden, yellow, soft white
        if random.random() > 0.3:
            new_colors[i] = (random.randint(150, 255), random.randint(220, 255), 255) # Yellowish white BGR
        else:
            new_colors[i] = (255, 255, 255) # Pure white
            
    _PARTICLES = np.vstack([_PARTICLES, new_pos])
    _VELOCITIES = np.vstack([_VELOCITIES, new_vel])
    _LIVES = np.concatenate([_LIVES, new_lives])
    _MAX_LIVES = np.concatenate([_MAX_LIVES, new_max_lives])
    _SIZES = np.concatenate([_SIZES, new_sizes])
    _COLORS = np.vstack([_COLORS, new_colors])
    
def apply(canvas: np.ndarray, pose: PoseResult, **kwargs) -> np.ndarray:
    global _prev_y, _last_jump_time, _prev_facing, _last_turn_time
    global _PARTICLES, _VELOCITIES, _LIVES, _MAX_LIVES, _COLORS, _SIZES
    
    h, w = canvas.shape[:2]
    t = time.time()
    
    if pose.detected:
        lm = pose.landmarks
        vis = pose.visibility
        mask = pose.segmentation_mask
        
        # Determine Reference Scale
        scale_ref = 100.0
        if 11 in lm and 12 in lm and (vis.get(11,0) > 0.3 or vis.get(12,0) > 0.3):
            scale_ref = max(40.0, float(np.hypot(lm[11][0] - lm[12][0], lm[11][1] - lm[12][1])))
            
        jump_thresh = max(10.0, scale_ref * 0.15) # Much more forgiving jump threshold
        
        # 1. Detect Jump
        if 23 in lm and 24 in lm:
            curr_hip_y = (lm[23][1] + lm[24][1]) // 2
            
            # Use smaller cooldown to catch quick repeated jumps
            if _prev_y != 0 and (_prev_y - curr_hip_y) > jump_thresh and (t - _last_jump_time) > 0.6:
                _last_jump_time = t
                _trigger_positive_burst(lm, vis, mask, h, w)
                
            # Keep history sticky for a few frames to handle fast blurring
            _prev_y = _prev_y * 0.5 + curr_hip_y * 0.5 if _prev_y != 0 else curr_hip_y
            
        # 2. Detect Turn-Around
        facing_now = _is_facing_camera(lm, vis)
        if _prev_facing is not None and facing_now != _prev_facing:
            # Drop cooldown so spinning works
            if (t - _last_turn_time) > 0.8: 
                _last_turn_time = t
                _trigger_positive_burst(lm, vis, mask, h, w)
        _prev_facing = facing_now

    # Update & Render Particles
    if len(_PARTICLES) > 0:
        _LIVES -= 0.05
        alive = _LIVES > 0
        
        _PARTICLES = _PARTICLES[alive]
        _VELOCITIES = _VELOCITIES[alive]
        _LIVES = _LIVES[alive]
        _MAX_LIVES = _MAX_LIVES[alive]
        _COLORS = _COLORS[alive]
        _SIZES = _SIZES[alive]
        
    if len(_PARTICLES) > 0:
        _VELOCITIES *= 0.94 # Slow down smoothly
        _PARTICLES += _VELOCITIES
        
        # Rising effect (anti-gravity for positive energy)
        _PARTICLES[:, 1] -= 0.5 
        
        overlay = np.zeros_like(canvas)
        
        # Optimize rendering arrays
        px = _PARTICLES[:, 0].astype(np.int32)
        py = _PARTICLES[:, 1].astype(np.int32)
        
        for i in range(len(_PARTICLES)):
            if 0 <= px[i] < w and 0 <= py[i] < h:
                ratio = _LIVES[i] / _MAX_LIVES[i]
                alpha = ratio if ratio < 0.3 else (1.0 - ratio) / 0.7 # Fade in and out
                alpha = np.clip(alpha * 1.5, 0, 1)
                
                c = tuple(int(ch * alpha) for ch in _COLORS[i])
                sz = max(1, int(_SIZES[i] * (0.5 + 0.5 * ratio)))
                
                cv2.circle(overlay, (px[i], py[i]), sz, c, -1, cv2.LINE_AA)
                cv2.circle(overlay, (px[i], py[i]), sz + 2, tuple(int(ch * 0.5) for ch in c), 1, cv2.LINE_AA)
                
        # Glow blend
        cv2.addWeighted(canvas, 1.0, overlay, 1.0, 0, canvas)
        
    return canvas
