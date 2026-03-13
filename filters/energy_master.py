"""
Energy Master Filter (Energy Ball & Flames)
Gesture: Bring both hands close together
Effect:
A glowing energy sphere forms between the hands.
When hands move apart -> ball grows.
When hands close -> ball compresses.
Release hands (move wide apart quickly) -> energy blast forward.

Gesture: Palm facing forward (fingers above wrist)
Effect:
Flames appear in the palm and follow hand.
Flames appear in the palm and follow hand.
"""
import cv2
import numpy as np
import random
import time
from pose_detector import PoseResult, CONNECTIONS

# --- Energy Ball State ---
_ball_charge = 0.0
_blast_active = False
_blast_pos = [0.0, 0.0]
_blast_radius = 0.0
_blast_alpha = 0.0
_blast_vel = [0.0, 0.0]
_smoothed_mid = None
_smoothed_dist = 0.0

# --- Swipe State ---
_prev_wrists = {"left": None, "right": None}
_firewaves = [] # list of dicts: pos, vel, alpha, size
_FIRE_PARTICLES = np.zeros((0, 2), dtype=np.float32)
_FIRE_VEL = np.zeros((0, 2), dtype=np.float32)
_FIRE_LIFE = np.zeros(0, dtype=np.float32)
_FIRE_MAXLIFE = np.zeros(0, dtype=np.float32)
_FIRE_COLOR = np.zeros((0, 3), dtype=np.float32)

def _spawn_fire(x, y, count=5, vx_range=(-2, 2), vy_range=(-5, -1), color_type="orange", size_mult=1.0):
    global _FIRE_PARTICLES, _FIRE_VEL, _FIRE_LIFE, _FIRE_MAXLIFE, _FIRE_COLOR
    if count <= 0 or len(_FIRE_PARTICLES) > 800: return
    
    pos = np.full((count, 2), [x, y], dtype=np.float32) + np.random.uniform(-10*size_mult, 10*size_mult, (count, 2))
    vel = np.column_stack([
        np.random.uniform(vx_range[0], vx_range[1], count),
        np.random.uniform(vy_range[0], vy_range[1], count)
    ]).astype(np.float32)
    
    life = np.random.uniform(0.5, 1.2, count).astype(np.float32) * size_mult
    
    colors = np.zeros((count, 3), dtype=np.float32)
    for i in range(count):
        if color_type == "orange":
            colors[i] = (random.randint(0, 50), random.randint(100, 180), 255) # BGR
        elif color_type == "blue":
            colors[i] = (255, random.randint(150, 200), random.randint(0, 50))
            
    _FIRE_PARTICLES = np.vstack([_FIRE_PARTICLES, pos])
    _FIRE_VEL = np.vstack([_FIRE_VEL, vel])
    _FIRE_LIFE = np.concatenate([_FIRE_LIFE, life])
    _FIRE_MAXLIFE = np.concatenate([_FIRE_MAXLIFE, life.copy()])
    _FIRE_COLOR = np.vstack([_FIRE_COLOR, colors])

def _update_fire(canvas):
    global _FIRE_PARTICLES, _FIRE_VEL, _FIRE_LIFE, _FIRE_MAXLIFE, _FIRE_COLOR
    if len(_FIRE_PARTICLES) == 0: return
    
    _FIRE_LIFE -= 0.04
    alive = _FIRE_LIFE > 0
    _FIRE_PARTICLES = _FIRE_PARTICLES[alive]
    _FIRE_VEL = _FIRE_VEL[alive]
    _FIRE_LIFE = _FIRE_LIFE[alive]
    _FIRE_MAXLIFE = _FIRE_MAXLIFE[alive]
    _FIRE_COLOR = _FIRE_COLOR[alive]
    
    if len(_FIRE_PARTICLES) == 0: return
    
    _FIRE_PARTICLES += _FIRE_VEL
    
    overlay = np.zeros_like(canvas)
    
    # Scale fire with life
    for i in range(len(_FIRE_PARTICLES)):
        x, y = int(_FIRE_PARTICLES[i, 0]), int(_FIRE_PARTICLES[i, 1])
        if 0 <= x < canvas.shape[1] and 0 <= y < canvas.shape[0]:
            ratio = _FIRE_LIFE[i] / _FIRE_MAXLIFE[i]
            sz = max(1, int(15 * ratio))
            c = tuple(int(ch * ratio) for ch in _FIRE_COLOR[i])
            cv2.circle(overlay, (x, y), sz, c, -1, cv2.LINE_AA)
            cv2.circle(overlay, (x, y), sz // 2, (255, 255, 255), -1, cv2.LINE_AA)
            
    cv2.addWeighted(canvas, 1.0, overlay, 0.8, 0, canvas)


def apply(canvas: np.ndarray, pose: PoseResult, **kwargs) -> np.ndarray:
    global _ball_charge, _blast_active, _blast_pos, _blast_radius, _blast_alpha, _blast_vel
    global _smoothed_mid, _smoothed_dist
    global _prev_wrists, _firewaves
    
    h, w = canvas.shape[:2]
    t = time.time()
    
    _update_fire(canvas)
    
    # Update Fire waves
    active_waves = []
    for wave in _firewaves:
        wave['pos'][0] += wave['vel'][0]
        wave['pos'][1] += wave['vel'][1]
        wave['alpha'] -= 0.05
        wave['size'] += wave['growth']
        if wave['alpha'] > 0:
            active_waves.append(wave)
            _spawn_fire(wave['pos'][0], wave['pos'][1], count=6, vx_range=(-4, 4), vy_range=(-4, 4), color_type="orange", size_mult=2.0)
    _firewaves = active_waves
    
    # Blast logic (energy blast shooting forward / expanding screen wide)
    if _blast_active:
        _blast_pos[0] += _blast_vel[0]
        _blast_pos[1] += _blast_vel[1]
        _blast_radius += 25.0
        _blast_alpha -= 0.03
        if _blast_alpha <= 0:
            _blast_active = False
        else:
            overlay = canvas.copy()
            c = (255, 255, 100) # cyan-ish white
            cv2.circle(overlay, (int(_blast_pos[0]), int(_blast_pos[1])), int(_blast_radius), (255, 200, 100), -1, cv2.LINE_AA)
            cv2.circle(overlay, (int(_blast_pos[0]), int(_blast_pos[1])), int(_blast_radius*0.7), (255, 255, 200), -1, cv2.LINE_AA)
            cv2.circle(overlay, (int(_blast_pos[0]), int(_blast_pos[1])), int(_blast_radius*0.4), (255, 255, 255), -1, cv2.LINE_AA)
            cv2.addWeighted(overlay, _blast_alpha, canvas, 1.0, 0, canvas)

    if not pose.detected:
        return canvas
        
    lm = pose.landmarks
    vis = pose.visibility

    # --- Energy Ball Logic ---
    def get_hand_center(side):
        w_idx = 15 if side == "left" else 16
        i_idx = 19 if side == "left" else 20
        pts = []
        if w_idx in lm and vis.get(w_idx, 0) > 0.25:
            pts.append(np.array(lm[w_idx]))
        if i_idx in lm and vis.get(i_idx, 0) > 0.25:
            pts.append(np.array(lm[i_idx]))
        if len(pts) > 0:
            return sum(pts) / len(pts)
        return None
        
    left_center = get_hand_center("left")
    right_center = get_hand_center("right")
    
    # Independent fallbacks for each side
    if left_center is None and 15 in lm:
        left_center = np.array(lm[15])
    if right_center is None and 16 in lm:
        right_center = np.array(lm[16])
    
    if left_center is not None and right_center is not None:
        dist = np.linalg.norm(left_center - right_center)
        mid = (left_center + right_center) / 2
        
        # Smooth calculations for jitter-free tracking
        if _smoothed_mid is None:
            _smoothed_mid = mid.copy()
            _smoothed_dist = dist
        else:
            _smoothed_mid = _smoothed_mid * 0.70 + mid * 0.30
            _smoothed_dist = _smoothed_dist * 0.80 + dist * 0.20
            
        s_dist = _smoothed_dist
        s_mid = _smoothed_mid
        
        # Bring close -> start charging
        if s_dist < 150: # Increased threshold for easier detection
            _ball_charge = min(1.0, _ball_charge + 0.04) # Smoother charge rate
        # Move apart -> ball grows
        elif s_dist < 400 and _ball_charge > 0.2: # increased from 350
            # maintain charge
            pass 
        # Release!
        elif s_dist > 400 and _ball_charge > 0.5:
            # Trigger blast
            _blast_active = True
            _blast_pos = [s_mid[0], s_mid[1]]
            _blast_radius = 40.0 + _ball_charge * 50.0
            _blast_alpha = 1.0
            _blast_vel = [0, 0] 
            _ball_charge = 0.0
        else:
            _ball_charge = max(0.0, _ball_charge - 0.08) # discharge slowly
            
        if _ball_charge > 0.05:
            # Draw energy ball using smoothed variables
            ball_radius = int(20 + (s_dist * 0.4) * _ball_charge)
            bx, by = int(s_mid[0]), int(s_mid[1])
            
            pulse = np.sin(t * 15) * 8
            ov = canvas.copy()
            cv2.circle(ov, (bx, by), int(ball_radius + pulse + 25), (255, 100, 50), -1, cv2.LINE_AA)
            cv2.circle(ov, (bx, by), int(ball_radius + pulse + 10), (255, 200, 100), -1, cv2.LINE_AA)
            cv2.circle(ov, (bx, by), int(ball_radius * 0.6 + pulse), (255, 255, 255), -1, cv2.LINE_AA)
            cv2.addWeighted(ov, 0.4 + 0.3 * _ball_charge, canvas, 1.0, 0, canvas)
            
            # Electric arcs bouncing between hands and ball core
            for _ in range(4):
                angle = random.uniform(0, 2*np.pi)
                r1 = random.uniform(0, ball_radius * 0.5)
                r2 = ball_radius + random.uniform(10, 40)
                x1 = int(bx + np.cos(angle) * r1)
                y1 = int(by + np.sin(angle) * r1)
                x2 = int(bx + np.cos(angle) * r2)
                y2 = int(by + np.sin(angle) * r2)
                cv2.line(canvas, (x1, y1), (x2, y2), (255, 255, random.randint(150, 255)), 2, cv2.LINE_AA)
    else:
        # Reset smoothed tracking if hands drop out of view
        _ball_charge = max(0.0, _ball_charge - 0.1)
        _smoothed_mid = None
        _smoothed_dist = 0.0

    # --- Flame Logic ---
    for side, wrist_idx, index_idx in [("left", 15, 19), ("right", 16, 20)]:
        if wrist_idx in lm and index_idx in lm and vis.get(wrist_idx, 0) > 0.15 and vis.get(index_idx, 0) > 0.15:
            wy = lm[wrist_idx][1]
            wx = lm[wrist_idx][0]
            iy = lm[index_idx][1]
            ix = lm[index_idx][0]
            
            # Palm facing up/forward check (fingers above wrist)
            # Make sure it's a prominent open hand
            dist_w_i = np.hypot(wx - ix, wy - iy)
            if iy < wy - 30 and dist_w_i > 40:
                _spawn_fire(wx, wy - 30, count=6, vy_range=(-12, -3))
                
            # Store wrist position to keep track
            _prev_wrists[side] = (wx, wy)
        elif wrist_idx in lm:
            _prev_wrists[side] = lm[wrist_idx]
        else:
            _prev_wrists[side] = None

    return canvas
