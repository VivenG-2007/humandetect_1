"""
Bubble Filter — Optimized Clean Trails & Interaction.
- Orange/Warm bubble particles follow index fingertip movement.
- Fading blue glow trail on fingertips for visual weight.
- Dynamic Rainbow Burst when fingertips (35, 36) meet.
- High-performance vectorized physics.
"""
import cv2
import numpy as np
import random
import time
from pose_detector import PoseResult, CONNECTIONS

# ── Particle state arrays (NumPy optimized) ──────────────────────────────────
_POSITIONS     = np.zeros((0, 2), dtype=np.float32)
_VELOCITIES    = np.zeros((0, 2), dtype=np.float32)
_SIZES         = np.zeros(0,      dtype=np.float32)
_COLORS        = np.zeros((0, 3), dtype=np.float32)
_START_TIMES   = np.zeros(0,      dtype=np.float32)
_LIFETIMES     = np.zeros(0,      dtype=np.float32)
_WOBBLE_CONFIG = np.zeros((0, 2), dtype=np.float32)

# ── Trail state ───────────────────────────────────────────────────────────────
_TRAIL_LEFT    = []
_TRAIL_RIGHT   = []
TRAIL_MAX_LEN  = 25
TRAIL_FADE_SEC = 0.3

# ── Config ────────────────────────────────────────────────────────────────────
_prev_landmarks     = {}
_last_spawn_time    = {35: 0.0, 36: 0.0}

STRICT_MOVE         = 25.0    # Smoother emission with stabilized landmarks
MAX_TELEPORT        = 200.0   # ignore large tracking jumps
MAX_BUBBLES         = 1500    # high cap for multiple bursts
VIS_THRESHOLD       = 0.60    # very strict visibility for spawning
SPAWN_COOLDOWN      = 0.03    # ~30 Hz spawn max

_was_pinching_left  = False
_was_pinching_right = False
_was_joining        = False

# ── Color Palettes (BGR) ──────────────────────────────────────────────────
ORANGE_COLORS = [
    (20, 160, 255),  # Pure Orange
    (70, 190, 255),  # Light Orange
    (20, 110, 240)   # Deep Orange
]

RAINBOW_COLORS = [
    (0, 0, 255), (0, 165, 255), (0, 255, 255), (0, 255, 0), 
    (255, 0, 0), (255, 0, 255), (255, 50, 150)
]

BLUE_TRAIL_COLORS = [
    (255, 200, 100), (255, 180, 50), (255, 150, 20)
]

# ── Implementation ────────────────────────────────────────────────────────────

def _spawn_bubbles(x, y, count, size_range=(3, 7), colors=None, 
                   vx_range=(-1.5, 1.5), vy_range=(0.5, 2.8),
                   lifetime_range=(0.8, 1.8), rainbow=False):
    global _POSITIONS, _VELOCITIES, _SIZES, _COLORS, _START_TIMES, _LIFETIMES, _WOBBLE_CONFIG
    
    # Cap count based on available space
    remaining = MAX_BUBBLES - len(_POSITIONS)
    count = max(0, min(count, remaining))
    if count <= 0: return

    new_pos = np.full((count, 2), [x, y], dtype=np.float32) + np.random.uniform(-4, 4, (count, 2))
    
    # Velocity: mostly upward (y decreases in CV2)
    vx = np.random.uniform(vx_range[0], vx_range[1], count)
    vy = np.random.uniform(vy_range[0], vy_range[1], count)
    new_vel = np.stack([vx, vy], axis=1).astype(np.float32)
    
    new_sz = np.random.uniform(size_range[0], size_range[1], count).astype(np.float32)
    
    if rainbow:
        new_colors = np.array([random.choice(RAINBOW_COLORS) for _ in range(count)], dtype=np.float32)
    elif colors is None:
        new_colors = np.tile([255, 180, 80], (count, 1)).astype(np.float32)
    else:
        new_colors = np.array(colors, dtype=np.float32)
        if new_colors.ndim == 1: new_colors = np.tile(new_colors, (count, 1))

    now = time.time()
    _POSITIONS = np.vstack([_POSITIONS, new_pos])
    _VELOCITIES = np.vstack([_VELOCITIES, new_vel])
    _SIZES = np.concatenate([_SIZES, new_sz])
    _COLORS = np.vstack([_COLORS, new_colors])
    _START_TIMES = np.concatenate([_START_TIMES, np.full(count, now, dtype=np.float32)])
    _LIFETIMES = np.concatenate([_LIFETIMES, np.random.uniform(lifetime_range[0], lifetime_range[1], count).astype(np.float32)])
    _WOBBLE_CONFIG = np.vstack([_WOBBLE_CONFIG, np.random.uniform(2.0, 5.0, (count, 2)).astype(np.float32)])

def apply(canvas: np.ndarray, pose: PoseResult, **kwargs) -> np.ndarray:
    global _POSITIONS, _VELOCITIES, _SIZES, _COLORS, _START_TIMES, _LIFETIMES, _WOBBLE_CONFIG
    global _prev_landmarks, _TRAIL_LEFT, _TRAIL_RIGHT, _was_joining, _last_spawn_time

    h, w = canvas.shape[:2]
    t = time.time()

    # 1. Update Physics (Vectorized)
    if len(_POSITIONS) > 0:
        elapsed = t - _START_TIMES
        alive = elapsed < _LIFETIMES
        if not np.all(alive):
            _POSITIONS = _POSITIONS[alive]; _VELOCITIES = _VELOCITIES[alive]
            _SIZES = _SIZES[alive]; _COLORS = _COLORS[alive]
            _START_TIMES = _START_TIMES[alive]; _LIFETIMES = _LIFETIMES[alive]
            _WOBBLE_CONFIG = _WOBBLE_CONFIG[alive]
            elapsed = elapsed[alive]

        if len(_POSITIONS) > 0:
            _POSITIONS[:, 1] -= _VELOCITIES[:, 1]
            _POSITIONS[:, 0] += _VELOCITIES[:, 0] + np.sin(elapsed * _WOBBLE_CONFIG[:, 0]) * _WOBBLE_CONFIG[:, 1] * 0.5
            
            # Rendering: Single combined overlay for efficiency
            life_ratios = elapsed / _LIFETIMES
            alphas = np.clip(np.where(life_ratios > 0.6, 1.0 - (life_ratios - 0.6)/0.4, 1.0), 0, 1) ** 1.5
            
            for i in range(len(_POSITIONS)):
                px, py = int(_POSITIONS[i, 0]), int(_POSITIONS[i, 1])
                if 0 <= px < w and 0 <= py < h:
                    sz = int(_SIZES[i])
                    a = alphas[i]
                    c = tuple(map(int, _COLORS[i] * a))
                    cv2.circle(canvas, (px, py), sz, c, 1, cv2.LINE_AA)
                    # Glint
                    cv2.circle(canvas, (px-max(1, sz//3), py-max(1, sz//3)), max(1, sz//4), (255, 255, 255), -1, cv2.LINE_AA)

    # 2. Gesture Logic
    if pose.detected:
        lm, vis = pose.landmarks, pose.visibility
        
        # Fingertip Trails (35, 36)
        tip_info = {35: _TRAIL_LEFT, 36: _TRAIL_RIGHT}
        
        # Batch collect trail points to render in one sub-layer
        trail_overlay = np.zeros_like(canvas)
        for idx, trail in tip_info.items():
            if idx in lm and vis.get(idx, 0) > VIS_THRESHOLD:
                pt = lm[idx]
                trail.append((pt[0], pt[1], t))
                if len(trail) > TRAIL_MAX_LEN: trail.pop(0)
                
                # Render Trail (Blue Glow)
                n = len(trail)
                for i, (tx, ty, t_start) in enumerate(trail):
                    age = t - t_start
                    if age > TRAIL_FADE_SEC: continue
                    alpha = (1.0 - age / TRAIL_FADE_SEC) * (i / n)
                    color = BLUE_TRAIL_COLORS[i % len(BLUE_TRAIL_COLORS)]
                    cv2.circle(trail_overlay, (tx, ty), int(4 + i*0.8), color, -1, cv2.LINE_AA)
                
                # Spawn actual bubbles from moving fingers
                if idx in _prev_landmarks and vis.get(idx, 0) > 0.85:
                    p1_arr = np.array(pt)
                    p2_arr = np.array(_prev_landmarks[idx])
                    dist = np.linalg.norm(p1_arr - p2_arr)
                    
                    if STRICT_MOVE < dist < MAX_TELEPORT:
                        _last_spawn_time[idx] = t
                        # 1 bubble per 50px of movement
                        _spawn_bubbles(pt[0], pt[1], int(dist/50)+1, colors=random.choice(ORANGE_COLORS))
                
                # Update memory ONLY if visibility is high to prevent teleport lines
                if vis.get(idx, 0) > 0.75:
                    _prev_landmarks[idx] = pt

        cv2.addWeighted(trail_overlay, 0.4, canvas, 1.0, 0, canvas)
        # Hysteresis reset: Reset burst if hands are separated
        if 35 in lm and 36 in lm:
            d = np.linalg.norm(np.array(lm[35]) - np.array(lm[36]))
            if d > 75:
                _was_joining = False

        # Rainbow Burst (Hands Meeting)
        is_join = False
        meeting_pt = (0, 0)
        # Check against index fingertips (35, 36)
        if 35 in lm and 36 in lm and vis.get(35,0) > 0.4 and vis.get(36,0) > 0.4:
            d = np.linalg.norm(np.array(lm[35]) - np.array(lm[36]))
            # Precision meeting point
            if d < 45: 
                is_join = True
                meeting_pt = (int((lm[35][0] + lm[36][0]) // 2), 
                              int((lm[35][1] + lm[36][1]) // 2))
        
        if is_join and not _was_joining:
            # Giga Burst: High density radial expansion exactly at midpoint
            # 1. Flash Core
            _spawn_bubbles(meeting_pt[0], meeting_pt[1], 70, 
                           size_range=(12, 26), rainbow=True, 
                           vx_range=(-20, 20), vy_range=(-20, 20))
            
            # 2. Expanding Ring
            for ang in np.linspace(0, 2*np.pi, 18):
                vx, vy = np.cos(ang) * 18, np.sin(ang) * 18
                _spawn_bubbles(meeting_pt[0], meeting_pt[1], 15, 
                               size_range=(4, 14), rainbow=True,
                               vx_range=(vx-5, vx+5), vy_range=(vy-5, vy+5))
            _was_joining = True # Lock until separated

        # Fingertip Glow Rings
        for idx in [35, 36]:
            if idx in lm and vis.get(idx, 0) > 0.2:
                pt = lm[idx]
                c = (255, 150, 255) if _was_joining else (255, 220, 100)
                cv2.circle(canvas, pt, 20, c, 2, cv2.LINE_AA)
                cv2.circle(canvas, (pt[0]-6, pt[1]-6), 5, (255, 255, 255), -1, cv2.LINE_AA)

    return canvas