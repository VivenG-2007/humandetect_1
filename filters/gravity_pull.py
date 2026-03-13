"""
Gravity Pull Filter
Gesture: Close fist
Effect: Particles on the screen get pulled into the hand.
Open hand -> particles explode outward.
"""
import cv2
import numpy as np
import random
import time
from pose_detector import PoseResult, CONNECTIONS

# --- Particle State ---
NUM_PARTICLES = 600
_PARTICLES = np.zeros((0, 2), dtype=np.float32)
_VELOCITIES = np.zeros((0, 2), dtype=np.float32)
_COLORS = np.zeros((0, 3), dtype=np.float32)
_SIZES = np.zeros(0, dtype=np.int32)

_initialized = False

# --- Hand State ---
_fist_history = {"left": [], "right": []}
_was_closed = {"left": False, "right": False}
_HISTORY_LEN = 5
_FIST_VOTES = 3

def _init_particles(w, h):
    global _PARTICLES, _VELOCITIES, _COLORS, _SIZES, _initialized
    _PARTICLES = np.column_stack((
        np.random.uniform(0, w, NUM_PARTICLES),
        np.random.uniform(0, h, NUM_PARTICLES)
    )).astype(np.float32)
    
    _VELOCITIES = np.random.uniform(-1, 1, (NUM_PARTICLES, 2)).astype(np.float32)
    
    _COLORS = np.zeros((NUM_PARTICLES, 3), dtype=np.float32)
    for i in range(NUM_PARTICLES):
        # Blueish / Cyan / Purple cosmic particles
        _COLORS[i] = (255, random.randint(100, 255), random.randint(50, 200))
        
    _SIZES = np.random.randint(1, 4, NUM_PARTICLES)
    _initialized = True

def _vote(history, new_val, max_len, votes_needed):
    history.append(new_val)
    if len(history) > max_len:
        history.pop(0)
    return history.count(True) >= votes_needed

def _is_fist(lm, vis, wrist_idx, elbow_idx, knuckle_idxs, tip_idxs):
    if wrist_idx not in lm: 
        return False
        
    wrist = np.array(lm[wrist_idx], dtype=np.float32)
    
    # Hand Tracker sets visibility exactly to 1.0 for highly detailed fingers.
    # If active, we have accurate knuckles AND accurate varying tips.
    hand_tracker_active = any(vis.get(ti, 0) == 1.0 for ti in tip_idxs)
    
    if hand_tracker_active:
        curled = 0
        for ki, ti in zip(knuckle_idxs, tip_idxs):
            if ki in lm and ti in lm:
                d_knuckle = max(1.0, np.linalg.norm(np.array(lm[ki]) - wrist))
                d_tip = np.linalg.norm(np.array(lm[ti]) - wrist)
                if d_tip < d_knuckle * 1.3:
                    curled += 1
        return curled >= 2
    else:
        # Distance fallback! Hand is far from camera. 
        # The POSE model lumps knuckles and tips together into broader "hand extents" (17, 19, 21).
        # We simply check if these points are clumped closely to the wrist!
        if elbow_idx not in lm: return False
        
        elbow = np.array(lm[elbow_idx], dtype=np.float32)
        d_forearm = max(10.0, np.linalg.norm(wrist - elbow))
        
        dists = [np.linalg.norm(np.array(lm[ki]) - wrist) for ki in knuckle_idxs if ki in lm]
        if not dists: return False
        
        avg_dist = sum(dists) / len(dists)
        
        # When hand is open, the spread is 40-50% forearm length. 
        # Clamped closed fist shrinks the bounding hand shape to < 35% of the arm
        return (avg_dist / d_forearm) < 0.35

def apply(canvas: np.ndarray, pose: PoseResult, **kwargs) -> np.ndarray:
    global _PARTICLES, _VELOCITIES, _COLORS, _SIZES, _initialized
    global _fist_history, _was_closed
    
    h, w = canvas.shape[:2]
    if not _initialized:
        _init_particles(w, h)
        
    overlay = np.zeros_like(canvas)
    
    if pose.detected:
        lm = pose.landmarks
        vis = pose.visibility
        
        # 1. Detect fists
        # Landmarks: wrists (15/16), elbows (13/14), [Pinky base, Index base, Thumb base], [Pinky tip, Index tip, Thumb tip]
        lf_raw = _is_fist(lm, vis, 15, 13, [17, 19, 21], [33, 35, 37]) # Left hand
        rf_raw = _is_fist(lm, vis, 16, 14, [18, 20, 22], [34, 36, 38]) # Right hand
        
        left_fist = _vote(_fist_history["left"], lf_raw, _HISTORY_LEN, _FIST_VOTES)
        right_fist = _vote(_fist_history["right"], rf_raw, _HISTORY_LEN, _FIST_VOTES)
        
        for side, is_fist, wrist_idx in [("left", left_fist, 15), ("right", right_fist, 16)]:
            if wrist_idx in lm and vis.get(wrist_idx, 0) > 0.15: # Forgiving visibility
                idx_base = 19 if side == "left" else 20
                if idx_base in lm and vis.get(idx_base, 0) > 0.15:
                    hx = int((lm[wrist_idx][0] + lm[idx_base][0]) / 2)
                    hy = int((lm[wrist_idx][1] + lm[idx_base][1]) / 2)
                else:
                    hx, hy = lm[wrist_idx]
                
                # Draw hand core marker
                color = (0, 0, 255) if is_fist else (255, 200, 100)
                cv2.circle(canvas, (hx, hy), 15, color, -1, cv2.LINE_AA)
                cv2.circle(canvas, (hx, hy), 25, color, 2, cv2.LINE_AA)
                
                # Gravity / Explosion logic
                if is_fist:
                    # Pull particles towards hand
                    dx = hx - _PARTICLES[:, 0]
                    dy = hy - _PARTICLES[:, 1]
                    dist = np.sqrt(dx**2 + dy**2) + 1.0
                    
                    # Gravity force increases as it gets closer, up to a limit
                    force = 1500.0 / (dist + 50.0)
                    
                    _VELOCITIES[:, 0] += (dx / dist) * force * 0.1
                    _VELOCITIES[:, 1] += (dy / dist) * force * 0.1
                    
                    # Friction when closed to gather them in a dense ball
                    _VELOCITIES *= 0.85 
                    
                else:
                    if _was_closed[side]:
                        # Explode! (Fist transitioned to open)
                        dx = _PARTICLES[:, 0] - hx
                        dy = _PARTICLES[:, 1] - hy
                        dist = np.sqrt(dx**2 + dy**2) + 1.0
                        
                        # Base outward push based on distance
                        push_power = 1500.0 / (dist + 30.0)
                        push_power = np.clip(push_power, 2.0, 15.0)
                        
                        # Because particles cluster into a tiny singularity, dx/dist forms an unnatural straight line.
                        # We must inject a random radial starburst scatter to ensure they explode in a beautiful circle!
                        angles = np.random.uniform(0, 2 * np.pi, NUM_PARTICLES)
                        speeds = np.random.uniform(10.0, 45.0, NUM_PARTICLES)
                        
                        rand_vx = np.cos(angles) * speeds
                        rand_vy = np.sin(angles) * speeds
                        
                        _VELOCITIES[:, 0] = (dx / dist) * push_power + rand_vx
                        _VELOCITIES[:, 1] = (dy / dist) * push_power + rand_vy
                
                _was_closed[side] = is_fist

    # 2. Update and render particles
    _PARTICLES += _VELOCITIES
    _VELOCITIES *= 0.94 # Smoother friction slowing down the particles
    
    # Boundary wrap-around (so particles don't get lost forever)
    _PARTICLES[:, 0] = np.mod(_PARTICLES[:, 0], w)
    _PARTICLES[:, 1] = np.mod(_PARTICLES[:, 1], h)
    
    # Draw particles (vectorized for speed)
    px = _PARTICLES[:, 0].astype(np.int32)
    py = _PARTICLES[:, 1].astype(np.int32)
    
    # Render with loops to handle colors
    for i in range(NUM_PARTICLES):
        cv2.circle(overlay, (px[i], py[i]), _SIZES[i], tuple(map(int, _COLORS[i])), -1, cv2.LINE_AA)

    # Add particle overlay to canvas with a soft glow addition
    cv2.addWeighted(canvas, 1.0, overlay, 0.9, 0, canvas)
    
    return canvas
