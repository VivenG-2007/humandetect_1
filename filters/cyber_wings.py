"""
Cyber Wings Filter — large, geometric neon wings that respond to movement.
Optimized for performance and includes a face-capture point cloud.
"""
import cv2
import numpy as np
import time
import random
from pose_detector import PoseResult

VIS_THRESHOLD = 0.4

def apply(canvas: np.ndarray, pose: PoseResult, **kwargs) -> np.ndarray:
    if not pose.detected:
        return canvas

    h, w = canvas.shape[:2]
    lm = pose.landmarks
    vis = pose.visibility
    t = time.time()
    original_frame = kwargs.get('original_frame')

    def get_pt(idx): return np.array(lm[idx]) if idx in lm and vis.get(idx, 0) > VIS_THRESHOLD else None

    # 1. Neon Body Core (Cyber Spine)
    sh_l, sh_r = get_pt(11), get_pt(12)
    hip_l, hip_r = get_pt(23), get_pt(24)
    
    pulse = (np.sin(t * 8) + 1) / 2
    neon_cyan = (255, 255, 0) # Cyan in BGR
    glow_color = tuple(int(c * (0.7 + 0.3 * pulse)) for c in neon_cyan)

    if sh_l is not None and sh_r is not None:
        mid_sh = (sh_l + sh_r) // 2
        # Neck point
        nose = get_pt(0)
        if nose is not None:
            cv2.line(canvas, tuple(mid_sh), tuple(nose), glow_color, 2, cv2.LINE_AA)

        if hip_l is not None and hip_r is not None:
            mid_hip = (hip_l + hip_r) // 2
            # Main energy spine
            cv2.line(canvas, tuple(mid_sh), tuple(mid_hip), glow_color, 8, cv2.LINE_AA)
            cv2.line(canvas, tuple(mid_sh), tuple(mid_hip), (255, 255, 255), 2, cv2.LINE_AA)

    # 2. Geometric Cyber Wings (Power Wings)
    if sh_l is not None and sh_r is not None:
        base_flap = np.sin(t * 6) * 0.1 # Subtle idle animation
        for side in [-1, 1]:
            # Mirrored: user's right shoulder (12) is on left of screen (-X). We want it to point LEFT (-X), so side should be -1.
            # User's left shoulder (11) is on right of screen (+X). We want it to point RIGHT (+X), so side should be 1.
            anchor = sh_r if side == -1 else sh_l
            hand_tip = get_pt(36) if side == -1 else get_pt(35) # Using index fingertips instead of wrists
            
            hand_flap = 0.0
            if hand_tip is not None:
                # Calculate hand elevation relative to shoulder
                dy = hand_tip[1] - anchor[1]
                dx = hand_tip[0] - anchor[0]
                dist = np.hypot(dx, dy)
                if dist > 5:
                    norm_y = dy / dist # 1 = straight down, -1 = straight up
                    spread_amt = (1.0 - norm_y) * 0.5 # ranges 0 to 1
                    hand_flap = spread_amt * (np.pi / 2.0) # Larger lift for satisfying feel

            flap = base_flap + hand_flap

            # Draw multiple layers of feathers
            for layer in range(2):
                l_pulse = (np.sin(t * 4 + layer) + 1) / 2
                layer_color = tuple(int(c * (0.5 + 0.5 * l_pulse)) for c in neon_cyan)
                
                for i in range(5):
                    # Base spread for feather array (pointing downwards)
                    spread = (np.pi / 2.2) - (i * 0.22)
                    
                    if side == 1: 
                        # Right Wing (must point Right: X > 0)
                        angle = spread - flap
                    else:         
                        # Left Wing (must point Left: X < 0)
                        angle = np.pi - (spread - flap)
                        
                    length = 180 + layer * 40 - i * 15
                    
                    end = anchor + np.array([np.cos(angle) * length, np.sin(angle) * length])
                    
                    # Outer energy stroke
                    cv2.line(canvas, tuple(anchor.astype(int)), tuple(end.astype(int)), layer_color, 4 - layer, cv2.LINE_AA)
                    # Inner white core
                    cv2.line(canvas, tuple(anchor.astype(int)), tuple(end.astype(int)), (255, 255, 255), 1)
                    
                    # Power node at end
                    cv2.circle(canvas, tuple(end.astype(int)), 3, (255, 255, 255), -1, cv2.LINE_AA)
                    cv2.circle(canvas, tuple(end.astype(int)), 6, layer_color, 1, cv2.LINE_AA)

    # 3. FACE HUD (Tech Graphics)
    if 0 in lm and vis.get(0, 0) > VIS_THRESHOLD:
        cp = lm[0]
        radius = 30
        # Octagon-like tech ring
        pts = []
        for i in range(8):
            ang = t * 2 + i * (np.pi / 4)
            pts.append([int(cp[0] + np.cos(ang) * radius), int(cp[1] + np.sin(ang) * radius)])
        cv2.polylines(canvas, [np.array(pts)], True, neon_cyan, 1, cv2.LINE_AA)
        
        # Face particles
        for _ in range(8):
            px = int(cp[0] + random.uniform(-20, 20))
            py = int(cp[1] + random.uniform(-20, 20))
            if 0 <= px < w and 0 <= py < h:
                color = original_frame[py, px].tolist() if original_frame is not None else (255, 255, 255)
                cv2.circle(canvas, (px, py), 1, color, -1)

    return canvas
