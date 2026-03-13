"""
Biomechanical Overlay Filter
Body parts are partially replaced with robotic structural components, pistons, and glowing cyborg circuit nodes.
"""
import cv2
import numpy as np
import random
from pose_detector import PoseResult, CONNECTIONS

def apply(canvas: np.ndarray, pose: PoseResult, **kwargs) -> np.ndarray:
    if not pose.detected:
        return canvas

    h, w = canvas.shape[:2]
    original = kwargs.get('original_frame')
    
    # Render a desaturated, gritty background to emphasize the cybernetic parts
    if original is not None:
        gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
        bg = cv2.merge([(gray * 0.4).astype(np.uint8), (gray * 0.5).astype(np.uint8), (gray * 0.4).astype(np.uint8)])
        canvas[:] = bg
    
    layer = np.zeros_like(canvas)
    lm = pose.landmarks
    vis = pose.visibility

    # Cyborg Structural Pistons
    thick_pistons = [
        # Arms
        ((11, 13), (200, 200, 220), 16),
        ((13, 15), (150, 150, 170), 12),
        ((12, 14), (200, 200, 220), 16),
        ((14, 16), (150, 150, 170), 12),
        # Legs
        ((23, 25), (200, 200, 220), 22),
        ((25, 27), (150, 150, 170), 18),
        ((24, 26), (200, 200, 220), 22),
        ((26, 28), (150, 150, 170), 18),
        # Torso spine
        ((11, 23), (80, 80, 90),   25),
        ((12, 24), (80, 80, 90),   25),
    ]

    for (a, b), color, thickness in thick_pistons:
        if a in lm and b in lm and vis.get(a, 0) > 0.3 and vis.get(b, 0) > 0.3:
            pt1 = lm[a]
            pt2 = lm[b]
            # Draw industrial steel armature
            cv2.line(layer, pt1, pt2, (40, 40, 50), thickness + 6, cv2.LINE_AA) # Shadow/Housing
            cv2.line(layer, pt1, pt2, color, thickness, cv2.LINE_AA) # Core metal
            
            # Add mechanical joints along the piston (hydraulic bands)
            mid = ((pt1[0] + pt2[0]) // 2, (pt1[1] + pt2[1]) // 2)
            cv2.circle(layer, mid, thickness + 4, (30, 30, 30), -1, cv2.LINE_AA)
            cv2.circle(layer, mid, thickness + 2, (0, 150, 255), 2, cv2.LINE_AA) # Orange glowing ring

    # Sci-Fi Circuit Nodes for joints
    for idx, pt in lm.items():
        if vis.get(idx, 0) > 0.4:
            # Mechanical Gear Center
            cv2.circle(layer, pt, 15, (50, 50, 60), -1, cv2.LINE_AA)
            # Inner rotating element
            cv2.circle(layer, pt, 8, (200, 200, 220), -1, cv2.LINE_AA)
            # Glowing sensor core
            cv2.circle(layer, pt, 4, (0, 255, 255), -1, cv2.LINE_AA)
            
            # Random circuit traces ejecting from nodes
            if random.random() < 0.3:
                cx, cy = pt
                tx = cx + random.randint(-40, 40)
                ty = cy + random.randint(-40, 40)
                # Right angle circuit trace
                cv2.line(layer, (cx, cy), (tx, cy), (0, 255, 0), 2, cv2.LINE_AA)
                cv2.line(layer, (tx, cy), (tx, ty), (0, 255, 0), 2, cv2.LINE_AA)
                cv2.circle(layer, (tx, ty), 3, (0, 255, 0), -1, cv2.LINE_AA)

    # Face plate / HUD Visor
    if all(i in lm and vis.get(i, 0) > 0.4 for i in [0, 2, 5]):
        nose = lm[0]
        eye_l = lm[2]
        eye_r = lm[5]
        
        # Cyber-viso drawing across eyes
        v_left = (eye_r[0] - 25, eye_r[1] - 15)
        v_right = (eye_l[0] + 25, eye_l[1] + 15)
        cv2.rectangle(layer, tuple(min(x) for x in zip(v_left, v_right)), tuple(max(x) for x in zip(v_left, v_right)), (0, 0, 0), -1)
        # Red glowing scanline
        scan_y = int((nose[1] - 15) + np.sin(time.time() * 5) * 10)
        cv2.line(layer, (v_left[0], scan_y), (v_right[0], scan_y), (0, 0, 255), 3, cv2.LINE_AA)

    # Composite blending
    canvas[:] = cv2.addWeighted(canvas, 1.0, layer, 1.0, 0)
    
    return canvas
