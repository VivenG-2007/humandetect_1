"""
Skeleton Renderer — draws stick-figure skeleton on a black canvas.
Handles partial detections gracefully (waist-up, head-only, etc.).
Optimized for batch drawing and reduced per-frame overhead.
"""
import cv2
import numpy as np
import time
from typing import Optional, Tuple
from pose_detector import PoseResult, CONNECTIONS

VIS_THRESHOLD = 0.45

class SkeletonRenderer:
    def __init__(
        self,
        line_color: tuple = (255, 242, 0),    # Cyan/Neon (BGR)
        joint_color: tuple = (255, 255, 255), # White
        torso_color: tuple = (40, 40, 50),    # Dark Tech Grey
        line_thickness: int = 3,
        joint_radius: int = 4,
    ):
        self.line_color = line_color
        self.joint_color = joint_color
        self.torso_color = torso_color
        self.line_thickness = line_thickness
        self.joint_radius = joint_radius
        self._start_time = time.time()
        
        # Pre-calculate limb connection groups for faster drawing
        self.limb_connections = [
            (11, 13), (13, 15), # Left arm
            (12, 14), (14, 16), # Right arm
            (15, 17), (17, 33), # Left pinky
            (15, 19), (19, 35), # Left index
            (15, 21), (21, 37), # Left thumb
            (15, 40), (15, 42), # L Middle, L Ring
            (16, 18), (18, 34), # Right pinky
            (16, 20), (20, 36), # Right index
            (16, 22), (22, 38), # Right thumb
            (16, 41), (16, 43), # R Middle, R Ring
            (23, 25), (25, 27), # Left leg
            (24, 26), (26, 28), # Right leg
            (27, 29), (27, 31), # Left foot
            (28, 30), (28, 32), # Right foot
            (11, 12), (23, 24), # Shoulder/Hip lines
        ]
        
        # All tracking indices for nodes
        self.node_indices = [
            11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 
            23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 
            33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43
        ]

    def render(
        self,
        pose: PoseResult,
        frame_shape: tuple,
        canvas: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        h, w = frame_shape[:2]
        if canvas is None:
            canvas = np.zeros((h, w, 3), dtype=np.uint8)

        if not pose.detected:
            return canvas

        lm = pose.landmarks
        vis = pose.visibility
        t = time.time() - self._start_time
        
        # Pulsing Glow Alpha (pre-calculated factor)
        pulse = (np.sin(t * 4) + 1) / 2 * 0.4 + 0.6 

        # 1. DRAW TORSO (Batch Poly)
        if all(i in lm and vis.get(i, 0) > VIS_THRESHOLD for i in [11, 12, 23, 24]):
            pts = np.array([lm[11], lm[12], lm[24], lm[23]], np.int32)
            overlay = canvas.copy()
            cv2.fillPoly(overlay, [pts], self.torso_color)
            cv2.addWeighted(overlay, 0.4, canvas, 0.6, 0, canvas)
            cv2.polylines(canvas, [pts], True, self.line_color, 1, cv2.LINE_AA)
            cv2.polylines(canvas, [pts], True, self.line_color, 4, cv2.LINE_AA)

        # 2. DRAW LIMBS (Batch Optimized)
        glow_layer = np.zeros_like(canvas)
        for (a, b) in self.limb_connections:
            if a in lm and b in lm and vis.get(a, 0) > VIS_THRESHOLD and vis.get(b, 0) > VIS_THRESHOLD:
                p1, p2 = lm[a], lm[b]
                cv2.line(canvas, p1, p2, self.line_color, self.line_thickness, cv2.LINE_AA)
                cv2.line(glow_layer, p1, p2, self.line_color, self.line_thickness + 6, cv2.LINE_AA)

        # Batch Blur Proxy for Glow
        if np.any(glow_layer):
            glow_small = cv2.resize(glow_layer, (w // 4, h // 4), interpolation=cv2.INTER_LINEAR)
            glow_blur_small = cv2.GaussianBlur(glow_small, (7, 7), 0)
            glow_blur = cv2.resize(glow_blur_small, (w, h), interpolation=cv2.INTER_LINEAR)
            cv2.addWeighted(canvas, 1.0, glow_blur, 0.8 * pulse, 0, dst=canvas)

        # 3. DRAW HEAD (Enhanced crown tracking 39)
        if 0 in lm and vis.get(0, 0) > VIS_THRESHOLD:
            head_center = lm[0]
            # Neck line
            if 11 in lm and 12 in lm:
                mid_sh = ((lm[11][0] + lm[12][0]) // 2, (lm[11][1] + lm[12][1]) // 2)
                cv2.line(canvas, head_center, mid_sh, self.line_color, 2, cv2.LINE_AA)
            
            # Head Circle
            cv2.circle(canvas, head_center, 18, (20, 20, 20), -1, cv2.LINE_AA)
            cv2.circle(canvas, head_center, 18, self.line_color, 2, cv2.LINE_AA)
            
            # Spine to Crown (39 is top of head)
            if 39 in lm and vis.get(39, 0) > VIS_THRESHOLD:
                cv2.line(canvas, head_center, lm[39], self.line_color, 2, cv2.LINE_AA)
            
            # HUD details
            for i in range(3):
                ang = t * 2 + i * (2 * np.pi / 3)
                dx, dy = int(np.cos(ang) * 26), int(np.sin(ang) * 26)
                cv2.circle(canvas, (head_center[0] + dx, head_center[1] + dy), 2, self.line_color, -1, cv2.LINE_AA)

        # 4. DRAW JOINTS (Tech Nodes)
        for idx in self.node_indices:
            if idx in lm and vis.get(idx, 0) > VIS_THRESHOLD:
                pt = lm[idx]
                cv2.circle(canvas, pt, self.joint_radius + 2, self.line_color, 1, cv2.LINE_AA)
                cv2.circle(canvas, pt, self.joint_radius - 1, self.joint_color, -1, cv2.LINE_AA)

        return canvas

    def render_with_custom_color(self, pose, frame_shape, canvas, line_color, joint_color, thickness=2):
        old_c, old_j, old_t = self.line_color, self.joint_color, self.line_thickness
        self.line_color, self.joint_color, self.line_thickness = line_color, joint_color, thickness
        res = self.render(pose, frame_shape, canvas)
        self.line_color, self.joint_color, self.line_thickness = old_c, old_j, old_t
        return res
