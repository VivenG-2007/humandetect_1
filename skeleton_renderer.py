"""
Skeleton Renderer — draws stick-figure skeleton on a black canvas.
Handles partial detections gracefully (waist-up, head-only, etc.).
"""
import cv2
import numpy as np
from typing import Optional, Tuple
from pose_detector import PoseResult, CONNECTIONS

# Visibility threshold below which we skip a landmark / connection
VIS_THRESHOLD = 0.4
import cv2
import numpy as np
import time
from typing import Optional, Tuple
from pose_detector import PoseResult, CONNECTIONS

# Visibility threshold below which we skip a landmark / connection
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
        
        # ── Pulsing Glow Alpha ──
        pulse = (np.sin(t * 4) + 1) / 2 * 0.4 + 0.6 # Pulse between 0.6 and 1.0

        # 1. DRAW TORSO (Semi-transparent Tech Box)
        if all(i in lm and vis.get(i, 0) > VIS_THRESHOLD for i in [11, 12, 23, 24]):
            pts = np.array([lm[11], lm[12], lm[24], lm[23]], np.int32)
            
            # Fill torso with dark tech color
            overlay = canvas.copy()
            cv2.fillPoly(overlay, [pts], self.torso_color)
            cv2.addWeighted(overlay, 0.4, canvas, 0.6, 0, canvas)
            
            # Draw glowing border for torso
            cv2.polylines(canvas, [pts], True, self.line_color, 1, cv2.LINE_AA)
            # Outer glow
            cv2.polylines(canvas, [pts], True, self.line_color, 4, cv2.LINE_AA)

        # 2. DRAW LIMBS with Neon Glow
        limb_connections = [
            (11, 13), (13, 15), # Left arm
            (12, 14), (14, 16), # Right arm
            (23, 25), (25, 27), # Left leg
            (24, 26), (26, 28), # Right leg
            (11, 12), (23, 24), # Shoulder/Hip lines
        ]
        
        # Create a glow layer
        glow_layer = np.zeros_like(canvas)
        for (a, b) in limb_connections:
            if a in lm and b in lm and vis.get(a, 0) > VIS_THRESHOLD and vis.get(b, 0) > VIS_THRESHOLD:
                # Main line
                cv2.line(canvas, lm[a], lm[b], self.line_color, self.line_thickness, cv2.LINE_AA)
                # Glow effect
                cv2.line(glow_layer, lm[a], lm[b], self.line_color, self.line_thickness + 6, cv2.LINE_AA)

        # Blur the glow layer and add it (optimized for large resolution)
        glow_small = cv2.resize(glow_layer, (w // 4, h // 4))
        glow_blur_small = cv2.GaussianBlur(glow_small, (7, 7), 0)
        glow_blur = cv2.resize(glow_blur_small, (w, h))
        canvas = cv2.addWeighted(canvas, 1.0, glow_blur, 0.8 * pulse, 0)

        # 3. DRAW HEAD & NECK
        if 0 in lm and vis.get(0, 0) > VIS_THRESHOLD:
            head_center = lm[0]
            radius = 18
            
            # Neck line
            if 11 in lm and 12 in lm:
                mid_shoulder = ((lm[11][0] + lm[12][0]) // 2, (lm[11][1] + lm[12][1]) // 2)
                cv2.line(canvas, head_center, mid_shoulder, self.line_color, 2, cv2.LINE_AA)

            # Circular Tech Head
            cv2.circle(canvas, head_center, radius, (20, 20, 20), -1, cv2.LINE_AA)
            cv2.circle(canvas, head_center, radius, self.line_color, 2, cv2.LINE_AA)
            # Floating "HUD" dots near head
            for i in range(3):
                ang = t * 2 + i * (2 * np.pi / 3)
                dx = int(np.cos(ang) * (radius + 8))
                dy = int(np.sin(ang) * (radius + 8))
                cv2.circle(canvas, (head_center[0] + dx, head_center[1] + dy), 2, self.line_color, -1, cv2.LINE_AA)

        # 4. DRAW JOINTS (Tech Nodes)
        for idx in [11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]:
            if idx in lm and vis.get(idx, 0) > VIS_THRESHOLD:
                # Outer ring
                cv2.circle(canvas, lm[idx], self.joint_radius + 2, self.line_color, 1, cv2.LINE_AA)
                # Inner dot
                cv2.circle(canvas, lm[idx], self.joint_radius - 1, self.joint_color, -1, cv2.LINE_AA)

        return canvas

    def render_with_custom_color(self, pose, frame_shape, canvas, line_color, joint_color, thickness=2):
        old_c, old_j, old_t = self.line_color, self.joint_color, self.line_thickness
        self.line_color, self.joint_color, self.line_thickness = line_color, joint_color, thickness
        res = self.render(pose, frame_shape, canvas)
        self.line_color, self.joint_color, self.line_thickness = old_c, old_j, old_t
        return res
