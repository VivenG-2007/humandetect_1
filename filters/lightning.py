"""
Lightning Filter — electric arcs drawn along the skeleton's bones.
"""
import cv2
import numpy as np
import random
from pose_detector import PoseResult, CONNECTIONS
from skeleton_renderer import SkeletonRenderer

_renderer = SkeletonRenderer()

VIS_THRESHOLD = 0.25


def _lightning_bolt(img, pt1, pt2, color, segments=8, jitter=12, thickness=2):
    """Draw a jagged lightning bolt between two points."""
    pts = [pt1]
    for i in range(1, segments):
        t = i / segments
        mx = int(pt1[0] + (pt2[0] - pt1[0]) * t + random.randint(-jitter, jitter))
        my = int(pt1[1] + (pt2[1] - pt1[1]) * t + random.randint(-jitter, jitter))
        pts.append((mx, my))
    pts.append(pt2)

    for i in range(len(pts) - 1):
        # Draw glow
        cv2.line(img, pts[i], pts[i + 1], tuple(c // 4 for c in color), thickness + 6, cv2.LINE_AA)
        cv2.line(img, pts[i], pts[i + 1], color, thickness, cv2.LINE_AA)


def apply(canvas: np.ndarray, pose: PoseResult, **kwargs) -> np.ndarray:
    if not pose.detected:
        return canvas

    h, w = canvas.shape[:2]
    lm = pose.landmarks
    vis = pose.visibility

    # Wrist indices: 15 (L), 16 (R). Ankle indices: 27 (L), 28 (R)
    effector_indices = [15, 16, 27, 28]

    for idx in effector_indices:
        if idx in lm and vis.get(idx, 0) > VIS_THRESHOLD:
            pt = lm[idx]
            # Draw several electric arcs shooting from the hand
            for _ in range(random.randint(3, 5)):
                # Target point near the hand
                angle = random.uniform(0, 2 * np.pi)
                dist = random.uniform(30, 100)
                target = (
                    int(pt[0] + np.cos(angle) * dist),
                    int(pt[1] + np.sin(angle) * dist)
                )
                
                color = random.choice([
                    (255, 255, 255),  # White
                    (255, 220, 100),  # Electric blue/cyan look (BGR)
                    (255, 150, 50),   # Deep blue (BGR)
                ])
                _lightning_bolt(canvas, pt, target, color, segments=6, jitter=15, thickness=2)
                
            # Bright core at the hand
            cv2.circle(canvas, pt, 6, (255, 255, 255), -1, cv2.LINE_AA)
            cv2.circle(canvas, pt, 12, (255, 200, 100), 2, cv2.LINE_AA)

    # 3. Draw Stick Figure
    _renderer.render(pose, (h, w), canvas)

    return canvas
