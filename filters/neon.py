"""
Neon Filter — vibrant neon outline with shifting rainbow colors.
Enhanced for performance with downscaled blurring.
Stick figure hidden.
"""
import cv2
import numpy as np
import time
from pose_detector import PoseResult, CONNECTIONS

VIS_THRESHOLD = 0.25

def apply(canvas: np.ndarray, pose: PoseResult, **kwargs) -> np.ndarray:
    if not pose.detected:
        return canvas

    h, w = canvas.shape[:2]
    lm = pose.landmarks
    vis = pose.visibility
    t = time.time()

    # 1. CYBER DIGIT GRID BACKGROUND
    grid_gap = 40
    grid_color = (40, 20, 10) # Dark purple/blue
    # Pulse the grid
    pulse_grid = (np.sin(t * 2) + 1) / 2 * 10
    grid_color = tuple(int(c + pulse_grid) for c in grid_color)
    
    for x in range(0, w, grid_gap):
        cv2.line(canvas, (x, 0), (x, h), grid_color, 1)
    for y in range(0, h, grid_gap):
        cv2.line(canvas, (0, y), (w, y), grid_color, 1)

    # 2. Shifting Neon Color
    hue = int((t * 60) % 180)
    neon_hsv = np.uint8([[[hue, 255, 255]]])
    neon_bgr = cv2.cvtColor(neon_hsv, cv2.COLOR_HSV2BGR)[0][0]
    neon_color = tuple(int(c) for c in neon_bgr)
    
    # Create a mask of the body profile
    mask = np.zeros((h, w), dtype=np.uint8)
    for (a, b) in CONNECTIONS:
        if a not in lm or b not in lm: continue
        if vis.get(a, 0) < VIS_THRESHOLD or vis.get(b, 0) < VIS_THRESHOLD: continue
        cv2.line(mask, lm[a], lm[b], 255, 30, cv2.LINE_AA)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # 3. Draw Glow Layer (Double Layer Glow)
    neon_layer = np.zeros_like(canvas)
    cv2.drawContours(neon_layer, contours, -1, neon_color, 15, cv2.LINE_AA)
    
    glow_small = cv2.resize(neon_layer, (w//4, h//4))
    glow_small = cv2.GaussianBlur(glow_small, (15, 15), 0)
    glow = cv2.resize(glow_small, (w, h))
    canvas[:] = cv2.addWeighted(canvas, 1.0, glow, 1.5, 0)
    
    # 4. Inner Bright Line and Accents
    cv2.drawContours(canvas, contours, -1, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.drawContours(canvas, contours, -1, neon_color, 4, cv2.LINE_AA)
    
    # 5. SCANLINE EFFECT
    for y in range(0, h, 4):
        canvas[y:y+1] = (canvas[y:y+1].astype(float) * 0.85).astype(np.uint8)

    return canvas
