"""
Hologram Filter — Sci-fi ghostly projection of the human body.
Uses segmentation mask for a perfect silhouette with digital scan lines.
"""
import cv2
import numpy as np
import random
import time

def apply(canvas: np.ndarray, pose, **kwargs) -> np.ndarray:
    if not pose.detected or pose.segmentation_mask is None:
        return canvas

    h, w = canvas.shape[:2]
    mask = pose.segmentation_mask
    t = time.time()
    
    # 1. Base Hologram Silhouette
    # Use mask to extract the body shape
    hologram_color = (255, 180, 50) # BGR Cyan/Blue
    
    # Create a soft body silhouette from the mask
    mask_small = cv2.resize(mask, (w // 4, h // 4))
    soft_mask_small = cv2.GaussianBlur(mask_small, (11, 11), 0)
    soft_mask = cv2.resize(soft_mask_small, (w, h))

    # Color the silhouette
    alpha = (soft_mask * 0.6).astype(np.float32)
    body_layer = (alpha[:, :, np.newaxis] * hologram_color).astype(np.uint8)

    # 2. Chromatic Aberration (Shifted ghost layers)
    b, g, r = cv2.split(body_layer)
    shift = int(5 * np.sin(t * 3))
    b = np.roll(b, shift, axis=1)
    r = np.roll(r, -shift, axis=1)
    ghost_layer = cv2.merge([b, g, r])

    # 3. Dynamic Scan lines
    scan_layer = np.zeros_like(canvas)
    line_y = int((t * 100) % h)
    cv2.line(scan_layer, (0, line_y), (w, line_y), (255, 255, 255), 2, cv2.LINE_AA)
    
    # Static-ish scan lines
    for i in range(0, h, 6):
        intensity = (np.sin(i * 0.2 + t * 10) + 1) / 2 * 0.2
        cv2.line(ghost_layer, (0, i), (w, i), (0, 0, 0), 1)

    # 4. Digital Glitches
    if random.random() < 0.1:
        # Horizontal shift glitch
        gy = random.randint(0, h-20)
        gh = random.randint(5, 15)
        ghost_layer[gy:gy+gh] = np.roll(ghost_layer[gy:gy+gh], random.randint(-40, 40), axis=1)

    # 5. Combine with background
    canvas = cv2.add(canvas, ghost_layer)
    canvas = cv2.addWeighted(canvas, 1.0, scan_layer, 0.3, 0)

    return canvas
