"""
Matrix Filter — green code rain flowing ONLY inside the body outline.
Background is solid black. Stick figure is hidden.
"""
import cv2
import numpy as np
import random
import string
from pose_detector import PoseResult, CONNECTIONS

# Column state: [y position, speed, char_list]
_columns: dict[int, list] = {}
_FONT = cv2.FONT_HERSHEY_SIMPLEX
_CHARS = string.ascii_letters + string.digits + "@#$%&"
_CHAR_SIZE = 8  # Smaller size = higher density columns
_INIT_DONE = False
VIS_THRESHOLD = 0.4

def _init_columns(w, h):
    global _columns, _INIT_DONE
    cols = w // _CHAR_SIZE
    for c in range(cols):
        # [y_pos, speed]
        _columns[c] = [random.randint(-h, 0), random.randint(3, 7)]
    _INIT_DONE = True

def apply(canvas: np.ndarray, pose: PoseResult, **kwargs) -> np.ndarray:
    global _INIT_DONE, _columns
    h, w = canvas.shape[:2]

    if not _INIT_DONE:
        _init_columns(w, h)

    # 1. Create Body Mask
    mask = np.zeros((h, w), dtype=np.uint8)
    if pose.detected:
        lm = pose.landmarks
        vis = pose.visibility
        
        for (a, b) in CONNECTIONS:
            if a not in lm or b not in lm:
                continue
            if vis.get(a, 0) < VIS_THRESHOLD or vis.get(b, 0) < VIS_THRESHOLD:
                continue
            # Thick lines to fill the body volume
            cv2.line(mask, lm[a], lm[b], 255, 35, cv2.LINE_AA)

    # 2. Generate Matrix Rain on a separate layer
    rain_layer = np.zeros_like(canvas)
    
    for col_idx, state in _columns.items():
        x = col_idx * _CHAR_SIZE + 4
        y = int(state[0])
        
        # Draw the falling characters
        # Bright leading character
        char = random.choice(_CHARS)
        cv2.putText(rain_layer, char, (x, y), _FONT, 0.25, (180, 255, 180), 1, cv2.LINE_AA)
        
        # Trail - longer for higher concentration
        for t in range(1, 18):
            ty = y - t * _CHAR_SIZE
            if 0 < ty < h:
                alpha = max(0, 255 - t * 14)
                cv2.putText(rain_layer, random.choice(_CHARS), (x, ty), _FONT, 0.25,
                            (0, alpha, 0), 1, cv2.LINE_AA)

        # Update position
        state[0] += state[1]
        if state[0] > h + (_CHAR_SIZE * 12):
            state[0] = random.randint(-100, 0)
            state[1] = random.randint(3, 7)

    # 3. Apply Mask: Only keep rain where the body is
    # Convert mask to 3 channels
    mask_3ch = cv2.merge([mask, mask, mask])
    
    # Use bitwise_and to cut the rain to the body shape
    canvas[:] = cv2.bitwise_and(rain_layer, mask_3ch)

    return canvas
