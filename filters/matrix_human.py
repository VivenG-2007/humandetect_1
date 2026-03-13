"""
Matrix Human Filter — Digital rain effect that intensifies on the human body.
Uses segmentation mask to differentiate between background and subject.
"""
import cv2
import numpy as np
import random
import string
import time

class MatrixRain:
    def __init__(self, w, h):
        self.w = w
        self.h = h
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.chars = string.ascii_letters + string.digits + "!@#$%^&*"
        self.char_size = 6 # EVEN SMALLER size = max columns = max density
        self.num_cols = (w // self.char_size) * 2 # DOUBLE density
        
        # Column state: list of [head_y, speed, length, characters]
        self.cols = []
        for _ in range(self.num_cols):
            self.cols.append(self._reset_col())

    def _reset_col(self):
        return [
            random.randint(-self.h, 0),         # head_y
            random.randint(6, 16),              # speed
            random.randint(20, 50),             # trail length (extremely long trails)
            [random.choice(self.chars) for _ in range(55)] # pre-gen chars
        ]

    def draw(self, canvas, mask):
        h, w = canvas.shape[:2]
        
        for i in range(self.num_cols):
            # Interleave odd columns to create extreme density
            x_base = (i // 2) * self.char_size
            offset = (self.char_size // 2) if i % 2 == 1 else 0
            x = x_base + offset + 1
            
            head_y, speed, length, chars = self.cols[i]
            
            mask_col = mask[:, min(x, w-1)] if mask is not None else None
            
            for j in range(length):
                y = int(head_y - j * self.char_size)
                if y < 0 or y >= h:
                    continue
                
                # Check if this specific point is on the person
                is_person = False
                if mask_col is not None:
                    if mask_col[y] > 0.3: # Even lower threshold to wrap tightly
                        is_person = True
                
                # Brightness and Color logic
                if j == 0: # Head character
                    color = (255, 255, 255) if is_person else (180, 255, 180)
                    scale = 0.3 if is_person else 0.2
                    thickness = 2 if is_person else 1
                else:
                    # Fade out the trail
                    alpha = max(0, 255 - int(j * (255 / length)))
                    if is_person:
                        color = (0, 255, 120) # Bright neon green
                        color = (int(color[0] * (alpha/255)), int(color[1] * (alpha/255)), int(color[2] * (alpha/255)))
                        scale = 0.25 # Slightly smaller to prevent absolute clutter at max density
                        thickness = 1
                    else:
                        color = (0, int(60 * (alpha/255)), 0) # Very faint background
                        scale = 0.2
                        thickness = 1
                
                char = chars[j % len(chars)]
                cv2.putText(canvas, char, (x, y), self.font, scale, color, thickness, cv2.LINE_AA)

            # Update state
            self.cols[i][0] += speed
            if self.cols[i][0] > h + (length * self.char_size):
                self.cols[i] = self._reset_col()
            
            # Randomly change characters in trail
            if random.random() < 0.1:
                self.cols[i][3][random.randint(0, 29)] = random.choice(self.chars)

_rain_inst = None

def apply(canvas: np.ndarray, pose, **kwargs) -> np.ndarray:
    global _rain_inst
    h, w = canvas.shape[:2]
    
    if _rain_inst is None or _rain_inst.w != w or _rain_inst.h != h:
        _rain_inst = MatrixRain(w, h)
    
    # We start with the canvas passed to us (which has PIP and is mostly black)
    # Removing canvas.fill(0) to preserve the PIP background    
    mask = pose.segmentation_mask if pose.detected else None
    
    _rain_inst.draw(canvas, mask)
    
    return canvas
