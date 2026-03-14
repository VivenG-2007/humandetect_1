import math
import time
import cv2
import numpy as np

class OneEuroFilter:
    def __init__(self, freq=30, min_cutoff=1.0, beta=0.0, d_cutoff=1.0):
        self.freq = freq
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self.x_prev = None
        self.dx_prev = None

    def _low_pass_filter(self, x, alpha, x_prev):
        return alpha * x + (1.0 - alpha) * x_prev

    def _alpha(self, cutoff):
        tau = 1.0 / (2 * math.pi * cutoff)
        te = 1.0 / self.freq
        return 1.0 / (1.0 + tau / te)

    def __call__(self, x, freq=None):
        if freq is not None:
            self.freq = freq
        
        if self.x_prev is None:
            self.x_prev = x
            self.dx_prev = 0.0
            return x

        dx = (x - self.x_prev) * self.freq
        edx = self._low_pass_filter(dx, self._alpha(self.d_cutoff), self.dx_prev)
        cutoff = self.min_cutoff + self.beta * abs(edx)
        alpha = self._alpha(cutoff)
        result = self._low_pass_filter(x, alpha, self.x_prev)
        
        self.x_prev = result
        self.dx_prev = edx
        return result

class PointSmoothing:
    """Helper to smooth 2D points (x, y) using OneEuroFilter."""
    def __init__(self, min_cutoff=0.8, beta=0.015):
        self.x_filter = OneEuroFilter(min_cutoff=min_cutoff, beta=beta)
        self.y_filter = OneEuroFilter(min_cutoff=min_cutoff, beta=beta)

    def __call__(self, x, y):
        return self.x_filter(x), self.y_filter(y)

class MaskSmoothing:
    """Temporal smoothing for segmentation masks using EMA."""
    def __init__(self, alpha=0.35):
        self.alpha = alpha
        self.prev_mask = None

    def __call__(self, mask):
        if mask is None:
            return None
        if self.prev_mask is None or self.prev_mask.shape != mask.shape:
            self.prev_mask = mask.astype(np.float32)
            return mask
        
        # Apply EMA: new = prev * (1-a) + current * a
        self.prev_mask = cv2.addWeighted(self.prev_mask, 1.0 - self.alpha, 
                                          mask.astype(np.float32), self.alpha, 0)
        return self.prev_mask
