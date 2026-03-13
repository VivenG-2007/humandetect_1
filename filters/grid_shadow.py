"""
Space-Time Warp Filter — Heavily optimized for speed.
Key changes:
  - Batched vectorized warp (all keypoints at once, no Python loop over grid)
  - Removed adaptiveThreshold (was the #1 bottleneck)
  - Cached grid coordinates (recomputed only on resolution change)
  - Eliminated repeated canvas float32 conversions
  - Pre-sliced polyline arrays to avoid per-row np.stack
"""
import cv2
import numpy as np
import random
from pose_detector import PoseResult, CONNECTIONS

VIS_THRESHOLD = 0.25

# --- Module-level cache ---
_stars: list = []
_grid_cache: dict = {}   # keyed by (w, h) → (warped_x_base, warped_y_base, wx_cols, wy_rows)
_KEY_INDICES = [0, 11, 12, 15, 16, 23, 24, 27, 28]
_GRID_GAP = 40
_INFLUENCE_RADIUS = 150.0
_GLOW_COLOR = np.array([[[255, 100, 50]]], dtype=np.float32)   # shape (1,1,3) for broadcasting
_GRID_COLOR = (150, 80, 40)
_SKEL_COLOR = (255, 255, 255)


def _init_stars(w: int, h: int) -> None:
    global _stars
    _stars = [
        (random.randint(0, w), random.randint(0, h), random.uniform(1, 2))
        for _ in range(60)
    ]


def _build_grid(w: int, h: int):
    """Build and cache base grid arrays for a given resolution."""
    key = (w, h)
    if key in _grid_cache:
        return _grid_cache[key]

    cols = (w // _GRID_GAP) + 2
    rows = (h // _GRID_GAP) + 2
    gx, gy = np.meshgrid(
        np.arange(cols) * _GRID_GAP - _GRID_GAP / 2,
        np.arange(rows) * _GRID_GAP - _GRID_GAP / 2,
    )
    base_x = gx.astype(np.float32)
    base_y = gy.astype(np.float32)

    # Pre-build index arrays for fast polyline slicing (rows × cols × 2)
    _grid_cache[key] = (base_x, base_y, rows, cols)
    return _grid_cache[key]


def apply(canvas: np.ndarray, pose: PoseResult, **kwargs) -> np.ndarray:
    h, w = canvas.shape[:2]

    # --- Stars ---
    if not _stars:
        _init_stars(w, h)
    for sx, sy, sz in _stars:
        cv2.circle(canvas, (sx, sy), int(sz), (200, 200, 200), -1)

    # --- Grid setup (cached) ---
    base_x, base_y, rows, cols = _build_grid(w, h)
    warped_x = base_x.copy()
    warped_y = base_y.copy()

    # --- Batched warp: collect all visible keypoints, then apply in one pass ---
    if pose.detected:
        lm, vis = pose.landmarks, pose.visibility

        kp_xy = np.array(
            [lm[idx] for idx in _KEY_INDICES
             if idx in lm and vis.get(idx, 0) >= VIS_THRESHOLD],
            dtype=np.float32,
        )  # shape (K, 2)

        if len(kp_xy):
            # Expand dims for broadcasting: grid (R,C) vs keypoints (K,)
            # dx[k, r, c] = warped_x[r,c] - kp_xy[k,0]
            dx = warped_x[np.newaxis] - kp_xy[:, 0, np.newaxis, np.newaxis]  # (K,R,C)
            dy = warped_y[np.newaxis] - kp_xy[:, 1, np.newaxis, np.newaxis]  # (K,R,C)
            dist = np.sqrt(dx * dx + dy * dy)                                  # (K,R,C)

            # Clamp & compute strength where within radius
            ratio = np.clip(1.0 - dist / _INFLUENCE_RADIUS, 0.0, None)        # (K,R,C)
            strength = ratio * ratio * 0.4                                      # (K,R,C)

            # Sum displacement from all keypoints
            warped_x -= np.sum(dx * strength, axis=0)
            warped_y -= np.sum(dy * strength, axis=0)

    # --- Draw grid lines ---
    pts_xy = np.stack((warped_x, warped_y), axis=-1).astype(np.int32)  # (R,C,2)

    # cv2.polylines requires contiguous arrays of shape (N,1,2)
    for r in range(rows):
        cv2.polylines(canvas, [np.ascontiguousarray(pts_xy[r, :, np.newaxis, :])],
                      False, _GRID_COLOR, 1, cv2.LINE_AA)
    for c in range(cols):
        cv2.polylines(canvas, [np.ascontiguousarray(pts_xy[:, c, np.newaxis, :])],
                      False, _GRID_COLOR, 1, cv2.LINE_AA)

    # --- Body glow + skeleton ---
    if pose.detected:
        lm, vis = pose.landmarks, pose.visibility

        # Build thick body mask
        mask = np.zeros((h, w), dtype=np.uint8)
        for (a, b) in CONNECTIONS:
            if (a in lm and b in lm
                    and vis.get(a, 0) > VIS_THRESHOLD
                    and vis.get(b, 0) > VIS_THRESHOLD):
                cv2.line(mask, lm[a], lm[b], 255, 30, cv2.LINE_AA)

        # Blur on downscaled mask (fast)
        scale = 4
        small = cv2.resize(mask, (w // scale, h // scale), interpolation=cv2.INTER_NEAREST)
        small = cv2.GaussianBlur(small, (15, 15), 0)
        mask_blur = cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)

        # Glow blend — single float32 pass
        alpha = mask_blur * (0.6 / 255.0)                              # (H,W) float32
        alpha3 = alpha[:, :, np.newaxis]                                # (H,W,1)
        canvas_f = canvas.astype(np.float32)
        np.multiply(canvas_f, 1.0 - alpha3, out=canvas_f)
        canvas_f += _GLOW_COLOR * alpha3
        np.clip(canvas_f, 0, 255, out=canvas_f)
        canvas = canvas_f.astype(np.uint8)

        # Ghost from original frame — simple threshold instead of adaptiveThreshold
        original_frame = kwargs.get('original_frame')
        if original_frame is not None:
            gray = cv2.cvtColor(original_frame, cv2.COLOR_BGR2GRAY)
            _, ghost = cv2.threshold(gray, 60, 255, cv2.THRESH_BINARY_INV)
            ghost_3ch = cv2.cvtColor(ghost, cv2.COLOR_GRAY2BGR)
            body_ghost = cv2.bitwise_and(ghost_3ch, ghost_3ch, mask=mask)
            # Cyan tint: zero out red channel in-place
            body_ghost[:, :, 2] = (body_ghost[:, :, 2].astype(np.uint16) * 40 // 255).astype(np.uint8)
            cv2.addWeighted(canvas, 1.0, body_ghost, 0.5, 0, dst=canvas)

        # Clean neon skeleton
        for (a, b) in CONNECTIONS:
            if (a in lm and b in lm
                    and vis.get(a, 0) > VIS_THRESHOLD
                    and vis.get(b, 0) > VIS_THRESHOLD):
                cv2.line(canvas, lm[a], lm[b], _SKEL_COLOR, 2, cv2.LINE_AA)

    return canvas