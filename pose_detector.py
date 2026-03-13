"""
Pose Detector — wraps MediaPipe Pose to detect body landmarks.
Returns landmark positions in pixel coordinates for downstream use.

Compatible with both mediapipe 0.9.x and 0.10.x+
"""
import cv2
import numpy as np
from dataclasses import dataclass
from typing import Optional, Dict, Tuple
import time
import os

try:
    import mediapipe as mp
    from mediapipe.tasks import python
    from mediapipe.tasks.python import vision
except ImportError as e:
    print(f"CRITICAL: MediaPipe Task API not found. Error: {e}")
    raise e

# ── Landmark indices we care about ────────────────────────────────────────
LANDMARK_NAMES = {
    0:  "nose",
    2:  "left_eye",
    5:  "right_eye",
    7:  "left_ear",
    8:  "right_ear",
    11: "left_shoulder",
    12: "right_shoulder",
    13: "left_elbow",
    14: "right_elbow",
    15: "left_wrist",
    16: "right_wrist",
    17: "left_pinky",
    18: "right_pinky",
    19: "left_index",
    20: "right_index",
    21: "left_thumb",
    22: "right_thumb",
    23: "left_hip",
    24: "right_hip",
    25: "left_knee",
    26: "right_knee",
    27: "left_ankle",
    28: "right_ankle",
    29: "left_heel",
    30: "right_heel",
    31: "left_foot_index",
    32: "right_foot_index",
}

# Skeleton connections (pairs of landmark indices)
CONNECTIONS = [
    # Torso
    (11, 12), (11, 23), (12, 24), (23, 24),
    # Left arm & hand
    (11, 13), (13, 15),
    (15, 17), (15, 19), (15, 21),  # L wrist to pinky, index, thumb
    # Right arm & hand
    (12, 14), (14, 16),
    (16, 18), (16, 20), (16, 22),  # R wrist to pinky, index, thumb
    # Left leg & foot
    (23, 25), (25, 27),
    (27, 29), (27, 31),  # L ankle to heel, foot index
    # Right leg & foot
    (24, 26), (26, 28),
    (28, 30), (28, 32),  # R ankle to heel, foot index
    # Head to shoulders
    (0, 11), (0, 12),
]


@dataclass
class PoseResult:
    landmarks: Dict[int, Tuple[int, int]]   # {index: (x_px, y_px)}
    visibility: Dict[int, float]             # {index: confidence 0–1}
    raw_landmarks: object                    # mp NormalizedLandmarkList
    detected: bool
    segmentation_mask: Optional[np.ndarray] = None


class PoseDetector:
    def __init__(
        self,
        model_path: str = "pose_landmarker_lite.task",
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
        smoothing_factor: float = 0.4,
    ):
        self.smoothing_factor = smoothing_factor
        self.prev_landmarks: Dict[int, Tuple[float, float]] = {}
        self.latest_result = PoseResult({}, {}, None, False, None)
        
        # Initialize MediaPipe Task
        model_abs_path = model_path
        if not os.path.isabs(model_path):
            model_abs_path = os.path.join(os.path.dirname(__file__), model_path)
            
        base_options = python.BaseOptions(model_asset_path=model_abs_path)
        options = vision.PoseLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.VIDEO,
            num_poses=1,
            min_pose_detection_confidence=min_detection_confidence,
            min_pose_presence_confidence=min_tracking_confidence,
            min_tracking_confidence=min_tracking_confidence,
            output_segmentation_masks=True
        )
        self.landmarker = vision.PoseLandmarker.create_from_options(options)
        self.frame_count = 0
        self.start_time_ms = int(time.time() * 1000)

    def _process_result(self, result: vision.PoseLandmarkerResult, h: int, w: int) -> PoseResult:
        if not result.pose_landmarks:
            return PoseResult({}, {}, None, False, None)

        lm_list = result.pose_landmarks[0] # Single pose
        
        landmarks: Dict[int, Tuple[int, int]] = {}
        visibility: Dict[int, float] = {}

        for idx in LANDMARK_NAMES:
            if idx >= len(lm_list): continue
            lm = lm_list[idx]
            curr_x, curr_y = lm.x * w, lm.y * h
            
            # Smoothing
            if idx in self.prev_landmarks:
                prev_x, prev_y = self.prev_landmarks[idx]
                smooth_x = prev_x + self.smoothing_factor * (curr_x - prev_x)
                smooth_y = prev_y + self.smoothing_factor * (curr_y - prev_y)
            else:
                smooth_x, smooth_y = curr_x, curr_y
            
            self.prev_landmarks[idx] = (smooth_x, smooth_y)
            landmarks[idx] = (int(smooth_x), int(smooth_y))
            visibility[idx] = lm.visibility

        mask = None
        if result.segmentation_masks:
            # The mask is an mp.Image. Use .numpy_view()
            mask = result.segmentation_masks[0].numpy_view()

        return PoseResult(landmarks, visibility, lm_list, True, mask)

    def detect(self, bgr_frame: np.ndarray) -> PoseResult:
        # Convert BGR to RGB and then to mp.Image
        rgb = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        
        # Use monotonic time in ms for the timestamp
        timestamp = int(time.time() * 1000)
        # Ensure timestamp is strictly increasing for VIDEO mode
        if timestamp <= self.start_time_ms:
            timestamp = self.start_time_ms + 1
        self.start_time_ms = timestamp
        
        # Sync call - blocks until done but guarantees perfect tracking lock with frame
        result = self.landmarker.detect_for_video(mp_image, timestamp)
        
        # Process and return immediately
        return self._process_result(result, rgb.shape[0], rgb.shape[1])
    def close(self):
        if self.landmarker:
            self.landmarker.close()