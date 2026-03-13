"""
Pose Detector — processes video frames to locate human skeletal landmarks.
Uses MediaPipe Pose Landmarker for body tracking and MediaPipe Hands for fine finger detail.
Optimized for low-latency concurrent processing.
"""
import mediapipe as mp
import numpy as np
import cv2
import time
from typing import Dict, Optional, Tuple, List
from dataclasses import dataclass
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision import PoseLandmarkerOptions, RunningMode

# Explicit landmark IDs for key joints
# 0-32: standard MediaPipe Pose
# 33-38: Extrapolated Fingertips (Pinky, Index, Thumb)
# 40-43: Extrapolated Middle/Ring
# 39: Top of head (crown)
LANDMARK_NAMES = list(range(33)) + list(range(33, 39)) + list(range(40, 44))

CONNECTIONS = [
    (11, 12), (11, 13), (13, 15), (12, 14), (14, 16), # Shoulders / Arms
    (11, 23), (12, 24), (23, 24),                   # Torso
    (23, 25), (25, 27), (24, 26), (26, 28),         # Legs
    (27, 29), (27, 31), (28, 30), (28, 32),         # Feet
    # Fingertips (Extended connections)
    (15, 33), (16, 34), # Pinkies
    (15, 35), (16, 36), # Indices
    (15, 37), (16, 38), # Thumbs
    (15, 40), (16, 41), # Middle
    (15, 42), (16, 43)  # Ring
]

# Static Mapping for Hand Overrides
_HAND_MAPPINGS = {
    4: (37, 38),   # Thumb tip
    8: (35, 36),   # Index tip
    12: (40, 41),  # Middle tip
    16: (42, 43),  # Ring tip
    20: (33, 34),  # Pinky tip
    2: (21, 22),   # Thumb base
    5: (19, 20),   # Index base
    17: (17, 18),  # Pinky base
}

@dataclass
class PoseResult:
    landmarks: Dict[int, Tuple[int, int]]
    visibility: Dict[int, float]
    segmentation_mask: Optional[np.ndarray]
    detected: bool
    mp_result: Optional[vision.PoseLandmarkerResult]

class PoseDetector:
    def __init__(
        self,
        model_path: str = 'pose_landmarker_lite.task',
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
        smoothing_factor: float = 0.45
    ):
        self.smoothing_factor = smoothing_factor
        self.prev_landmarks: Dict[int, Tuple[float, float]] = {}
        
        # Base Pose Landmarker Options
        options = vision.PoseLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=model_path),
            running_mode=vision.RunningMode.VIDEO,
            min_pose_detection_confidence=min_detection_confidence,
            min_pose_presence_confidence=min_tracking_confidence,
            min_tracking_confidence=min_tracking_confidence,
            output_segmentation_masks=True
        )
        self.landmarker = vision.PoseLandmarker.create_from_options(options)
        
        # Advanced Multi-Hand Tracker for fine gestures
        self.hand_tracker = mp.solutions.hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )

    def detect(self, frame: np.ndarray, timestamp_ms: int) -> PoseResult:
        # 1. Run Pose Detect (Requires MP Image format)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
        pose_result = self.landmarker.detect_for_video(mp_image, timestamp_ms)
        
        # 2. Run Hand Tracking concurrently on same frame
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        hand_result = self.hand_tracker.process(rgb_frame)
        
        h, w = frame.shape[:2]
        return self._process_result(pose_result, h, w, hand_result)

    def _process_result(self, result: vision.PoseLandmarkerResult, h: int, w: int, hand_result=None) -> PoseResult:
        if not result.pose_landmarks:
            return PoseResult({}, {}, None, False, None)

        lm_list = result.pose_landmarks[0]
        landmarks: Dict[int, Tuple[int, int]] = {}
        visibility: Dict[int, float] = {}
        
        # Faster dictionary building with smoothing
        sf = self.smoothing_factor
        isf = 1.0 - sf
        
        for idx in range(min(33, len(lm_list))):
            lm = lm_list[idx]
            cx, cy = lm.x * w, lm.y * h
            
            if idx in self.prev_landmarks:
                px, py = self.prev_landmarks[idx]
                cx = px * isf + cx * sf
                cy = py * isf + cy * sf
            
            self.prev_landmarks[idx] = (cx, cy)
            landmarks[idx] = (int(cx), int(cy))
            visibility[idx] = lm.visibility


        # Fingertip fallbacks (Pinky, Index, Thumb)
        # Side-agnostic configs: (tip, base, wrist, elbow, scale)
        f_cfg = [
            (33, 17, 15, 13, 1.8), # Left Pinky
            (34, 18, 16, 14, 1.8), # Right Pinky
            (35, 19, 15, 13, 2.1), # Left Index
            (36, 20, 16, 14, 2.1), # Right Index
            (37, 21, 15, 13, 1.5), # Left Thumb
            (38, 22, 16, 14, 1.5)  # Right Thumb
        ]
        for tip, base, wrist, elbow, scale in f_cfg:
            if wrist in landmarks:
                wx, wy = landmarks[wrist]
                # Primary: Use actual palm landmark if available
                if base in landmarks:
                    dx, dy = landmarks[base][0] - wx, landmarks[base][1] - wy
                # Fallback: Project from wrist using forearm vector
                elif elbow in landmarks:
                    dx, dy = (wx - landmarks[elbow][0]) * 0.25, (wy - landmarks[elbow][1]) * 0.25
                else: continue
                
                landmarks[tip] = (int(wx + dx * scale), int(wy + dy * scale))
                visibility[tip] = visibility.get(base, visibility[wrist])

        # Middle/Ring interpolation (Always derive from Index and Pinky)
        for s in [0, 1]: # Left, Right
            i_tip, p_tip, m_tip, r_tip = (35+s, 33+s, 40+s, 42+s)
            if i_tip in landmarks and p_tip in landmarks:
                ix, iy = landmarks[i_tip]; px, py = landmarks[p_tip]
                landmarks[m_tip] = (int(ix + (px-ix)*0.35), int(iy + (py-iy)*0.35))
                landmarks[r_tip] = (int(ix + (px-ix)*0.65), int(iy + (py-iy)*0.65))
                visibility[m_tip] = visibility.get(i_tip, 0.5)
                visibility[r_tip] = visibility.get(p_tip, 0.5)

        # --- Override with fine hand tracking ---
        if hand_result and hand_result.multi_hand_landmarks:
            for hlms in hand_result.multi_hand_landmarks:
                h_base = hlms.landmark[0]
                hx, hy = h_base.x * w, h_base.y * h
                
                # Fast distance check for hand assignment
                dl = ((landmarks[15][0]-hx)**2 + (landmarks[15][1]-hy)**2) if 15 in landmarks else 1e9
                dr = ((landmarks[16][0]-hx)**2 + (landmarks[16][1]-hy)**2) if 16 in landmarks else 1e9
                
                side_idx = 0 if (dl < dr and dl < 22500) else 1 if (dr < dl and dr < 22500) else -1
                if side_idx != -1:
                    for hd_idx, p_indices in _HAND_MAPPINGS.items():
                        target = p_indices[side_idx]
                        landmarks[target] = (int(hlms.landmark[hd_idx].x * w), int(hlms.landmark[hd_idx].y * h))
                        visibility[target] = 1.0

        mask = result.segmentation_masks[0].numpy_view() if result.segmentation_masks else None
        return PoseResult(landmarks, visibility, mask, True, result)

    def close(self):
        self.landmarker.close()
        self.hand_tracker.close()