"""
Camera Thread — captures frames from webcam in a dedicated QThread.
Emits a Qt signal for each new frame so the UI can update safely.
"""
import cv2
import numpy as np
from typing import Optional
from PyQt5.QtCore import QThread, pyqtSignal


from pose_detector import PoseDetector, PoseResult

class CameraThread(QThread):
    # Emits (BGR frame, PoseResult)
    frame_ready = pyqtSignal(np.ndarray, object)

    def __init__(self, camera_index: int = 0, width: int = 640, height: int = 480, detector: PoseDetector = None):
        super().__init__()
        self.camera_index = camera_index
        self.width = width
        self.height = height
        self._running = False
        self.cap: Optional[cv2.VideoCapture] = None
        self.detector = detector

    # ------------------------------------------------------------------ #
    def run(self):
        self._running = True
        # Using CAP_DSHOW for faster startup on Windows
        self.cap = cv2.VideoCapture(self.camera_index, cv2.CAP_DSHOW)
        
        # Optimize capture settings
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1) # Reduce latency

        while self._running:
            ret, frame = self.cap.read()
            if ret:
                # Mirror the frame
                frame = cv2.flip(frame, 1)
                
                # Run detection in the background thread
                pose = None
                if self.detector:
                    pose = self.detector.detect(frame)
                
                self.frame_ready.emit(frame, pose)

        if self.cap:
            self.cap.release()

    # ------------------------------------------------------------------ #
    def stop(self):
        self._running = False
        self.wait()
