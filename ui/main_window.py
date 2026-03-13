"""
Main Window — PyQt5 GUI for the Camera Stick-Figure Filter App.

Features:
  - Live camera preview (640×480)
  - Filter selector buttons sidebar
  - FPS counter overlay
  - Screenshot button
  - Fullscreen toggle
  - Dark themed UI
"""
import sys
import os
import time
import cv2
import numpy as np
from datetime import datetime

from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QHBoxLayout, QVBoxLayout,
    QPushButton, QLabel, QSizePolicy, QFrame, QApplication, QScrollArea
)
from PyQt5.QtCore import Qt, pyqtSlot, QSize
from PyQt5.QtGui import QImage, QPixmap, QFont, QColor, QPalette

# Project imports
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from camera import CameraThread
from pose_detector import PoseDetector
from skeleton_renderer import SkeletonRenderer
from filters import FILTER_REGISTRY, FILTER_NAMES

# ──────────────────────────────────────────────────────────────────────────────
# Style constants
# ──────────────────────────────────────────────────────────────────────────────
BG_COLOR       = "#08080c"
SIDEBAR_BG     = "rgba(20, 20, 28, 0.85)"
ACCENT         = "#00f2ff"  # Cyan/Neon
ACCENT_MUTED   = "rgba(0, 242, 255, 0.15)"
TEXT_COLOR     = "#f0f0f5"
SIDEBAR_WIDTH  = 200

STYLESHEET = f"""
QMainWindow, QWidget {{
    background-color: {BG_COLOR};
    color: {TEXT_COLOR};
    font-family: 'Segoe UI', 'Roboto', sans-serif;
}}
#sidebar {{
    background-color: {SIDEBAR_BG};
    border-right: 1px solid rgba(255, 255, 255, 0.1);
    min-width: {SIDEBAR_WIDTH}px;
    max-width: {SIDEBAR_WIDTH}px;
}}
QPushButton.filterBtn {{
    background-color: rgba(255, 255, 255, 0.05);
    color: {TEXT_COLOR};
    border: 1px solid rgba(255, 255, 255, 0.1);
    border-radius: 12px;
    padding: 12px 10px;
    font-size: 14px;
    margin: 5px 15px;
    text-align: left;
    padding-left: 20px;
}}
QPushButton.filterBtn:hover {{
    background-color: {ACCENT_MUTED};
    border-color: {ACCENT};
}}
QPushButton.filterBtn[active="true"] {{
    background-color: {ACCENT};
    color: #000;
    font-weight: bold;
    border: none;
}}
QPushButton#screenshotBtn {{
    background-color: rgba(0, 255, 127, 0.1);
    color: #00ff7f;
    border: 1px solid #00ff7f;
    border-radius: 12px;
    padding: 12px;
    font-size: 14px;
    margin: 10px 15px;
    font-weight: bold;
}}
QPushButton#screenshotBtn:hover {{
    background-color: #00ff7f;
    color: #000;
}}
QPushButton#fullscreenBtn {{
    background-color: rgba(124, 58, 237, 0.1);
    color: #a78bfa;
    border: 1px solid #7c3aed;
    border-radius: 12px;
    padding: 12px;
    font-size: 14px;
    margin: 5px 15px;
}}
QPushButton#fullscreenBtn:hover {{
    background-color: #7c3aed;
    color: white;
}}
#titleLabel {{
    color: {ACCENT};
    font-size: 20px;
    font-weight: 800;
    padding: 30px 20px 10px 20px;
    letter-spacing: 2px;
}}
#fpsLabel {{
    color: {ACCENT};
    font-size: 14px;
    font-weight: bold;
    padding: 10px 20px;
}}
#statusLabel {{
    color: #8a8a9a;
    font-size: 12px;
    padding: 10px 20px 20px 20px;
}}
#videoLabel {{
    background-color: #000;
    border-radius: 0px;
}}
QScrollArea {{
    border: none;
    background: transparent;
}}
#scrollContent {{
    background: transparent;
}}
QScrollBar:vertical {{
    border: none;
    background: rgba(255, 255, 255, 0.05);
    width: 6px;
    margin: 0px;
}}
QScrollBar::handle:vertical {{
    background: rgba(255, 255, 255, 0.2);
    min-height: 30px;
    border-radius: 3px;
}}
QScrollBar::handle:vertical:hover {{
    background: rgba(0, 242, 255, 0.5);
}}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
    border: none;
    background: none;
    height: 0px;
}}
"""


# ──────────────────────────────────────────────────────────────────────────────
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("VISION X - Human Sensing")
        self.resize(1100, 700)
        self.setStyleSheet(STYLESHEET)

        # State
        self.active_filter = "Default"
        self._filter_buttons: dict[str, QPushButton] = {}
        self._fps_times: list[float] = []
        self._is_fullscreen = False

        # Core components
        self.pose_detector = PoseDetector(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
            smoothing_factor=0.4 # Lower = smoother but slight lag. 0.25 is very buttery.
        )
        self.base_renderer = SkeletonRenderer(
            line_color=(0, 255, 255),    # Yellow
            joint_color=(220, 220, 220),  # Grey
            torso_color=(40, 160, 40),   # Green
            line_thickness=6,
            joint_radius=5,
        )

        self._build_ui()
        self._start_camera()

    # ── UI Construction ────────────────────────────────────────────────────────
    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root_layout = QHBoxLayout(central)
        root_layout.setContentsMargins(0, 0, 0, 0)
        root_layout.setSpacing(0)

        # ── Sidebar ──
        sidebar = QFrame()
        sidebar.setObjectName("sidebar")
        sidebar_layout = QVBoxLayout(sidebar)
        sidebar_layout.setContentsMargins(0, 0, 0, 0)
        sidebar_layout.setSpacing(0)

        title = QLabel("VISION X")
        title.setObjectName("titleLabel")
        sidebar_layout.addWidget(title)

        subtitle = QLabel("AI SENSING CORE")
        subtitle.setStyleSheet("color: rgba(255,255,255,0.4); font-size: 9px; padding: 0 20px 20px 20px; letter-spacing: 3px;")
        sidebar_layout.addWidget(subtitle)

        # ── Scroll Area for Filters ──
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        
        scroll_content = QWidget()
        scroll_content.setObjectName("scrollContent")
        scroll_layout = QVBoxLayout(scroll_content)
        scroll_layout.setContentsMargins(0, 0, 0, 0)
        scroll_layout.setSpacing(0)

        # Filter buttons
        for name in FILTER_NAMES:
            btn = QPushButton(name)
            btn.setProperty("class", "filterBtn")
            btn.setCheckable(False)
            btn.setCursor(Qt.PointingHandCursor)
            btn.clicked.connect(lambda _, n=name: self._set_filter(n))
            self._filter_buttons[name] = btn
            scroll_layout.addWidget(btn)

        scroll_layout.addStretch()
        scroll.setWidget(scroll_content)
        sidebar_layout.addWidget(scroll)

        # ── Bottom Section ──
        bottom_frame = QFrame()
        bottom_layout = QVBoxLayout(bottom_frame)
        bottom_layout.setContentsMargins(0, 10, 0, 15)

        self.fps_label = QLabel("FPS: --")
        self.fps_label.setObjectName("fpsLabel")
        bottom_layout.addWidget(self.fps_label)

        self.status_label = QLabel("Starting camera…")
        self.status_label.setObjectName("statusLabel")
        self.status_label.setWordWrap(True)
        bottom_layout.addWidget(self.status_label)

        # Screenshot
        ss_btn = QPushButton("📸 Screenshot")
        ss_btn.setObjectName("screenshotBtn")
        ss_btn.setCursor(Qt.PointingHandCursor)
        ss_btn.clicked.connect(self._take_screenshot)
        bottom_layout.addWidget(ss_btn)

        # Fullscreen toggle
        fs_btn = QPushButton("⛶ Fullscreen")
        fs_btn.setObjectName("fullscreenBtn")
        fs_btn.setCursor(Qt.PointingHandCursor)
        fs_btn.clicked.connect(self._toggle_fullscreen)
        bottom_layout.addWidget(fs_btn)

        sidebar_layout.addWidget(bottom_frame)

        root_layout.addWidget(sidebar)

        # ── Video display ──
        self.video_label = QLabel()
        self.video_label.setObjectName("videoLabel")
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.video_label.setMinimumSize(1, 1)  # Prevent label from pushing layout boundaries
        root_layout.addWidget(self.video_label)

        # Mark default filter as active
        self._update_button_states()

    # ── Camera ────────────────────────────────────────────────────────────────
    def _start_camera(self):
        self.camera = CameraThread(
            camera_index=0, 
            width=640, 
            height=480, 
            detector=self.pose_detector
        )
        self.camera.frame_ready.connect(self._on_frame)
        self.camera.start()
        self.status_label.setText("Camera active")

    # ── Frame Processing ───────────────────────────────────────────────────────
    @pyqtSlot(np.ndarray, object)
    def _on_frame(self, bgr_frame: np.ndarray, pose):
        t0 = time.perf_counter()

        # Pose is now pre-detected in the background thread
        if pose is None:
            return

        h, w = bgr_frame.shape[:2]
        
        # Determine background
        filter_mod = FILTER_REGISTRY.get(self.active_filter)
        if filter_mod is None:
            # Default: Show camera feed with skeleton
            canvas = bgr_frame.copy()
            self.base_renderer.render(pose, (h, w), canvas)
        else:
            # Creative filters: Start with black or their custom internal logic
            canvas = np.zeros((h, w, 3), dtype=np.uint8)
            
            # ── PIP: Original Feed Overlay (Top Left) ──
            # We draw this BEFORE the filter so the filter (e.g. wings) can go ABOVE the UI
            pip_w = w // 4
            pip_h = int(pip_w * (h / w))
            pip_small = cv2.resize(bgr_frame, (pip_w, pip_h), interpolation=cv2.INTER_AREA)
            
            # Position (with margin)
            mx, my = 15, 15
            
            # Draw Glassy Border / Neon Border
            # Outer white/accent border
            cv2.rectangle(canvas, (mx-2, my-2), (mx + pip_w + 2, my + pip_h + 2), (255, 255, 255), 1, cv2.LINE_AA)
            # Accent color bottom-right shadow-like line
            cv2.line(canvas, (mx + pip_w + 5, my + 5), (mx + pip_w + 5, my + pip_h + 5), (120, 0, 255), 2, cv2.LINE_AA)
            cv2.line(canvas, (mx + 5, my + pip_h + 5), (mx + pip_w + 5, my + pip_h + 5), (120, 0, 255), 2, cv2.LINE_AA)
            
            # Place PIP onto canvas
            canvas[my:my+pip_h, mx:mx+pip_w] = pip_small
            
            # Label the PIP
            cv2.putText(
                canvas, "ORIGINAL", (mx + 5, my + pip_h - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1, cv2.LINE_AA
            )

            # Now apply the filter ON TOP of the PIP
            canvas = filter_mod.apply(canvas, pose, original_frame=bgr_frame)

        # ── FPS counter ──
        self._fps_times.append(t0)
        self._fps_times = [t for t in self._fps_times if t0 - t < 1.0]
        fps = len(self._fps_times)
        
        # Overlay FPS on status bar instead of just on canvas
        self.fps_label.setText(f"ENGINES: ACTIVE | FPS: {fps}")

        # Filter PIP is now drawn before filter_mod.apply in the else block.

        # ── Display ──
        rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
        qimg = QImage(rgb.data, w, h, 3 * w, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qimg)

        # Scale to label size while keeping aspect ratio (SmoothTransformation for better quality)
        label_size = self.video_label.size()
        pixmap = pixmap.scaled(label_size, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.video_label.setPixmap(pixmap)

    # ── Filter Selection ───────────────────────────────────────────────────────
    def _set_filter(self, name: str):
        self.active_filter = name
        self._update_button_states()
        self.status_label.setText(f"Filter: {name}")

    def _update_button_states(self):
        for name, btn in self._filter_buttons.items():
            is_active = (name == self.active_filter)
            btn.setProperty("active", "true" if is_active else "false")
            btn.style().unpolish(btn)
            btn.style().polish(btn)

    # ── Screenshot ─────────────────────────────────────────────────────────────
    def _take_screenshot(self):
        pixmap = self.video_label.pixmap()
        if pixmap:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = os.path.join(os.path.expanduser("~"), "Pictures", f"stickfig_{ts}.png")
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            pixmap.save(filename)
            self.status_label.setText(f"Saved!\n{os.path.basename(filename)}")

    # ── Fullscreen ─────────────────────────────────────────────────────────────
    def _toggle_fullscreen(self):
        if self._is_fullscreen:
            self.showNormal()
        else:
            self.showFullScreen()
        self._is_fullscreen = not self._is_fullscreen

    # ── Keyboard shortcuts ─────────────────────────────────────────────────────
    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Escape and self._is_fullscreen:
            self._toggle_fullscreen()
        elif event.key() == Qt.Key_F11:
            self._toggle_fullscreen()
        elif event.key() == Qt.Key_S:
            self._take_screenshot()
        else:
            super().keyPressEvent(event)

    # ── Cleanup ────────────────────────────────────────────────────────────────
    def closeEvent(self, event):
        self.camera.stop()
        self.pose_detector.close()
        event.accept()
