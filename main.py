"""
main.py — Entry point for the Camera Stick-Figure Filter App.

Usage:
    python main.py

Requirements:
    pip install opencv-python mediapipe numpy pyqt5 scipy pillow
"""
import sys
import os

# Ensure project root is on the path so all submodules import cleanly
sys.path.insert(0, os.path.dirname(__file__))

from PyQt5.QtWidgets import QApplication
from PyQt5.QtGui import QFont
from ui.main_window import MainWindow


def main():
    app = QApplication(sys.argv)
    app.setApplicationName("Stick-Figure Filter Cam")

    # Use a clean, modern font
    font = QFont("Segoe UI", 10)
    app.setFont(font)

    window = MainWindow()
    window.show()

    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
