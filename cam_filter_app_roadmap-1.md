# Camera Stick-Figure Filter App Roadmap

Technologies: Python, OpenCV, MediaPipe, PyQt5

## Goal

Build a real‑time camera application that: - Detects a human using
MediaPipe pose detection - Converts the human body into a stick‑figure
skeleton - Blacks out the entire background - Applies creative visual
filters on the stick figure outline - Allows selecting filters via a GUI

------------------------------------------------------------------------

# Phase 1 -- Environment Setup

Install Python 3.10

Create virtual environment: python -m venv venv

Activate: Windows: venv`\Scripts`{=tex}`\activate`{=tex}

Linux/Mac: source venv/bin/activate

Install dependencies: pip install opencv-python mediapipe numpy pyqt5
scipy pillow

------------------------------------------------------------------------

# Phase 2 -- Camera Input

Goal: Capture frames from webcam.

Steps: 1. Initialize OpenCV VideoCapture 2. Read frames continuously 3.
Convert BGR to RGB for MediaPipe 4. Display frames

------------------------------------------------------------------------

# Phase 3 -- Human Detection

Use MediaPipe Pose.

Steps: 1. Initialize MediaPipe Pose model 2. Detect landmarks 3. Extract
body keypoints

Key landmarks: Head, Shoulders, Elbows, Wrists, Hips, Knees, Ankles

------------------------------------------------------------------------

# Phase 4 -- Background Removal

Goal: Hide original camera frame.

Method: 1. Create black image same size as frame 2. Draw only detected
skeleton on black canvas

Result: Pure black background with stick figure.

------------------------------------------------------------------------

# Phase 5 -- Stick Figure Rendering

Steps: 1. Connect landmarks with lines 2. Draw joints as circles 3.
Control thickness and color

Result: Human represented as animated stick person.

------------------------------------------------------------------------

# Phase 6 -- GUI Interface

Use PyQt5.

UI Components: - Camera preview window - Filter buttons - Toggle
effects - Screenshot button

Clicking button changes active filter.

------------------------------------------------------------------------

# Phase 7 -- Filter Engine

Create modular filter system.

Structure:

filters/ aura.py fire.py neon.py lightning.py

Each filter receives: - skeleton coordinates - frame canvas

Returns modified frame.

------------------------------------------------------------------------

# Phase 8 -- Aura Filter

Effect: Glowing outline around skeleton.

Implementation: 1. Duplicate skeleton layer 2. Blur with Gaussian blur
3. Blend with original skeleton

------------------------------------------------------------------------

# Phase 9 -- Firecracker Head Animation

Goal: particles above head.

Steps: 1. Detect head landmark 2. Spawn particles 3. Animate upward
sparks 4. Fade particles over time

------------------------------------------------------------------------

# Phase 10 -- Creative Filters

Ideas: - Neon Skeleton - Lightning Body - Energy Pulse - Matrix Code
Rain - Hologram Mode - Shadow Clone

------------------------------------------------------------------------

# Phase 11 -- Motion Tracking Effects

Use landmark velocity.

Examples: - Hand trails - Foot sparks - Jump glow

velocity = current_position - previous_position

------------------------------------------------------------------------

# Phase 12 -- Performance Optimization

Techniques: - Reduce frame resolution - Use threading for UI - Cache
filters - Limit particle counts

Goal: maintain smooth FPS.

------------------------------------------------------------------------

# Phase 13 -- Project Architecture

project/

main.py camera.py pose_detector.py skeleton_renderer.py

filters/ aura.py firecracker.py neon.py

ui/ main_window.py

utils/ particle_system.py

------------------------------------------------------------------------

# Phase 14 -- Final Features

-   Filter switching
-   Screenshot capture
-   Video recording
-   FPS counter
-   Fullscreen mode

------------------------------------------------------------------------

# Phase 15 -- Future Improvements

-   AI style transfer
-   Gesture filter switching
-   AR objects
-   3D skeleton rendering
-   GPU acceleration
