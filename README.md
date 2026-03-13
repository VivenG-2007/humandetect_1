# 🎭 Stick-Figure Filter Cam

A real-time AI-powered camera application that detects human poses and transforms them into stylized, animated stick figures with premium visual effects.

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)
![OpenCV](https://img.shields.io/badge/OpenCV-4.x-green?logo=opencv)
![MediaPipe](https://img.shields.io/badge/MediaPipe-Pose-red)
![PyQt5](https://img.shields.io/badge/PyQt5-GUI-orange)

## ✨ Key Features

- **🚀 Real-time Pose Detection**: Powered by MediaPipe for high-accuracy body tracking.
- **🎨 Premium Stick Figure Design**: Features a rigid "square" (rectangular) torso, stylized head with outlines, and mechanical joint aesthetics.
- **🌈 Creative Filter Suite**:
    - **Extreme FX**: A unified high-energy filter featuring action-triggered lightning/sparkles, shields, and supernovas.
    - **Stick Figure**: High-contrast stylized figure on a black canvas.
    - **Boss Mode**: Intimidating giant red glowing aura and dark aesthetics.
    - **Portal**: High-energy circular spark portal inspired by Doctor Strange.
    - **Aura**: Movement-based neon smoke trails with glowing cores.
    - **Lightning**: Electric arcs and bolts shooting from hands and feet.
    - **Bubbles**: Interactive floating bubbles spawned by movement.
    - **And many more...** (Neon, Hologram, Magma Flow, Cyber Wings, Prism, etc.)
- **🖼️ Picture-in-Picture (PIP)**: Small window in the top-left showing the original camera feed for reference.
- **📸 Screenshot Capture**: Quick button and shortcut (`S`) to save filtered frames.
- **🖥️ Fullscreen Mode**: Toggle via button or F11 for an immersive experience.

## 🛠️ Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/VivenG-2007/humandetect.git
   cd humandetect
   ```

2. **Install dependencies**:
   Ensure you have Python installed, then run:
   ```bash
   pip install -r requirements.txt
   ```
   *Dependencies include: `opencv-python`, `mediapipe`, `numpy`, `pyqt5`, `scipy`, `pillow`.*

## 🚀 Usage

Run the main entry script to launch the application:
```bash
python main.py
```

### ⌨️ Keyboard Shortcuts
- **`F11`**: Toggle Fullscreen.
- **`S`**: Take a Screenshot.
- **`ESC`**: Exit Fullscreen.

## 📁 Project Structure

```text
humandetect/
├── main.py              # Entry point
├── camera.py            # Async camera thread
├── pose_detector.py      # MediaPipe Pose wrapper
├── skeleton_renderer.py  # Premium stick-figure rendering engine
├── filters/             # Modular filter system
│   ├── aura.py
│   ├── lightning.py
│   ├── bubbles.py
│   └── ...
├── ui/                  # PyQt5 GUI components
│   └── main_window.py
└── utils/               # Particle systems and math helpers
```

## 🏗️ Technical Highlights

- **Hybrid Rendering**: Combines fast OpenCV primitives with stylized logic for a unique "rigid" torso look.
- **Performance Optimized**: Uses downsampling and Gaussian blurring tricks to maintain high FPS even with complex visual effects.
- **Modular Filtering**: Each filter is a standalone module that can be easily extended or modified.

---
Developed with ❤️ by [VivenG-2007](https://github.com/VivenG-2007)
