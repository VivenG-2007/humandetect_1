"""
Firecracker Filter — sparkling particle explosion above the head landmark.
"""
import cv2
import numpy as np
import random
from pose_detector import PoseResult
from utils.particle_system import ParticleSystem

_system = ParticleSystem(max_particles=400)


def _fire_color():
    r = random.choice([
        (0, random.randint(100, 200), 255),   # orange
        (0, 200, 255),                          # yellow-orange
        (0, 50, 255),                           # red
        (50, 220, 255),                         # bright yellow
    ])
    return r


def apply(canvas: np.ndarray, pose: PoseResult, **kwargs) -> np.ndarray:
    _system.update()

    if pose.detected:
        nose = pose.landmarks.get(0)
        if nose and pose.visibility.get(0, 0) > 0.4:
            x, y = nose
            # Spawn sparks above head
            _system.spawn(
                x, y - 20,
                count=6,
                color_fn=_fire_color,
                size_range=(2, 4),
                lifetime_range=(25, 55),
                speed_scale=1.2,
            )
            # Occasional burst
            if random.random() < 0.15:
                _system.spawn(
                    x + random.randint(-15, 15),
                    y - 10,
                    count=12,
                    color_fn=_fire_color,
                    size_range=(3, 6),
                    lifetime_range=(30, 60),
                    speed_scale=1.8,
                )


    _system.draw(canvas)
    return canvas
    return canvas
