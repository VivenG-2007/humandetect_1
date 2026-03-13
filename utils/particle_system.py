"""
Particle System — generates and animates particles for visual effects.
"""
import numpy as np
import random
from typing import List, Optional, Tuple


class Particle:
    def __init__(self, x, y, color, velocity=None, lifetime=40, size=3):
        self.x = float(x)
        self.y = float(y)
        self.color = color  # (B, G, R)
        self.velocity = velocity if velocity is not None else (
            random.uniform(-2.5, 2.5),
            random.uniform(-5.0, -1.5)
        )
        self.lifetime = lifetime
        self.max_lifetime = lifetime
        self.size = size
        self.gravity = 0.15

    def update(self):
        self.x += self.velocity[0]
        self.y += self.velocity[1]
        self.velocity = (self.velocity[0] * 0.97, self.velocity[1] + self.gravity)
        self.lifetime -= 1

    @property
    def is_alive(self):
        return self.lifetime > 0

    @property
    def alpha(self):
        return self.lifetime / self.max_lifetime


class ParticleSystem:
    def __init__(self, max_particles: int = 300):
        self.particles: List[Particle] = []
        self.max_particles = max_particles

    def spawn(self, x, y, count=5, color_fn=None, size_range=(2, 5),
              lifetime_range=(20, 50), speed_scale=1.0):
        """Spawn `count` particles at (x, y)."""
        for _ in range(count):
            if len(self.particles) >= self.max_particles:
                break
            color = color_fn() if color_fn else (
                random.randint(100, 255),
                random.randint(100, 255),
                random.randint(200, 255)
            )
            vx = random.uniform(-2.5, 2.5) * speed_scale
            vy = random.uniform(-5.0, -1.0) * speed_scale
            size = random.randint(*size_range)
            lifetime = random.randint(*lifetime_range)
            self.particles.append(Particle(x, y, color, (vx, vy), lifetime, size))

    def update(self):
        self.particles = [p for p in self.particles if p.is_alive]
        for p in self.particles:
            p.update()

    def draw(self, canvas: np.ndarray):
        import cv2
        for p in self.particles:
            if not p.is_alive:
                continue
            px, py = int(p.x), int(p.y)
            if 0 <= px < canvas.shape[1] and 0 <= py < canvas.shape[0]:
                alpha = p.alpha
                color = tuple(int(c * alpha) for c in p.color)
                cv2.circle(canvas, (px, py), p.size, color, -1)

    def clear(self):
        self.particles.clear()
