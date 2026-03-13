"""
Particle System — generates and animates particles for visual effects.
Optimized with vectorized NumPy processing for massive scale.
"""
import numpy as np
import random
import time
from typing import List, Optional, Tuple, Callable

class ParticleSystem:
    def __init__(self, max_particles: int = 1000):
        self.max_particles = max_particles
        
        # State Arrays
        self.pos = np.zeros((0, 2), dtype=np.float32)      # [x, y]
        self.vel = np.zeros((0, 2), dtype=np.float32)      # [vx, vy]
        self.colors = np.zeros((0, 3), dtype=np.uint8)    # [B, G, R]
        self.sizes = np.zeros(0, dtype=np.int32)
        self.lifetimes = np.zeros(0, dtype=np.int32)
        self.max_lifetimes = np.zeros(0, dtype=np.int32)
        self.gravity = 0.15
        
    def spawn_batch(self, points: np.ndarray, count_per_point=1, color_fn: Optional[Callable] = None,
                   size_range=(2, 5), lifetime_range=(20, 50), speed_scale=1.0):
        """Spawn particles at multiple points simultaneously."""
        if len(self.pos) >= self.max_particles:
            return
            
        total_to_spawn = min(len(points) * count_per_point, self.max_particles - len(self.pos))
        if total_to_spawn <= 0:
            return
            
        # Broadcast points to match total count
        spawn_points = np.repeat(points, count_per_point, axis=0)[:total_to_spawn]
        
        new_pos = spawn_points.astype(np.float32)
        new_vel = np.stack([
            np.random.uniform(-2.5, 2.5, total_to_spawn) * speed_scale,
            np.random.uniform(-5.0, -1.0, total_to_spawn) * speed_scale
        ], axis=1).astype(np.float32)
        
        if color_fn:
            new_colors = np.array([color_fn() for _ in range(total_to_spawn)], dtype=np.uint8)
        else:
            new_colors = np.random.randint(150, 255, (total_to_spawn, 3), dtype=np.uint8)
            
        new_sizes = np.random.randint(size_range[0], size_range[1] + 1, total_to_spawn, dtype=np.int32)
        new_lives = np.random.randint(lifetime_range[0], lifetime_range[1] + 1, total_to_spawn, dtype=np.int32)
        
        self.pos = np.vstack([self.pos, new_pos])
        self.vel = np.vstack([self.vel, new_vel])
        self.colors = np.vstack([self.colors, new_colors])
        self.sizes = np.concatenate([self.sizes, new_sizes])
        self.lifetimes = np.concatenate([self.lifetimes, new_lives])
        self.max_lifetimes = np.concatenate([self.max_lifetimes, new_lives])

    def spawn(self, x, y, count=5, color_fn: Optional[Callable] = None, 
              size_range=(2, 5), lifetime_range=(20, 50), speed_scale=1.0,
              velocity: Optional[Tuple[float, float]] = None):
        """Spawn `count` particles at (x, y)."""
        if len(self.pos) >= self.max_particles:
            return
            
        spawn_count = min(count, self.max_particles - len(self.pos))
        if spawn_count <= 0:
            return
            
        # Initialize new particles
        new_pos = np.full((spawn_count, 2), [x, y], dtype=np.float32)
        if velocity is not None:
            new_vel = np.tile(np.array(velocity, dtype=np.float32), (spawn_count, 1))
        else:
            new_vel = np.stack([
                np.random.uniform(-2.5, 2.5, spawn_count) * speed_scale,
                np.random.uniform(-5.0, -1.0, spawn_count) * speed_scale
            ], axis=1).astype(np.float32)
        
        if color_fn:
            new_colors = np.array([color_fn() for _ in range(spawn_count)], dtype=np.uint8)
        else:
            new_colors = np.random.randint(150, 255, (spawn_count, 3), dtype=np.uint8)
            
        new_sizes = np.random.randint(size_range[0], size_range[1] + 1, spawn_count, dtype=np.int32)
        new_lives = np.random.randint(lifetime_range[0], lifetime_range[1] + 1, spawn_count, dtype=np.int32)
        
        # Append to state arrays
        self.pos = np.vstack([self.pos, new_pos])
        self.vel = np.vstack([self.vel, new_vel])
        self.colors = np.vstack([self.colors, new_colors])
        self.sizes = np.concatenate([self.sizes, new_sizes])
        self.lifetimes = np.concatenate([self.lifetimes, new_lives])
        self.max_lifetimes = np.concatenate([self.max_lifetimes, new_lives])

    def update(self):
        if len(self.pos) == 0:
            return
            
        # 1. Physics update
        self.pos += self.vel
        self.vel[:, 0] *= 0.97  # Friction
        self.vel[:, 1] += self.gravity # Gravity
        
        # 2. Lifecycle update
        self.lifetimes -= 1
        
        # 3. Filter dead particles
        alive_mask = self.lifetimes > 0
        self.pos = self.pos[alive_mask]
        self.vel = self.vel[alive_mask]
        self.colors = self.colors[alive_mask]
        self.sizes = self.sizes[alive_mask]
        self.lifetimes = self.lifetimes[alive_mask]
        self.max_lifetimes = self.max_lifetimes[alive_mask]

    def draw(self, canvas: np.ndarray):
        import cv2
        if len(self.pos) == 0:
            return
            
        h, w = canvas.shape[:2]
        
        # Calculate alpha and apply to colors
        alphas = self.lifetimes / self.max_lifetimes
        
        # Draw batch (OpenCV doesn't have a vectorized 'circles' for individual alphas yet)
        for i in range(len(self.pos)):
            px, py = int(self.pos[i, 0]), int(self.pos[i, 1])
            if 0 <= px < w and 0 <= py < h:
                # Optimized color scaling
                a = alphas[i]
                c = (int(self.colors[i, 0] * a), int(self.colors[i, 1] * a), int(self.colors[i, 2] * a))
                cv2.circle(canvas, (px, py), self.sizes[i], c, -1)

    def clear(self):
        self.pos = np.zeros((0, 2), dtype=np.float32)
        self.vel = np.zeros((0, 2), dtype=np.float32)
        self.colors = np.zeros((0, 3), dtype=np.uint8)
        self.sizes = np.zeros(0, dtype=np.int32)
        self.lifetimes = np.zeros(0, dtype=np.int32)
        self.max_lifetimes = np.zeros(0, dtype=np.int32)
