"""
Flora Infusion Filter
Limbs grow semi-organic vines, drift falling leaves, and bloom bright flowers at the joints.
"""
import cv2
import numpy as np
import random
import math
from pose_detector import PoseResult, CONNECTIONS

class FallingLeaf:
    def __init__(self, x, y):
        self.x = float(x)
        self.y = float(y)
        self.vx = random.uniform(-2.0, 2.0)
        self.vy = random.uniform(1.0, 4.0)
        self.angle = random.uniform(0, 360)
        self.spin = random.uniform(-10, 10)
        self.scale = random.uniform(0.5, 1.2)
        # 10% chance to be a pink flower petal instead of a green leaf
        self.is_petal = (random.random() < 0.1)

_leaves = []

def draw_vine(img, pt1, pt2, thickness, color):
    # Draw a wobbly bezier/spline to emulate organic vine growth mechanics
    dist = np.hypot(pt2[0]-pt1[0], pt2[1]-pt1[1])
    if dist < 5:
        cv2.line(img, pt1, pt2, color, thickness, cv2.LINE_AA)
        return
        
    pts = [pt1]
    segments = int(dist // 15) + 2
    for i in range(1, segments):
        t = i / segments
        mx = pt1[0] + (pt2[0]-pt1[0])*t
        my = pt1[1] + (pt2[1]-pt1[1])*t
        
        # Add organic sine wobble for natural irregularities
        offset_amp = 8
        wx = mx + math.sin(t * math.pi * 4 + pt1[0]) * offset_amp
        wy = my + math.cos(t * math.pi * 4 + pt1[1]) * offset_amp
        pts.append((int(wx), int(wy)))
        
    pts.append(pt2)
    
    for i in range(len(pts)-1):
        cv2.line(img, pts[i], pts[i+1], color, thickness, cv2.LINE_AA)

def apply(canvas: np.ndarray, pose: PoseResult, **kwargs) -> np.ndarray:
    global _leaves
    
    h, w = canvas.shape[:2]
    original = kwargs.get('original_frame')
    if original is not None:
        canvas[:] = original

    if pose.detected:
        lm = pose.landmarks
        vis = pose.visibility
        
        layer = np.zeros_like(canvas)
        
        # 1. Draw Vines along skeleton length
        for a, b in CONNECTIONS:
            if a in lm and b in lm and vis.get(a, 0) > 0.3 and vis.get(b, 0) > 0.3:
                # Main thick trunk/vine
                draw_vine(layer, lm[a], lm[b], 8, (40, 120, 20)) # Dark green
                # Thin intertwined lighter vine rotating backwards
                draw_vine(layer, lm[b], lm[a], 3, (80, 200, 50)) # Bright green
                
                # Spawn falling leaves randomly off the arms and torso
                if random.random() < 0.20:
                    pt1, pt2 = lm[a], lm[b]
                    t = random.random()
                    lx = pt1[0] + (pt2[0] - pt1[0]) * t
                    ly = pt1[1] + (pt2[1] - pt1[1]) * t
                    _leaves.append(FallingLeaf(lx, ly))

        # 2. Draw Bloom Flowers natively bolted at structural Joints
        for idx in [11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 0]:
            if idx in lm and vis.get(idx, 0) > 0.4:
                cx, cy = lm[idx]
                # Golden Flower Center
                cv2.circle(layer, (cx, cy), 12, (0, 200, 255), -1, cv2.LINE_AA)
                # Outer Petal Ring
                for angle in range(0, 360, 45):
                    rad = math.radians(angle + (id(idx) % 90))
                    px = int(cx + math.cos(rad) * 16)
                    py = int(cy + math.sin(rad) * 16)
                    cv2.circle(layer, (px, py), 9, (200, 100, 255), -1, cv2.LINE_AA)
                    
        # Crisp Alpha blend for the organic plant layer covering the canvas
        mask = cv2.cvtColor(layer, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(mask, 1, 255, cv2.THRESH_BINARY)
        mask_inv = cv2.bitwise_not(mask)
        canvas_bg = cv2.bitwise_and(canvas, canvas, mask=mask_inv)
        canvas[:] = cv2.add(canvas_bg, layer)

    # 3. Ambient Engine: Simulate and Draw falling leaves in physics space
    new_leaves = []
    for leaf in _leaves:
        leaf.x += leaf.vx
        leaf.y += leaf.vy
        leaf.angle += leaf.spin
        
        # Subtly sway the leaf side-to-side based on vertical Y position
        leaf.vx += math.sin(leaf.y / 20.0) * 0.1
        
        # Kill leaves that sink offscreen
        if leaf.y < h + 20 and 0 < leaf.x < w:
            new_leaves.append(leaf)
            
            col = (200, 100, 255) if leaf.is_petal else (50, 200, 50)
            sz = int(8 * leaf.scale)
            
            # Procedural leaf shape calculation (Diamond/Ellipse)
            M = cv2.getRotationMatrix2D((leaf.x, leaf.y), leaf.angle, 1.0)
            pts = np.array([
                [-sz, 0], [0, sz*2], [sz, 0], [0, -sz]
            ], np.float32)
            
            pts += np.array([leaf.x, leaf.y], np.float32)
            pts = pts.reshape(-1, 1, 2)
            pts_rot = cv2.transform(pts, M)
            pts_int = np.int32(pts_rot)
            
            cv2.fillPoly(canvas, [pts_int], col, cv2.LINE_AA)
            # Add central stem vein inside the leaf structure
            cv2.polylines(canvas, [pts_int[0:2]], False, (0, 0, 0), 1, cv2.LINE_AA)

    _leaves = new_leaves[-250:] # Performance strict cap
    return canvas
