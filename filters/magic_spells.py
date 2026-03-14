"""
Magic Spells Filter  —  MERGED FINAL
=====================================
Combines the best of two filter versions into one cohesive effect.

Gestures (MediaPipe Pose landmarks):
  Both wrists raised + hands close  → 🛡  Hex-grid energy shield
                                         Ambient particles bounce off it
  One wrist raised (not both)       → 🌀  Mystic rune circle on that palm
                                         3 dashed rings · Elder Futhark runes · orange glow
                                         Circle follows the hand
  Index extended beyond wrist       → ⚡  Arcane lightning beam from fingertip
                                         Jittery forked lightning + particle trail
                                         Intensity scales with hand speed

Landmark reference (MediaPipe Pose):
  11/12 = L/R shoulder   13/14 = L/R elbow
  15/16 = L/R wrist      17/18 = L/R pinky-base
  19/20 = L/R index-base 21/22 = L/R thumb-base
"""

import cv2
import numpy as np
import math
import random
import time
from pose_detector import PoseResult

# ══════════════════════════════════════════════════════════════════════════════
#  TUNING CONSTANTS
# ══════════════════════════════════════════════════════════════════════════════
VIS               = 0.30    # minimum landmark visibility

# Shield
SHIELD_GROW       = 0.10    # buttery smooth growth
SHIELD_SHRINK     = 0.75    # decay multiplier when inactive
SHIELD_COLOR      = (255, 200, 50)    # BGR gold-cyan
SHIELD_GLOW       = (200, 100,  0)    # darker for hex cells
SHIELD_EDGE       = (200, 240, 255)   # bright rim highlight

# Rune circle
RUNE_GROW         = 0.12
RUNE_SHRINK       = 0.72
RUNE_SYMBOLS      = list("ᚠᚢᚦᚨᚱᚲᚷᚹᚺᚾᛁᛃᛇᛈᛉᛊᛏᛒᛖᛗᛚᛜᛞᛟ")
RUNE_COLORS       = [(0, 130, 255), (0, 190, 255), (30, 240, 210)]  # orange tiers
RUNE_SPIN_SPEED   = [0.6, -0.9, 1.3]   # rad/s per ring

# Beam
BEAM_MAX          = 600
BEAM_SPEED_BASE   = 20.0
BEAM_SPEED_MAX    = 45.0
BEAM_DRAG         = 0.91
BEAM_LIFE         = 0.50
BEAM_CORE         = (255, 255, 255)
BEAM_GLOW         = (200,  60, 255)   # purple-pink lightning

# Gesture stability
HISTORY_LEN       = 10
VOTES_NEEDED      = 7

# Ambient particles
NUM_AMBIENT       = 150


# ══════════════════════════════════════════════════════════════════════════════
#  STATE  (single dict — easy to refactor into a class later)
# ══════════════════════════════════════════════════════════════════════════════
def _make_state(w: int = 640, h: int = 480) -> dict:
    rng = np.random
    return {
        # Shield
        "shield_r":    0.0,
        "shield_cx":   w // 2,
        "shield_cy":   h // 2,
        # Rune
        "rune_r":      {"left": 0.0, "right": 0.0},
        "rune_angle":  {"left": 0.0, "right": 0.0},
        "rune_pos":    {"left": (w // 4, h // 2), "right": (3 * w // 4, h // 2)},
        # Beam particles
        "bp_pos":      np.zeros((0, 2), np.float32),
        "bp_vel":      np.zeros((0, 2), np.float32),
        "bp_born":     np.zeros(0, np.float32),
        "bp_life":     np.zeros(0, np.float32),
        "bp_size":     np.zeros(0, np.float32),
        # Ambient falling particles
        "ap_pos":      np.column_stack((
                           rng.uniform(0, w, NUM_AMBIENT),
                           rng.uniform(0, h, NUM_AMBIENT),
                       )).astype(np.float32),
        "ap_vel":      np.column_stack((
                           rng.uniform(-1.0, 1.0, NUM_AMBIENT),
                           rng.uniform( 2.0, 5.0, NUM_AMBIENT),
                       )).astype(np.float32),
        # Speed / gesture tracking
        "prev_idx":    {"left": None, "right": None},
        "hand_speed":  {"left": 0.0,  "right": 0.0},
        "g_hist":      {k: [] for k in
                        ["left_open", "right_open", "left_point", "right_point",
                         "left_straight", "right_straight"]},
        "prev_t":      time.time(),
        "initialized": True,
        "frame_wh":    (w, h),
    }


_S: dict = {}


def _ensure_init(w: int, h: int) -> None:
    global _S
    if not _S.get("initialized") or _S["frame_wh"] != (w, h):
        _S = _make_state(w, h)


# ══════════════════════════════════════════════════════════════════════════════
#  GESTURE HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _vote(history: list, new_val: bool) -> bool:
    history.append(new_val)
    if len(history) > HISTORY_LEN:
        history.pop(0)
    return history.count(True) >= VOTES_NEEDED


def _wrist_raised(lm: dict, vis: dict, wrist: int, shoulder: int) -> bool:
    return (vis.get(wrist,    0) > VIS and
            vis.get(shoulder, 0) > VIS and
            lm[wrist][1] < lm[shoulder][1])


def _is_open_palm(lm: dict, vis: dict,
                  wrist_i: int, elbow_i: int,
                  knuckle_idxs: list, tip_idxs: list) -> bool:
    """
    Open palm: majority of available tip landmarks are further from the
    wrist than their knuckle (fingers extended).  Falls back to a
    forearm-proportion check when hand-tracker tips are absent.
    """
    if wrist_i not in lm:
        return False
    wrist = np.array(lm[wrist_i], np.float32)

    hand_active = any(vis.get(ti, 0) >= 0.95 for ti in tip_idxs)

    if hand_active:
        extended = 0
        for ki, ti in zip(knuckle_idxs, tip_idxs):
            if ki in lm and ti in lm:
                d_k = max(1.0, np.linalg.norm(np.array(lm[ki]) - wrist))
                d_t = np.linalg.norm(np.array(lm[ti]) - wrist)
                if d_t > d_k * 1.15:
                    extended += 1
        return extended >= 2
    else:
        if elbow_i not in lm:
            return False
        elbow     = np.array(lm[elbow_i], np.float32)
        d_forearm = max(10.0, np.linalg.norm(wrist - elbow))
        dists     = [np.linalg.norm(np.array(lm[ti]) - wrist)
                     for ti in tip_idxs if ti in lm]
        if not dists:
            return False
        return (sum(dists) / len(dists) / d_forearm) > 0.42


def _is_pointing(lm: dict, vis: dict,
                 wrist_i: int, elbow_i: int,
                 idx_base: int, idx_tip: int,
                 other_knuckles: list, other_tips: list) -> bool:
    """
    Pointing: index-base is further from wrist than pinky-base, or when
    hand-tracker is active, index tip is extended while others are curled.
    """
    if wrist_i not in lm:
        return False
    wrist = np.array(lm[wrist_i], np.float32)

    hand_active = (vis.get(idx_tip, 0) >= 0.95 or
                   any(vis.get(ti, 0) >= 0.95 for ti in other_tips))

    if hand_active:
        if idx_base not in lm or idx_tip not in lm:
            return False
        d_idx_k = max(1.0, np.linalg.norm(np.array(lm[idx_base]) - wrist))
        d_idx_t = np.linalg.norm(np.array(lm[idx_tip]) - wrist)
        if d_idx_t < d_idx_k * 1.15:
            return False
        curled = sum(
            1 for ki, ti in zip(other_knuckles, other_tips)
            if ki in lm and ti in lm and
               np.linalg.norm(np.array(lm[ti]) - wrist) <
               max(1.0, np.linalg.norm(np.array(lm[ki]) - wrist)) * 1.3
        )
        return curled >= 1
    else:
        if elbow_i not in lm or idx_tip not in lm:
            return False
        elbow     = np.array(lm[elbow_i], np.float32)
        d_forearm = max(10.0, np.linalg.norm(wrist - elbow))
        d_idx     = np.linalg.norm(np.array(lm[idx_tip]) - wrist) / d_forearm
        others    = [np.linalg.norm(np.array(lm[ti]) - wrist) / d_forearm
                     for ti in other_tips if ti in lm]
        return d_idx > 0.40 and (sum(others) / max(1, len(others))) < 0.35


def _beam_dir(lm: dict, wrist_i: int, index_i: int) -> np.ndarray:
    w   = np.array(lm[wrist_i],  np.float32)
    idx = np.array(lm[index_i],  np.float32)
    d   = idx - w
    n   = float(np.linalg.norm(d))
    return d / n if n > 1e-3 else np.array([1.0, 0.0], np.float32)


def _is_arm_straight(lm: dict, vis: dict,
                     shoulder_i: int, elbow_i: int, wrist_i: int,
                     min_angle_deg: float = 155.0) -> bool:
    """
    Returns True when the elbow angle (shoulder→elbow→wrist) exceeds
    min_angle_deg, meaning the arm is fully extended / straight.
    """
    if not all(vis.get(i, 0) > VIS for i in [shoulder_i, elbow_i, wrist_i]):
        return False
    s = np.array(lm[shoulder_i], np.float32)
    e = np.array(lm[elbow_i],    np.float32)
    wr = np.array(lm[wrist_i],   np.float32)
    v1 = s - e                          # shoulder → elbow
    v2 = wr - e                         # wrist    → elbow
    n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if n1 < 1e-3 or n2 < 1e-3:
        return False
    cos_a = float(np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0))
    return math.degrees(math.acos(cos_a)) > min_angle_deg


def _arm_fire_dir(lm: dict, shoulder_i: int, wrist_i: int) -> np.ndarray:
    """Unit vector from shoulder through wrist — the aim direction of a straight arm."""
    s  = np.array(lm[shoulder_i], np.float32)
    wr = np.array(lm[wrist_i],    np.float32)
    d  = wr - s
    n  = float(np.linalg.norm(d))
    return d / n if n > 1e-3 else np.array([1.0, 0.0], np.float32)


# ══════════════════════════════════════════════════════════════════════════════
#  SHIELD  (hex-grid interior + bounding hexagon + glow)
# ══════════════════════════════════════════════════════════════════════════════

def _hexagon_pts(cx: int, cy: int, r: float, angle_off: float) -> np.ndarray:
    pts = [(int(cx + r * math.cos(angle_off + i * math.pi / 3)),
            int(cy + r * math.sin(angle_off + i * math.pi / 3)))
           for i in range(6)]
    return np.array(pts, np.int32)


def _draw_shield(canvas: np.ndarray, cx: int, cy: int,
                 r: float, t: float, h: int, w: int) -> None:
    if r < 8:
        return

    pulse  = 0.04 * math.sin(t * 5)
    r_out  = r * (1.0 + pulse)
    spin   = t * 0.35
    spin2  = t * -0.22

    overlay = np.zeros_like(canvas)

    # ── Hex-grid interior (V2 design, pulsing per-cell) ──────────────────
    hex_size = max(6, int(r * 0.30))
    off_x    = hex_size * math.sqrt(3)
    off_y    = hex_size * 1.5
    
    base_hex = np.array([
        [int((hex_size - 2) * math.cos(math.radians(60 * i + 30))),
         int((hex_size - 2) * math.sin(math.radians(60 * i + 30)))]
        for i in range(6)
    ], np.int32)

    for q in range(-3, 4):
        for rr in range(-3, 4):
            if abs(q + rr) > 3:
                continue
            hx = cx + int(off_x * (q + rr / 2.0))
            hy = cy + int(off_y * rr)
            # Only draw cells within the shield radius
            if math.hypot(hx - cx, hy - cy) > r_out * 0.92:
                continue
            cell_pulse = (math.sin(t * 5 + q + rr) + 1.0) / 2.0
            c = (
                int(SHIELD_GLOW[0] + (SHIELD_COLOR[0] - SHIELD_GLOW[0]) * cell_pulse),
                int(SHIELD_GLOW[1] + (SHIELD_COLOR[1] - SHIELD_GLOW[1]) * cell_pulse),
                int(SHIELD_GLOW[2] + (SHIELD_COLOR[2] - SHIELD_GLOW[2]) * cell_pulse),
            )
            pts = (base_hex + [hx, hy]).reshape((-1, 1, 2))
            cv2.polylines(overlay, [pts], True, c, 1, cv2.LINE_AA)
            if cell_pulse > 0.80:
                fill_c = (c[0] // 4, c[1] // 4, c[2] // 4)
                cv2.fillPoly(overlay, [pts], fill_c)

    # ── Bounding hexagon rings (V1 design) ────────────────────────────────
    for rad, col, th in [
        (r_out,        SHIELD_COLOR, 3),
        (r_out * 0.80, (80, 140, 255), 2),
    ]:
        hex_pts = _hexagon_pts(cx, cy, rad, spin)
        cv2.polylines(overlay, [hex_pts], True, col, th, cv2.LINE_AA)

    # Counter-rotating inner star
    star_pts = _hexagon_pts(cx, cy, r_out * 0.42, spin2)
    for i in range(6):
        cv2.line(overlay, tuple(star_pts[i]),
                 tuple(star_pts[(i + 2) % 6]), SHIELD_EDGE, 1, cv2.LINE_AA)

    # Rotating spokes
    for deg in range(0, 360, 60):
        a  = math.radians(deg) + spin
        x1 = int(cx + r_out * 0.28 * math.cos(a))
        y1 = int(cy + r_out * 0.28 * math.sin(a))
        x2 = int(cx + r_out * 0.90 * math.cos(a))
        y2 = int(cy + r_out * 0.90 * math.sin(a))
        cv2.line(overlay, (x1, y1), (x2, y2), SHIELD_EDGE, 1, cv2.LINE_AA)

    # Central crystal core
    core_r = max(8, int(r * 0.12))
    cv2.circle(overlay, (cx, cy), core_r,     SHIELD_COLOR, -1, cv2.LINE_AA)
    cv2.circle(overlay, (cx, cy), core_r + 5, SHIELD_EDGE,   2, cv2.LINE_AA)

    # ── Glow (quarter-res blur) ───────────────────────────────────────────
    ov_s = cv2.resize(overlay, (w // 4, h // 4))
    glow = cv2.GaussianBlur(ov_s, (21, 21), 0)
    cv2.add(canvas, cv2.resize(glow, (w, h)), dst=canvas)
    cv2.add(canvas, overlay, dst=canvas)


# ══════════════════════════════════════════════════════════════════════════════
#  RUNE CIRCLE  (dashed rings + Elder Futhark symbols + tick marks + glow)
# ══════════════════════════════════════════════════════════════════════════════

def _draw_dashed_circle(img: np.ndarray, cx: int, cy: int, radius: int,
                        color: tuple, thick: int,
                        start_deg: float, n_dashes: int, dash_ratio: float) -> None:
    arc_len = 360.0 / n_dashes
    for i in range(n_dashes):
        a1 = start_deg + i * arc_len
        a2 = a1 + arc_len * dash_ratio
        cv2.ellipse(img, (cx, cy), (radius, radius), 0,
                    a1, a2, color, thick, cv2.LINE_AA)


def _draw_rune_circle(canvas: np.ndarray, cx: int, cy: int,
                      r: float, base_angle: float,
                      t: float, h: int, w: int) -> None:
    if r < 8:
        return

    overlay    = np.zeros_like(canvas)
    font       = cv2.FONT_HERSHEY_COMPLEX
    font_scale = max(0.28, r / 170)

    for ring in range(3):
        frac    = 0.45 + ring * 0.27          # 0.45 / 0.72 / 0.99 of r
        ring_r  = int(r * frac)
        if ring_r < 5:
            continue

        angle_deg = math.degrees(base_angle + t * RUNE_SPIN_SPEED[ring])
        col       = RUNE_COLORS[ring % len(RUNE_COLORS)]
        n_dashes  = 8 + ring * 4              # more dashes on outer rings
        n_runes   = 6 + ring * 2

        # Dashed ring (V2 style)
        _draw_dashed_circle(overlay, cx, cy, ring_r,
                            col, 2, angle_deg, n_dashes, 0.65)

        # Precompute approx text dimensions for speed
        (tw, th_), _ = cv2.getTextSize("H", font, font_scale, 1)
        
        # Elder Futhark runes vectorized positions
        a_runes = base_angle + t * RUNE_SPIN_SPEED[ring] + np.arange(n_runes) * (2 * math.pi / n_runes)
        rx = (cx + ring_r * np.cos(a_runes)).astype(np.int32)
        ry = (cy + ring_r * np.sin(a_runes)).astype(np.int32)

        for k in range(n_runes):
            sym = RUNE_SYMBOLS[(ring * 7 + k) % len(RUNE_SYMBOLS)]
            cv2.putText(overlay, sym,
                        (rx[k] - tw // 2, ry[k] + th_ // 2),
                        font, font_scale, col, 1, cv2.LINE_AA)

        # Tick marks vectorized
        a_ticks = base_angle + t * RUNE_SPIN_SPEED[ring] + np.arange(n_runes * 2) * (math.pi / n_runes)
        x1 = (cx + ring_r * 0.88 * np.cos(a_ticks)).astype(np.int32)
        y1 = (cy + ring_r * 0.88 * np.sin(a_ticks)).astype(np.int32)
        x2 = (cx + ring_r * 1.10 * np.cos(a_ticks)).astype(np.int32)
        y2 = (cy + ring_r * 1.10 * np.sin(a_ticks)).astype(np.int32)
        
        for k in range(n_runes * 2):
            cv2.line(overlay, (x1[k], y1[k]), (x2[k], y2[k]), col, 1, cv2.LINE_AA)

    # Central sigil
    core_r = max(4, int(r * 0.10))
    cv2.circle(overlay, (cx, cy), core_r,     RUNE_COLORS[0], -1, cv2.LINE_AA)
    cv2.circle(overlay, (cx, cy), core_r + 4, RUNE_COLORS[1],  1, cv2.LINE_AA)

    # Glow blend (V2 addWeighted style)
    glow = cv2.GaussianBlur(overlay, (15, 15), 0)
    cv2.addWeighted(canvas, 1.0, glow,    1.5, 0, canvas)
    cv2.addWeighted(canvas, 1.0, overlay, 1.2, 0, canvas)


# ══════════════════════════════════════════════════════════════════════════════
#  BEAM  (lightning jitter + particle trail + muzzle flash)
# ══════════════════════════════════════════════════════════════════════════════

def _draw_lightning(canvas: np.ndarray,
                    start: tuple, angle: float, intensity: float,
                    t: float, w: int, h: int) -> None:
    """Jittery forked lightning stroke (V2 design)."""
    overlay = np.zeros_like(canvas)
    length  = max(w, h) * 1.5
    end_x   = int(start[0] + length * math.cos(angle))
    end_y   = int(start[1] + length * math.sin(angle))

    thick_core  = max(2, int(intensity * 8))
    thick_outer = max(5, int(intensity * 26))

    ratios = np.linspace(0, 1, 17)
    bx = start[0] + (end_x - start[0]) * ratios
    by = start[1] + (end_y - start[1]) * ratios
    
    jx = np.random.uniform(-18, 18, 17) * intensity
    jy = np.random.uniform(-18, 18, 17) * intensity
    jx[0] = jx[-1] = jy[0] = jy[-1] = 0

    pts = np.column_stack((bx + jx, by + jy)).astype(np.int32).reshape((-1, 1, 2))
    
    cv2.polylines(overlay, [pts], False, BEAM_GLOW, thick_outer, cv2.LINE_AA)
    cv2.polylines(overlay, [pts], False, BEAM_CORE, thick_core, cv2.LINE_AA)

    glow = cv2.GaussianBlur(overlay, (15, 15), 0)
    cv2.addWeighted(canvas, 1.0, glow,    1.5, 0, canvas)
    cv2.addWeighted(canvas, 1.0, overlay, 1.2, 0, canvas)


def _spawn_beam_particles(ox: float, oy: float,
                          direction: np.ndarray, speed: float,
                          count: int = 8) -> None:
    """Particle trail behind the lightning bolt (V1 design)."""
    s = _S
    if len(s["bp_pos"]) >= BEAM_MAX:
        return
    count   = min(count, BEAM_MAX - len(s["bp_pos"]))
    spread  = np.random.uniform(-0.20, 0.20, count)
    perp    = np.array([-direction[1], direction[0]], np.float32)
    speeds  = np.random.uniform(speed * 0.6, speed, count).astype(np.float32)
    vel     = (direction[np.newaxis, :] * speeds[:, np.newaxis] +
               perp[np.newaxis, :]     * (spread * speed)[:, np.newaxis]).astype(np.float32)
    pos     = np.full((count, 2), [ox, oy], np.float32)
    pos    += np.random.uniform(-4, 4, (count, 2))
    born    = np.full(count, time.time(), np.float32)
    life    = np.random.uniform(BEAM_LIFE * 0.5, BEAM_LIFE, count).astype(np.float32)
    size    = np.random.uniform(2, 6, count).astype(np.float32)

    s["bp_pos"]  = np.vstack([s["bp_pos"],  pos])
    s["bp_vel"]  = np.vstack([s["bp_vel"],  vel])
    s["bp_born"] = np.concatenate([s["bp_born"], born])
    s["bp_life"] = np.concatenate([s["bp_life"], life])
    s["bp_size"] = np.concatenate([s["bp_size"], size])


def _update_beam_particles(canvas: np.ndarray, h: int, w: int,
                           shield_active: bool,
                           scx: int, scy: int, sr: float) -> None:
    s = _S
    if len(s["bp_pos"]) == 0:
        return

    now     = time.time()
    elapsed = now - s["bp_born"]
    alive   = elapsed < s["bp_life"]

    # Vectorised shield deflection (V1 design)
    if shield_active and sr > 10:
        sc   = np.array([scx, scy], np.float32)
        diff = sc - s["bp_pos"]
        dist = np.linalg.norm(diff, axis=1)
        hit  = (dist < sr) & (dist > 1.0)
        if np.any(hit):
            normals          = diff[hit] / dist[hit, np.newaxis]
            v_hit            = s["bp_vel"][hit]
            dot              = np.einsum("ij,ij->i", v_hit, normals)[:, np.newaxis]
            s["bp_vel"][hit] = v_hit - 2 * dot * normals
            s["bp_pos"][hit] -= normals * 3

    if not np.all(alive):
        for key in ("bp_pos", "bp_vel", "bp_born", "bp_life", "bp_size"):
            s[key] = s[key][alive]
        elapsed = elapsed[alive]

    if len(s["bp_pos"]) == 0:
        return

    s["bp_vel"] *= BEAM_DRAG
    s["bp_pos"] += s["bp_vel"]

    lr   = elapsed / s["bp_life"]
    alph = np.clip(1.0 - lr, 0.0, 1.0) ** 1.2
    dp   = s["bp_pos"].astype(np.int32)
    ds   = np.maximum(s["bp_size"].astype(np.int32), 1)
    ok   = ((dp[:, 0] > -20) & (dp[:, 0] < w + 20) &
            (dp[:, 1] > -20) & (dp[:, 1] < h + 20))

    for i in np.where(ok)[0]:
        px, py = dp[i]; sz = ds[i]; a = float(alph[i])
        cv2.circle(canvas, (px, py), sz + 3,
                   tuple(int(c * a * 0.6) for c in BEAM_GLOW), -1, cv2.LINE_AA)
        cv2.circle(canvas, (px, py), max(1, sz),
                   tuple(int(c * a) for c in BEAM_CORE), -1, cv2.LINE_AA)


# ══════════════════════════════════════════════════════════════════════════════
#  AMBIENT PARTICLES  (fall down, bounce off shield)
# ══════════════════════════════════════════════════════════════════════════════

def _update_ambient(canvas: np.ndarray, h: int, w: int,
                    shield_active: bool,
                    scx: int, scy: int, sr: float) -> None:
    s   = _S
    pos = s["ap_pos"]
    vel = s["ap_vel"]

    # Shield collision (Vectorised)
    if shield_active and sr > 0:
        dx = pos[:, 0] - scx
        dy = pos[:, 1] - scy
        dist = np.hypot(dx, dy)
        hit = (dist < sr + 10) & (dist > 0.5)
        
        if np.any(hit):
            nx = dx[hit] / dist[hit]
            ny = dy[hit] / dist[hit]
            vx = vel[hit, 0]
            vy = vel[hit, 1]
            dot = vx * nx + vy * ny
            
            vel[hit, 0] = vx - 2 * dot * nx + nx * 5
            vel[hit, 1] = vy - 2 * dot * ny + ny * 5
            pos[hit, 0] = scx + nx * (sr + 15)
            pos[hit, 1] = scy + ny * (sr + 15)

    # Physics
    vel[:, 1] += 0.05
    vel       *= 0.99
    pos       += vel

    # Horizontal wrap; respawn off top when below bottom
    pos[:, 0] = np.mod(pos[:, 0], w)
    out_b = pos[:, 1] > h
    n_out = np.count_nonzero(out_b)
    if n_out > 0:
        pos[out_b, 0] = np.random.uniform(0, w, n_out)
        pos[out_b, 1] = -10
        vel[out_b, 0] = np.random.uniform(-1, 1, n_out)
        vel[out_b, 1] = np.random.uniform(2, 5, n_out)

    px_py = pos.astype(np.int32)
    valid = (px_py[:, 0] >= 0) & (px_py[:, 0] < w) & (px_py[:, 1] >= 0) & (px_py[:, 1] < h)
    for px, py in px_py[valid]:
        cv2.circle(canvas, (px, py), 2, (150, 150, 255), -1, cv2.LINE_AA)
        cv2.circle(canvas, (px, py), 4, (100, 100, 200),  1, cv2.LINE_AA)


# ══════════════════════════════════════════════════════════════════════════════
#  APPLY — main entry point
# ══════════════════════════════════════════════════════════════════════════════

def apply(canvas: np.ndarray, pose: PoseResult, **kwargs) -> np.ndarray:
    h, w = canvas.shape[:2]
    _ensure_init(w, h)

    t  = time.time()
    dt = max(0.001, t - _S["prev_t"])
    _S["prev_t"] = t

    # ── Defaults ──────────────────────────────────────────────────────────
    shield_active   = False
    left_open       = False
    right_open      = False
    left_point      = False
    right_point     = False
    left_straight   = False
    right_straight  = False
    both_straight   = False
    shoulder_w      = 120.0
    lm: dict        = {}
    vis: dict       = {}

    if pose.detected:
        lm  = pose.landmarks
        vis = pose.visibility

        if 11 in lm and 12 in lm:
            shoulder_w = max(40.0, float(np.hypot(
                lm[11][0] - lm[12][0], lm[11][1] - lm[12][1])))

        # ── Raw gesture detection ──────────────────────────────────────────
        gh = _S["g_hist"]

        lf_open_raw = _is_open_palm(lm, vis, 15, 13,
                                    [17, 19, 21], [33, 35, 37])
        rf_open_raw = _is_open_palm(lm, vis, 16, 14,
                                    [18, 20, 22], [34, 36, 38])

        lf_pt_raw = (_is_pointing(lm, vis, 15, 13, 19, 35,
                                  [17, 21], [33, 37])
                     if not lf_open_raw else False)
        rf_pt_raw = (_is_pointing(lm, vis, 16, 14, 20, 36,
                                  [18, 22], [34, 38])
                     if not rf_open_raw else False)

        left_open  = _vote(gh["left_open"],   lf_open_raw)
        right_open = _vote(gh["right_open"],  rf_open_raw)
        left_point = _vote(gh["left_point"],  lf_pt_raw)
        right_point= _vote(gh["right_point"], rf_pt_raw)

        # ── Straight-arm detection ─────────────────────────────────────────
        # Shoulder(11/12) → Elbow(13/14) → Wrist(15/16) angle > 155°
        left_straight  = _vote(gh["left_straight"],
                               _is_arm_straight(lm, vis, 11, 13, 15))
        right_straight = _vote(gh["right_straight"],
                               _is_arm_straight(lm, vis, 12, 14, 16))
        both_straight  = left_straight and right_straight

        # ── Hand speed tracking ────────────────────────────────────────────
        for side, idx_lm in [("left", 19), ("right", 20)]:
            if idx_lm in lm and vis.get(idx_lm, 0) > VIS:
                cur = np.array(lm[idx_lm], np.float32)
                if _S["prev_idx"][side] is not None:
                    spd = float(np.linalg.norm(cur - _S["prev_idx"][side])) / dt
                    _S["hand_speed"][side] = _S["hand_speed"][side] * 0.6 + spd * 0.4
                _S["prev_idx"][side] = cur
            else:
                _S["prev_idx"][side]   = None
                _S["hand_speed"][side] = 0.0

        # ── Gesture priority ──────────────────────────────────────────────
        # Shield   : both wrists come within ~60% of shoulder width of each other
        # Rune     : one palm open, not pointing, shield not active
        # Beam     : pointing (per-hand)
        if 15 in lm and 16 in lm and vis.get(15, 0) > VIS and vis.get(16, 0) > VIS:
            lx, ly = lm[15]; rx, ry = lm[16]
            hand_dist = math.hypot(lx - rx, ly - ry)
            if hand_dist < shoulder_w * 0.60:
                shield_active = True

    # ══════════════════════════════════════════════════════════════════════
    #  1.  MAGIC SHIELD
    # ══════════════════════════════════════════════════════════════════════
    if shield_active and 15 in lm and 16 in lm:
        cx = (lm[15][0] + lm[16][0]) // 2
        cy = (lm[15][1] + lm[16][1]) // 2
        _S["shield_cx"] = cx
        _S["shield_cy"] = cy
        target_r = shoulder_w * 1.65
        _S["shield_r"] = _S["shield_r"] * (1 - SHIELD_GROW) + target_r * SHIELD_GROW
    else:
        _S["shield_r"] = max(0.0, _S["shield_r"] * SHIELD_SHRINK)

    _draw_shield(canvas,
                 _S["shield_cx"], _S["shield_cy"],
                 _S["shield_r"], t, h, w)

    # ══════════════════════════════════════════════════════════════════════
    #  2.  AMBIENT PARTICLES  (drawn early so shield/beam render on top)
    # ══════════════════════════════════════════════════════════════════════
    _update_ambient(canvas, h, w,
                    shield_active,
                    _S["shield_cx"], _S["shield_cy"], _S["shield_r"])

    # ══════════════════════════════════════════════════════════════════════
    #  3.  MYSTIC RUNE CIRCLE
    # ══════════════════════════════════════════════════════════════════════
    rune_sides = []
    if not shield_active:
        if left_open  and not left_point:
            rune_sides.append(("left",  15))
        if right_open and not right_point:
            rune_sides.append(("right", 16))

    active_rune_sides = {s for s, _ in rune_sides}
    for side, wrist_lm in rune_sides:
        cx, cy = lm[wrist_lm][0], lm[wrist_lm][1]
        _S["rune_pos"][side]   = (cx, cy)
        target_r = shoulder_w * 0.45
        _S["rune_r"][side]     = (_S["rune_r"][side] * (1 - RUNE_GROW)
                                  + target_r           * RUNE_GROW)
        _S["rune_angle"][side] += dt * RUNE_SPIN_SPEED[0]

    for side in ("left", "right"):
        if side not in active_rune_sides:
            _S["rune_r"][side] = max(0.0, _S["rune_r"][side] * RUNE_SHRINK)

    for side in ("left", "right"):
        if _S["rune_r"][side] > 4:
            cx, cy = _S["rune_pos"][side]
            _draw_rune_circle(canvas, cx, cy,
                              _S["rune_r"][side],
                              _S["rune_angle"][side],
                              t, h, w)

    # ══════════════════════════════════════════════════════════════════════
    #  4.  ARCANE ENERGY BEAM
    #      • Both arms straight → dual beam fired from both wrists
    #      • Single arm pointing → single beam from that hand
    # ══════════════════════════════════════════════════════════════════════
    if both_straight and not shield_active:
        # Dual simultaneous attack — high intensity, full speed
        for shoulder_lm, wrist_lm in [(11, 15), (12, 16)]:
            if shoulder_lm in lm and wrist_lm in lm:
                direction = _arm_fire_dir(lm, shoulder_lm, wrist_lm)
                ox = float(lm[wrist_lm][0])
                oy = float(lm[wrist_lm][1])
                angle = math.atan2(direction[1], direction[0])

                _draw_lightning(canvas, (int(ox), int(oy)),
                                angle, 1.0, t, w, h)
                _spawn_beam_particles(ox, oy, direction, BEAM_SPEED_MAX, count=18)

                # Large muzzle flash
                cv2.circle(canvas, (int(ox), int(oy)), 28, BEAM_GLOW, -1, cv2.LINE_AA)
                cv2.circle(canvas, (int(ox), int(oy)), 14, BEAM_CORE, -1, cv2.LINE_AA)
    else:
        # Single-arm pointing beam (original behaviour)
        for side, wrist_lm, index_lm, is_pt in [
            ("left",  15, 19, left_point),
            ("right", 16, 20, right_point),
        ]:
            if is_pt and wrist_lm in lm and index_lm in lm:
                spd       = _S["hand_speed"][side]
                direction = _beam_dir(lm, wrist_lm, index_lm)
                ox        = float(lm[index_lm][0])
                oy        = float(lm[index_lm][1])
                angle     = math.atan2(direction[1], direction[0])

                intensity = float(np.interp(spd, [0, 1000], [0.25, 1.0]))
                beam_spd  = float(np.clip(BEAM_SPEED_BASE + spd * 0.06,
                                          BEAM_SPEED_BASE, BEAM_SPEED_MAX))
                pcount    = int(np.interp(spd, [0, 300], [5, 16]))

                _draw_lightning(canvas, (int(ox), int(oy)),
                                angle, intensity, t, w, h)
                _spawn_beam_particles(ox, oy, direction, beam_spd, pcount)

                tip_glow = int(np.interp(spd, [0, 300], [8, 24]))
                cv2.circle(canvas, (int(ox), int(oy)),
                           tip_glow, BEAM_GLOW, -1, cv2.LINE_AA)
                cv2.circle(canvas, (int(ox), int(oy)),
                           max(4, tip_glow // 2), BEAM_CORE, -1, cv2.LINE_AA)

    _update_beam_particles(canvas, h, w,
                           shield_active,
                           _S["shield_cx"], _S["shield_cy"], _S["shield_r"])

    return canvas