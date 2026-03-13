"""
Extreme Filter — Thermal / Volcanic Theme
=========================================
  Default (open hand)  →  BLUE skeleton
  Closed fist          →  RED skeleton + SHIELD around that fist + palm glow
  Jump / Turn-around   →  WHITE bubble burst
"""
import cv2
import numpy as np
import random
import time
from pose_detector import PoseResult, CONNECTIONS
from skeleton_renderer import SkeletonRenderer
from utils.particle_system import ParticleSystem

# ── Components ────────────────────────────────────────────────────────────────
_renderer = SkeletonRenderer(line_color=(255, 50, 0))
_sparks   = ParticleSystem(max_particles=300)

# ── White bubble arrays ───────────────────────────────────────────────────────
_WB_POS    = np.zeros((0, 2), dtype=np.float32)
_WB_VEL    = np.zeros((0, 2), dtype=np.float32)
_WB_SIZES  = np.zeros(0,      dtype=np.float32)
_WB_STARTS = np.zeros(0,      dtype=np.float32)
_WB_LIVES  = np.zeros(0,      dtype=np.float32)
WB_MAX     = 200

# ── Shield radius per hand (smooth animate) ───────────────────────────────────
_shield_r = {"left": 0.0, "right": 0.0}

# ── Persistent state ──────────────────────────────────────────────────────────
_prev_y         = 0
_prev_lms       = {}
_last_jump_time = 0.0
_prev_facing    = None
_last_turn_time = 0.0
_VIS_THRESH     = 0.40

_fist_history = {"left": [], "right": []}
_HISTORY_LEN  = 6
_FIST_VOTES   = 4


# ══════════════════════════════════════════════════════════════════════════════
#  WHITE BUBBLE HELPERS
# ══════════════════════════════════════════════════════════════════════════════
def _spawn_white_bubbles(x, y, count, speed=8.0):
    global _WB_POS, _WB_VEL, _WB_SIZES, _WB_STARTS, _WB_LIVES
    count = max(0, min(count, WB_MAX - len(_WB_POS)))
    if count == 0:
        return
    pos    = np.full((count, 2), [x, y], dtype=np.float32) \
             + np.random.uniform(-10, 10, (count, 2))
    angles = np.random.uniform(0, 2 * np.pi, count)
    speeds = np.random.uniform(speed * 0.4, speed, count)
    vel    = np.stack([np.cos(angles)*speeds,
                       np.sin(angles)*speeds], axis=1).astype(np.float32)
    _WB_POS    = np.vstack([_WB_POS, pos])
    _WB_VEL    = np.vstack([_WB_VEL, vel])
    _WB_SIZES  = np.concatenate([_WB_SIZES,
                                  np.random.uniform(4, 14, count).astype(np.float32)])
    _WB_STARTS = np.concatenate([_WB_STARTS,
                                  np.full(count, time.time(), dtype=np.float32)])
    _WB_LIVES  = np.concatenate([_WB_LIVES,
                                  np.random.uniform(0.8, 1.8, count).astype(np.float32)])


def _update_draw_white_bubbles(canvas, h, w):
    global _WB_POS, _WB_VEL, _WB_SIZES, _WB_STARTS, _WB_LIVES
    if len(_WB_POS) == 0:
        return
    elapsed = time.time() - _WB_STARTS
    alive   = elapsed < _WB_LIVES
    if not np.all(alive):
        _WB_POS    = _WB_POS[alive];    _WB_VEL    = _WB_VEL[alive]
        _WB_SIZES  = _WB_SIZES[alive];  _WB_STARTS = _WB_STARTS[alive]
        _WB_LIVES  = _WB_LIVES[alive];  elapsed    = elapsed[alive]
    if len(_WB_POS) == 0:
        return
    _WB_VEL      *= 0.96
    _WB_POS      += _WB_VEL
    _WB_POS[:, 1] -= 0.8
    lr   = elapsed / _WB_LIVES
    alph = np.where(lr > 0.4,
                    np.clip(1.0 - (lr - 0.4) / 0.6, 0.0, 1.0), 1.0) ** 1.4
    dp   = _WB_POS.astype(np.int32)
    ds   = np.maximum(_WB_SIZES.astype(np.int32), 2)
    ok   = (dp[:,0]>-20)&(dp[:,0]<w+20)&(dp[:,1]>-20)&(dp[:,1]<h+20)
    for i in np.where(ok)[0]:
        px, py = dp[i]; sz = ds[i]; v = int(255 * alph[i])
        cv2.circle(canvas, (px, py), sz, (v, v, v), 1, cv2.LINE_AA)
        hl = (px - max(1, int(sz*0.3)), py - max(1, int(sz*0.3)))
        cv2.circle(canvas, hl, max(1, int(sz*0.2)), (255,255,255), -1, cv2.LINE_AA)


# ══════════════════════════════════════════════════════════════════════════════
#  GESTURE HELPERS
# ══════════════════════════════════════════════════════════════════════════════
def _vote(history, new_val, max_len, votes_needed):
    history.append(new_val)
    if len(history) > max_len:
        history.pop(0)
    return history.count(True) >= votes_needed


def _is_fist(lm, vis, wrist_idx, knuckle_idxs, tip_idxs):
    """
    Fist = majority of fingertips are closer to the wrist than their
    own base knuckle.  Low visibility threshold so it works while moving.
    """
    if wrist_idx not in lm or vis.get(wrist_idx, 0) < 0.25:
        return False
    wrist = np.array(lm[wrist_idx], dtype=np.float32)
    curled, checked = 0, 0
    for ki, ti in zip(knuckle_idxs, tip_idxs):
        if ki not in lm or ti not in lm:
            continue
        checked += 1
        d_knuckle = np.linalg.norm(np.array(lm[ki]) - wrist)
        d_tip     = np.linalg.norm(np.array(lm[ti]) - wrist)
        if d_tip < d_knuckle * 0.85:
            curled += 1
    return checked >= 2 and (curled / checked) >= 0.60


def _draw_shield(canvas, cx, cy, rad, t, fist_active, h, w):
    """Render the energy shield ring + glow + rotating spokes."""
    sc   = (0,   0, 255) if fist_active else (255, 80,   0)
    sc_d = (0,   0, 100) if fist_active else (120, 30,   0)
    sc_h = (80, 80, 255) if fist_active else (255, 160,  80)

    ov = np.zeros_like(canvas)
    cv2.circle(ov, (cx, cy), rad,      sc,   3, cv2.LINE_AA)
    cv2.circle(ov, (cx, cy), rad + 12, sc_d, 2, cv2.LINE_AA)
    cv2.circle(ov, (cx, cy), rad + 24, sc_d, 1, cv2.LINE_AA)

    small = cv2.resize(ov, (w // 4, h // 4))
    glow  = cv2.GaussianBlur(small, (21, 21), 0)
    cv2.add(canvas, cv2.resize(glow, (w, h)), dst=canvas)
    cv2.add(canvas, ov, dst=canvas)

    # Rotating hex spokes
    for deg in range(0, 360, 60):
        ar = np.radians(deg + t * 50)
        x1 = int(cx + rad * 0.55 * np.cos(ar))
        y1 = int(cy + rad * 0.55 * np.sin(ar))
        x2 = int(cx + rad * 0.92 * np.cos(ar))
        y2 = int(cy + rad * 0.92 * np.sin(ar))
        cv2.line(canvas, (x1, y1), (x2, y2), sc_h, 1, cv2.LINE_AA)


def _is_facing_camera(lm, vis):
    return (vis.get(0,  0) > 0.50 and
            vis.get(11, 0) > 0.40 and
            vis.get(12, 0) > 0.40)


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN APPLY
# ══════════════════════════════════════════════════════════════════════════════
def apply(canvas: np.ndarray, pose: PoseResult, **kwargs) -> np.ndarray:
    global _prev_y, _prev_lms, _last_jump_time, _prev_facing, _last_turn_time
    global _fist_history, _shield_r

    h, w = canvas.shape[:2]
    t    = time.time()

    _sparks.update()
    _update_draw_white_bubbles(canvas, h, w)

    # Defaults
    left_fist  = False
    right_fist = False
    skel_line  = (255,  50,  0)   # BLUE
    skel_joint = (255, 200, 150)
    skel_thick = 4
    pulse_base  = (200,  30,  0)
    pulse_outer = (255,  80, 30)
    pulse_inner = (255, 220, 200)

    if pose.detected:
        lm  = pose.landmarks
        vis = pose.visibility

        # ── 1. SCALE ──────────────────────────────────────────────────────
        scale_ref = 100.0
        if 11 in lm and 12 in lm and (vis.get(11,0)>0.3 or vis.get(12,0)>0.3):
            scale_ref = max(40.0, float(np.hypot(
                lm[11][0] - lm[12][0], lm[11][1] - lm[12][1])))
        jump_thresh   = max(20.0, scale_ref * 0.25)
        shield_target = scale_ref * 0.80   # shield radius when fist is closed

        # ── 2. FIST DETECTION ─────────────────────────────────────────────
        lf_raw = _is_fist(lm, vis, 15, [33,35,40,42], [37,39,41,43])
        rf_raw = _is_fist(lm, vis, 16, [34,36,40,42], [38,36,41,43])
        left_fist  = _vote(_fist_history["left"],  lf_raw, _HISTORY_LEN, _FIST_VOTES)
        right_fist = _vote(_fist_history["right"], rf_raw, _HISTORY_LEN, _FIST_VOTES)
        any_fist   = left_fist or right_fist

        # ── 3. COLOURS ────────────────────────────────────────────────────
        if any_fist:
            skel_line   = (0,   0, 255)   # RED
            skel_joint  = (255, 255, 255)
            skel_thick  = 6
            pulse_base  = (0,   0, 180)
            pulse_outer = (60,  60, 255)
            pulse_inner = (255, 255, 255)

        # ── 4. PLASMA PULSE ───────────────────────────────────────────────
        pulse_t = (t * 4) % 1.0
        for (a, b) in CONNECTIONS:
            if (a in lm and b in lm
                    and vis.get(a, 0) > _VIS_THRESH
                    and vis.get(b, 0) > _VIS_THRESH):
                p1 = np.array(lm[a]); p2 = np.array(lm[b])
                cv2.line(canvas, tuple(p1), tuple(p2), pulse_base, 2, cv2.LINE_AA)
                pp = (p1*(1-pulse_t) + p2*pulse_t).astype(int)
                cv2.circle(canvas, tuple(pp), 4, pulse_inner, -1, cv2.LINE_AA)
                cv2.circle(canvas, tuple(pp), 8, pulse_outer,  1, cv2.LINE_AA)

        # ── 5. SHIELD — one per fist, centred on that wrist ───────────────
        for side, wrist_idx, is_fist_hand in [
            ("left",  15, left_fist),
            ("right", 16, right_fist),
        ]:
            if is_fist_hand and wrist_idx in lm and vis.get(wrist_idx, 0) > 0.25:
                # Smoothly grow toward target radius
                _shield_r[side] = _shield_r[side] * 0.78 + shield_target * 0.22
                rad = max(20, int(_shield_r[side]))
                cx, cy = lm[wrist_idx]

                _draw_shield(canvas, cx, cy, rad, t, True, h, w)

                # Palm glow in the centre
                glow_rad = int(scale_ref * 0.25)
                ov2 = canvas.copy()
                cv2.circle(ov2, (cx, cy), glow_rad, (0, 0, 255), -1, cv2.LINE_AA)
                cv2.addWeighted(ov2, 0.45, canvas, 0.55, 0.0, canvas)
                cv2.circle(canvas, (cx,cy), max(10, int(glow_rad*0.4)),
                           (255,255,255), -1, cv2.LINE_AA)
                cv2.circle(canvas, (cx,cy), max(12, int(glow_rad*0.5)),
                           (0,255,255), 2, cv2.LINE_AA)

                # Sparks
                if random.random() < 0.4:
                    _sparks.spawn(cx, cy, count=4,
                                  color_fn=lambda: (0, random.randint(50,150), 255),
                                  speed_scale=4.0)
            else:
                # Smoothly shrink when fist opens
                _shield_r[side] = max(0.0, _shield_r[side] * 0.65)

        # ── 6. JUMP → white bubbles ───────────────────────────────────────
        if 23 in lm and 24 in lm:
            curr_hip_y = (lm[23][1] + lm[24][1]) // 2
            if (_prev_y != 0
                    and (_prev_y - curr_hip_y) > jump_thresh
                    and (t - _last_jump_time) > 0.8):
                _last_jump_time = t
                for hi in [23, 24]:
                    _spawn_white_bubbles(lm[hi][0], lm[hi][1], 35, speed=10.0)
            _prev_y = curr_hip_y

        # ── 7. TURN-AROUND → white bubbles ───────────────────────────────
        facing_now = _is_facing_camera(lm, vis)
        if _prev_facing is not None and facing_now != _prev_facing:
            if (t - _last_turn_time) > 1.0:
                _last_turn_time = t
                if 11 in lm and 12 in lm:
                    cx_b = (lm[11][0]+lm[12][0])//2
                    cy_b = (lm[11][1]+lm[12][1])//2
                    _spawn_white_bubbles(cx_b, cy_b, 50, speed=12.0)
        _prev_facing = facing_now

        _prev_lms = {idx: pt for idx, pt in lm.items()}

    # ── Particles & skeleton ──────────────────────────────────────────────────
    _sparks.draw(canvas)

    if pose.detected:
        _renderer.render_with_custom_color(
            pose, (h, w), canvas,
            line_color  = skel_line,
            joint_color = skel_joint,
            thickness   = skel_thick,
        )

    return canvas