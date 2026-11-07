"""Synthetic IR eye images and eye-movement traces with known ground truth."""

from __future__ import annotations

import cv2
import numpy as np

RNG = np.random.default_rng(0)


def eye_image(cx: float, cy: float, axes=(40, 40), angle: float = 0.0, size=(640, 400),
              noise: float = 4.0, glint: bool = True) -> np.ndarray:
    """Dark-pupil IR eye: grey skin, darker iris, black pupil ellipse, optional corneal glint."""
    w, h = size
    img = np.full((h, w), 150, np.uint8)
    cv2.circle(img, (int(cx), int(cy)), int(max(axes) * 1.9), 95, -1)            # iris
    cv2.ellipse(img, ((cx, cy), axes, angle), 15, -1)                             # pupil
    if glint:
        cv2.circle(img, (int(cx + axes[0] * 0.35), int(cy - axes[1] * 0.35)), 4, 255, -1)
    img = cv2.GaussianBlur(img, (5, 5), 1.2)
    img = np.clip(img + RNG.normal(0, noise, img.shape), 0, 255).astype(np.uint8)
    return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)


def sphere_ellipse(center, radius, yaw_deg, pitch_deg, pupil_radius=20.0):
    """RotatedRect of a pupil circle on an eyeball sphere under orthographic projection."""
    yaw, pitch = np.radians(yaw_deg), np.radians(pitch_deg)
    v = np.array([np.sin(yaw) * np.cos(pitch), np.sin(pitch), -np.cos(yaw) * np.cos(pitch)])
    px, py = center[0] + radius * v[0], center[1] + radius * v[1]
    cos_phi = -v[2]
    minor, major = 2 * pupil_radius * cos_phi, 2 * pupil_radius
    radial = np.degrees(np.arctan2(v[1], v[0])) if np.hypot(v[0], v[1]) > 1e-9 else 0.0
    # first axis (width) along the radial direction = minor axis
    return ((px, py), (minor, major), radial)


def step_trace(fs=120.0, n_steps=10, step_deg=10.0, hold_s=1.0, sacc_ms=40.0, noise=0.05):
    """Horizontal square-wave of saccades between two targets; returns t, x, y, true onsets."""
    n = int(fs * hold_s * n_steps)
    t = np.arange(n) / fs
    x = np.zeros(n)
    onsets = []
    for k in range(1, n_steps):
        t0 = k * hold_s
        onsets.append(t0)
        target = step_deg if k % 2 else 0.0
        prev = 0.0 if k % 2 else step_deg
        tau = sacc_ms / 1000
        s = np.clip((t - t0) / tau, 0, 1)
        prog = 0.5 - 0.5 * np.cos(np.pi * s)  # smooth bell-shaped velocity profile
        sel = t >= t0
        x[sel] = prev + (target - prev) * prog[sel]
    x += RNG.normal(0, noise, n)
    y = RNG.normal(0, noise, n)
    return t, x, y, onsets


def _rot(axis: str, deg: float) -> np.ndarray:
    c, s = np.cos(np.radians(deg)), np.sin(np.radians(deg))
    if axis == "x":
        return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def gaze_to_eye_image(theta_x: float, theta_y: float, mount: dict, size=(640, 400)) -> np.ndarray:
    """Eye-camera image of an eye looking at angles (theta_x right, theta_y down) relative to the head.

    mount: {"center": (x, y), "radius": px, "pitch": deg, "yaw": deg} - how this eye camera sits; two eyes
    get different mounts to model cameras that aren't placed symmetrically.
    """
    tx, ty = np.radians(theta_x), np.radians(theta_y)
    v = np.array([np.sin(tx) * np.cos(ty), np.sin(ty), -np.cos(tx) * np.cos(ty)])
    v = _rot("y", mount.get("yaw", 0)) @ _rot("x", mount.get("pitch", 0)) @ v
    yaw, pitch = np.degrees(np.arctan2(v[0], -v[2])), np.degrees(np.arcsin(np.clip(v[1], -1, 1)))
    (cx, cy), axes, ang = sphere_ellipse(mount["center"], mount["radius"], yaw, pitch, pupil_radius=20)
    w, h = size
    img = np.full((h, w), 150, np.uint8)
    cv2.circle(img, (int(cx), int(cy)), 60, 95, -1)
    cv2.ellipse(img, ((cx, cy), axes, ang), 15, -1)
    img = cv2.GaussianBlur(img, (5, 5), 1.0)
    return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)


def write_board_session(session, eye_fs=60, scene_fs=30, dwell=1.2, lead=1.0, spread=15.0, head_deg=2.0,
                        mounts=None, scene_size=(1280, 720), scene_f=700.0, two_eyes=True):
    """A calibration-board session with known truth: eye.avi (+ eye2.avi), scene.avi, *_ts.csv, markers.csv,
    meta.json. The head turns slowly by up to head_deg, so markers drift in the scene image while the wearer
    keeps looking at them. Returns a function truth(t_s) -> scene px of the target being looked at.
    """
    import json
    from pathlib import Path
    import pandas as pd
    from eyetrack.calibration.board import GRID, target_labels
    from eyetrack.calibration.targets import marker_image

    session = Path(session)
    session.mkdir(parents=True, exist_ok=True)
    mounts = mounts or {"eye": {"center": (320, 200), "radius": 110, "pitch": 20, "yaw": 0},
                        "eye2": {"center": (300, 190), "radius": 125, "pitch": 30, "yaw": -8}}
    roles = ["eye", "eye2"] if two_eyes else ["eye"]
    labels, mfl = target_labels()
    T = lead + dwell * len(labels)
    W, H = scene_size
    markers_img = {m: cv2.resize(marker_image(size=60, marker_id=m), (70, 70), interpolation=cv2.INTER_NEAREST)
                   for m in GRID}

    def head(t):
        return head_deg * np.sin(2 * np.pi * t / 7.0), 0.5 * head_deg * np.sin(2 * np.pi * t / 5.0)

    def cam_angle(m, t):
        hx, hy = head(t)
        gx, gy = GRID[m]
        return gx * spread - hx, gy * spread - hy

    def to_px(ax, ay):
        return W / 2 + scene_f * np.tan(np.radians(ax)), H / 2 + scene_f * np.tan(np.radians(ay))

    def current(t):
        if t < lead:
            return 4
        i = min(int((t - lead) // dwell), len(labels) - 1)
        return mfl[labels[i]]

    t0 = 10**12
    offsets = {"eye": 0, "eye2": int(3e6), "scene": int(1e6)}
    for role in roles:
        n = int(T * eye_fs)
        t = np.arange(n) / eye_fs
        wr = cv2.VideoWriter(str(session / f"{role}.avi"), cv2.VideoWriter_fourcc(*"MJPG"), eye_fs, (640, 400))
        for ti in t:
            ax, ay = cam_angle(current(ti), ti)
            wr.write(gaze_to_eye_image(ax, ay, mounts[role]))
        wr.release()
        pd.DataFrame({"frame_index": np.arange(n), "t_ns": t0 + offsets[role] + (t * 1e9).astype(np.int64)}).to_csv(
            session / f"{role}_ts.csv", index=False)
    n = int(T * scene_fs)
    t = np.arange(n) / scene_fs
    wr = cv2.VideoWriter(str(session / "scene.avi"), cv2.VideoWriter_fourcc(*"MJPG"), scene_fs, (W, H))
    for ti in t:
        img = np.full((H, W, 3), 170, np.uint8)
        for m, mimg in markers_img.items():
            x, y = to_px(*cam_angle(m, ti))
            x0, y0 = int(x - 35), int(y - 35)
            if 0 <= x0 and x0 + 70 <= W and 0 <= y0 and y0 + 70 <= H:
                img[y0:y0 + 70, x0:x0 + 70] = mimg
        wr.write(img)
    wr.release()
    pd.DataFrame({"frame_index": np.arange(n), "t_ns": t0 + offsets["scene"] + (t * 1e9).astype(np.int64)}).to_csv(
        session / "scene_ts.csv", index=False)
    rows = [{"t_ns": t0 + int((lead + i * dwell) * 1e9), "label": lab} for i, lab in enumerate(labels)]
    pd.DataFrame(rows).to_csv(session / "markers.csv", index=False)
    (session / "meta.json").write_text(json.dumps({"calibration": {"type": "board", "marker_for_label": mfl}}))

    def truth(t_s):
        """t_s on the first eye's clock -> scene px of the target being looked at."""
        return np.array([to_px(*cam_angle(current(ti), ti)) for ti in np.atleast_1d(t_s)])
    return truth
