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
