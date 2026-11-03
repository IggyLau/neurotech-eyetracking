"""Cross-platform USB camera wrapper with a background grab thread.

Every frame is stamped with time.perf_counter_ns() the moment read() returns, so
eye and scene streams share one clock and can be aligned offline. Dropped frames
show up as gaps in those timestamps.
"""

from __future__ import annotations

import platform
import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Callable

import cv2
import numpy as np


def default_backend() -> int:
    system = platform.system()
    if system == "Darwin":
        return cv2.CAP_AVFOUNDATION
    if system == "Windows":
        return cv2.CAP_MSMF
    return cv2.CAP_V4L2


def open_capture(index: int | str, width: int | None = None, height: int | None = None,
                 fps: float | None = None, fourcc: str | None = None,
                 exposure: float | None = None) -> cv2.VideoCapture:
    """index may also be a video file path, to replay a recording through the live tools."""
    if isinstance(index, str):
        cap = cv2.VideoCapture(index)
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video {index}")
        return cap
    cap = cv2.VideoCapture(index, default_backend())
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera {index}. Run `eyetrack probe` to list cameras.")
    # FOURCC must be set before size/fps on most UVC drivers
    if fourcc:
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*fourcc))
    if width:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    if height:
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    if fps:
        cap.set(cv2.CAP_PROP_FPS, fps)
    if exposure is not None:
        cap.set(cv2.CAP_PROP_EXPOSURE, exposure)
    return cap


@dataclass
class Frame:
    image: np.ndarray
    t_ns: int          # perf_counter_ns at grab
    index: int         # sequential frame number from this camera


class Camera:
    """Grabs frames on a thread so slow consumers (preview, processing) never throttle the camera.

    `on_frame` runs on the grab thread for every frame (use it for recording);
    `latest()` returns the newest frame for previews.
    """

    # Frame-rate watchdog: the U20CAM-9281M has dropped from 120 to 30 fps part-way through sessions and
    # stayed there until the camera was reopened (docs/design-history/hardware.md).
    WATCHDOG_WINDOW_S = 2.0
    WATCHDOG_MAX_WAIT_S = 30.0  # after each reopen that didn't help, wait twice as long before the next

    def __init__(self, index: int, width=None, height=None, fps=None, fourcc=None,
                 exposure=None, flip=False, name="camera", uvc=None, min_fps=None):
        self.name = name
        self.uvc_cfg = uvc          # UVC controls set over USB once streaming starts (see uvc.py)
        self.uvc_state: dict | str | None = None
        self.index = index
        self.flip = flip
        self._open_args = (index, width, height, fps, fourcc, exposure)
        self.cap = open_capture(*self._open_args)
        # below this for WATCHDOG_WINDOW_S: re-send UVC controls, then reopen. Off for video files
        self.min_fps = None if isinstance(index, str) else min_fps
        self.recoveries: list[dict] = []
        self._recent_ns: deque[int] = deque()
        self._stream_start_ns = 0
        self._step = 0              # 0 = healthy, 1 = UVC re-sent, waiting to see if it helped
        self._reopens = 0           # reopens in a row without the frame rate recovering
        self._need_uvc = bool(uvc)
        self.on_frame: Callable[[Frame], None] | None = None
        self._latest: Frame | None = None
        self._lock = threading.Lock()
        self._running = False
        self._thread: threading.Thread | None = None
        self._count = 0

    @classmethod
    def from_config(cls, cam_cfg: dict, name: str) -> "Camera":
        return cls(cam_cfg["index"], cam_cfg.get("width"), cam_cfg.get("height"),
                   cam_cfg.get("fps"), cam_cfg.get("fourcc"), cam_cfg.get("exposure"),
                   cam_cfg.get("flip", False), name=name, uvc=cam_cfg.get("uvc"),
                   min_fps=cam_cfg.get("watchdog_min_fps"))

    @property
    def actual(self) -> dict:
        return {
            "width": int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "height": int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            "fps_reported": float(self.cap.get(cv2.CAP_PROP_FPS)),
            "exposure": float(self.cap.get(cv2.CAP_PROP_EXPOSURE)),
            "uvc": self.uvc_state,
        }

    @property
    def frame_count(self) -> int:
        return self._count

    def start(self) -> "Camera":
        self._running = True
        self._stream_start_ns = time.perf_counter_ns()
        self._thread = threading.Thread(target=self._loop, name=f"{self.name}-grab", daemon=True)
        self._thread.start()
        return self

    def _loop(self):
        while self._running:
            ok, img = self.cap.read()
            t_ns = time.perf_counter_ns()
            if self.min_fps:
                self._watchdog(t_ns, ok)
            if not ok:
                time.sleep(0.001)
                continue
            if self.flip:
                img = cv2.flip(img, -1)
            if self._need_uvc:
                self._need_uvc = False
                self._apply_uvc()   # after the stream starts: opening the stream can reset the controls
            frame = Frame(img, t_ns, self._count)
            self._count += 1
            if self.on_frame is not None:
                self.on_frame(frame)
            with self._lock:
                self._latest = frame

    def _apply_uvc(self):
        from .uvc import apply
        try:
            self.uvc_state = apply(self.uvc_cfg)
            print(f"{self.name}: UVC controls {self.uvc_state}")
        except Exception as e:      # never stop a recording over this; the frame-rate indicator will show it
            self.uvc_state = f"failed: {e}"
            print(f"{self.name}: could not set UVC controls ({e}); exposure stays automatic")

    def _watchdog(self, t_ns: int, ok: bool):
        window_ns = int(self.WATCHDOG_WINDOW_S * 1e9)
        if ok:
            self._recent_ns.append(t_ns)
        while self._recent_ns and self._recent_ns[0] < t_ns - window_ns:
            self._recent_ns.popleft()
        wait_s = min(self.WATCHDOG_WINDOW_S * 2 ** self._reopens, self.WATCHDOG_MAX_WAIT_S)
        if t_ns - self._stream_start_ns < wait_s * 1e9:
            return                  # not long enough since the stream (re)started
        fps = len(self._recent_ns) / self.WATCHDOG_WINDOW_S
        if fps >= self.min_fps:
            self._step = self._reopens = 0
            return
        if self._step == 0 and self.uvc_cfg:
            action = "resend_uvc"
            self._apply_uvc()
            self._step = 1
        else:
            action = "reopen"
            self.cap.release()
            try:
                self.cap = open_capture(*self._open_args)
            except RuntimeError as e:   # e.g. unplugged; read() fails and the next window tries again
                action = f"reopen failed: {e}"
            self._need_uvc = bool(self.uvc_cfg)
            self._step = 0
            self._reopens += 1
        self.recoveries.append({"t_ns": t_ns, "frame": self._count, "fps": round(fps, 1), "action": action})
        print(f"{self.name}: {fps:.0f} fps, below {self.min_fps:g}; {action.replace('_', ' ')}")
        self._recent_ns.clear()
        self._stream_start_ns = t_ns

    def latest(self) -> Frame | None:
        with self._lock:
            return self._latest

    def stop(self):
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=2)
        self.cap.release()

    def __enter__(self):
        return self.start()

    def __exit__(self, *exc):
        self.stop()
