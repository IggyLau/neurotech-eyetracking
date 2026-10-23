"""Cross-platform USB camera wrapper with a background grab thread.

Every frame is stamped with time.perf_counter_ns() the moment read() returns, so
eye and scene streams share one clock and can be aligned offline. Dropped frames
show up as gaps in those timestamps.
"""

from __future__ import annotations

import platform
import threading
import time
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

    def __init__(self, index: int, width=None, height=None, fps=None, fourcc=None,
                 exposure=None, flip=False, name="camera"):
        self.name = name
        self.index = index
        self.flip = flip
        self.cap = open_capture(index, width, height, fps, fourcc, exposure)
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
                   cam_cfg.get("flip", False), name=name)

    @property
    def actual(self) -> dict:
        return {
            "width": int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "height": int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            "fps_reported": float(self.cap.get(cv2.CAP_PROP_FPS)),
            "exposure": float(self.cap.get(cv2.CAP_PROP_EXPOSURE)),
        }

    @property
    def frame_count(self) -> int:
        return self._count

    def start(self) -> "Camera":
        self._running = True
        self._thread = threading.Thread(target=self._loop, name=f"{self.name}-grab", daemon=True)
        self._thread.start()
        return self

    def _loop(self):
        while self._running:
            ok, img = self.cap.read()
            t_ns = time.perf_counter_ns()
            if not ok:
                time.sleep(0.001)
                continue
            if self.flip:
                img = cv2.flip(img, -1)
            frame = Frame(img, t_ns, self._count)
            self._count += 1
            if self.on_frame is not None:
                self.on_frame(frame)
            with self._lock:
                self._latest = frame

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
