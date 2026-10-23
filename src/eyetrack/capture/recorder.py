"""Record raw eye + scene video with per-frame timestamps.

Session layout (data/<session>/):
    eye.avi, eye_ts.csv        frame_index, t_ns
    scene.avi, scene_ts.csv
    markers.csv                t_ns, label   (keys pressed during recording)
    meta.json                  config + actual camera settings

Raw video is kept so every metric can be recomputed later with better settings.
"""

from __future__ import annotations

import csv
import json
import platform
import threading
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from .camera import Camera, Frame

# Keys that drop a labelled marker into markers.csv, for test protocols
MARKER_KEYS = {
    ord("1"): "lights_off", ord("2"): "lights_on",
    ord("3"): "task_start", ord("4"): "task_end",
    ord("b"): "blink_cue", ord("m"): "marker",
}


class StreamWriter:
    def __init__(self, path: Path, ts_path: Path, codec: str, fps: float):
        self.path, self.codec, self.fps = path, codec, fps
        self.writer: cv2.VideoWriter | None = None
        self.ts_file = open(ts_path, "w", newline="")
        self.ts = csv.writer(self.ts_file)
        self.ts.writerow(["frame_index", "t_ns"])
        self.n = 0
        self.lock = threading.Lock()

    def write(self, frame: Frame):
        with self.lock:
            if self.writer is None:
                h, w = frame.image.shape[:2]
                self.writer = cv2.VideoWriter(str(self.path), cv2.VideoWriter_fourcc(*self.codec),
                                              self.fps, (w, h))
            self.writer.write(frame.image)
            self.ts.writerow([self.n, frame.t_ns])
            self.n += 1

    def close(self):
        with self.lock:
            if self.writer is not None:
                self.writer.release()
            self.ts_file.close()


def new_session_dir(root: str | Path, name: str | None = None) -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = Path(root) / (f"{stamp}_{name}" if name else stamp)
    path.mkdir(parents=True, exist_ok=False)
    return path


def record(cfg: dict, name: str | None = None, duration: float | None = None,
           use_scene: bool = True, preview: bool = True, display=None) -> Path:
    """Record until q / duration / display finishes.

    display: optional object with .frame(elapsed_s) -> (image, label) | None, shown
    full-screen instead of the preview (calibration targets). A marker is written
    every time its label changes.
    """
    session = new_session_dir(cfg["recording"]["output_dir"], name)
    rec = cfg["recording"]
    cams: dict[str, Camera] = {"eye": Camera.from_config(cfg["eye_camera"], "eye")}
    if use_scene:
        cams["scene"] = Camera.from_config(cfg["scene_camera"], "scene")

    writers = {}
    for key, cam in cams.items():
        fps = cfg[f"{key}_camera"]["fps"]
        writers[key] = StreamWriter(session / f"{key}.avi", session / f"{key}_ts.csv",
                                    rec[f"{key}_codec"], fps)
        cam.on_frame = writers[key].write

    markers_file = open(session / "markers.csv", "w", newline="")
    markers = csv.writer(markers_file)
    markers.writerow(["t_ns", "label"])

    meta = {
        "created": datetime.now().isoformat(timespec="seconds"),
        "platform": platform.platform(),
        "clock": "time.perf_counter_ns",
        "config": cfg,
        "cameras": {k: c.actual for k, c in cams.items()},
    }

    print(f"Recording to {session}")
    print("Keys: q=stop  1=lights off  2=lights on  3=task start  4=task end  b=blink cue  m=marker")
    for cam in cams.values():
        cam.start()
    if display is not None:
        cv2.namedWindow("eyetrack targets", cv2.WINDOW_NORMAL)
        cv2.setWindowProperty("eyetrack targets", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    t0 = time.perf_counter()
    last_label = None
    try:
        while True:
            if duration and time.perf_counter() - t0 >= duration:
                break
            key = -1
            if display is not None:
                out = display.frame(time.perf_counter() - t0)
                if out is None:
                    break
                img, label = out
                if label != last_label and label is not None:
                    markers.writerow([time.perf_counter_ns(), label])
                    markers_file.flush()
                last_label = label
                cv2.imshow("eyetrack targets", img)
                key = cv2.waitKey(5) & 0xFF
            elif preview:
                tiles = []
                for k, cam in cams.items():
                    f = cam.latest()
                    if f is not None:
                        img = cv2.resize(f.image, (480, int(480 * f.image.shape[0] / f.image.shape[1])))
                        cv2.putText(img, f"{k}: {cam.frame_count}", (8, 22),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                        tiles.append(img)
                if tiles:
                    h = max(t.shape[0] for t in tiles)
                    tiles = [cv2.copyMakeBorder(t, 0, h - t.shape[0], 0, 0, cv2.BORDER_CONSTANT) for t in tiles]
                    cv2.imshow("eyetrack record", np.hstack(tiles))
                key = cv2.waitKey(15) & 0xFF
            else:
                time.sleep(0.05)
            if key == ord("q"):
                break
            if key in MARKER_KEYS:
                t_ns = time.perf_counter_ns()
                markers.writerow([t_ns, MARKER_KEYS[key]])
                markers_file.flush()
                print(f"  marker: {MARKER_KEYS[key]}")
    finally:
        elapsed = time.perf_counter() - t0
        for cam in cams.values():
            cam.stop()
        for w in writers.values():
            w.close()
        markers_file.close()
        if preview or display is not None:
            cv2.destroyAllWindows()
        meta["duration_s"] = round(elapsed, 3)
        meta["frames"] = {k: w.n for k, w in writers.items()}
        meta["fps_measured"] = {k: round(w.n / elapsed, 2) for k, w in writers.items()}
        (session / "meta.json").write_text(json.dumps(meta, indent=2))
        print(f"Saved {meta['frames']} frames in {elapsed:.1f}s -> fps {meta['fps_measured']}")
    return session
