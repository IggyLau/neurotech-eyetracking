"""Record raw eye + scene video with per-frame timestamps.

Session layout (data/<session>/):
    eye.avi, eye_ts.csv        frame_index, t_ns
    scene.avi, scene_ts.csv
    markers.csv                t_ns, label   (keys pressed during recording)
    meta.json                  config + actual camera settings
    live_pupil.csv             real-time per-frame results (realtime.enabled; see realtime/worker.py)
    live_blinks.csv            blink / dropout events found live
    live_summary.json          live processing stats: dropped frames, latency percentiles

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

from ..pupil.detector import DetectorParams, detect_pupil, draw_result
from ..viz.live import PupilTrace, with_trace
from ..realtime.replay import draw_live, live_stats_text
from .camera import Camera, Frame

# Keys that drop a labelled marker into markers.csv, for test protocols
MARKER_KEYS = {
    ord("1"): "lights_off", ord("2"): "lights_on",
    ord("3"): "task_start", ord("4"): "task_end",
    ord("b"): "blink_cue", ord("m"): "marker",
    ord(" "): "solved",          # arithmetic: pressed the moment a problem is solved
}
# A held key auto-repeats every ~0.1 s; presses of the same key closer than this count once.
KEY_DEBOUNCE_S = 1.0


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
           use_scene: bool = True, preview: bool = True, display=None, realtime: bool | None = None,
           cameras: dict | None = None, eye_model=None, backend: str | None = None,
           block_when_full: bool = False, write_video: bool = True) -> Path:
    """Record until q / duration / display finishes (or a replayed video ends).

    display: optional object with .frame(elapsed_s) -> (image, label) | None, shown
    full-screen instead of the preview (calibration targets). A marker is written
    every time its label changes.
    realtime: run detection + causal metrics on every eye frame and save live_pupil.csv
    (default: config realtime.enabled). cameras: {"eye": ..., "scene": ...} sources to use instead
    of opening the configured cameras (e.g. realtime.replay.ReplayCamera). block_when_full: make the
    eye source wait for the worker instead of dropping (as-fast-as-possible replays only).
    """
    session = new_session_dir(cfg["recording"]["output_dir"], name)
    rec = cfg["recording"]
    if cameras is not None:
        cams = dict(cameras)
    else:
        cams = {"eye": Camera.from_config(cfg["eye_camera"], "eye")}
        if use_scene:
            cams["scene"] = Camera.from_config(cfg["scene_camera"], "scene")

    if realtime is None:
        realtime = cfg.get("realtime", {}).get("enabled", False)
    proc = None
    if realtime:
        from ..realtime.worker import RealtimeProcessor
        if eye_model is None and cfg.get("realtime", {}).get("eye_model"):
            from ..realtime.online import load_eye_model
            eye_model = load_eye_model(cfg["realtime"]["eye_model"])
        proc = RealtimeProcessor(cfg, session, backend=backend, eye_model=eye_model,
                                 fps=cfg["eye_camera"]["fps"]).start()

    writers = {}
    for key, cam in cams.items():
        fps = cfg[f"{key}_camera"]["fps"]
        if write_video:
            writers[key] = StreamWriter(session / f"{key}.avi", session / f"{key}_ts.csv",
                                        rec[f"{key}_codec"], fps)
        write = writers[key].write if write_video else None
        if key == "eye" and proc is not None:
            def on_eye(frame, write=write):
                proc.submit(frame, block=block_when_full)   # hand off first: the encode below takes ~1 ms
                if write is not None:
                    write(frame)
            cam.on_frame = on_eye
        else:
            cam.on_frame = write

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

    # Without the real-time worker: preview-only pupil detection on the main thread, on the newest
    # frame at preview rate. With it, the preview just shows the worker's latest result.
    params = DetectorParams.from_config(cfg["pupil"])
    trace = PupilTrace(min_confidence=cfg["pupil"]["min_confidence"],
                       baseline_s=cfg["pupil_size"]["baseline_ms"] / 1000)
    last_eye_index, eye_vis = -1, None
    mirror = cfg["eye_camera"].get("mirror_preview", False)

    print(f"Recording to {session}")
    print("Keys: q=stop  1=lights off  2=lights on  3=task start  4=task end  b=blink cue  m=marker")
    for cam in cams.values():
        cam.start()
    if display is not None:
        cv2.namedWindow("eyetrack targets", cv2.WINDOW_NORMAL)
        cv2.setWindowProperty("eyetrack targets", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    t0 = time.perf_counter()
    last_label = None
    last_press: dict[int, float] = {}
    try:
        while True:
            if duration and time.perf_counter() - t0 >= duration:
                break
            if getattr(cams["eye"], "finished", False):      # replayed video ended
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
                    if f is None:
                        continue
                    if k == "eye" and proc is not None:
                        for row in proc.drain():
                            trace.add(row["t_capture_ns"] / 1e9, _as_result(row))
                        pf, row = proc.latest()
                        if pf is not None and pf.index != last_eye_index:
                            last_eye_index = pf.index
                            eye_vis = draw_live(pf.image, row, mirror)
                        if eye_vis is None:
                            continue
                        src = eye_vis
                    elif k == "eye":
                        if f.index != last_eye_index:
                            last_eye_index = f.index
                            img = cv2.flip(f.image, 1) if mirror else f.image   # preview only
                            r = detect_pupil(img, params)
                            trace.add(f.t_ns / 1e9, r)
                            eye_vis = draw_result(img, r)
                        src = eye_vis
                    else:
                        src = f.image
                    img = cv2.resize(src, (480, int(480 * src.shape[0] / src.shape[1])))
                    label = live_stats_text(proc, cam) if (k == "eye" and proc is not None) else f"{k}: {cam.frame_count}"
                    cv2.putText(img, label, (8, img.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    tiles.append(with_trace(img, trace) if k == "eye" else img)
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
                now = time.perf_counter()
                repeat = now - last_press.get(key, -1e9) < KEY_DEBOUNCE_S
                last_press[key] = now        # keeps a held key suppressed for as long as it repeats
                if repeat:
                    continue
                t_ns = time.perf_counter_ns()
                markers.writerow([t_ns, MARKER_KEYS[key]])
                markers_file.flush()
                trace.mark(t_ns / 1e9, MARKER_KEYS[key])
                print(f"  marker: {MARKER_KEYS[key]}")
                if display is not None and hasattr(display, "on_key"):
                    display.on_key(MARKER_KEYS[key], now - t0)
    finally:
        elapsed = time.perf_counter() - t0
        for cam in cams.values():
            cam.stop()
        if proc is not None:
            live = proc.stop()           # finishes every queued frame before returning
            meta["realtime"] = live
            print(f"Live processing: {live.get('processed')} frames at {live.get('processing_fps')} fps, "
                  f"dropped {live['dropped']}, latency p50 {live.get('latency_ms', {}).get('p50')} ms "
                  f"p95 {live.get('latency_ms', {}).get('p95')} ms")
        for w in writers.values():
            w.close()
        markers_file.close()
        if preview or display is not None:
            cv2.destroyAllWindows()
        meta["duration_s"] = round(elapsed, 3)
        meta["frames"] = {k: w.n for k, w in writers.items()} or {k: c.frame_count for k, c in cams.items()}
        meta["fps_measured"] = {k: round(n / elapsed, 2) for k, n in meta["frames"].items()}
        (session / "meta.json").write_text(json.dumps(meta, indent=2))
        print(f"Saved {meta['frames']} frames in {elapsed:.1f}s -> fps {meta['fps_measured']}")
    return session


def _as_result(row: dict):
    """Live row -> PupilResult, for the PupilTrace strip."""
    from ..pupil.detector import PupilResult
    return PupilResult(found=bool(row["found"]), axis_minor=row["axis_minor"], axis_major=row["axis_major"],
                       confidence=row["confidence"] if row["valid"] else 0.0)
