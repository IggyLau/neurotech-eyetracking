"""Real-time processing worker: every eye frame, in order, off the camera and display threads.

    camera grab thread --submit()--> bounded queue --> worker --> OnlineProcessor --> live_pupil.csv
                                                                       \\--> latest()/drain() for the display

submit() never blocks the grab thread (unless block=True, used for as-fast-as-possible replays): when the
queue is full the frame is dropped from live processing and counted. The raw video is still recorded, so
`eyetrack process` sees every frame. Latency is measured from the frame's capture timestamp
(perf_counter_ns stamped by the grab thread when read() returns) to its result being ready; USB transfer
and the driver's MJPG decode happen before that stamp and are not included.

Backends:
  thread   worker thread in this process. OpenCV releases the GIL in its heavy calls, so detection runs
           alongside the grab and video-writer threads (measured to keep up at 120 fps: docs/realtime.md).
  process  worker process; frames go through a shared-memory ring, results come back over a queue.
           Isolates the worker from the GIL entirely at ~1 ms extra hand-off cost.
"""

from __future__ import annotations

import csv
import json
import math
import multiprocessing as mp
import queue
import threading
import time
from collections import deque
from pathlib import Path

import numpy as np

from .online import BLINK_COLUMNS, COLUMNS

_STOP = None


def _fmt(v):
    if isinstance(v, (bool, np.bool_)):
        return int(v)
    if isinstance(v, float):
        return "" if math.isnan(v) else f"{v:.4f}".rstrip("0").rstrip(".")
    return v


class LiveWriter:
    """CSV of per-frame live results (+ live_blinks.csv), flushed every flush_rows rows / flush_s seconds."""

    def __init__(self, out_dir: Path, flush_rows: int = 60, flush_s: float = 0.5):
        out_dir = Path(out_dir)
        self.f = open(out_dir / "live_pupil.csv", "w", newline="")
        self.w = csv.writer(self.f)
        self.w.writerow(COLUMNS)
        self.bf = open(out_dir / "live_blinks.csv", "w", newline="")
        self.bw = csv.writer(self.bf)
        self.bw.writerow(BLINK_COLUMNS)
        self.flush_rows, self.flush_s = flush_rows, flush_s
        self.n_unflushed, self.t_flush = 0, time.monotonic()
        self.n_events = 0

    def write(self, row: dict, events: list[dict]):
        self.w.writerow([_fmt(row.get(c, "")) for c in COLUMNS])
        self.n_unflushed += 1
        new = events[self.n_events:]
        for ev in new:
            self.bw.writerow([_fmt(ev[c]) for c in BLINK_COLUMNS])
        self.n_events = len(events)
        if new:
            self.bf.flush()
        if self.n_unflushed >= self.flush_rows or time.monotonic() - self.t_flush >= self.flush_s:
            self.f.flush()
            self.n_unflushed, self.t_flush = 0, time.monotonic()

    def close(self, events: list[dict]):
        for ev in events[self.n_events:]:
            self.bw.writerow([_fmt(ev[c]) for c in BLINK_COLUMNS])
        self.f.close()
        self.bf.close()


class _Stats:
    def __init__(self, late_ms: float):
        self.late_ms = late_ms
        self.latency: list[float] = []
        self.proc: list[float] = []
        self.first_t = self.last_done = None
        self.n_sacc_onsets = 0
        self._prev_flag = False
        self.n_valid = 0

    def add(self, row: dict, t_done_ns: int):
        self.latency.append(row["latency_ms"])
        self.proc.append(row["proc_ms"])
        if self.first_t is None:
            self.first_t = row["t_capture_ns"]
        self.last_done = t_done_ns
        self.n_valid += bool(row["valid"])
        if row["saccade_flag"] and not self._prev_flag:
            self.n_sacc_onsets += 1
        self._prev_flag = bool(row["saccade_flag"])

    def summary(self) -> dict:
        lat = np.array(self.latency) if self.latency else np.array([np.nan])
        n = len(self.latency)
        span = (self.last_done - self.first_t) / 1e9 if n > 1 else 0.0
        return {"processed": n, "processing_fps": round(n / span, 2) if span else 0.0,
                "latency_ms": {k: round(float(np.percentile(lat, q)), 3) for k, q in
                               (("p50", 50), ("p95", 95), ("p99", 99), ("max", 100))},
                "late_frames": int((lat > self.late_ms).sum()) if n else 0, "late_ms": self.late_ms,
                "proc_ms_mean": round(float(np.mean(self.proc)), 3) if n else None,
                "valid_rate": round(self.n_valid / n, 4) if n else 0.0,
                "saccade_onsets": self.n_sacc_onsets}


def _run_one(proc, img, t_ns, t_media_ns, index, t_dequeued_ns):
    """t_ns: capture stamp on this machine's perf_counter (latency); t_media_ns: time used by the metrics."""
    row = proc.process(img, t_media_ns, index)
    t_done = time.perf_counter_ns()
    row["queue_ms"] = (t_dequeued_ns - t_ns) / 1e6
    row["proc_ms"] = (t_done - t_dequeued_ns) / 1e6
    row["latency_ms"] = (t_done - t_ns) / 1e6
    row["t_capture_ns"] = t_ns          # not saved: same as t_ns for a camera, wall time for a replay
    return row, t_done


def _make_processor(cfg, eye_model_dict, fps):
    from ..pupil.eye_model import EyeModel
    from .online import OnlineProcessor
    em = EyeModel(**eye_model_dict) if eye_model_dict else None
    return OnlineProcessor(cfg, em, fps)


class RealtimeProcessor:
    """Front end used by the recorder / live view / replay bench; see module docstring."""

    def __init__(self, cfg: dict, out_dir: str | Path | None = None, backend: str | None = None,
                 eye_model=None, fps: float | None = None):
        rt = cfg.get("realtime", {})
        self.cfg = cfg
        self.out_dir = Path(out_dir) if out_dir else None
        self.backend = backend or rt.get("backend", "thread")
        self.queue_frames = int(rt.get("queue_frames", 240))
        self.late_ms = float(rt.get("late_ms", 25))
        self.flush_rows = int(rt.get("flush_rows", 60))
        self.fps = fps or cfg["eye_camera"].get("fps") or 120
        self.eye_model = eye_model
        self._em_dict = ({k: getattr(eye_model, k) for k in ("cx", "cy", "radius", "n_used", "residual_px")}
                         if eye_model is not None else None)
        self.submitted = self.dropped = 0
        self._recent: deque = deque(maxlen=32)          # recent frames by index, for latest()
        self._rows: deque = deque(maxlen=4096)          # results not yet drained by the display
        self._latest_row = None
        self._summary: dict | None = None
        self._t_start = None
        self._cpu0 = None

    # ------------------------------------------------------------------ lifecycle
    def start(self) -> "RealtimeProcessor":
        self._t_start = time.perf_counter()
        self._cpu0 = time.process_time()
        if self.backend == "thread":
            self._q: queue.Queue = queue.Queue(maxsize=self.queue_frames)
            self._thread = threading.Thread(target=self._thread_loop, name="eye-realtime", daemon=True)
            self._thread.start()
        elif self.backend == "process":
            ctx = mp.get_context("spawn")
            self._slots = self.queue_frames + 2
            self._free = ctx.Semaphore(self.queue_frames)
            self._cmd = ctx.Queue()
            self._res = ctx.Queue()
            ready = ctx.Event()
            self._child = ctx.Process(target=_process_main, name="eye-realtime", daemon=True,
                                      args=(self.cfg, str(self.out_dir) if self.out_dir else None, self._em_dict,
                                            self.fps, self.late_ms, self.flush_rows, self._cmd, self._res,
                                            self._free, ready))
            self._child.start()
            if not ready.wait(30):
                raise RuntimeError("real-time worker process did not start")
            self._shm = None
            self._seq = 0
            self._collector = threading.Thread(target=self._collect_loop, daemon=True)
            self._collector.start()
        else:
            raise ValueError(f"backend must be thread or process, got {self.backend!r}")
        return self

    def submit(self, frame, block: bool = False) -> bool:
        """Queue a camera Frame (image, t_ns, index). False if it was dropped because the worker is behind."""
        self.submitted += 1
        self._recent.append(frame)
        if self.backend == "thread":
            try:
                self._q.put(frame, block=block)
                return True
            except queue.Full:
                self.dropped += 1
                return False
        if not self._free.acquire(block):
            self.dropped += 1
            return False
        img = frame.image
        if self._shm is None:
            from multiprocessing import shared_memory
            self._shape, self._dtype = img.shape, img.dtype
            self._shm = shared_memory.SharedMemory(create=True, size=img.nbytes * self._slots)
            self._ring = np.ndarray((self._slots, *img.shape), img.dtype, buffer=self._shm.buf)
            self._cmd.put(("shm", self._shm.name, img.shape, img.dtype.str, self._slots))
        if img.shape != self._shape:
            self._free.release()
            self.dropped += 1
            return False
        slot = self._seq % self._slots
        self._seq += 1
        self._ring[slot] = img
        self._cmd.put(("frame", slot, frame.t_ns, getattr(frame, "t_media_ns", frame.t_ns), frame.index))
        return True

    def stop(self) -> dict:
        """Process everything still queued, close the files, return (and save) the summary."""
        if self._summary is not None:
            return self._summary
        if self.backend == "thread":
            self._q.put(_STOP)
            self._thread.join()
            worker = self._worker_summary
        else:
            self._cmd.put(_STOP)
            self._collector.join()
            self._child.join(10)
            worker = self._worker_summary
            if self._shm is not None:
                del self._ring
                self._shm.close()
                self._shm.unlink()
        wall = time.perf_counter() - self._t_start
        s = {"backend": self.backend, "submitted": self.submitted, "dropped": self.dropped,
             "queue_frames": self.queue_frames, **worker,
             "unit": "deg" if self.eye_model is not None else "px",
             "eye_model": self._em_dict, "wall_s": round(wall, 3)}
        s["cpu_cores_this_process"] = round((time.process_time() - self._cpu0) / wall, 3) if wall else None
        self._summary = s
        if self.out_dir is not None:
            (self.out_dir / "live_summary.json").write_text(json.dumps(s, indent=2, default=float))
        return s

    # ------------------------------------------------------------------ results for the display
    def drain(self) -> list[dict]:
        out = []
        while self._rows:
            out.append(self._rows.popleft())
        return out

    def latest(self):
        """(frame, row) of the newest processed frame, or (None, None)."""
        row = self._latest_row
        if row is None:
            return None, None
        for f in reversed(self._recent):
            if f.index == row["frame_index"]:
                return f, row
        return None, row

    def _publish(self, row):
        self._rows.append(row)
        self._latest_row = row

    # ------------------------------------------------------------------ thread backend
    def _thread_loop(self):
        proc = _make_processor(self.cfg, self._em_dict, self.fps)
        writer = LiveWriter(self.out_dir, self.flush_rows) if self.out_dir else None
        stats = _Stats(self.late_ms)
        cpu0 = time.thread_time()
        while True:
            frame = self._q.get()
            if frame is _STOP:
                break
            row, t_done = _run_one(proc, frame.image, frame.t_ns, getattr(frame, "t_media_ns", frame.t_ns),
                                   frame.index, time.perf_counter_ns())
            stats.add(row, t_done)
            if writer:
                writer.write(row, proc.blinks.events)
            self._publish(row)
        proc.close()
        if writer:
            writer.close(proc.blinks.events)
        self._worker_summary = {**stats.summary(), "worker_cpu_s": round(time.thread_time() - cpu0, 3),
                                "blink_events": _count_kinds(proc.blinks.events)}

    # ------------------------------------------------------------------ process backend
    def _collect_loop(self):
        while True:
            try:
                msg = self._res.get(timeout=0.5)
            except queue.Empty:
                if not self._child.is_alive():
                    self._worker_summary = {"error": f"worker process exited ({self._child.exitcode})"}
                    return
                continue
            if msg[0] == "row":
                self._publish(msg[1])
            elif msg[0] == "done":
                self._worker_summary = msg[1]
                return


def _count_kinds(events):
    out = {}
    for e in events:
        out[e["kind"]] = out.get(e["kind"], 0) + 1
    return out


def _process_main(cfg, out_dir, em_dict, fps, late_ms, flush_rows, cmd, res, free, ready):
    from multiprocessing import shared_memory
    proc = _make_processor(cfg, em_dict, fps)
    writer = LiveWriter(Path(out_dir), flush_rows) if out_dir else None
    stats = _Stats(late_ms)
    cpu0 = time.process_time()
    ready.set()
    shm = ring = None
    while True:
        msg = cmd.get()
        if msg is _STOP:
            break
        if msg[0] == "shm":
            _, name, shape, dtype, slots = msg
            shm = shared_memory.SharedMemory(name=name, track=False)
            ring = np.ndarray((slots, *shape), np.dtype(dtype), buffer=shm.buf)
            continue
        _, slot, t_ns, t_media_ns, index = msg
        t_deq = time.perf_counter_ns()      # perf_counter is system-wide on macOS/Linux: same clock as the parent
        row, t_done = _run_one(proc, ring[slot], t_ns, t_media_ns, index, t_deq)
        free.release()
        stats.add(row, t_done)
        if writer:
            writer.write(row, proc.blinks.events)
        res.put(("row", row))
    proc.close()
    if writer:
        writer.close(proc.blinks.events)
    res.put(("done", {**stats.summary(), "worker_cpu_s": round(time.process_time() - cpu0, 3),
                      "blink_events": _count_kinds(proc.blinks.events)}))
    del ring
    if shm is not None:
        shm.close()
