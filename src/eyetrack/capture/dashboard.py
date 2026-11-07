"""`eyetrack dashboard`: live health check of every camera, with pupil tracking on the NIR eye cameras.

Finding the cameras: every camera index is opened briefly. The OV9281 eye cameras are monochrome, so their frames
arrive as three identical colour channels; those are taken as the NIR cameras (eye, then eye2, in index order;
press s to swap them). A camera that sends only black frames at the OV9281's native 1280x800 is also counted as
NIR, with a warning, because that's how the eye camera looks with its exposure reset. The configured scene camera
is shown too if it isn't one of them. --no-scan uses the configured indices instead.

Per camera: measured frame rate against the requested one, dropped frames (gaps over 1.5x the median frame
interval), time since the last frame, and image brightness (all-black frames are flagged).
Per NIR camera, from pupil detection on the newest frame at display rate:
    confidence of the latest detection and a 10 s trace with the config's min_confidence line
    valid fraction over the last 2 s (found and confidence >= min_confidence, as in the offline pipeline)
    precision: RMS of the pupil centre's frame-to-frame movement over the last 1 s, in eye-image px
    where the pupil sits: the search-zone box and a trail of recent pupil centres
The status thresholds follow realtime/status.py (valid >= 80% ok, < 30% lost; brightness < 10 = dark).

The frame-rate watchdog is off here, so a camera that slows down or stops shows as such rather than being reset.

Keys: q quit, s swap the two eye cameras, r reset counters, p save a snapshot to the output folder.
Nothing else is saved; the summary printed on exit can be pasted into notes.
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from ..pupil.detector import DetectorParams, PupilResult, detect_pupil
from .camera import Camera, default_backend

OV9281_NATIVE = (1280, 800)
COL = {"ok": (60, 190, 60), "weak": (0, 170, 230), "bad": (50, 50, 230)}     # BGR
PANEL_W, IMG_H, INFO_H = 520, 325, 150


# ------------------------------------------------------------------ finding the NIR cameras

def is_mono(img: np.ndarray | None) -> bool:
    """Three (nearly) identical channels, as the OV9281 delivers."""
    if img is None:
        return False
    if img.ndim < 3:
        return True
    small = img[::8, ::8].astype(np.int16)
    return bool(np.abs(small[..., 0] - small[..., 1]).mean() < 2 and np.abs(small[..., 1] - small[..., 2]).mean() < 2)


def classify(img: np.ndarray | None, size: tuple[int, int]) -> str:
    """'nir', 'nir_black' (black frames at the OV9281 native size), 'black', 'colour' or 'none'."""
    if img is None:
        return "none"
    if img.max() == 0:
        return "nir_black" if tuple(size) == OV9281_NATIVE else "black"
    return "nir" if is_mono(img) else "colour"


def scan(max_index: int = 6, reads: int = 20) -> list[dict]:
    """Open each index briefly: [{"index", "size", "kind"}] for every camera that opens."""
    found = []
    for i in range(max_index):
        cap = cv2.VideoCapture(i, default_backend())
        if not cap.isOpened():
            continue
        img = None
        for _ in range(reads):
            ok, f = cap.read()
            if ok and f is not None:
                img = f
                if f.max() > 0:
                    break
        size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        cap.release()
        found.append({"index": i, "size": size, "kind": classify(img, size)})
    return found


def assign_roles(found: list[dict], scene_index: int | None) -> dict[str, int]:
    """NIR cameras -> eye, eye2 (index order); the configured scene index if it isn't NIR."""
    nir = [c["index"] for c in found if c["kind"] in ("nir", "nir_black")]
    roles = {}
    for role, idx in zip(("eye", "eye2"), nir):
        roles[role] = idx
    if scene_index is not None and scene_index not in nir and any(c["index"] == scene_index for c in found):
        roles["scene"] = scene_index
    return roles


# ------------------------------------------------------------------ per-camera statistics

@dataclass
class CamStats:
    """Frame timing from the camera's grab thread, and pupil results from the display loop."""
    target_fps: float
    min_confidence: float = 0.5
    stamps: deque = field(default_factory=lambda: deque(maxlen=4000))
    frames: int = 0
    gaps: int = 0
    pupils: deque = field(default_factory=lambda: deque(maxlen=2000))   # (t_s, found, conf, cx, cy, valid)
    _dt: float = 0.0            # running frame interval (cheap enough for the 120 fps grab thread)

    def on_frame(self, frame):
        t = frame.t_ns / 1e9
        if self.stamps:
            dt = t - self.stamps[-1]
            if self._dt and dt > 1.5 * self._dt:
                self.gaps += 1
            self._dt = dt if not self._dt else 0.95 * self._dt + 0.05 * min(dt, 3 * self._dt)
        self.stamps.append(t)
        self.frames += 1

    def measured_fps(self, now: float | None = None, window: float = 2.0) -> float:
        if len(self.stamps) < 2:
            return 0.0
        now = self.stamps[-1] if now is None else now
        recent = [s for s in self.stamps if s >= now - window]
        if len(recent) < 2:
            return 0.0
        return (len(recent) - 1) / max(recent[-1] - recent[0], 1e-6)

    def age(self, now: float) -> float:
        return now - self.stamps[-1] if self.stamps else float("inf")

    def add_pupil(self, t: float, r: PupilResult):
        valid = bool(r.found and r.confidence >= self.min_confidence)
        self.pupils.append((t, bool(r.found), float(r.confidence), float(r.cx), float(r.cy), valid))

    def recent(self, now: float, window: float) -> list[tuple]:
        return [p for p in self.pupils if p[0] >= now - window]

    def valid_fraction(self, now: float, window: float = 2.0) -> float:
        rows = self.recent(now, window)
        return float(np.mean([p[5] for p in rows])) if rows else float("nan")

    def precision_px(self, now: float, window: float = 1.0) -> float:
        xy = np.array([(p[3], p[4]) for p in self.recent(now, window) if p[5]])
        if len(xy) < 3:
            return float("nan")
        return float(np.sqrt(np.mean(np.sum(np.diff(xy, axis=0) ** 2, axis=1))))

    def reset(self):
        self.stamps.clear()
        self.pupils.clear()
        self.frames = self.gaps = 0
        self._dt = 0.0


def camera_status(st: CamStats, now: float, brightness: float, nir: bool) -> tuple[str, str]:
    """(level, reason) for the tile header."""
    fps = st.measured_fps(now)
    if st.age(now) > 1.0:
        return "bad", "NO FRAMES"
    if brightness < 1:
        return "bad", "BLACK FRAMES"
    if fps < 0.5 * st.target_fps:
        return "bad", f"LOW FPS ({fps:.0f}/{st.target_fps:g})"
    if brightness < 10:
        return "weak", "DARK (LEDs off / lens covered?)"
    if fps < 0.9 * st.target_fps:
        return "weak", f"FPS {fps:.0f}/{st.target_fps:g}"
    if nir:
        v = st.valid_fraction(now)
        if np.isfinite(v) and v < 0.3:
            return "bad", f"PUPIL LOST ({v:.0%} valid)"
        if np.isfinite(v) and v < 0.8:
            return "weak", f"PUPIL WEAK ({v:.0%} valid)"
    return "ok", "OK"


# ------------------------------------------------------------------ drawing

def _sparkline(st: CamStats, now: float, w: int, h: int, threshold: float, window: float = 10.0) -> np.ndarray:
    img = np.full((h, w, 3), 30, np.uint8)
    y_thr = int(h - threshold * (h - 4) - 2)
    cv2.line(img, (0, y_thr), (w, y_thr), (90, 90, 160), 1)
    pts = [(int((p[0] - (now - window)) / window * (w - 1)), int(h - p[2] * (h - 4) - 2), p[5])
           for p in st.recent(now, window)]
    for (x0, y0, _), (x1, y1, v) in zip(pts, pts[1:]):
        cv2.line(img, (x0, y0), (x1, y1), COL["ok"] if v else COL["bad"], 1)
    cv2.putText(img, "confidence, last 10 s", (4, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (170, 170, 170), 1)
    return img


def render_tile(role: str, index: int, img: np.ndarray | None, st: CamStats, now: float, nir: bool,
                result: PupilResult | None = None, zone=None, mirror: bool = False, warn: str = "") -> np.ndarray:
    """One camera's panel: image (with pupil, trail and search zone on NIR cameras) and a stats strip."""
    if img is None:
        view = np.zeros((IMG_H, PANEL_W, 3), np.uint8)
        brightness = 0.0
    else:
        vis = img.copy() if img.ndim == 3 else cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        brightness = float(vis[::8, ::8].mean())
        h, w = vis.shape[:2]
        if nir:
            if zone:
                x0, y0, x1, y1 = zone
                cv2.rectangle(vis, (int(x0 * w), int(y0 * h)), (int(x1 * w), int(y1 * h)), (80, 200, 80), 1)
            for p in st.recent(now, 2.0):
                if p[5]:
                    cv2.circle(vis, (int(p[3]), int(p[4])), 2, (255, 160, 0), -1)
            if result is not None and result.found:
                colour = (20, 255, 255) if result.confidence >= st.min_confidence else (0, 140, 255)
                cv2.ellipse(vis, result.ellipse, colour, 2)
                cv2.circle(vis, (int(result.cx), int(result.cy)), 3, (255, 255, 0), -1)
        if mirror:
            vis = cv2.flip(vis, 1)
        scale = min(PANEL_W / w, IMG_H / h)
        small = cv2.resize(vis, (int(w * scale), int(h * scale)))
        view = np.zeros((IMG_H, PANEL_W, 3), np.uint8)
        oy, ox = (IMG_H - small.shape[0]) // 2, (PANEL_W - small.shape[1]) // 2
        view[oy:oy + small.shape[0], ox:ox + small.shape[1]] = small
    level, reason = camera_status(st, now, brightness, nir)
    head = np.full((34, PANEL_W, 3), COL[level], np.uint8)
    kind = "NIR eye camera" if nir else "colour camera"
    cv2.putText(head, f"{role}  index {index}  ({kind})", (8, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(head, reason, (PANEL_W - 10 - 9 * len(reason), 23), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    info = np.full((INFO_H, PANEL_W, 3), 18, np.uint8)
    size = f"{img.shape[1]}x{img.shape[0]}" if img is not None else "-"
    lines = [f"fps {st.measured_fps(now):5.1f} / {st.target_fps:g}   dropped {st.gaps}   frames {st.frames}   {size}",
             f"brightness {brightness:5.1f}   last frame {min(st.age(now), 99) * 1000:4.0f} ms ago"]
    if nir:
        conf = result.confidence if (result is not None and result.found) else 0.0
        v, prec = st.valid_fraction(now), st.precision_px(now)
        lines.append(f"confidence {conf:4.2f}   valid (2 s) {v:4.0%}   precision {prec:4.2f} px RMS".replace("nan%", "  - "))
        if result is not None and result.found:
            lines.append(f"pupil at ({result.cx:.0f}, {result.cy:.0f})  diameter {result.diameter_px:.1f} px")
        else:
            lines.append("no pupil found")
    if warn:
        lines.append(warn)
    for k, line in enumerate(lines):
        cv2.putText(info, line, (8, 20 + 20 * k), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    (80, 180, 255) if line == warn else (220, 220, 220), 1)
    if nir:
        info[INFO_H - 42:INFO_H - 2, 4:PANEL_W - 4] = _sparkline(st, now, PANEL_W - 8, 40, st.min_confidence)
    return np.vstack([head, view, info])


def compose(tiles: list[np.ndarray], cols: int = 2) -> np.ndarray:
    if not tiles:
        return np.zeros((200, PANEL_W, 3), np.uint8)
    blank = np.zeros_like(tiles[0])
    rows = []
    for k in range(0, len(tiles), cols):
        row = tiles[k:k + cols]
        row += [blank] * (cols - len(row))
        rows.append(np.hstack([cv2.copyMakeBorder(t, 4, 4, 4, 4, cv2.BORDER_CONSTANT) for t in row]))
    return np.vstack(rows)


# ------------------------------------------------------------------ main loop

def run_dashboard(cfg: dict, use_scan: bool = True, max_index: int = 6):
    warns: dict[str, str] = {}
    if use_scan:
        print("Scanning camera indices...")
        found = scan(max_index)
        for c in found:
            print(f"  index {c['index']}: {c['size'][0]}x{c['size'][1]}  {c['kind']}")
        roles = assign_roles(found, cfg["scene_camera"]["index"])
        for c in found:
            if c["kind"] == "nir_black":
                role = next((r for r, i in roles.items() if i == c["index"]), None)
                if role:
                    warns[role] = "sent only black frames during the scan (exposure reset?); recheck below"
        n_nir = sum(r.startswith("eye") for r in roles)
        if n_nir < 2:
            print(f"  found {n_nir} NIR camera(s); expected 2 for both eyes")
    else:
        roles = {r: cfg[f"{r}_camera"]["index"] for r in ("eye", "eye2", "scene")}
    if not roles:
        raise SystemExit("No cameras found.")

    params = DetectorParams.from_config(cfg["pupil"])
    min_conf = cfg["pupil"]["min_confidence"]
    cams, stats = {}, {}

    def open_role(role: str, index: int):
        ccfg = dict(cfg["eye_camera" if role.startswith("eye") else "scene_camera"])
        if role == "eye2":
            ccfg = dict(cfg["eye2_camera"])
        ccfg["index"] = index
        # Observe only: no frame-rate watchdog, so the dashboard reports what the camera really does instead of
        # re-sending settings and reopening cameras (with two identical eye cameras, a reopen by index can land
        # on the wrong one). Exposure is still set once when streaming starts.
        ccfg["watchdog_min_fps"] = None
        cam = Camera.from_config(ccfg, role)
        st = CamStats(float(ccfg.get("fps") or 30), min_conf)
        cam.on_frame = st.on_frame
        return cam.start(), st

    try:
        for role, idx in roles.items():
            try:
                cams[role], stats[role] = open_role(role, idx)
            except RuntimeError as e:
                print(f"  {role} (index {idx}): {e}")
        results: dict[str, PupilResult | None] = {r: None for r in cams}
        last_index = {r: -1 for r in cams}
        t_start = time.perf_counter()
        print("Dashboard. q quit, s swap eye cameras, r reset counters, p snapshot")
        while True:
            now = time.perf_counter()
            tiles = []
            for role in ("eye", "eye2", "scene"):
                if role not in cams:
                    continue
                cam, st = cams[role], stats[role]
                f = cam.latest()
                nir = role.startswith("eye")
                if nir and f is not None and f.index != last_index[role]:
                    last_index[role] = f.index
                    r = detect_pupil(f.image, params)
                    results[role] = r
                    st.add_pupil(f.t_ns / 1e9, r)
                ccfg = cfg["eye2_camera" if role == "eye2" else ("eye_camera" if nir else "scene_camera")]
                warn = warns.get(role, "")
                if nir and f is not None and not is_mono(f.image) and f.image.max() > 0:
                    warn = "colour image: this may not be an NIR eye camera"
                tiles.append(render_tile(role, roles[role], f.image if f is not None else None, st, now, nir,
                                         results.get(role), params.search_zone if nir else None,
                                         bool(ccfg.get("mirror_preview")) if nir else False, warn))
            board = compose(tiles)
            cv2.putText(board, f"{time.perf_counter() - t_start:5.0f} s", (board.shape[1] - 80, board.shape[0] - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (150, 150, 150), 1)
            cv2.imshow("eyetrack camera dashboard", board)
            key = cv2.waitKey(15) & 0xFF
            if key == ord("q"):
                break
            if key == ord("s") and "eye" in cams and "eye2" in cams:
                for d in (cams, stats, results, last_index, roles):
                    d["eye"], d["eye2"] = d["eye2"], d["eye"]
                warns["eye"], warns["eye2"] = warns.get("eye2", ""), warns.get("eye", "")
            if key == ord("r"):
                for st in stats.values():
                    st.reset()
            if key == ord("p"):
                out = Path(cfg["recording"]["output_dir"]) / f"dashboard_{datetime.now():%Y%m%d_%H%M%S}.png"
                out.parent.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(out), board)
                print(f"  saved {out}")
    finally:
        for c in cams.values():
            c.stop()
        cv2.destroyAllWindows()
        now = time.perf_counter()
        print("Summary:")
        for role, st in stats.items():
            line = f"  {role:5s} index {roles.get(role)}: {st.measured_fps(now, 5):.1f} fps (target {st.target_fps:g}), " \
                   f"{st.gaps} dropped of {st.frames}"
            if role.startswith("eye") and st.pupils:
                conf = np.mean([p[2] for p in st.pupils if p[1]] or [0])
                valid = np.mean([p[5] for p in st.pupils])
                line += f", mean confidence {conf:.2f}, valid {valid:.0%}"
            print(line)
