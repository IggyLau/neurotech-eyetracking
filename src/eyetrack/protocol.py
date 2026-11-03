"""Guided validation session: `eyetrack validate`.

1. Alignment: a live view of the eye camera with checks (pupil found, centred, sized right,
   frame rate) that turn green once the camera is placed well. SPACE starts the protocol.
2. Protocol: a full-screen, timed sequence recorded in one session, with markers at every step:
     baseline      20 s fixation on mid-grey
     targets       a dot at each of the 9 calibration positions, 2.5 s each, to measure the eye model's
                   radius (degrees per pixel) when screen.width_cm is set
     blinks        10 cued blinks, one every 3 s
     light         pupil light reflex: 8 s dark, then 3 x (3 s white screen, 7 s dark)
     reading       45 s reading a passage, black on white
     digit span    12 spoken digit lists to hold and repeat back, 4 each of 3, 5 and 7 digits, interleaved.
                   Each trial: 4 s fixation, digits spoken at 1 per second, 2 s hold, then say the digits
   The light and digit-span parts follow standard pupillometry paradigms: a luminance step after dark
   adaptation (pupil light reflex), and task-evoked dilation from holding digits in memory at constant
   luminance (Kahneman & Beatty 1966), where longer lists dilate more.
   Mental arithmetic (Hess & Polt 1964; Ahern & Beatty 1979) is available instead with
   `--effort arithmetic` or `--only arithmetic`: 2 multiplications each of easy, medium and hard.
3. Report: the session is processed and summarised (light reflex, effort dilation, blinks,
   reading, signal quality), shown on screen and saved as validation_report.json.

Before starting: screen brightness fixed and high, auto-brightness / True Tone / Night Shift off,
room lighting constant, ~57 cm from the screen.
"""

from __future__ import annotations

import copy
import dataclasses
import json
import platform
import subprocess
import tempfile
import textwrap
import wave
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import cv2
import numpy as np
import pandas as pd

GREY, BLACK, WHITE = 128, 0, 255
LEVELS = ("task_easy", "task_medium", "task_hard")
ANSWER_S = 4.0     # time to say the answer after the prompt
MIN_SOLVE_WINDOW_S = 1.0   # shortest stretch of solving (before any speech) worth measuring dilation over
SPAN_LENGTHS = (3, 5, 7)           # digits per list: low / medium / high load (7 stays within memory span)
SPAN_LEVELS = tuple(f"span_{n}" for n in SPAN_LENGTHS)
SPAN_DIGIT_S = 1.0                  # one digit per second, as in Kahneman & Beatty (1966)
SPAN_HOLD_S = 2.0                   # silent hold between the last digit and the recall prompt
SPAN_FIX_S = 4.0                    # fixation before each list; the epoch baseline is its last 0.5 s
EPOCH_LABELS = ["lights_on", "lights_off", *LEVELS, *SPAN_LEVELS]
EFFORT_PARTS = ("digit_span", "arithmetic")
PARTS = ("baseline", "targets", "blinks", "light", "reading", "digit_span")   # default; arithmetic is opt-in
ALL_PARTS = ("baseline", "targets", "blinks", "light", "reading", "digit_span", "arithmetic")
TARGET_S = 2.5                      # per target: ~0.3 s to get there, the last 1.2 s are measured

READING_PASSAGE = (
    "Rivers rarely follow the shortest path to the sea. Over thousands of years they bend, split and "
    "wander across their valleys, cutting into one bank while laying down sand and gravel on the other. "
    "A small change upstream, such as a fallen tree or a new bridge, can shift the current enough to move "
    "a whole bend within a few seasons. Engineers who manage rivers have learned that forcing water into a "
    "straight channel often makes floods worse downstream, because the water arrives faster and all at "
    "once. Many towns now do the opposite: they give the river room to spread out across meadows and "
    "wetlands, where it slows down, drops its sediment and soaks into the ground. The result looks untidy "
    "on a map, but it protects the houses further along the valley and gives fish and birds more places "
    "to live."
)


@dataclass
class Trial:
    level: str      # task_easy | task_medium | task_hard
    a: int
    b: int

    @property
    def problem(self) -> str:
        return f"{self.a} × {self.b}"


def arithmetic_trials(per_level: int = 5, seed: int | None = None) -> list[Trial]:
    """per_level problems at each difficulty, no repeats, interleaved so every level is spread across the
    session: each consecutive group of three holds one of each level in random order.

    easy: a small times-table fact (2-4 x 2-9); medium: two digits x one digit (12-29 x 2-9), getting harder
    through the session; hard: two digits x two digits (12-19 x 12-19). Multiples of 10 are left out (too easy).

    Medium problems are ranked by their product (a proxy for difficulty), split into per_level equal bins and one
    is drawn from each bin, so the k-th medium problem always comes from the k-th easiest bin.
    """
    rng = np.random.default_rng(seed)
    pools = {
        "task_easy": sorted({tuple(sorted((a, b))) for a in range(2, 5) for b in range(2, 10)}),
        "task_hard": [(a, b) for a in range(12, 20) for b in range(12, 20) if a < b],
    }
    picked = {lvl: [pool[i] for i in rng.choice(len(pool), per_level, replace=False)] for lvl, pool in pools.items()}
    medium = sorted(((a, b) for a in range(12, 30) for b in range(2, 10) if a % 10), key=lambda p: (p[0] * p[1], p))
    picked["task_medium"] = [bin_[rng.integers(len(bin_))] for bin_ in np.array_split(np.array(medium), per_level)]
    trials = []
    for k in range(per_level):
        for lvl in rng.permutation(LEVELS):
            a, b = picked[str(lvl)][k]
            trials.append(Trial(str(lvl), *(int(v) for v in rng.permutation([a, b]))))
    return trials


@dataclass
class SpanTrial:
    digits: str     # e.g. "4729"

    @property
    def level(self) -> str:
        return f"span_{len(self.digits)}"

    def recall_s(self) -> float:
        return 3.0 + 0.5 * len(self.digits)   # time to say the digits back


def span_trials(per_level: int = 4, seed: int | None = None) -> list[SpanTrial]:
    """per_level digit lists at each length (3, 5, 7), interleaved like the arithmetic: each consecutive group
    of three holds one of each length in random order. Digits 1-9, none repeated within a list (0 is left out:
    "zero" / "oh" is transcribed less reliably)."""
    rng = np.random.default_rng(seed)
    trials = []
    for _ in range(per_level):
        for n in rng.permutation(SPAN_LENGTHS):
            trials.append(SpanTrial("".join(str(d) for d in rng.choice(np.arange(1, 10), int(n), replace=False))))
    return trials


class DigitVoice:
    """Spoken digit lists from the macOS `say` voice, one digit every SPAN_DIGIT_S, played with afplay.

    Each digit is rendered once to a WAV and lists are assembled with exact spacing, so the timing comes
    from the audio file rather than from the speech engine. available is False off macOS (the digits are
    then shown on screen instead)."""

    def __init__(self):
        self.available = platform.system() == "Darwin"
        self.dir = Path(tempfile.mkdtemp(prefix="eyetrack_span_"))
        self._clips: dict[str, np.ndarray] = {}
        self.sr = 22050
        if self.available:
            try:
                for d in "123456789":
                    path = self.dir / f"{d}.wav"
                    subprocess.run(["say", "-o", str(path), f"--data-format=LEI16@{self.sr}", d],
                                   check=True, capture_output=True, timeout=20)
                    with wave.open(str(path)) as w:
                        self._clips[d] = np.frombuffer(w.readframes(w.getnframes()), np.int16)
            except (OSError, subprocess.SubprocessError) as e:
                print(f"Spoken digits unavailable ({e}); showing them on screen instead.")
                self.available = False

    def prepare(self, digits: str) -> Path:
        out = np.zeros(int(self.sr * SPAN_DIGIT_S * len(digits)), np.int16)
        for k, d in enumerate(digits):
            clip = self._clips[d][:int(self.sr * SPAN_DIGIT_S)]
            start = int(self.sr * SPAN_DIGIT_S * k)
            out[start:start + len(clip)] = clip
        path = self.dir / f"list_{digits}.wav"
        with wave.open(str(path), "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(self.sr)
            w.writeframes(out.tobytes())
        return path

    def play(self, path: Path):
        subprocess.Popen(["afplay", str(path)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


# ---------------------------------------------------------------- drawing

_FONTS: dict[int, object] = {}


def _font(size: int):
    from PIL import ImageFont
    if size not in _FONTS:
        for path in ("/System/Library/Fonts/Helvetica.ttc", "/System/Library/Fonts/SFNS.ttf",
                     "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", "C:/Windows/Fonts/arial.ttf"):
            try:
                _FONTS[size] = ImageFont.truetype(path, size)
                break
            except OSError:
                continue
        else:
            _FONTS[size] = ImageFont.load_default(size)
    return _FONTS[size]


def text_block(img: np.ndarray, lines: list[str], size: int, color: int | tuple, top: int | None = None,
               left: int | None = None, spacing: float = 1.45) -> np.ndarray:
    """Draw lines of text; centred horizontally unless `left` is given, vertically unless `top` is."""
    from PIL import Image, ImageDraw
    pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil)
    font = _font(size)
    rgb = (color, color, color) if isinstance(color, int) else tuple(reversed(color))
    step = int(size * spacing)
    y = top if top is not None else (img.shape[0] - step * len(lines)) // 2
    for line in lines:
        w = draw.textlength(line, font=font)
        x = left if left is not None else int((img.shape[1] - w) / 2)
        draw.text((x, y), line, font=font, fill=rgb)
        y += step
    return cv2.cvtColor(np.asarray(pil), cv2.COLOR_RGB2BGR)


def canvas(screen: tuple[int, int], level: int) -> np.ndarray:
    return np.full((screen[1], screen[0], 3), level, np.uint8)


def cross(img: np.ndarray, color: int, size: int = 18, thickness: int = 3) -> np.ndarray:
    h, w = img.shape[:2]
    c = (color, color, color)
    cv2.line(img, (w // 2 - size, h // 2), (w // 2 + size, h // 2), c, thickness, cv2.LINE_AA)
    cv2.line(img, (w // 2, h // 2 - size), (w // 2, h // 2 + size), c, thickness, cv2.LINE_AA)
    return img


# ---------------------------------------------------------------- protocol timeline

@dataclass
class Step:
    label: str          # written to markers.csv when it changes
    duration: float
    draw: Callable[[float], np.ndarray]   # seconds into this step -> screen image
    static: bool = True  # image doesn't change within the step, so it is drawn once
    skip_on: str | None = None   # a key marker that ends the step early ("solved": SPACE in arithmetic)
    min_skip_s: float = 0.3      # ignore that key for this long after the step starts


def target_order() -> list[int]:
    """The 9 target positions in a fixed shuffled order, so no two neighbours follow each other in a line."""
    return [4, 0, 8, 2, 6, 1, 7, 3, 5]


def build_steps(screen=(1440, 900), short: bool = False, parts=PARTS, trials: list[Trial] | None = None,
                break_s: float = 20, spans: list[SpanTrial] | None = None,
                voice: DigitVoice | None = None) -> list[Step]:
    """The protocol. short=True keeps every part but with fewer repetitions (for a quick try);
    parts selects which parts to run; trials are the arithmetic problems (default: 2 per level), spans the
    digit lists (default: 4 per length); voice speaks the digits (None: they are shown on screen)."""
    S = screen
    n_blinks, n_light = (3, 1) if short else (10, 3)
    if trials is None:
        trials = arithmetic_trials(1 if short else 2)
    if spans is None:
        spans = span_trials(1 if short else 4)
    steps: list[Step] = []
    number = iter(range(1, 10))

    def intro(label, title, body, seconds, level=GREY):
        def draw(t, title=title, body=body):
            img = canvas(S, level)
            ink = BLACK if level > 100 else 200
            img = text_block(img, [title], 44, ink, top=S[1] // 2 - 170)
            img = text_block(img, textwrap.wrap(body, 60), 28, ink, top=S[1] // 2 - 90)
            return text_block(img, [f"starting in {max(0, int(np.ceil(seconds - t)))}"], 26, ink, top=S[1] - 140)
        steps.append(Step(label, seconds, draw, static=False))

    def fixation(label, seconds, level, ink, note=None):
        def draw(t):
            img = cross(canvas(S, level), ink)
            return text_block(img, [note], 40, ink, top=S[1] // 2 + 60) if note else img
        steps.append(Step(label, seconds, draw))

    intro("intro", "Eye-tracker validation",
          "Keep your head still and look at the cross whenever one is shown. Each part starts with "
          "instructions. Press Q to stop at any time.", 8)

    if "baseline" in parts:
        intro("baseline_intro", f"{next(number)}. Baseline", "Look at the cross and relax. Blink normally.", 4)
        fixation("baseline", 20, GREY, BLACK)

    if "targets" in parts:
        from .calibration.eye_radius import TARGET_POINTS
        intro("targets_intro", f"{next(number)}. Dots", "Look at the centre of each dot as soon as it "
              "appears, and keep looking at it until it moves. Keep your head still.", 5)
        for k in target_order():
            x, y = int(TARGET_POINTS[k][0] * S[0]), int(TARGET_POINTS[k][1] * S[1])

            def dot(t, x=x, y=y):
                img = canvas(S, GREY)
                cv2.circle(img, (x, y), 14, BLACK, -1, cv2.LINE_AA)
                cv2.circle(img, (x, y), 3, WHITE, -1, cv2.LINE_AA)
                return img
            steps.append(Step(f"target_{k}", TARGET_S, dot))

    if "blinks" in parts:
        intro("blink_intro", f"{next(number)}. Blinks",
              "Look at the cross. Blink once each time the word BLINK appears.", 4)
        for _ in range(n_blinks):
            fixation("blink_cue", 0.6, GREY, BLACK, note="BLINK")
            fixation("blink_wait", 2.4, GREY, BLACK)

    if "light" in parts:
        intro("light_intro", f"{next(number)}. Light response",
              "Keep looking at the cross. The screen will go dark, then flash white a few times.", 5)
        fixation("lights_off", 8, BLACK, 90)
        for _ in range(n_light):
            fixation("lights_on", 3, WHITE, 60)
            fixation("lights_off", 7, BLACK, 90)

    if "reading" in parts:
        intro("reading_intro", f"{next(number)}. Reading", "Read the passage at your normal pace. It stays up "
              "for 45 seconds; if you finish early, start again from the top.", 5, level=WHITE)

        # Lines span the full screen width (4% margins), so the eye covers the whole horizontal range
        margin = int(S[0] * 0.04)
        per_char = _font(30).getlength(READING_PASSAGE) / len(READING_PASSAGE)
        lines = textwrap.wrap(READING_PASSAGE, int((S[0] - 2 * margin) / per_char))

        def reading(t):
            img = canvas(S, WHITE)
            return text_block(img, lines, 30, BLACK, top=(S[1] - int(30 * 1.6) * len(lines)) // 2,
                              left=margin, spacing=1.6)
        steps.append(Step("reading", 45, reading))

    if "digit_span" in parts:
        spoken = voice is not None and voice.available
        intro("span_intro", f"{next(number)}. Remembering digits",
              f"You'll {'hear' if spoken else 'see'} {len(spans)} short lists of digits, one digit per second. Keep "
              "looking at the cross and hold the list in your head, silently. Wait for 'say the digits' before "
              "saying anything, even under your breath: talking moves the pupil and spoils the measurement. Then "
              "say them out loud in order.", 12)
        for trial in spans:
            fixation("span_fix", SPAN_FIX_S, GREY, BLACK)
            n = len(trial.digits)
            clip = voice.prepare(trial.digits) if spoken else None

            def listen(t, digits=trial.digits, clip=clip, state={"played": False}):
                if clip is not None and not state["played"]:
                    state["played"] = True
                    voice.play(clip)
                img = cross(canvas(S, GREY), BLACK)
                k = int(t // SPAN_DIGIT_S)
                if clip is None and k < len(digits) and t % SPAN_DIGIT_S < 0.6:
                    img = text_block(img, [digits[k]], 40, 40, top=S[1] // 2 - 90)   # small, like the arithmetic
                return img
            steps.append(Step(trial.level, n * SPAN_DIGIT_S + SPAN_HOLD_S, listen, static=False))
            fixation("recall", trial.recall_s(), GREY, BLACK, note="say the digits")

    if "arithmetic" in parts:
        intro("arith_intro", f"{next(number)}. Mental arithmetic",
              f"{len(trials)} multiplications, easy to hard in mixed order. Each appears for 2 seconds. Solve it in "
              "your head while looking at the cross. Press SPACE (once) as soon as you have the answer, then say "
              "it out loud. After 10 seconds without SPACE, you'll be asked for your answer anyway.", 10)
        halfway = len(trials) // 2 if len(trials) >= 6 else None
        for k, trial in enumerate(trials):
            if k == halfway and break_s:
                intro("break", "Short break", "Rest your eyes and blink freely. Keep your head where it is.", break_s)
            fixation("arith_fix", 4, GREY, BLACK)

            def task(t, problem=trial.problem):
                img = cross(canvas(S, GREY), BLACK)
                if t < 2:   # problem shown small, just above the cross, so screen luminance barely changes
                    img = text_block(img, [problem], 40, 40, top=S[1] // 2 - 90)
                return img
            steps.append(Step(trial.level, 10, task, static=False, skip_on="solved"))
            fixation("answer", ANSWER_S, GREY, BLACK, note="say your answer")

    steps.append(Step("done", 2, lambda t: text_block(canvas(S, GREY), ["Done - processing..."], 40, BLACK)))
    return steps


class ProtocolDisplay:
    """Callable display for recorder.record(): (image, label) for each tick, None when finished."""

    def __init__(self, steps: list[Step], inset: bool = False, zone=None):
        self.steps = steps
        self.inset_zone = tuple(zone or ALIGN_ZONE) if inset else None   # recorder draws the live eye view bottom-right
        self.starts = np.cumsum([0.0] + [s.duration for s in steps])
        self._cache: dict[int, np.ndarray] = {}
        self.skipped = 0.0     # seconds jumped over by steps that ended early

    def _locate(self, elapsed: float) -> tuple[int, float]:
        e = elapsed + self.skipped
        i = int(np.searchsorted(self.starts, e, side="right")) - 1
        return i, e - (self.starts[i] if i < len(self.steps) else 0.0)

    def on_key(self, label: str, elapsed: float):
        """A key marker from the recorder. SPACE ("solved") during an arithmetic problem moves straight on
        to the answer prompt."""
        i, t = self._locate(elapsed)
        if i < len(self.steps) and self.steps[i].skip_on == label and t >= self.steps[i].min_skip_s:
            self.skipped += self.steps[i].duration - t

    @property
    def total_s(self) -> float:
        return float(self.starts[-1])

    def frame(self, elapsed: float):
        i, t = self._locate(elapsed)
        if i >= len(self.steps):
            return None
        step = self.steps[i]
        if step.static:
            if i not in self._cache:
                self._cache = {i: step.draw(0.0)}   # keep only the current step's image
            return self._cache[i], step.label
        return step.draw(t), step.label


# ---------------------------------------------------------------- alignment

@dataclass
class AlignState:
    found: bool = False
    centred: bool = False
    size_ok: bool = False
    fps_ok: bool = False
    valid_fraction: float = 0.0
    hint: str = ""

    @property
    def ready(self) -> bool:
        return self.found and self.centred and self.size_ok and self.fps_ok and self.valid_fraction >= 0.9


ALIGN_ZONE = (0.25, 0.20, 0.75, 0.80)   # target box for the pupil, as fractions of the eye image


def align_zone(cfg: dict) -> tuple[float, float, float, float]:
    """The box the pupil should be in: pupil.search_zone (where the detector looks) if set, else ALIGN_ZONE."""
    return tuple(cfg["pupil"].get("search_zone") or ALIGN_ZONE)


def mirror_zone(zone: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
    x0, y0, x1, y1 = zone
    return (1 - x1, y0, 1 - x0, y1)


def assess_alignment(result, shape: tuple[int, int], fps: float, recent_valid: list[bool],
                     zone=ALIGN_ZONE, size_range=(0.06, 0.25), min_conf=0.7) -> AlignState:
    """Is the pupil found, inside the target zone, a sensible size, at a good frame rate?"""
    h, w = shape
    st = AlignState(fps_ok=fps >= 100)
    st.found = bool(result.found and result.confidence >= min_conf)
    st.valid_fraction = float(np.mean(recent_valid)) if recent_valid else 0.0
    if not result.found:
        st.hint = "No pupil found: aim the camera at the eye and check the IR LEDs are on"
        return st
    x0, y0, x1, y1 = zone
    nx, ny = result.cx / w, result.cy / h
    st.centred = x0 <= nx <= x1 and y0 <= ny <= y1
    frac = result.diameter_px / w
    st.size_ok = size_range[0] <= frac <= size_range[1]
    if not st.centred:
        horiz = "left" if nx < x0 else "right" if nx > x1 else ""
        vert = "top" if ny < y0 else "bottom" if ny > y1 else ""
        st.hint = f"Pupil is too far to the {' '.join(p for p in (vert, horiz) if p)} of the image: re-aim the camera"
    elif not st.size_ok:
        st.hint = "Pupil looks small: move the camera closer" if frac < size_range[0] else \
            "Pupil looks large: move the camera further away"
    elif not st.found:
        st.hint = "Pupil found but confidence is low: refocus, or check for reflections on the pupil"
    elif not st.fps_ok:
        st.hint = f"Camera is running at {fps:.0f} fps: plug it in directly and close other apps"
    elif st.valid_fraction < 0.9:
        st.hint = "Tracking drops out: hold still and keep your eye open"
    else:
        st.hint = "Looks good. Look left and right: the pupil should stay tracked."
    return st


def draw_alignment(eye_vis: np.ndarray, st: AlignState, fps: float, ready_for: float, screen,
                   zone=ALIGN_ZONE) -> np.ndarray:
    S = screen
    img = canvas(S, 35)
    h, w = eye_vis.shape[:2]
    scale = min(S[0] * 0.62 / w, S[1] * 0.8 / h)
    vis = cv2.resize(eye_vis, (int(w * scale), int(h * scale)))
    vh, vw = vis.shape[:2]
    ox, oy = 40, (S[1] - vh) // 2
    img[oy:oy + vh, ox:ox + vw] = vis
    zx0, zy0, zx1, zy1 = zone
    cv2.rectangle(img, (ox + int(zx0 * vw), oy + int(zy0 * vh)), (ox + int(zx1 * vw), oy + int(zy1 * vh)),
                  (80, 200, 80) if st.centred else (60, 60, 230), 2)
    rows = [("Pupil found (confidence >= 0.7)", st.found), ("Pupil inside the box", st.centred),
            ("Pupil size", st.size_ok), (f"Frame rate {fps:.0f} fps (>= 100)", st.fps_ok),
            (f"Tracked {100 * st.valid_fraction:.0f}% of the last 2 s (>= 90%)", st.valid_fraction >= 0.9)]
    lx = ox + vw + 50
    img = text_block(img, ["Aim the eye camera"], 38, 235, top=oy, left=lx)
    y = oy + 80
    for label, ok in rows:
        color = (90, 210, 90) if ok else (80, 80, 235)
        cv2.circle(img, (lx + 10, y + 16), 9, color, -1, cv2.LINE_AA)
        img = text_block(img, [label], 24, 220, top=y, left=lx + 32)
        y += 46
    img = text_block(img, textwrap.wrap(st.hint, 34), 24, 200, top=y + 20, left=lx)
    status = ("Adjust until every check is green" if not st.ready else
              "Ready - press SPACE to start" if ready_for >= 2 else f"Hold still... {2 - ready_for:.1f} s")
    img = text_block(img, [status], 30, 240, top=S[1] - 150, left=lx)
    return text_block(img, ["SPACE start   F start anyway   Q quit"], 20, 150, top=S[1] - 90, left=lx)


def align_camera(cfg: dict, screen=(1440, 900)) -> bool:
    """Live alignment screen. Returns True to start the protocol, False if the user quit."""
    from .capture.camera import Camera
    from .pupil.detector import DetectorParams, detect_pupil, draw_result

    params = DetectorParams.from_config(cfg["pupil"])
    mirror = cfg["eye_camera"].get("mirror_preview", False)
    zone = mirror_zone(align_zone(cfg)) if mirror else align_zone(cfg)   # detection runs on the mirrored view
    if params.search_zone:
        params = dataclasses.replace(params, search_zone=zone)
    cam = Camera.from_config(cfg["eye_camera"], "eye").start()
    win = "eyetrack align"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(win, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    recent: list[tuple[float, bool]] = []
    counts: list[tuple[float, int]] = []   # (time, camera frame count) over the last second, for the frame rate
    ready_since = None
    last_index = -1
    try:
        while True:
            f = cam.latest()
            if f is None or f.index == last_index:
                if cv2.waitKey(2) & 0xFF == ord("q"):
                    return False
                continue
            last_index = f.index
            now = time.perf_counter()
            counts = [c for c in counts if c[0] > now - 1.0] + [(now, cam.frame_count)]
            span = counts[-1][0] - counts[0][0]
            fps = (counts[-1][1] - counts[0][1]) / span if span > 0.3 else 0.0
            img = cv2.flip(f.image, 1) if mirror else f.image   # mirror like Photo Booth (preview only)
            r = detect_pupil(img, params)
            recent = [(t, v) for t, v in recent if t > now - 2.0] + [(now, bool(r.found and r.confidence >= 0.5))]
            st = assess_alignment(r, img.shape[:2], float(fps), [v for _, v in recent], zone=zone)
            ready_since = (ready_since or now) if st.ready else None
            ready_for = now - ready_since if ready_since else 0.0
            cv2.imshow(win, draw_alignment(draw_result(img, r), st, float(fps), ready_for, screen, zone))
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                return False
            if key == ord("f") or (key == ord(" ") and ready_for >= 2):
                return True
    finally:
        cam.stop()
        cv2.destroyWindow(win)


# ---------------------------------------------------------------- analysis

def _phases(markers: pd.DataFrame, end_s: float) -> pd.DataFrame:
    m = markers.sort_values("t_s").reset_index(drop=True)
    m["end_s"] = list(m.t_s[1:]) + [end_s]
    return m


def analyze(session: str | Path, cfg: dict | None = None) -> dict:
    """cfg: the run's config (its `digit_span` block sets the digit-span cleaning); default config if None."""
    session = Path(session)
    if cfg is None:
        from .config import load_config
        cfg = load_config()
    summary = json.loads((session / "metrics_summary.json").read_text())
    pupil = pd.read_csv(session / "pupil.csv")
    markers = pd.read_csv(session / "markers_s.csv")
    blinks = pd.read_csv(session / "blinks.csv")
    sacc = pd.read_csv(session / "saccades.csv")
    fix = pd.read_csv(session / "fixations.csv")
    ep_path = session / "pupil_epochs_summary.csv"
    epochs = pd.read_csv(ep_path) if ep_path.exists() else pd.DataFrame(columns=["label"])
    ph = _phases(markers, float(pupil.t_s.max()))
    long_ep = pd.read_csv(session / "pupil_epochs.csv") if (session / "pupil_epochs.csv").exists() else None

    def window(label):
        return [(r.t_s, r.end_s) for r in ph.itertuples() if r.label == label]

    def in_windows(t, wins):
        return np.any([(t >= a) & (t < b) for a, b in wins], axis=0) if wins else np.zeros(len(t), bool)

    out: dict = {"session": session.name, "signal": {
        k: summary.get(k) for k in ("sample_rate_hz", "valid_rate", "mean_confidence", "noise_rms_s2s_deg", "unit")}}

    # baseline: pupil-size steadiness during plain fixation
    base = pupil[in_windows(pupil.t_s.to_numpy(), window("baseline"))]
    good = base[base.valid & ~base.in_blink].diameter_clean_px.dropna()
    out["baseline"] = {"valid_rate": float(base.valid.mean()) if len(base) else None,
                       "pupil_mean_px": float(good.mean()) if len(good) else None,
                       "pupil_cv_pct": float(100 * good.std() / good.mean()) if len(good) > 10 else None}

    # blinks: a cue counts as hit if a blink starts within 1.5 s after it
    cues = markers[markers.label == "blink_cue"].t_s.to_numpy()
    real = blinks[blinks.kind == "blink"].onset_s.to_numpy() if len(blinks) else np.array([])
    hits = sum(bool(np.any((real >= c) & (real <= c + 1.5))) for c in cues)
    out["blinks"] = {"cued": int(len(cues)), "detected_after_cue": int(hits),
                     "hit_rate": float(hits / len(cues)) if len(cues) else None}

    # pupil light reflex: change after each white flash, and recovery in the dark
    def ep(label, col):
        sel = epochs[epochs.label == label]
        return float(sel[col].mean()) if len(sel) else None
    on = epochs[epochs.label == "lights_on"]
    out["light_reflex"] = {"flashes": int(len(on)),
                           "constriction_pct": ep("lights_on", "min_change_pct"),
                           "time_to_max_constriction_s": ep("lights_on", "t_min_s"),
                           "redilation_in_dark_pct": ep("lights_off", "max_change_pct")}

    # reading: fixations and saccades inside the reading window
    rw = window("reading")
    rf = fix[in_windows(fix.onset_s.to_numpy(), rw)] if len(fix) else fix
    rs = sacc[in_windows(sacc.onset_s.to_numpy(), rw)] if len(sacc) else sacc
    dur = sum(b - a for a, b in rw)
    out["reading"] = {"duration_s": dur, "fixations": int(len(rf)),
                      "fixation_duration_median_ms": float(rf.duration_ms.median()) if len(rf) else None,
                      "saccades_per_s": float(len(rs) / dur) if dur else None,
                      "valid_rate": float(pupil[in_windows(pupil.t_s.to_numpy(), rw)].valid.mean()) if rw else None}

    # effort: task-evoked dilation per difficulty, at constant luminance. Each epoch runs from the problem
    # appearing to the answer prompt (10 s); its change is relative to the 0.5 s of fixation before.
    tasks = epochs[epochs.label.isin(LEVELS)]
    if "onset_s" in tasks:      # no epochs at all (session stopped before any marker epoch) -> no columns
        tasks = tasks.sort_values("onset_s")
    tasks = tasks.reset_index(drop=True)
    trials_path = session / "trials.csv"
    if trials_path.exists() and len(tasks):
        tr = pd.read_csv(trials_path)
        n = min(len(tr), len(tasks))
        tasks.loc[:n - 1, "problem"] = tr.problem[:n].to_numpy()
    answers_path = session / "arithmetic_answers.csv"
    if answers_path.exists() and len(tasks):
        ans = pd.read_csv(answers_path)
        n = min(len(ans), len(tasks))
        for col in ("rt_s", "heard", "correct", "spoke_during_solve", "voice_onset_after_prompt_s", "transcript"):
            if col in ans:
                tasks.loc[:n - 1, col] = ans[col][:n].to_numpy()
    if answers_path.exists() and len(tasks) and "solve_speech_onset_s" in ans:
        tasks.loc[:n - 1, "solve_speech_onset_s"] = ans.solve_speech_onset_s[:n].to_numpy()
    # A trial where you spoke while solving is measured only up to the moment speech started (as Hess &
    # Polt measured dilation up to the answer), if that leaves at least MIN_SOLVE_WINDOW_S of solving.
    tasks["measured_to_s"] = np.nan
    if long_ep is not None and "solve_speech_onset_s" in tasks:
        task_events = epochs[epochs.label.isin(LEVELS)].sort_values("onset_s").event.to_numpy()
        for i, row in tasks.iterrows():
            cut = row.get("solve_speech_onset_s")
            if row.get("spoke_during_solve") is True or row.get("spoke_during_solve") == 1.0:
                if pd.notna(cut) and cut >= MIN_SOLVE_WINDOW_S and i < len(task_events):
                    ev = long_ep[(long_ep.event == task_events[i]) & (long_ep.t_rel_s >= 0) & (long_ep.t_rel_s < cut)]
                    tasks.loc[i, "mean_change_pct"] = ev.change_pct.mean()
                    tasks.loc[i, "measured_to_s"] = cut
    # dilation is taken from correct answers, when answers were scored (wrong or abandoned problems dilate
    # less and would dilute the effect), excluding talking that started too early to measure; else all trials
    scored = "correct" in tasks and tasks.correct.notna().any()
    if scored:
        talked = tasks.get("spoke_during_solve", pd.Series(False, index=tasks.index)).eq(True)
        clean = tasks.correct.eq(True) & (~talked | tasks.measured_to_s.notna())
    else:
        clean = pd.Series(True, index=tasks.index)
    if len(tasks):
        cols = [c for c in ("label", "problem", "onset_s", "rt_s", "heard", "correct", "spoke_during_solve",
                            "solve_speech_onset_s", "measured_to_s", "baseline_px", "mean_change_pct",
                            "max_change_pct", "t_max_s", "valid_fraction")
                if c in tasks]
        tasks[cols].to_csv(session / "arithmetic_trials.csv", index=False)
    out["arithmetic"] = {"dilation_from": "correct answers; trials with early talking measured up to the speech"
                         if scored else "all trials"}
    for lab in LEVELS:
        level = tasks[tasks.label == lab] if len(tasks) else tasks
        sel = level[clean.reindex(level.index, fill_value=False)].mean_change_pct.dropna() if len(level) else \
            pd.Series(dtype=float)
        every = level.mean_change_pct.dropna() if len(level) else pd.Series(dtype=float)
        entry = {
            "trials": int(len(sel)),
            "mean_change_pct": float(sel.mean()) if len(sel) else None,
            "se_pct": float(sel.std() / np.sqrt(len(sel))) if len(sel) > 1 else None,
            "peak_change_pct": float(level.max_change_pct.mean()) if len(level) else None,
            "mean_change_all_trials_pct": float(every.mean()) if len(every) else None,
        }
        if scored:
            answered = level.correct.notna()
            entry.update({
                "correct": int(level.correct.eq(True).sum()), "presented": int(len(level)),
                "accuracy": float(level.correct.eq(True).sum() / len(level)) if len(level) else None,
                "unanswered": int((~answered).sum()),
                "solve_time_median_s": float(level.rt_s.median()) if level.rt_s.notna().any() else None,
                "solved_by_space": int(level.rt_s.notna().sum()),
                "talked_while_solving": int(level.get("spoke_during_solve", pd.Series(dtype=bool)).eq(True).sum()),
            })
        out["arithmetic"][lab] = entry

    # digit span: dilation per list length from the last digit through the hold, on the cleaned trace; trials
    # with too little usable data are dropped (and counted), recall accuracy is reported alongside
    span = span_responses(pupil, markers, cfg.get("digit_span"))
    ans_path = session / "span_answers.csv"
    if ans_path.exists() and len(span):
        sa = pd.read_csv(ans_path, dtype={"digits": str, "heard": str})
        n = min(len(sa), len(span))
        for col in ("digits", "heard", "correct", "positions_correct", "spoke_during_hold"):
            if col in sa:
                span.loc[:n - 1, col] = sa[col][:n].to_numpy()
        if "spoke_during_hold" in span:          # talking moves the pupil: the hold is no longer a clean measure
            talked = span.spoke_during_hold.astype(str).eq("True") & ~span.rejected.astype(bool)
            span.loc[talked, "rejected"] = True
            span.loc[talked, "reason"] = "spoke during the hold"
        span.to_csv(session / "span_trials_pupil.csv", index=False)
    elif len(span):
        span.to_csv(session / "span_trials_pupil.csv", index=False)
    out["digit_span"] = {}
    for lab in SPAN_LEVELS:
        lv = span[span.label == lab] if len(span) else span
        kept = lv[~lv.rejected.astype(bool)] if len(lv) else lv
        vals = kept.change_pct.dropna() if len(kept) else pd.Series(dtype=float)
        px = kept.change_px.dropna() if len(kept) else pd.Series(dtype=float)
        entry = {"trials": int(len(vals)), "rejected": int(lv.rejected.astype(bool).sum()) if len(lv) else 0,
                 "mean_change_px": float(px.mean()) if len(px) else None,
                 "se_px": float(px.std() / np.sqrt(len(px))) if len(px) > 1 else None,
                 "mean_change_pct": float(vals.mean()) if len(vals) else None,
                 "se_pct": float(vals.std() / np.sqrt(len(vals))) if len(vals) > 1 else None}
        if "correct" in lv and lv.correct.notna().any():
            entry.update({"recalled": int(lv.correct.eq(True).sum()), "presented": int(len(lv))})
        out["digit_span"][lab] = entry

    # plain-language verdicts
    c = out["light_reflex"]["constriction_pct"]
    easy, med, hard = (out["arithmetic"][lab]["mean_change_pct"] for lab in LEVELS)
    rts = [out["arithmetic"][lab].get("solve_time_median_s") for lab in LEVELS]
    se = [out["arithmetic"][lab]["se_pct"] for lab in ("task_easy", "task_hard")]
    gap_sig = None not in (easy, hard) and None not in se and (hard - easy) > 2 * np.hypot(*se)
    cv = out["baseline"]["pupil_cv_pct"]
    lo, mid, hi = (out["digit_span"][lab]["mean_change_pct"] for lab in SPAN_LEVELS)
    sse = [out["digit_span"][lab]["se_pct"] for lab in (SPAN_LEVELS[0], SPAN_LEVELS[-1])]
    span_sig = None not in (lo, hi) and None not in sse and (hi - lo) > 2 * np.hypot(*sse)
    out["verdict"] = {
        "span_dilation": "n/a" if hi is None else
        ("graded (3 < 5 < 7 digits)" if None not in (lo, mid) and lo < mid < hi and span_sig else
         "clear (7 > 3 digits)" if span_sig else
         "weak" if hi >= 2 and (lo is None or hi > lo) else "not detected"),
        "light_reflex": "n/a" if c is None else "clear" if c <= -10 else "weak" if c <= -5 else "not detected",
        "effort_dilation": "n/a" if hard is None else
        ("graded (easy < medium < hard)" if None not in (easy, med) and easy < med < hard and gap_sig else
         "clear (hard > easy)" if gap_sig else
         "weak" if hard >= 2 and (easy is None or hard > easy) else "not detected"),
        "pupil_noise": "n/a" if cv is None else "low" if cv <= 3 else "moderate" if cv <= 8 else "high",
        "difficulty_check": "n/a" if None in rts else
        "solve time rises with difficulty" if rts[0] < rts[1] < rts[2] else "solve times not ordered by difficulty",
        "blinks": "n/a" if out["blinks"]["hit_rate"] is None else
        "good" if out["blinks"]["hit_rate"] >= 0.9 else "partial" if out["blinks"]["hit_rate"] >= 0.6 else "poor",
    }
    from .calibration.eye_radius import measure_radius
    out["eye_radius"] = measure_radius(session, cfg)
    out["eye_model_warning"] = summary.get("eye_model_warning")
    (session / "validation_report.json").write_text(json.dumps(out, indent=2))
    return out


def score_arithmetic(session: str | Path) -> pd.DataFrame | None:
    """Per arithmetic trial: solve time (SPACE after the problem appeared), the spoken answer heard in the
    answer window and whether it is correct, and whether there was talking during the solve window
    (which would contaminate the pupil measurement). Writes arithmetic_answers.csv."""
    from .answers import load_wav, score_answers, voice_onset
    session = Path(session)
    paths = [session / f for f in ("trials.csv", "markers.csv", "audio.wav", "audio_meta.json")]
    if not all(p.exists() for p in paths):
        return None
    trials = pd.read_csv(paths[0])
    m = pd.read_csv(paths[1]).sort_values("t_ns").reset_index(drop=True)
    audio, sr = load_wav(paths[2])
    t0 = json.loads(paths[3].read_text())["t0_ns"]
    sec = lambda t_ns: (t_ns - t0) / 1e9   # noqa: E731  (audio time)

    task_rows = m[m.label.isin(LEVELS)].reset_index()
    rows, answer_windows, solve_windows = [], [], []
    for k, task in task_rows.iterrows():
        after = m.iloc[task["index"] + 1:]
        ans = after[after.label == "answer"]
        t_ans = ans.t_ns.iloc[0] if len(ans) else task.t_ns + 10e9
        solved = after[(after.label == "solved") & (after.t_ns < t_ans)]
        nxt = after[(after.t_ns > t_ans) & ~after.label.isin(["solved", "answer"])]
        t_end = nxt.t_ns.iloc[0] if len(nxt) else t_ans + ANSWER_S * 1e9
        answer_windows.append((sec(t_ans) - 0.3, sec(t_end) + 0.5))
        solve_windows.append((sec(task.t_ns), sec(t_ans) - 0.3))
        tr = trials.iloc[k] if k < len(trials) else None
        rows.append({"trial": k + 1, "level": task.label,
                     "problem": None if tr is None else tr.problem, "answer": None if tr is None else int(tr.answer),
                     "rt_s": (solved.t_ns.iloc[0] - task.t_ns) / 1e9 if len(solved) else None})
    scores = score_answers(audio, sr, answer_windows, [r["answer"] for r in rows])
    for r, sc, (a, _), (s0, s1) in zip(rows, scores, answer_windows, solve_windows):
        r.update({"heard": sc["heard"], "correct": sc["correct"] if sc["heard"] is not None else None,
                  "transcript": sc["transcript"],
                  "voice_onset_after_prompt_s": None if sc["voice_onset_s"] is None else sc["voice_onset_s"] - (a + 0.3),
                  "spoke_during_solve": voice_onset(audio, sr, s0, s1) is not None, "answered_early": False})
        speech = voice_onset(audio, sr, s0, s1)
        r["solve_speech_onset_s"] = None if speech is None else speech - s0   # from the problem appearing
    # answers said before the prompt (while solving) still count for correctness; the pupil window is
    # still flagged as contaminated by speech
    early = [k for k, r in enumerate(rows) if r["heard"] is None and r["spoke_during_solve"]]
    if early:
        rescued = score_answers(audio, sr, [solve_windows[k] for k in early], [rows[k]["answer"] for k in early])
        for k, sc in zip(early, rescued):
            if sc["heard"] is not None:
                rows[k].update({"heard": sc["heard"], "correct": sc["correct"], "transcript": sc["transcript"],
                                "answered_early": True})
    df = pd.DataFrame(rows)
    df.to_csv(session / "arithmetic_answers.csv", index=False)
    return df


def score_span(session: str | Path) -> pd.DataFrame | None:
    """Per digit-span trial: the digits said in the recall window, whether they match (exact, in order), how
    many positions are right, and whether there was speech while listening or holding (which would
    contaminate the pupil measurement). Writes span_answers.csv."""
    from .answers import DIGITS_PROMPT, has_sound, load_wav, parse_digits, transcribe_clips, voice_onset
    session = Path(session)
    paths = [session / f for f in ("span_trials.csv", "markers.csv", "audio.wav", "audio_meta.json")]
    if not all(p.exists() for p in paths):
        return None
    trials = pd.read_csv(paths[0], dtype={"digits": str})
    m = pd.read_csv(paths[1]).sort_values("t_ns").reset_index(drop=True)
    audio, sr = load_wav(paths[2])
    t0 = json.loads(paths[3].read_text())["t0_ns"]
    sec = lambda t_ns: (t_ns - t0) / 1e9   # noqa: E731  (audio time)
    rows, clips = [], []
    for k, task in m[m.label.isin(SPAN_LEVELS)].reset_index().iterrows():
        after = m.iloc[task["index"] + 1:]
        rec = after[after.label == "recall"]
        if not len(rec):
            continue
        t_rec = rec.t_ns.iloc[0]
        nxt = after[after.t_ns > t_rec]
        t_end = nxt.t_ns.iloc[0] if len(nxt) else t_rec + 6e9
        a, b = sec(t_rec) - 0.3, sec(t_end) + 0.5
        audible = voice_onset(audio, sr, a, b) is not None or has_sound(audio, sr, a, b)
        clips.append(audio[int(max(0, a) * sr):int(min(len(audio) / sr, b) * sr)] if audible else np.zeros(0, np.float32))
        # the digits themselves come out of the speakers, so only count speech in the silent hold
        hold0 = sec(task.t_ns) + int(task.label[5:]) * SPAN_DIGIT_S + 0.3
        expected = trials.digits.iloc[k] if k < len(trials) else None
        rows.append({"trial": k + 1, "level": task.label, "digits": expected,
                     "spoke_during_hold": voice_onset(audio, sr, hold0, sec(t_rec) - 0.1) is not None})
    texts = transcribe_clips(clips, sr, prompt=DIGITS_PROMPT) if any(len(c) for c in clips) else [""] * len(clips)
    for r, text in zip(rows, texts):
        heard = parse_digits(text)
        exp = r["digits"]
        r.update({"transcript": text, "heard": heard,
                  "correct": None if heard is None or exp is None else heard == exp,
                  "positions_correct": None if heard is None or exp is None else
                  sum(x == y for x, y in zip(heard, exp))})
    df = pd.DataFrame(rows)
    df.to_csv(session / "span_answers.csv", index=False)
    return df


SPAN_DEFAULTS = {"baseline_s": 1.0, "lowpass_hz": 4.0, "interpolate_max_ms": 250, "min_usable": 0.8, "off_track_px": 25}


def clean_pupil_trace(pupil: pd.DataFrame, lowpass_hz: float = 4.0, interpolate_max_ms: float = 250,
                      off_track_px: float = 25) -> tuple[np.ndarray, np.ndarray]:
    """Pupil diameter (px) ready for task-evoked analysis, and which samples were usable before filling.

    Untrusted samples are removed first: invalid frames, size jumps (size_outlier) and frames whose centre sits
    more than off_track_px from the local track of valid centres (a lash or the lid taken for the pupil). Gaps up
    to interpolate_max_ms are filled linearly; longer gaps stay empty. Each unbroken stretch is then low-pass
    filtered (zero-phase Butterworth): a task-evoked response lasts seconds, so faster changes are noise."""
    from scipy.signal import butter, filtfilt
    t = pupil.t_s.to_numpy(float)
    usable = pupil.valid.to_numpy(bool).copy()
    if "size_outlier" in pupil:
        usable &= ~pupil.size_outlier.fillna(False).to_numpy(bool)
    v = pupil[usable]
    if len(v) > 10:
        mx = v.cx.rolling(61, center=True, min_periods=10).median()
        my = v.cy.rolling(61, center=True, min_periods=10).median()
        off = (np.hypot(v.cx - mx, v.cy - my) > off_track_px).to_numpy()
        usable[np.flatnonzero(usable)[off]] = False
    d = np.where(usable, pupil.diameter_clean_px.to_numpy(float), np.nan)
    d[~np.isfinite(d)] = np.nan
    ok = np.isfinite(d)
    if ok.sum() < 2:
        return d, ok & usable
    # fill short gaps only
    filled = np.interp(t, t[ok], d[ok])
    gap_id = np.cumsum(ok)
    for g in np.unique(gap_id[~ok]):
        idx = np.flatnonzero((gap_id == g) & ~ok)
        before, after = idx[0] - 1, idx[-1] + 1
        if before < 0 or after >= len(t) or (t[after] - t[before]) * 1000 > interpolate_max_ms:
            filled[idx] = np.nan
    out = filled.copy()
    fs = 1.0 / np.median(np.diff(t)) if len(t) > 1 else 120.0
    if lowpass_hz and lowpass_hz < fs / 2:
        b, a = butter(2, lowpass_hz / (fs / 2))
        fin = np.isfinite(filled)
        edges = np.flatnonzero(np.diff(np.r_[0, fin.astype(int), 0]))
        min_len = 9 * max(len(a), len(b))      # shorter stretches are left unfiltered
        for s0, s1 in zip(edges[::2], edges[1::2]):
            if s1 - s0 > min_len:
                out[s0:s1] = filtfilt(b, a, filled[s0:s1])
    return out, usable


def span_responses(pupil: pd.DataFrame, markers: pd.DataFrame, params: dict | None = None) -> pd.DataFrame:
    """One row per digit-span trial, measured on the cleaned trace (clean_pupil_trace).

    Baseline: median size over the baseline_s before the first digit. Response: mean size from the last digit to
    the end of the hold, where memory load peaks (Kahneman & Beatty 1966), minus the baseline, in px
    (subtractive, as Mathot et al. 2018 recommend: dividing by a noisy baseline inflates single trials).
    change_pct expresses it against the median baseline of all trials, one fixed reference for the session.
    A trial is rejected when less than min_usable of its baseline or window had usable samples."""
    p = {**SPAN_DEFAULTS, **(params or {})}
    cols = ["label", "onset_s", "baseline_px", "change_px", "change_pct", "baseline_usable", "window_usable",
            "rejected", "reason"]
    m = markers[markers.label.isin(SPAN_LEVELS)].sort_values("t_s")
    if not len(m) or not len(pupil):
        return pd.DataFrame(columns=cols)
    clean, usable = clean_pupil_trace(pupil, p["lowpass_hz"], p["interpolate_max_ms"], p["off_track_px"])
    t = pupil.t_s.to_numpy(float)
    rows = []
    for e in m.itertuples():
        n = int(e.label[5:])
        b = (t >= e.t_s - p["baseline_s"]) & (t < e.t_s)
        w = (t >= e.t_s + (n - 1) * SPAN_DIGIT_S) & (t < e.t_s + n * SPAN_DIGIT_S + SPAN_HOLD_S)
        bu, wu = (float(usable[b].mean()) if b.any() else 0.0), (float(usable[w].mean()) if w.any() else 0.0)
        base = float(np.nanmedian(clean[b])) if np.isfinite(clean[b]).any() else np.nan
        resp = float(np.nanmean(clean[w])) if np.isfinite(clean[w]).any() else np.nan
        reason = "baseline mostly unusable" if bu < p["min_usable"] else "window mostly unusable" if wu < p["min_usable"] else ""
        rows.append({"label": e.label, "onset_s": e.t_s, "baseline_px": base, "change_px": resp - base,
                     "baseline_usable": bu, "window_usable": wu, "rejected": bool(reason), "reason": reason})
    df = pd.DataFrame(rows)
    ref = df.baseline_px.median()
    df["change_pct"] = 100 * df.change_px / ref if ref and np.isfinite(ref) else np.nan
    return df[cols]


def report_lines(r: dict) -> list[str]:
    def f(v, unit="", fmt="{:+.1f}"):
        return "-" if v is None else fmt.format(v) + unit
    s, b, lr, rd, ar, v = r["signal"], r["baseline"], r["light_reflex"], r["reading"], r["arithmetic"], r["verdict"]
    lines = [
        f"Signal: {f(s['sample_rate_hz'], ' Hz', '{:.0f}')}, valid {f(s['valid_rate'], '', '{:.2f}')}, "
        f"confidence {f(s['mean_confidence'], '', '{:.2f}')}, noise {f(s['noise_rms_s2s_deg'], ' deg', '{:.2f}')}",
        f"Pupil steadiness at rest: CV {f(b['pupil_cv_pct'], '%', '{:.1f}')}  ->  {v['pupil_noise']}",
        f"Light reflex: {f(lr['constriction_pct'], '%')} after white flashes, peak at "
        f"{f(lr['time_to_max_constriction_s'], ' s', '{:.2f}')}, {f(lr['redilation_in_dark_pct'], '%')} in the dark  ->  {v['light_reflex']}",
    ]
    sp = r.get("digit_span", {})
    if any(sp.get(lab, {}).get("trials") for lab in SPAN_LEVELS):
        lines.append("Digit span (memory load): " + ", ".join(
            f"{lab[5:]} digits {f(sp[lab]['mean_change_pct'], '%')}"
            + (f" ± {sp[lab]['se_pct']:.1f}" if sp[lab]["se_pct"] is not None else "")
            + f" (n={sp[lab]['trials']}" + (f", {sp[lab]['rejected']} dropped" if sp[lab].get("rejected") else "") + ")"
            for lab in SPAN_LEVELS) + f"  ->  {v['span_dilation']}")
        lines.append("Recall: " + ("; ".join(f"{lab[5:]} digits {sp[lab]['recalled']}/{sp[lab]['presented']}"
                                              for lab in SPAN_LEVELS if "recalled" in sp[lab])
                                   or "not scored (no microphone recording)"))
    if any(ar.get(lab, {}).get("trials") for lab in LEVELS):
        lines.append("Mental arithmetic: " + ", ".join(
            f"{lab[5:]} {f(ar[lab]['mean_change_pct'], '%')}"
            + (f" ± {ar[lab]['se_pct']:.1f}" if ar[lab]["se_pct"] is not None else "") + f" (n={ar[lab]['trials']})"
            for lab in LEVELS) + f"  ->  {v['effort_dilation']}")
        lines.append("Answers: " + ("; ".join(
            f"{lab[5:]} {ar[lab]['correct']}/{ar[lab]['presented']} correct, solve "
            f"{f(ar[lab]['solve_time_median_s'], ' s', '{:.1f}')}"
            + (f", {ar[lab]['talked_while_solving']} talked early" if ar[lab]["talked_while_solving"] else "")
            for lab in LEVELS) + f"  ->  {v['difficulty_check']}"
            if "correct" in ar["task_hard"] else "not scored (no microphone recording)"))
    er = r.get("eye_radius")
    if er:
        lines.append(f"Eye model radius: {er['radius_px']:.0f} px from {er['targets_used']} targets "
                     f"(angles within {er['rms_error_deg']:.1f} deg RMS, {er['deg_per_px']:.2f} deg/px; this session "
                     f"used {er['session_radius_px']:.0f} px, {er['session_radius_source']}). To use it, add to "
                     f"config/local.yaml:  eye_model: {{radius_px: {er['radius_px']:.0f}}}"
                     if er.get("radius_px") else f"Eye model radius: not measured ({er.get('note')})")
    if r.get("eye_model_warning"):
        lines.append(f"Warning: {r['eye_model_warning']}")
    return lines + [
        f"Blinks: {r['blinks']['detected_after_cue']}/{r['blinks']['cued']} cues detected  ->  {v['blinks']}",
        f"Reading: {rd['fixations']} fixations, median {f(rd['fixation_duration_median_ms'], ' ms', '{:.0f}')}, "
        f"{f(rd['saccades_per_s'], '/s', '{:.1f}')} saccades, valid {f(rd['valid_rate'], '', '{:.2f}')}",
    ]


# ---------------------------------------------------------------- entry point

def _start_mic():
    """Microphone for the spoken answers, or None (with the reason) if it isn't available."""
    try:
        from .answers import MicRecorder
        mic = MicRecorder()
        mic.__enter__()
        return mic
    except Exception as e:   # no sounddevice, no microphone, or permission refused
        print(f"Microphone unavailable ({e}); answers won't be scored. "
              "Install with `pip install sounddevice faster-whisper` and allow microphone access.")
        return None


def run(cfg: dict, screen=(1440, 900), align: bool = True, short: bool = False, process_after: bool = True,
        parts=None, trials_per_level: int | None = None, seed: int | None = None, use_mic: bool = True,
        inset: bool = True, effort: str = "digit_span"):
    """parts: which parts to run (default: every part, with `effort` as the mental-effort task)."""
    from .capture.recorder import record
    from .pipeline import process

    default_run = not parts
    parts = tuple(parts) if parts else tuple(p for p in ALL_PARTS if p not in EFFORT_PARTS or p == effort)
    voice = DigitVoice() if "digit_span" in parts else None   # ~5 s to render, so before the alignment screen
    if align and not align_camera(cfg, screen):
        print("Alignment cancelled.")
        return None
    trials = arithmetic_trials(trials_per_level or (1 if short else 2), seed)
    spans = span_trials(trials_per_level or (1 if short else 4), seed)
    display = ProtocolDisplay(build_steps(screen, short, parts, trials, spans=spans, voice=voice), inset=inset,
                              zone=align_zone(cfg))
    print(f"Protocol: {display.total_s / 60:.1f} min")
    name = "validate" if default_run and effort == "digit_span" else "validate_" + "_".join(
        parts if not default_run else (effort,))
    mic = _start_mic() if use_mic and any(p in parts for p in EFFORT_PARTS) else None
    try:
        session = record(cfg, name=name, use_scene=False, preview=False, display=display)
    finally:
        if mic is not None:
            mic.__exit__(None, None, None)
    if "arithmetic" in parts:
        pd.DataFrame([{"trial": k + 1, "level": t.level, "problem": t.problem, "answer": t.a * t.b}
                      for k, t in enumerate(trials)]).to_csv(session / "trials.csv", index=False)
    if "targets" in parts:
        from .calibration.eye_radius import TARGET_POINTS
        pd.DataFrame([{"label": f"target_{k}", "x": TARGET_POINTS[k][0], "y": TARGET_POINTS[k][1],
                       "screen_w": screen[0], "screen_h": screen[1]} for k in target_order()]
                     ).to_csv(session / "targets.csv", index=False)
    if "digit_span" in parts:
        pd.DataFrame([{"trial": k + 1, "level": t.level, "digits": t.digits}
                      for k, t in enumerate(spans)]).to_csv(session / "span_trials.csv", index=False)
    if mic is not None and mic.t0_ns is not None:
        mic.save(session / "audio.wav")
        (session / "audio_meta.json").write_text(json.dumps({"t0_ns": mic.t0_ns, "sample_rate": mic.sr,
                                                             "clock": "time.perf_counter_ns"}))
    if not process_after:
        return session
    pcfg = copy.deepcopy(cfg)
    pcfg["pupil_size"]["epoch_labels"] = EPOCH_LABELS
    process(session, pcfg)
    if (session / "audio.wav").exists():
        print("Scoring spoken answers...")
        score_arithmetic(session)
        score_span(session)
    result = analyze(session, pcfg)
    lines = report_lines(result)
    print("\n".join(["", "Validation report", *lines, f"Saved to {session / 'validation_report.json'}"]))
    win = "eyetrack validation report"
    img = text_block(canvas(screen, 245), ["Validation report"], 40, 20, top=90)
    img = text_block(img, lines + ["", "Press any key to close"], 24, 30, top=190, left=80, spacing=2.0)
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(win, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    cv2.imshow(win, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return session
