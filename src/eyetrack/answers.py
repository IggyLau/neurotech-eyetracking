"""Spoken answers: microphone recording on the eye-tracking clock, voice-onset detection, and
offline transcription of numbers (faster-whisper).

Used by the arithmetic task's answer scheme: press SPACE the moment the problem is solved (a silent
solve time), then say the answer at the prompt (scored for correctness, with no speech inside the
pupil-measurement window).

`eyetrack answer-test` runs a short check of just this: numbers appear, you press SPACE and say
them, and it reports what it heard.

Needs: pip install sounddevice faster-whisper  (the model downloads once, then runs offline).
"""

from __future__ import annotations

import re
import threading
import time
import wave
from pathlib import Path

import numpy as np

SAMPLE_RATE = 16000   # what the speech model expects


class MicRecorder:
    """Records the default microphone; every block is stamped with time.perf_counter_ns, the same
    clock as the camera frames and markers, so speech times line up with the eye data."""

    def __init__(self, sample_rate: int = SAMPLE_RATE, device=None):
        import sounddevice as sd
        self.sr = sample_rate
        self._blocks: list[np.ndarray] = []
        self._lock = threading.Lock()
        self.t0_ns: int | None = None      # perf_counter_ns of the first recorded sample
        self.stream = sd.InputStream(samplerate=sample_rate, channels=1, dtype="float32",
                                     device=device, callback=self._callback, blocksize=0)

    def _callback(self, data, frames, time_info, status):
        now = time.perf_counter_ns()
        with self._lock:
            if self.t0_ns is None:
                self.t0_ns = now - int(frames / self.sr * 1e9)   # the block ends now
            self._blocks.append(data[:, 0].copy())

    def __enter__(self):
        self.stream.start()
        return self

    def __exit__(self, *exc):
        self.stream.stop()
        self.stream.close()

    @property
    def audio(self) -> np.ndarray:
        with self._lock:
            return np.concatenate(self._blocks) if self._blocks else np.zeros(0, np.float32)

    def save(self, path: str | Path):
        save_wav(path, self.audio, self.sr)


def save_wav(path: str | Path, audio: np.ndarray, sr: int = SAMPLE_RATE):
    pcm = (np.clip(audio, -1, 1) * 32767).astype(np.int16)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(pcm.tobytes())


def load_wav(path: str | Path) -> tuple[np.ndarray, int]:
    with wave.open(str(path), "rb") as w:
        sr, n, width, ch = w.getframerate(), w.getnframes(), w.getsampwidth(), w.getnchannels()
        raw = np.frombuffer(w.readframes(n), {1: np.int8, 2: np.int16, 4: np.int32}[width]).astype(np.float32)
    raw = raw.reshape(-1, ch).mean(axis=1) / float(2 ** (8 * width - 1))
    return raw, sr


# ---------------------------------------------------------------- voice onset

def _frame_rms(audio: np.ndarray, sr: int, frame_ms: float) -> np.ndarray:
    hop = int(sr * frame_ms / 1000)
    frames = audio[: len(audio) // hop * hop].reshape(-1, hop)
    return np.sqrt((frames ** 2).mean(axis=1))


def speech_threshold(rms: np.ndarray, factor: float = 6.0, min_rms: float = 0.003) -> float:
    """Loudness that counts as speech: well above the room's noise floor (20th percentile of frame RMS,
    mostly silence), with an absolute minimum so a near-silent room doesn't make every rustle count."""
    return max(factor * float(np.percentile(rms, 20)), min_rms)


def voice_onset(audio: np.ndarray, sr: int, start_s: float, end_s: float, frame_ms: float = 20,
                min_ms: float = 60, gap_ms: float = 60, factor: float = 6.0, min_rms: float = 0.003) -> float | None:
    """Time (s, from the start of `audio`) when speech starts inside [start_s, end_s), or None.

    Speech = frames above speech_threshold() lasting at least min_ms, where gaps up to gap_ms (between
    syllables) are bridged. A key click is a single short spike and doesn't qualify on its own."""
    if len(audio) < int(sr * frame_ms / 1000):
        return None
    rms = _frame_rms(audio, sr, frame_ms)
    loud = rms > speech_threshold(rms, factor, min_rms)
    gap = int(gap_ms / frame_ms)
    on = np.flatnonzero(loud)
    for a, b in zip(on[:-1], on[1:]):          # bridge short silences between loud frames
        if 1 < b - a <= gap + 1:
            loud[a:b] = True
    i0, i1 = max(0, int(start_s * 1000 / frame_ms)), min(len(loud), int(end_s * 1000 / frame_ms))
    need = max(1, int(min_ms / frame_ms))
    run = 0
    for i in range(i0, i1):
        run = run + 1 if loud[i] else 0
        if run >= need:
            return (i - need + 1) * frame_ms / 1000
    return None


def has_sound(audio: np.ndarray, sr: int, start_s: float, end_s: float, frame_ms: float = 20) -> bool:
    """Anything louder than half the speech threshold in the window? Silent windows aren't transcribed,
    because the model can invent words (or numbers) from silence."""
    rms = _frame_rms(audio, sr, frame_ms)
    i0, i1 = int(start_s * 1000 / frame_ms), int(end_s * 1000 / frame_ms)
    return bool(len(rms[i0:i1]) and rms[i0:i1].max() > speech_threshold(rms) / 2)


# ---------------------------------------------------------------- spoken numbers

_UNITS = {w: i for i, w in enumerate(
    "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen "
    "sixteen seventeen eighteen nineteen".split())}
_UNITS.update({"oh": 0, "o": 0, "a": 1})
_TENS = {w: 10 * i for i, w in enumerate("_ _ twenty thirty forty fifty sixty seventy eighty ninety".split()) if w != "_"}


def parse_number(text: str) -> int | None:
    """The number in a short transcript: digits ("204", "2,04") or words ("two hundred and four",
    "a hundred four", "ninety-six"). Returns None if there isn't one."""
    t = text.lower().replace(",", "").replace("-", " ")
    digits = re.findall(r"\d+", t)
    if digits:
        return int(digits[0])
    words = re.findall(r"[a-z]+", t)
    total, current, seen = 0, 0, False
    for k, w in enumerate(words):
        nxt = words[k + 1] if k + 1 < len(words) else ""
        if w == "a" and nxt not in ("hundred", "thousand"):
            continue   # "a" is only a number in "a hundred" / "a thousand"
        if w in _UNITS and not (w == "o" and seen):
            current += _UNITS[w]
            seen = True
        elif w in _TENS:
            current += _TENS[w]
            seen = True
        elif w == "hundred":
            current = max(current, 1) * 100
            seen = True
        elif w == "thousand":
            total += max(current, 1) * 1000
            current, seen = 0, True
        elif w in ("and",):
            continue
        elif seen:
            break   # the number has ended ("... is the answer")
    return total + current if seen else None


# ---------------------------------------------------------------- transcription

_MODEL = None


def transcribe(audio: np.ndarray, sr: int = SAMPLE_RATE, model_size: str = "base.en") -> str:
    """Offline transcription of one short clip. Numbers are hinted to the model so it prefers digits."""
    global _MODEL
    if _MODEL is None:
        from faster_whisper import WhisperModel
        _MODEL = WhisperModel(model_size, device="cpu", compute_type="int8")
    if sr != SAMPLE_RATE:
        n = int(len(audio) * SAMPLE_RATE / sr)
        audio = np.interp(np.linspace(0, len(audio), n, endpoint=False), np.arange(len(audio)), audio)
    peak = float(np.abs(audio).max()) if len(audio) else 0.0
    if peak > 0:
        audio = audio / peak * 0.5      # quiet speech is transcribed better at a normal level
    segments, _ = _MODEL.transcribe(audio.astype(np.float32), language="en", beam_size=5,
                                    initial_prompt="The answer is a number, for example 42, 204, 168.",
                                    vad_filter=False, condition_on_previous_text=False)
    return " ".join(s.text.strip() for s in segments).strip()


def _transcribe_all(clips: list[np.ndarray], sr: int) -> list[str]:
    return [transcribe(c, sr) if len(c) else "" for c in clips]


def transcribe_clips(clips: list[np.ndarray], sr: int = SAMPLE_RATE) -> list[str]:
    """Transcribe in a separate process. faster-whisper's video library (PyAV) and OpenCV each bundle
    their own copy of ffmpeg's libavdevice; loading both in one process gives duplicate-class warnings
    and can crash, so the speech model never shares a process with the camera code."""
    import multiprocessing as mp
    from concurrent.futures import ProcessPoolExecutor
    with ProcessPoolExecutor(max_workers=1, mp_context=mp.get_context("spawn")) as pool:
        return pool.submit(_transcribe_all, clips, sr).result()


def score_answers(audio: np.ndarray, sr: int, windows: list[tuple[float, float]], expected: list[int | None],
                  pad_s: float = 0.15) -> list[dict]:
    """Voice onset, transcript, parsed number and correctness for each answer window (s from audio start)."""
    onsets = [voice_onset(audio, sr, a, b) for a, b in windows]
    clips = []
    for (a, b), onset in zip(windows, onsets):
        audible = onset is not None or has_sound(audio, sr, a, b)
        clips.append(audio[int(max(0, a - pad_s) * sr):int(min(len(audio) / sr, b + pad_s) * sr)] if audible
                     else np.zeros(0, np.float32))
    texts = transcribe_clips(clips, sr) if any(len(c) for c in clips) else [""] * len(clips)
    out = []
    for onset, text, exp in zip(onsets, texts, expected):
        heard = parse_number(text)
        out.append({"voice_onset_s": onset, "transcript": text, "heard": heard,
                    "correct": None if exp is None or heard is None else heard == exp})
    return out


# ---------------------------------------------------------------- answer test

def run_answer_test(n: int = 6, out_dir: str | Path = "data", screen=(1280, 800), seed: int | None = None,
                    show_s: float = 2.0, max_solve_s: float = 6.0, answer_s: float = 4.0) -> Path:
    """Numbers appear one at a time. Press SPACE as soon as you've read it (standing in for 'solved'),
    then say the number when 'Say it now' appears. Reports SPACE timing, voice onset and the transcript."""
    import cv2

    from .capture.recorder import new_session_dir
    from .protocol import GREY, BLACK, canvas, cross, text_block

    rng = np.random.default_rng(seed)
    # a spread of number shapes: teens vs -ties, hundreds with and without "and", repeated digits
    pool = [13, 15, 17, 19, 30, 40, 50, 68, 77, 96, 112, 144, 168, 204, 216, 252, 276, 288, 306, 342]
    numbers = [int(v) for v in rng.choice(pool, n, replace=False)]
    session = new_session_dir(out_dir, "answer_test")
    win = "eyetrack answer test"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, *screen)

    def show(img, wait_ms=15):
        cv2.imshow(win, img)
        return cv2.waitKey(wait_ms) & 0xFF

    rows = []
    with MicRecorder() as mic:
        intro_end = time.perf_counter() + 5
        while time.perf_counter() < intro_end:
            img = text_block(canvas(screen, GREY), ["Answer test"], 40, BLACK, top=screen[1] // 2 - 140)
            img = text_block(img, ["A number appears. Press SPACE as soon as you have it,",
                                   "then say it out loud when 'Say it now' appears.",
                                   f"starting in {int(np.ceil(intro_end - time.perf_counter()))}"], 26, BLACK)
            if show(img) == ord("q"):
                return session
        for k, number in enumerate(numbers):
            t_show = time.perf_counter()
            pressed = None
            while time.perf_counter() - t_show < max_solve_s:   # number for show_s, then the cross until SPACE
                el = time.perf_counter() - t_show
                img = cross(canvas(screen, GREY), BLACK)
                if el < show_s:
                    img = text_block(img, [str(number)], 44, 40, top=screen[1] // 2 - 100)
                key = show(img, 5)
                if key == ord("q"):
                    return session
                if key == ord(" ") and pressed is None:
                    pressed = time.perf_counter()
                    break
            t_prompt = time.perf_counter()
            while time.perf_counter() - t_prompt < answer_s:
                if show(text_block(cross(canvas(screen, GREY), BLACK), ["Say it now"], 40, BLACK,
                                   top=screen[1] // 2 + 60)) == ord("q"):
                    return session
            rows.append({"trial": k + 1, "number": number, "shown_ns": int(t_show * 1e9),
                         "space_s": None if pressed is None else pressed - t_show,
                         "prompt_ns": int(t_prompt * 1e9), "prompt_end_ns": int((t_prompt + answer_s) * 1e9)})
            show(cross(canvas(screen, GREY), BLACK), 600)
        mic_t0 = mic.t0_ns
        audio = mic.audio
    cv2.destroyWindow(win)
    save_wav(session / "audio.wav", audio)

    for r in rows:
        r["audio_t0_ns"] = mic_t0
    import pandas as pd
    pd.DataFrame(rows).to_csv(session / "answers.csv", index=False)
    print(f"Recorded {len(audio) / SAMPLE_RATE:.1f} s of audio. Transcribing...")
    return score_answer_test(session)


def score_answer_test(session: str | Path) -> Path:
    """Score (or re-score) an answer-test session from its answers.csv and audio.wav."""
    import pandas as pd
    session = Path(session)
    df = pd.read_csv(session / "answers.csv")
    audio, sr = load_wav(session / "audio.wav")
    if "audio_t0_ns" in df:
        t0 = int(df.audio_t0_ns.iloc[0])
    else:   # early sessions didn't save it; recording began with the 5 s intro before the first number
        t0 = int(df.shown_ns.min() - 5.0e9)
        print("note: audio start time estimated from the intro length (+-0.1 s)")
    windows = [((r.prompt_ns - t0) / 1e9, (r.prompt_end_ns - t0) / 1e9) for r in df.itertuples()]
    scores = score_answers(audio, sr, windows, list(df.number))
    df["voice_onset_after_prompt_s"] = [None if s["voice_onset_s"] is None else s["voice_onset_s"] - w[0]
                                        for s, w in zip(scores, windows)]
    for key in ("transcript", "heard", "correct"):
        df[key] = [s[key] for s in scores]
    df.to_csv(session / "answers.csv", index=False)
    cols = ["trial", "number", "space_s", "voice_onset_after_prompt_s", "transcript", "heard", "correct"]
    print(df[cols].to_string(index=False, float_format=lambda v: f"{v:.2f}"))
    print(f"\n{int(df.correct.fillna(False).astype(bool).sum())}/{len(df)} transcribed correctly | "
          f"SPACE pressed on {int(df.space_s.notna().sum())}/{len(df)} | "
          f"voice detected on {int(df.voice_onset_after_prompt_s.notna().sum())}/{len(df)} | "
          f"speech level peak {float(np.abs(audio).max()):.3f} of full scale")
    print(f"Saved {session / 'answers.csv'}")
    return session
