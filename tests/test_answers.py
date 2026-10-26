"""Spoken-answer helpers that need no microphone or speech model."""

import numpy as np
import pytest

from eyetrack.answers import load_wav, parse_number, save_wav, voice_onset


@pytest.mark.parametrize("text, expected", [
    ("204", 204), ("204.", 204), (" 1,024", 1024), ("two hundred and four", 204), ("a hundred four", 104),
    ("one hundred sixty-eight", 168), ("ninety six", 96), ("fifteen", 15), ("fifty", 50),
    ("I think it's 168", 168), ("the answer is three hundred and six", 306), ("um, seventy seven", 77),
    ("a number", None), ("", None), ("I don't know", None),
])
def test_parse_number(text, expected):
    assert parse_number(text) == expected


def test_voice_onset_finds_speech_after_silence():
    sr = 16000
    rng = np.random.default_rng(0)
    audio = rng.normal(0, 0.002, sr * 4).astype(np.float32)          # 4 s of room noise
    t = np.arange(int(0.5 * sr)) / sr
    audio[int(2.3 * sr):int(2.8 * sr)] += 0.2 * np.sin(2 * np.pi * 220 * t)   # "speech" at 2.3 s
    assert voice_onset(audio, sr, 2.0, 4.0) == pytest.approx(2.3, abs=0.03)
    assert voice_onset(audio, sr, 0.0, 2.0) is None                    # nothing before


def test_wav_round_trip(tmp_path):
    audio = (np.sin(np.linspace(0, 100, 16000)) * 0.5).astype(np.float32)
    save_wav(tmp_path / "a.wav", audio)
    back, sr = load_wav(tmp_path / "a.wav")
    assert sr == 16000 and np.allclose(back, audio, atol=1e-3)


def test_voice_onset_catches_quiet_speech_and_ignores_key_clicks():
    sr = 16000
    rng = np.random.default_rng(1)
    audio = rng.normal(0, 0.0004, sr * 4).astype(np.float32)          # a very quiet room, like a real recording
    audio[int(1.0 * sr):int(1.03 * sr)] += 0.05                       # a 30 ms key click at 1.0 s
    t = np.arange(int(0.4 * sr)) / sr
    syllables = 0.015 * np.sin(2 * np.pi * 180 * t) * (np.sin(2 * np.pi * 4 * t) > -0.3)   # soft speech, short gaps
    audio[int(1.5 * sr):int(1.9 * sr)] += syllables
    assert voice_onset(audio, sr, 0.9, 3.0) == pytest.approx(1.5, abs=0.03)
