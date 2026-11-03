# Saccades and fixations

**Current choice:** I-VT (`metrics/events.py`, `detect_events`), configured as follows:
- velocity threshold of 30 °/s;
- candidates smaller than 3× the measured noise are rejected;
- runs of up to 4 missing frames (`fill_gap_ms: 35`) are interpolated first;
- one sample is subtracted from each duration (`duration_bias_samples: 1`).

The step detector (`method: step`) is kept for very noisy recordings, around 0.5° and above.

## What a saccade is (the target)
A saccade is a fast, ballistic eye jump of tens of milliseconds. Size, duration and peak speed follow the
**main sequence**: about 2.2 ms per degree + 21 ms, and roughly 300 °/s at 15°. In reading, they are the 1–2° jumps
between words, plus long return sweeps. Hessels et al. (2018) found researchers define saccades very differently,
so we judge by Lund2013's expert labels. Two experts agree at F1 0.97, the realistic ceiling.

## Iterations

### 1. Plain I-VT at 120 Hz (2026-10-23)
- Speed from the gaze trace; anything above 30 °/s is a saccade.
- On Lund2013 resampled to 120 Hz it worked without noise (P 0.86 / R 0.90). With noise it collapsed:
  F1 **0.43** at 0.2° and **0.23** at 0.5°. At 120 Hz, small position noise becomes large velocity noise.

### 2. Amplitude floor (default since then)
- Noise bursts are fast but tiny, so candidates smaller than 3× the sample-to-sample noise are rejected.
- F1 rose to **0.78 / 0.64** on held-out files at 0.2 / 0.5°.
- Also tried: the adaptive threshold (Nyström & Holmqvist 2010), which found only 38% of saccades at 120 Hz, and
  merging saccades split by post-saccadic wobble, which didn't help on Lund.

### 3. Frame-rate check
- At 24 fps a third of saccades were missed and durations inflated 2.5×.
- **At least 120 Hz** is needed, which settled the camera mode.

### 4. Real reading data: saccades disappeared (2026-10-26)
- On the first validate session, I-VT found almost no reading saccades.
- **Causes:** screen flicker dropped every ~10th frame, the blink merger turned those dropouts into an 18.8 s
  "closed" event (see [blinks.md](blinks.md)), and I-VT needs unbroken frames on both sides of a saccade.
- After the blink fix, I-VT alone still found only 38 saccades in 45 s of reading.

### 5. Step (edge-filter) detector
- **Idea:** a saccade is a step in position between two stable stretches. Score each frame by the median
  position over the next 60 ms minus the median over the previous 60 ms. Medians ride over lost frames and
  spikes.
- **Rules:** a step counts if it's ≥ 1° and ≥ 8× the noise of a window median. Peaks must be ≥ 120 ms apart, and
  onset/offset are taken where the position crosses 10% and 90% of the step.
- **Lund:** F1 0.84 / 0.81 vs I-VT 0.86 / 0.76 at 0.2 / 0.5°. Looser settings mistook smooth pursuit for steps.
- **Real data:** recovered reading, with 214 saccades (4.8/s) that the viewer showed landing on the jumps between
  words.
- **Problems:** median duration 55 ms against a real 25–35 ms; 24–28% of reading frames labelled saccade; saccades
  under ~120 ms apart merge; and **no speed check at all**, so a slow drift of ≥ 1° counts. It is a position
  change-point method, not a saccade detector by the textbook definition.

### 6. Asking what limits accuracy
- Researching data-quality metrics showed the real limit is noise, not the detector.
- Pupil-centre jitter is about 1 px. The eye model converts that at ~1.1°/px when it should be ~0.3–0.5°/px (see
  [eye-model-and-gaze.md](eye-model-and-gaze.md)).
- The cleanest stretches (dark screen, no reflection, roughly what the IR filter will give) imply **~0.1–0.2°**
  of noise once the eye model and bad frames are fixed. Comparisons should be made at that noise level.

### 7. Head-to-head at realistic noise
At 0.15° noise on Lund image viewing (closest to reading):

| Method | F1, no loss | F1, 5% lost frames | Start-time error | Duration (true → measured) |
|---|---|---|---|---|
| I-VT | 0.92 | 0.85 | 2.7 ms | 30 → 42 ms |
| Step | 0.88 | 0.88 | 3.0–3.3 ms | 31 → 33 ms (24 → 50 ms on video/dots) |

- I-VT won on detection and timing, but lost frames hurt it and its durations ran long.
- Step's durations looked good on still images, but doubled whenever the eye followed moving content.

### 8. Fixing I-VT's two weaknesses
- **Lost frames:** interpolating runs of ≤ 4 missing frames before computing speed (`fill_gap_ms: 35`) brought F1
  with 5% loss from 0.85 back to **0.92**, the no-loss level. Filled frames are excluded from the noise estimate.
  Longer gaps such as blinks are not filled.
- **Durations:** the signed error showed onsets were already unbiased and the excess was at the end, where
  velocity smoothing keeps the speed above threshold. It grows with noise: +5 / +9 / +13 ms at 0 / 0.15 / 0.2°.
  - *Tried and rejected:* placing onset and offset where the speed crosses the threshold, interpolated between
    frames. It made onsets ~6 ms early.
  - *Kept:* subtract one sample. The median error becomes −2 / +1 / +4 ms at 0.1 / 0.15 / 0.2°.
  - The per-saccade error stays about one frame (8–10 ms), the limit of 120 Hz.

### 9. Checking it wasn't luck
- An intermediate version fixed the random noise seed per file, for reproducibility. That was reverted, because
  the point of adding noise is that the detector must cope with any random draw.
- The benchmark now draws fresh noise each run. Settings being compared share the same draw; `repeats` shows the
  spread.
- Over 5 fresh draws at 0.2°: F1 0.86 ± 0.01 old and new; duration 42.5 ± 1.9 → **34 ± 1.9 ms** (true 28 ms).
  The improvement held on every draw.

## Why the final choice
At the noise the improved hardware should reach, I-VT with gap filling is the most accurate on all three things
we care about:
- **finding saccades:** F1 0.92 with 5% loss, vs 0.88 for step;
- **when they start:** about 3 ms error, vs 3–7 ms;
- **how big they are:** about 0.29° error, vs 0.34–0.37°.

Its duration bias is now small and consistent; step's isn't. Step only wins around 0.5° noise, where neither
gives usable durations.

## Status on our own data
On the 2026-10-26 reading section (old, noisy eye model):

| Method | Saccades | Median duration |
|---|---|---|
| Step | 214 | 55 ms |
| I-VT, no gap fill | 38 | 24 ms |
| **I-VT + gap fill** | 233 | 40 ms |

All of these still label ~29% of frames as saccade, against ~10% expected. That recording is around 0.8° of
noise, so the realistic-noise numbers only apply once the IR filter and eye-model fix are in.

The live (real-time) saccade flag is a simpler causal check. It matched only 14% of offline saccades on a noisy
reading replay, so use it as a monitor; `saccades.csv` is the measurement.

On 2026-11-02 saccades fired at 3–7/s in every part, including fixation. The cause was a 53 px fitted
eye-model radius, not the detector; see [eye-model-and-gaze.md](eye-model-and-gaze.md) iteration 5. Detector
settings are unchanged.

## Open questions
- No hand-labelled saccades from our own camera yet. Labelling ~60 s of a recording would replace the Lund proxy.
- Add a main-sequence plausibility check (duration and peak speed vs size).
- Post-saccadic wobble and smooth pursuit are not separate classes.
