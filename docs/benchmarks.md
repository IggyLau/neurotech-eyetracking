# Benchmarks against labelled data

Real, hand-labelled datasets are used to tune and validate each measurement. They are
**downloaded into `data/datasets/` and never committed**: their licences (GPL-3.0 and CC BY-NC 3.0)
differ from this repo's MIT licence.

```bash
git clone https://github.com/richardandersson/EyeMovementDetectorEvaluation data/datasets/lund2013
curl -L -o data/datasets/LPW.zip http://datasets.d2.mpi-inf.mpg.de/tonsen/LPW.zip && unzip data/datasets/LPW.zip -d data/datasets/lpw
eyetrack benchmark lund          # saccades, fixations, blinks
eyetrack benchmark lpw           # pupil detection
```

| Dataset | What it is | Used for |
|---|---|---|
| **Lund2013** (Andersson et al. 2017) | 500 Hz gaze from an SMI HiSpeed tracker. Each sample is hand-labelled by two experts as fixation, saccade, post-saccadic wobble, pursuit or blink. Covers image viewing, video and moving dots. | Saccades, fixations and blinks. **Resampled to 120 Hz** to simulate the OV9281, with 0 / 0.2 / 0.5° noise added to simulate a DIY tracker's precision. |
| **LPW** (Tonsen et al. 2016) | 66 head-mounted dark-pupil IR videos from a Pupil Pro 120 Hz eye camera at 640×480: 22 people, indoor and outdoor, some with glasses or make-up. The pupil center is hand-labelled in every frame. | Pupil detection. Frames are **center-cropped to 640×400** (our OV9281 mode, same pixel scale). |

To avoid overfitting, settings were **tuned on one half and reported on the other**: Lund image-viewing
files for tuning, video and dots files held out; LPW participants 1–11 for tuning, 12–22 held out.
LPW tuning optimises the **no-glasses** videos, because glasses and heavy make-up can be avoided in our
own sessions; results on all videos are still reported.

## Saccades, fixations and blinks (Lund2013)

Expert vs expert saccade agreement, the realistic ceiling: **F1 0.97**.

### Saccades at 120 Hz, held-out files (video + dots)

| Noise added | Old: fixed 30 °/s | Tuned: 30 °/s + amplitude ≥ 3× noise |
|---|---|---|
| 0° (lab-grade) | P 0.86 / R 0.90 | P 0.86 / R 0.90 |
| 0.2° | P 0.28 / R 0.91 (**F1 0.43**) | P 0.79 / R 0.78 (**F1 0.78**) |
| 0.5° | P 0.13 / R 0.97 (**F1 0.23**) | P 0.76 / R 0.55 (**F1 0.64**) |

P = precision (detected saccades that are real), R = recall (real saccades found).

At 120 Hz, position noise turns into large velocity noise, so a fixed 30 °/s threshold
fires on noise. Noise bursts are fast but **tiny**, so rejecting candidates smaller than
3× the measured sample-to-sample noise (RMS-S2S) removes most of them. The textbook adaptive
threshold (Nyström & Holmqvist 2010) was also tried: at 120 Hz it missed most small saccades
(recall 0.38), so it is an option but not the default.

### Effect of frame rate (tuned defaults, all files)

| Rate | Noise | Saccade P / R / F1 | Amplitude error | Onset error | Saccade duration (true → measured) | Fixation F1 (image viewing) |
|---|---|---|---|---|---|---|
| 24 Hz | 0° | 0.99 / 0.66 / 0.79 | 0.33° | 13 ms | 34 → 83 ms | 0.84 |
| 60 Hz | 0° | 0.93 / 0.88 / 0.90 | 0.36° | 4 ms | 28 → 50 ms | 0.95 |
| **120 Hz** | **0°** | **0.88 / 0.94 / 0.91** | **0.20°** | **3 ms** | **26 → 35 ms** | **0.96** |
| **120 Hz** | **0.2°** | **0.84 / 0.86 / 0.85** | 0.36° | 4 ms | 28 → 44 ms | 0.93 |
| 120 Hz | 0.5° | 0.82 / 0.71 / 0.76 | 0.75° | 18 ms | 32 → 88 ms | 0.84 |
| 500 Hz | 0° | 0.64 / 0.96 / 0.77 | 0.07° | 2 ms | 27 → 32 ms | 0.97 |

- **120 Hz is the sweet spot for our camera.** At 24 fps a third of saccades are missed and
  durations are inflated 2.5×. That's why the JEO sample video (24 fps) gave untrustworthy numbers.
- Durations are always overestimated by roughly one sample. Compare durations within a study,
  not against published values.
- The settings are tuned for 120 Hz. At 500 Hz with no noise, precision drops (0.64) because post-saccadic
  wobble gets split into extra "saccades". Re-tune if we ever record faster.

### Blinks at 120 Hz

Merging valid "flickers" shorter than 80 ms (a half-open eyelid) into the surrounding blink
raised blink F1 from 0.86 to **0.96**. Caveat: Lund has only 27 labelled blinks.

## Pupil detection (LPW, 640×400)

Metric: % of frames where the detected pupil center is within 5 px of the hand label
(the standard LPW metric). Held-out participants 12–22, every 10th frame.

| Detector | Normal videos (excl. heavy make-up, P22) | Indoor, normal | Participant 22 (mascara) | Speed |
|---|---|---|---|---|
| **Default (JEO-based, single seed)** | **79.7%** | **86.1%** | 14% | **1.3 ms/frame** |
| Shape scoring, single seed | 79.3% | 86.2% | – | 3.4 ms/frame |
| Multi-candidate (5 seeds, shape scoring) | 78.8% | – | **72%** | 12.4 ms/frame |

The median center error is 1.6 px. Glint removal, median blur, other threshold offsets, search box,
closing kernel and contour refinement were all tried (60+ configurations), and none beat
the default on normal footage. **The default stays the original detector.** The multi-candidate
mode only rescues heavy make-up and glasses footage, which can be avoided in our sessions. It is available via
`pupil: {n_seeds: 5, score: shape, block: 30}`.

### Which frames to trust (confidence cutoff + size filter)

| Setting | Frames kept | Kept frames that are wrong (>10 px) | Correct frames thrown away |
|---|---|---|---|
| No filtering | 99.5% | 11.4% | 0% |
| Old: confidence ≥ 0.6 | 84.7% | 3.1% | 4.9% |
| **New: confidence ≥ 0.5 + size-jump filter (1.5×)** | **88.0%** | **3.1%** | **2.1%** |

Chosen on participants 1–11 and confirmed on 12–21 (both without P22). The size filter drops
frames whose diameter jumps more than 1.5× against the 2 s rolling median. These are almost always the iris or a
shadow picked by mistake.

### Blinks vs glare

A dropout where the detector is still fairly confident (median confidence > 0.35) is now
labelled `low_confidence`, not `blink`. On the JEO sample video this cut "blinks" from 52 to 3.
**This threshold is not validated against labels yet** (LPW has no blink labels). Check it with the
cued-blink test (10 blinks on cue).

## What these numbers do and don't tell us

- Lund is a lab tower tracker with simulated degradation, not our hardware. Our real noise
  level is unknown until the glasses are built. Every processed session reports it as
  `noise_rms_s2s_deg` in `metrics_summary.json`. RMS-S2S ≈ noise SD × 1.4, so 0.2° of noise
  shows up as ~0.28° RMS-S2S. Use that to see which row above applies to you.
- LPW is the closest public match to our camera (head-mounted IR, 120 Hz, 640×480), but it has
  no blink or pupil-size labels. Pupil-size accuracy is only checked on synthetic images (within 5%).
- Final validation still needs the weekend test protocol on our own glasses.
