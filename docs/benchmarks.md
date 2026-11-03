# Benchmarks against labelled data

Each measurement is tuned and validated against public, hand-labelled datasets. The datasets are
not redistributed with this repository because their licences (GPL-3.0 and CC BY-NC 3.0) differ from its
MIT licence. Download them into `data/datasets/`, which is gitignored:

```bash
git clone https://github.com/richardandersson/EyeMovementDetectorEvaluation data/datasets/lund2013
curl -L -o data/datasets/LPW.zip http://datasets.d2.mpi-inf.mpg.de/tonsen/LPW.zip && unzip data/datasets/LPW.zip -d data/datasets/lpw
eyetrack benchmark lund          # saccades, fixations, blinks
eyetrack benchmark lpw           # pupil detection
eyetrack benchmark compare       # saccades vs pymovements and REMoDNaV (needs .[compare])
```

| Dataset | What it is | Used for |
|---|---|---|
| **Lund2013** (Andersson et al. 2017) | 500 Hz gaze from an SMI HiSpeed tracker. Each sample is hand-labelled by two experts as fixation, saccade, post-saccadic wobble, pursuit or blink. Covers image viewing, video and moving dots. | Saccades, fixations and blinks. **Resampled to 120 Hz** to match the OV9281 eye camera, with 0 / 0.2 / 0.5° white noise added to simulate the lower precision of a low-cost tracker. |
| **LPW** (Tonsen et al. 2016) | 66 head-mounted dark-pupil IR videos from a Pupil Pro 120 Hz eye camera at 640×480: 22 people, indoor and outdoor, some with glasses or make-up. The pupil center is hand-labelled in every frame. | Pupil detection. Frames are **center-cropped to 640×400** (the OV9281's 640×400 mode, same pixel scale). |

To avoid overfitting, settings were **tuned on one half and reported on the other**: Lund image-viewing
files for tuning, video and dots files held out; LPW participants 1–11 for tuning, 12–22 held out.
LPW tuning optimises the **no-glasses** videos, because glasses and heavy eye make-up can usually be
avoided in controlled recordings. Results on all videos are also reported.

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

### Step (edge-filter) detector vs I-VT at 120 Hz

`events.method: step` finds saccades as steps: the median position over the next 60 ms minus the median
over the previous 60 ms, peaks above 8× the noise of a window median and at least 1°. Because it compares
windows rather than consecutive samples, it tolerates short dropouts (for example screen flicker) that
break I-VT's velocity. On continuous Lund data it is about as good as I-VT, and better when noisy:

| Noise added | I-VT P / R / F1 | Step P / R / F1 |
|---|---|---|
| 0.2° | 0.86 / 0.86 / **0.86** | 0.82 / 0.87 / 0.84 |
| 0.5° | 0.84 / 0.70 / 0.76 | 0.80 / 0.83 / **0.81** |

Looser settings (4× noise, 0.5° minimum) found more saccades (recall 0.95) but mistook smooth pursuit in the
video and moving-dot files for steps (precision 0.48–0.64). Saccades less than ~120 ms apart merge into one.
I-VT stays the default. Short dropouts are now handled by `fill_gap_ms` (next section); `step` remains an option for very noisy recordings (around 0.5° and above).

### At the noise expected from an improved build (I-VT gap filling and duration correction)

With an IR-pass filter and a corrected eye model, the cleanest stretches of a real session (dark screen, no
corneal reflection) imply about 0.11–0.17° of per-sample noise once bad frames are removed. Lund2013 at 120 Hz,
image viewing (closest to reading and still screens; I-VT was tuned on these files) and video + dots (held out):

| Noise, lost frames | Method | F1 image / video+dots | Onset error | Amplitude error | Duration true → measured (image) |
|---|---|---|---|---|---|
| 0.15°, none | I-VT | 0.92 / 0.81 | 2.7 ms | 0.29° | 30 → 42 ms |
| 0.15°, none | I-VT + gap fill + duration fix | 0.92 / 0.81 | 2.7 ms | 0.29° | 30 → 33 ms |
| 0.15°, none | Step | 0.88 / 0.75 | 3.0 ms | 0.34° | 32 → 33 ms (video+dots 24 → 50) |
| 0.15°, 5% | I-VT | 0.85 / 0.76 | 2.7 ms | 0.29° | 30 → 42 ms |
| 0.15°, 5% | **I-VT + gap fill + duration fix** | **0.92 / 0.81** | 3.0 ms | 0.29° | 30 → 33 ms |
| 0.15°, 5% | Step | 0.88 / 0.75 | 3.3 ms | 0.37° | 32 → 33 ms (video+dots 24 → 50) |

Lost frames are simulated as random runs of 1–4 frames. Onset and amplitude errors are medians over matched
saccades (image viewing).

- `fill_gap_ms: 35` interpolates runs of up to 4 missing frames before the velocity is computed. It has no
  effect without losses and restores the no-loss F1 with them. Filled samples are excluded from the noise estimate.
- `duration_bias_samples: 1` removes one sample from each duration. The median duration error goes from +6 / +9 /
  +12 ms to −2 / +1 / +4 ms at 0.1 / 0.15 / 0.2° (image viewing). The typical per-saccade error stays about 8–10 ms,
  one frame, which is the limit of 120 Hz. Starts are already unbiased; the excess was at the end of the saccade.
- Interpolating the threshold crossing between frames was also tried; it made onsets about 6 ms early.
- End to end with the shipped config (all files, no simulated frame loss), over 5 fresh random noise draws
  (mean ± SD; both settings scored on the same draws), median saccade duration true → before → now:
  26 → 35 → **27 ms** at 0°, 28 → 42.5 ± 1.9 → **34 ± 1.9 ms** at 0.2°, 31 → 92 → 83 ms at 0.5°. F1 is unchanged
  (0.91 / 0.86 ± 0.01 / 0.76 ± 0.02), onset error 2.7 / 3.7 / 19 ms. The improvement held on every draw.
  Moving dots are over-corrected at 0° (22 → 17 ms).
- `benchmark.lund.run` draws new noise on every run, so results describe the detector's handling of noise in
  general rather than one draw. Use `repeats` to see the spread, and the same `seed` to compare two settings.
- Above about 0.3° noise none of this helps. At 0.5° the step detector finds more saccades (F1 0.86 vs 0.81 on
  image viewing), but both overestimate durations by 30–50 ms.

### Glitch rejection and saccades hidden by lost tracking

On a low-cost tracker, glare tends to hide the pupil during large saccades, and single bad frames make the
position jump out and back. `--loss` simulates lost frames after resampling:

```bash
eyetrack benchmark lund --rates 120 --noise 0.2 0.5 --loss saccade   # or --loss random
```

`random` loses runs of 1–4 frames (about 5% of frames). `saccade` loses tracking for 30–150 ms around 30% of
the expert-labelled saccades. Expert blink labels mark which gaps are blinks, standing in for the pipeline's
brightness-based blink classifier, since Lund has no images. Held-out files (video + dots), 120 Hz, 3 noise draws,
saccade F1 (precision / recall in brackets):

| Lost frames | Noise | Without the two rules | `persist_ratio: 0.5` + `bridge_gap_ms: 150` |
|---|---|---|---|
| none | 0.2° | 0.789 (0.80 / 0.78) | 0.784 (0.83 / 0.74) |
| none | 0.5° | 0.624 (0.67 / 0.59) | 0.628 (0.81 / 0.51) |
| random | 0.2° | 0.769 (0.76 / 0.78) | 0.779 (0.82 / 0.74) |
| random | 0.5° | 0.618 (0.67 / 0.58) | 0.628 (0.81 / 0.51) |
| saccade | 0.2° | 0.617 (0.73 / 0.53) | **0.718** (0.82 / 0.64) |
| saccade | 0.5° | 0.469 (0.61 / 0.38) | **0.590** (0.82 / 0.46) |

On the tuning files (image viewing) with saccade loss, bridge limits of 100 / 150 / 200 ms gave F1 0.81 / 0.85 /
0.86 at 0.2°, and persistence ratios of 0.3 / 0.5 / 0.7 gave 0.85 / 0.85 / 0.84. Persistence trades some recall for
precision; bridging recovers most of what lost tracking costs and changes nothing when no frames are lost.

### Compared with other open-source detectors

`eyetrack benchmark compare` runs this package's detectors and two established Python implementations on the
same signal: Lund2013 resampled to 120 Hz with noise added, all 34 files, scored with the same event matching.
It needs the optional packages from `pip install -e ".[compare]"`.

| Noise | eyetrack I-VT (default) | eyetrack step | pymovements (Engbert-Kliegl) | REMoDNaV* |
|---|---|---|---|---|
| 0° | 0.88 / 0.94 / **0.91** | 0.83 / 0.86 / 0.84 | 0.75 / 0.97 / 0.85 | 0.95 / 0.93 / **0.94** |
| 0.2° | 0.87 / 0.88 / **0.88** | 0.82 / 0.86 / 0.84 | 0.88 / 0.85 / **0.87** | 0.85 / 0.79 / 0.82 |
| 0.5° | 0.84 / 0.72 / 0.77 | 0.80 / 0.83 / **0.82** | 0.85 / 0.62 / 0.72 | 0.77 / 0.66 / 0.71 |
| Amplitude error at 0.2° | 0.34° | 0.33° | 0.32° | 0.38° |

Cells are precision / recall / F1 for saccades. \*REMoDNaV raised an error on the 8 files that contain blinks
(its filters cannot handle gaps), so its scores cover the other 26 files.

How the methods differ:

| | eyetrack I-VT | pymovements | REMoDNaV |
|---|---|---|---|
| Velocity | Sample-to-sample difference (Savitzky-Golay smoothing at higher rates); short gaps interpolated | 5-point moving-average difference | Median filter + Savitzky-Golay, then difference |
| Threshold | Fixed 30 °/s | Adaptive per axis: λ (6) × median-based velocity SD, elliptic | Adaptive and local: iterated median + 10 × MAD, recomputed in a 1 s window around each candidate |
| Noise handling | Rejects candidates smaller than 3× the measured RMS-S2S noise | Threshold rises with noise | Threshold rises with noise |
| Other classes | Fixations; blinks detected separately | Saccades only | Fixations, smooth pursuit, post-saccadic oscillations |

All three label runs of fast samples as saccades and drop runs shorter than a minimum duration. They differ
in how they keep noise out. The adaptive thresholds of pymovements and REMoDNaV rise with noise, which costs
small saccades at 120 Hz. eyetrack keeps the threshold fixed and rejects movements too small to be real.
REMoDNaV is the most precise on clean data because it separates post-saccadic oscillations, but it needs
gap-free input.

### Effect of frame rate (tuned defaults, all files)

| Rate | Noise | Saccade P / R / F1 | Amplitude error | Onset error | Saccade duration (true → measured) | Fixation F1 (image viewing) |
|---|---|---|---|---|---|---|
| 24 Hz | 0° | 0.99 / 0.66 / 0.79 | 0.33° | 13 ms | 34 → 83 ms | 0.84 |
| 60 Hz | 0° | 0.93 / 0.88 / 0.90 | 0.36° | 4 ms | 28 → 50 ms | 0.95 |
| **120 Hz** | **0°** | **0.88 / 0.94 / 0.91** | **0.20°** | **3 ms** | **26 → 35 ms** | **0.96** |
| **120 Hz** | **0.2°** | **0.84 / 0.86 / 0.85** | 0.36° | 4 ms | 28 → 44 ms | 0.93 |
| 120 Hz | 0.5° | 0.82 / 0.71 / 0.76 | 0.75° | 18 ms | 32 → 88 ms | 0.84 |
| 500 Hz | 0° | 0.64 / 0.96 / 0.77 | 0.07° | 2 ms | 27 → 32 ms | 0.97 |

- **Use at least ~120 Hz for saccade analysis.** At 24 fps a third of saccades are missed and
  durations are inflated 2.5×.
- Durations are always overestimated by roughly one sample. Compare durations within a study,
  not against published values.
- The settings are tuned for 120 Hz. At 500 Hz with no noise, precision drops (0.64) because post-saccadic
  wobble gets split into extra "saccades". Re-tune the event settings when recording at a much higher rate.

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
mode mainly helps footage with heavy eye make-up or glasses frames in view. It is available via
`pupil: {n_seeds: 5, score: shape, block: 30}`.

### Which frames to trust (confidence cutoff + size filter)

| Setting | Frames kept | Kept frames that are wrong (>10 px) | Correct frames thrown away |
|---|---|---|---|
| No filtering | 99.5% | 11.4% | 0% |
| Confidence ≥ 0.6 | 84.7% | 3.1% | 4.9% |
| **Default: confidence ≥ 0.5 + size-jump filter (1.5×)** | **88.0%** | **3.1%** | **2.1%** |

Chosen on participants 1–11 and confirmed on 12–21, excluding participant 22 (heavy make-up). The size filter drops
frames whose diameter jumps more than 1.5× against the 2 s rolling median. These are almost always the iris or a
shadow picked by mistake.

### Pupils partly covered by a reflection

A lit screen reflects on the cornea as a grey patch about as bright as the iris. When it overlaps the pupil, the
dark blob's outline runs along the reflection, so the default confidence (share of the outline on the fitted
ellipse) drops to about 0.65 even when the ellipse itself is right. Two alternatives were tested:

- `refine: visible`: drop outline points next to pixels much brighter than the pupil, then fit and score the rest.
- `method: edge`: the published Pupil Labs approach (Świrski et al. 2012; Kassner et al. 2014), written for this
  repository: coarse dark-centre search, Canny edges on the pupil side, RANSAC ellipse fit, confidence = edge
  support on the visible perimeter. Not tuned.

LPW held-out (participants 12–21), every 10th frame:

| Detector | Detection rate (≤5 px) | Kept at conf ≥ 0.8 | Wrong among kept (>10 px) |
|---|---|---|---|
| Default (threshold, `refine: angle`) | 79.7% | 69.3% | 1.8% |
| `refine: visible` | **79.9%** | **83.7%** | 2.7% |
| `method: edge` | 53.2% | 42.2% | 5.0% |

On a 640×400 validate session (OV9281, no IR-pass filter), the share of frames with confidence ≥ 0.8 rose from
0% to 82% (baseline on a grey screen), 11% to 96% (white screen) and 4% to 63% (reading) with `refine: visible`,
with the same frame-to-frame jitter (median 0.70 vs 0.74 px at fixation). The edge method jittered more (median
1.0 px at fixation) and lost the pupil on 37% of reading frames. Confidence values differ between settings, so re-choose
`min_confidence` when switching `refine`.

Pupil Labs' own 2D detector (`pupil-detectors` 2.0.2, built against OpenCV 4.14, `method: pupil_labs`) was also
run on consecutive frames of 640×480 LPW held-out videos (every 3rd video, first 400 frames, every 10th frame scored),
with only the pupil size range raised to 150 px: **26.8%** within 5 px (median error 173 px) vs **75.2%** for the
default detector on the same frames. In its debug view it usually found the pupil region and edge, then rejected the
pupil and fitted a small ellipse elsewhere (often next to the glint). Pupil Capture users normally set a region of
interest and tune intensity and size per camera, which was not done here.

### Blinks vs glare

A dropout where the detector is still fairly confident (median confidence > 0.35) is labelled
`low_confidence`, not `blink`. On the JEOresearch sample video, which has frequent glare, this cut
reported blinks from 52 to 3. **This threshold has not been validated against labelled blinks**
(LPW has none). Check it on your own hardware with a cued-blink recording (see
[hardware.md](hardware.md#verifying-a-build)).

## Limitations of these benchmarks

- Lund2013 comes from a lab-grade tower tracker with simulated degradation, not from a head-mounted
  USB camera. To find out which noise level applies to a given build, look at `noise_rms_s2s_deg` in
  any session's `metrics_summary.json`. RMS-S2S ≈ noise SD × 1.4, so 0.2° of noise shows up as about 0.28° RMS-S2S.
- LPW closely matches the reference eye camera (head-mounted IR, 120 Hz, 640×480), but it has no
  blink or pupil-size labels. Pupil-size accuracy is only verified on synthetic images (within 5%).
- Validate each new build on real recordings; see [hardware.md](hardware.md#verifying-a-build).
