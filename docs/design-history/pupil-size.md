# Pupil size

**Current choice:**
- the ellipse's **major axis** in pixels as the diameter, since the major axis isn't foreshortened when the eye
  turns;
- cleaned with the Kret & Sjak-Shie / Mathôt steps (`metrics/pupil_size.py`, `clean_diameter`): speed-based
  outlier removal, interpolation across gaps, smoothing;
- **baseline-corrected epochs** after each marker, each running to the next marker;
- a live 10 s pupil trace in the preview.

## Iterations

### 1. Diameter and cleaning (2026-10-23)
- Major axis chosen over minor axis or area.
- Accuracy checked only on synthetic images: within 5%.

### 2. Live trace and marker epochs (2026-10-24)
- Goal: see dilation while recording, and measure it after events: the light reflex (lights on/off) and
  task-evoked dilation (Hess & Polt 1964; Ahern & Beatty 1979).
- Epochs run to the next marker. `epoch_end_ignore: [solved]` stops SPACE presses from cutting arithmetic epochs
  short.

### 3. An idea that wasn't pursued
- Measuring the pupil relative to the iris (the iris doesn't change size) would give dilation in % that's robust
  to camera distance.
- Not implemented: the iris edge is low-contrast in IR and often cut off by the eyelids. Revisit if
  millimetre calibration proves hard.

### 4. First validate session results (2026-10-26)

| Measure | Result | Verdict |
|---|---|---|
| Light reflex (white screen) | −50% constriction, maximum after 2.1 s, 77% redilation in the dark | Clear |
| Baseline stability | coefficient of variation 4.2% | OK |
| Effort dilation | Not detected | See causes below |

Why effort dilation wasn't detected:
- only 2–3 usable trials per difficulty level;
- talking during the solve window;
- solve times weren't ordered by difficulty;
- the battery ran low, dropping the frame rate.

Trial-to-trial spread was 6–15%, and effort effects are 5–15%. Detecting a 5% effect needs about 12–30 trials
per level, so `--trials-per-level 10` is now suggested.

### 5. Gaze-position error found
- In reading, where the eye points explains ~18% of the variance in measured diameter, a spread of up to 12% across
  positions. That's as big as the effects we want.
- Planned: a per-session correction map from 9 fixation points at constant brightness (Hayes & Petrov 2016).
  Longer term, a refraction-corrected 3D size (Petersch & Dierkes 2022).

## Open questions
- **Millimetres:** no calibration from pixels to mm yet. Printed dots of known size at eye distance would give
  one.
- **Outline accuracy:** see [pupil-detection.md](pupil-detection.md).
- **Brightness confound:** keep screen brightness constant (fixed brightness, grey or dark backgrounds) in any
  effort study; light changes pupil size far more than thinking does.
