# Validation protocol: what can this build measure?

A short set of recordings that shows which measurements a particular headset gives reliably. Run it
on every new build, after moving the camera or LEDs, or before a study. It takes about 15 minutes.

The order matters: signal quality (T1) decides which later results can be trusted. For example,
if pupil size varies more than mental effort changes it, the digit-span test can't show anything.

## Quick guided version: `eyetrack validate`

```bash
.venv/bin/eyetrack --config config/local.yaml validate                      # ~6 min, everything
.venv/bin/eyetrack --config config/local.yaml validate --short              # ~3 min, fewer repetitions
.venv/bin/eyetrack --config config/local.yaml validate --only digit_span    # ~4 min, the effort test alone
.venv/bin/eyetrack --config config/local.yaml validate --effort arithmetic  # mental arithmetic instead of digit span
```

`--trials-per-level N` changes the number of digit lists per length (default 4) and of arithmetic
problems per difficulty (default 2). `--seed N` repeats the same lists and problems across sessions.

Set the screen size in `config/local.yaml` once, so the dots part can turn screen positions into angles:

```yaml
screen:
  width_cm: 30.4      # visible width of the display (measure it)
  distance_cm: 57     # eye to screen
```

The report then gives the measured eye model radius, with a line to paste into `config/local.yaml`
(`eye_model: {radius_px: ...}`). Use the median of a few sessions. Once it's set, every session uses that
scale instead of fitting its own, so saccade sizes and speeds can be compared between sessions. Re-measure
after moving the camera. Without a fixed radius, `eyetrack process` warns when the fitted one can't be
physical.

Run it from Terminal.app. First set screen brightness high and fixed (auto-brightness, True Tone
and Night Shift off), keep room lighting constant, and sit ~57 cm from the screen.

1. **Aim the camera.** A live eye view with a target box. Five checks must all be green for 2 s:
   pupil found (confidence ≥ 0.7), inside the box, a sensible size, ≥ 100 fps, tracked ≥ 90% of
   the last 2 s. Hints say what to adjust. SPACE starts; F starts anyway. The detector only accepts a
   pupil centred inside the box (`pupil.search_zone`), so dark hair or the glasses frame at the image
   edge can't be taken for the pupil. If the pupil drifts out of the box, it's reported as lost.
2. **The protocol** runs full screen and is recorded as one session (`data/<date>_validate/`), with
   a marker at every step. A small live eye view with the target box sits in the bottom-right corner, so
   you can check the camera hasn't moved. It adds a little light to the screen; `--no-inset` hides it.

   | Part | Screen | Duration | Paradigm |
   |---|---|---|---|
   | Baseline | Cross on mid-grey | 20 s | Resting pupil size and noise |
   | Dots | A dot at each of 9 positions (3 × 3 grid), 2.5 s each | 23 s | Measures the eye model's radius, which sets degrees per pixel |
   | Blinks | `BLINK` cue every 3 s | 30 s | Cued blinks, scored as detected within 1.5 s |
   | Light reflex | 8 s black, then 3 × (3 s white, 7 s black) | 38 s | Pupil light reflex after dark adaptation |
   | Reading | Black text on white | 45 s | Reading fixations and saccades |
   | Digit span | Cross on constant grey; digits spoken one per second, 2 s hold, then say the digits back | 12 × 13–20 s | Task-evoked dilation from memory load (Kahneman & Beatty 1966) |

   **Digit span.** 12 lists, 4 each of 3, 5 and 7 digits (1–9, no repeats within a list), mixed so every
   group of three has one of each length. The digits are spoken by the macOS voice (`say`), so nothing on
   screen changes; off macOS they're shown small above the cross instead. Each trial: 4 s fixation, the
   digits at one per second, 2 s of silent holding, then "say the digits" for 4.5–6.5 s. The pupil widens
   with each digit held and peaks before recall. Before measuring, untrusted frames are removed (invalid,
   size jumps, or an ellipse away from the pupil's track, such as a lash), gaps up to 250 ms are filled,
   and the trace is smoothed with a 4 Hz low-pass. Dilation is the mean size from the last digit to the end
   of the hold minus the median size over the 1 s before the first digit, in pixels; the report also gives
   it as a percentage of the session's typical baseline. A list is dropped (and counted) when less than 80%
   of its baseline or window is usable, or when there was speech during the hold. The settings are in the
   `digit_span` block of `config/default.yaml`. Wait for the prompt: saying a long list early is natural,
   but it moves the pupil far more than the memory load does. The recall is recorded,
   transcribed offline and scored (exact match, and digits in the right position) in `span_answers.csv`;
   speech during the hold is flagged. The lists are saved in `span_trials.csv`, and the per-trial
   dilation in `span_trials_pupil.csv`. Seven digits stays within normal memory span: longer lists stop
   dilating the pupil further once they can't be held (Granholm et al. 1996).

   **Arithmetic (optional, `--effort arithmetic` or `--only arithmetic`).** Cross on constant grey; the
   problem is shown for 2 s; press SPACE when solved, then say the answer (Hess & Polt 1964; Ahern &
   Beatty 1979). 2 problems at each level by default, never repeated within a session:

   | Level | Problems | Example |
   |---|---|---|
   | Easy | a small times-table fact (2–4 × 2–9) | 3 × 7 |
   | Medium | two digits × one digit (12–29 × 2–9), getting harder each time (by size of the product): the first is near 12 × 3, the last near 28 × 9 | 24 × 7 |
   | Hard | two digits × two digits (12–19) | 17 × 13 |

   The levels are mixed so each is spread across the session: every group of three trials has one
   of each, in random order. That way tiredness or drift late in the session affects all levels
   equally. The exception is medium, which gets harder through the session, so its late trials mix
   difficulty with time on task. The problems and answers are saved in `trials.csv`. With 6 or more
   problems there's a 20 s break halfway.

   **Answers.** Press **SPACE** once, the moment you have the answer; "say your answer" appears
   straight away and you say it out loud (4 s). Without SPACE, the prompt appears after 10 s.
   SPACE gives a silent **solve time**, and the pupil is measured from the problem appearing up to
   SPACE, so speaking stays outside the measurement. A held or double-tapped key counts once
   (presses within 1 s are ignored). The microphone records the effort part; afterwards each
   answer is transcribed offline and checked. An answer said too early (before SPACE) is still
   scored, but that trial is flagged as talking while solving and left out of the dilation average.

   Needs `pip install -e ".[answers]"` and microphone access for Terminal.app; without them the
   protocol still runs, with `--no-mic` behaviour. `eyetrack answer-test` checks SPACE timing, the
   microphone and transcription on their own in a minute.

   A dot in the bottom-right corner shows whether tracking is still working: green is fine; amber or red
   means the pupil is near the edge, lost, or the camera has stopped (see [realtime.md](realtime.md)). If it
   stays red, press Q and re-aim rather than finishing the session.

3. **The report** appears on screen and is saved as `validation_report.json`: light-reflex
   constriction and timing, digit-span dilation at each list length (mean ± standard error) and how
   many lists were recalled, pupil noise at rest, cued blinks detected, reading fixation and saccade
   figures, and signal quality.

   The digit-span verdict is **graded** if dilation rises 3 < 5 < 7 digits and 7 exceeds 3 by more than
   twice their combined standard error; **clear** if only 7 vs 3 passes that test; **weak** if 7 digits
   dilates by at least 2% but the gap is within the noise.

   For an arithmetic run, the report shows dilation and accuracy per difficulty instead. `arithmetic_trials.csv` lists every problem with its solve time, what was heard,
   whether it was correct, whether you talked while solving, and its dilation
   (`arithmetic_answers.csv` and `audio.wav` hold the raw answer data).

   When answers were scored, **dilation is averaged over correct answers with no talking while
   solving**, as studies usually exclude error trials: a wrong or abandoned problem dilates less.
   The all-trials average is kept as `mean_change_all_trials_pct`. The report also shows accuracy
   and median solve time per level; solve time rising easy < medium < hard confirms the difficulty
   levels really differ for you.

   The effort verdict is **graded** if dilation rises easy < medium < hard and hard exceeds easy by
   more than twice their combined standard error; **clear** if only hard vs easy passes that test;
   **weak** if hard dilates by at least 2% but the gap is within the noise. With few trials per level,
   small differences may not separate from the noise. The standard error shrinks with the square root
   of the number of trials, so doubling `--trials-per-level` cuts it by about 30%.

The separate recordings below (T1–T6) give the same information plus drift and saccade accuracy,
and let you repeat one part at a time.

## What the pipeline measures

`eyetrack process` writes these per session; `metrics_summary.json` holds the summary values.

| Group | Statistic | Output | Validated against |
|---|---|---|---|
| **Signal quality** | Sample rate, detection rate, valid rate, mean confidence | `sample_rate_hz`, `detection_rate`, `valid_rate`, `mean_confidence` | – |
| | Precision (sample-to-sample noise) | `noise_rms_s2s_deg` | Lund2013 with 0 / 0.2 / 0.5° noise |
| | Eyeball model (needed for degrees) | `eye_model` (`residual_px`) | Synthetic videos |
| **Pupil** | Centre, ellipse, diameter per frame | `pupil.csv` | LPW: 80–86% of frames within 5 px |
| | Diameter, cleaned (blinks removed, gaps filled, smoothed) | `diameter_clean_px`, `pupil_diameter_mean_px` / `_sd_px` | Synthetic only (within 5%); **pixels, not mm** |
| | Change after each marker (light reflex, task dilation) | `pupil_epochs_summary.csv`: `mean/min/max_change_pct` | – |
| **Blinks** | Count, rate, duration, type | `blinks.csv`, `blink_count`, `blink_rate_per_min` | Lund2013: F1 0.96 (27 blinks) |
| **Eye movements** | Saccades: count, rate, amplitude, duration, peak velocity | `saccades.csv`, `saccade_*` | Lund2013 at 120 Hz: F1 0.91 (clean), 0.85 (0.2° noise) |
| | Fixations: count, duration | `fixations.csv`, `fixation_*` | Lund2013: F1 0.96 |
| | Eye rotation in degrees | `yaw_deg`, `pitch_deg` | Synthetic videos |
| **Gaze** (needs scene camera + calibration) | Gaze in scene pixels, heatmap, areas of interest | `gaze.csv`, `heatmap.png`, `aoi.csv` | Validation error < 2° target |

**Not available on this build yet:** gaze on the screen (no scene camera), pupil size in mm,
microsaccades (too small for 120 Hz and DIY noise levels).

## Setup checklist

- [ ] Glasses on, with the camera mount tightened and the cable taped along the arm
- [ ] IR LEDs on, starting at 1 kΩ each from 3.3 V (~2 mA); see [hardware.md](hardware.md#eye-safety-rules)
- [ ] Room lighting steady; no window or screen reflected in the eye if you can avoid it
- [ ] Head steady: sit, elbows on the desk, about 57 cm from the screen (then 1 cm on screen ≈ 1°)
- [ ] Run everything from **Terminal.app**, which has camera permission

## Stage 0: live check (nothing is saved)

```bash
.venv/bin/eyetrack --config config/local.yaml live
```

| Check | Good |
|---|---|
| Camera fps (top of the image) | ~120 |
| Image | Black pupil, grey iris, yellow ellipse on the pupil |
| Confidence looking straight ahead | > 0.8 |
| Looking fully left, right, up, down | Pupil still found; both eye corners in frame |
| LED reflections | Small dots, not on the pupil edge |
| Pupil graph while staring at one point | Flat line, gaps only at blinks |

Fix anything failing here before recording: refocus, re-aim the camera, move the LEDs, or adjust
exposure.

## Recordings

Each test is one recording. The `--name` must end in the test ID so `eyetrack check` knows what
to grade:

```bash
.venv/bin/eyetrack --config config/local.yaml record --eye-only --duration 30 --name t1_fixation
.venv/bin/eyetrack --config config/local.yaml process data/<session_folder> --annotate
.venv/bin/eyetrack --config config/local.yaml check data/<session_folder>
```

Click the record window before pressing marker keys (`1` lights off, `2` lights on, `3` task start,
`4` task end, `b` blink cue, `m` marker).

| ID | Duration | Procedure | What it measures | Pass |
|---|---|---|---|---|
| **T1** `t1_fixation` | 30 s | Stare at one dot. Blink normally. | Precision, pupil-size noise, drift, valid rate | valid ≥ 0.95, noise ≤ 0.28°, pupil CV ≤ 3%, drift ≤ 0.5° |
| **T2** `t2_ninepoint` | 30 s | Look at 9 dots on a 3 × 3 grid across the screen, ~3 s each, in reading order | Detection at the edges of gaze, eye-model fit | valid ≥ 0.95 (≥ 0.85 usable), model residual ≤ 5 px |
| **T3** `t3_blinks` | 30 s | Look at a dot. 10 times: press `b`, then blink once | Blink detection | blinks found = 10 ± 1 |
| **T4** `t4_saccades` | 30 s | Two dots 10 cm apart. Switch between them once per second (use a metronome) | Saccade detection and amplitude | 25–34 target saccades, amplitude CV ≤ 25% |
| **T5** `t5_light` | 45 s | Look at a dot. 10 s rest → press `2`, shine a phone torch at the **other** eye for 10 s → press `1`, torch off for 10 s → repeat once | Pupil light reflex | constriction ≤ −10% after `lights_on`, recovery ≥ +5% after `lights_off` |
| **T6** `t6_arithmetic` | 60 s | Look at a dot on a plain grey screen. 15 s rest → press `3`, count back from 1000 in 7s silently for 30 s → press `4` → 15 s rest | Task-evoked pupil dilation (mental effort) | mean dilation ≥ +5% during the task |

Use `--duration 45` for T5 and `--duration 60` for T6. T5 lights the **other** eye so the camera
eye's lighting stays constant: both pupils react together. In T6 keep the screen unchanged, since
any brightness change swamps the effort effect.

## Reading the results

`eyetrack check` prints each value with a verdict:

- **pass**: good enough for that measurement.
- **usable**: works for large effects; treat small ones with care.
- **fail**: don't report this measurement from this build until it's fixed.

Where a target comes from the benchmarks, the check says so. The noise levels map onto
[benchmarks.md](benchmarks.md): RMS-S2S 0.28° ≈ 0.2° noise (saccade F1 0.85) and 0.7° ≈ 0.5°
(F1 0.76). The pupil-size and drift targets are **starting points** chosen from typical effect
sizes: the light reflex changes size by about 20–40%, mental effort by about 5–15%.

### What each result unlocks

| If this passes | You can measure |
|---|---|
| T1 valid rate + noise | Fixations, saccades ≥ ~2° (reading, free viewing) |
| T1 pupil CV ≤ 3% and drift ≤ 0.5° | Small pupil changes within a recording (mental effort) |
| T1 pupil CV ≤ 8% | Large pupil changes only (light reflex) |
| T2 | Eye movements across the whole field of view, in degrees |
| T3 | Blink rate and duration |
| T4 | Saccade counts, amplitudes, rates |
| T5 | Pupil responses to light: the pupil-size signal is real |
| T6 | Cognitive load from pupil dilation |

### If something fails

| Failure | Likely cause | Fix |
|---|---|---|
| Sample rate < 110 Hz or dropped frames | USB hub, CPU load, wrong mode | Plug in directly; close apps; check `config/local.yaml` is 640×400 @ 120 |
| Low valid rate / confidence | Low contrast, reflections, focus | LEDs closer or 470 Ω; refocus; block window reflections; IR-pass filter |
| High noise | Blur, LED flicker, glints on the pupil edge | Refocus; steady LED supply (no PWM); move LEDs |
| High drift | Camera or glasses moving | Tighten the mount; tape the cable; strap or ear hooks |
| Blink count off | Glare dropouts counted as blinks, or partial blinks missed | Check `blinks.csv` types; adjust `blinks.max_blink_confidence` |
| No T5 constriction | Pupil not tracked, or torch too weak | Check T1 first; use a brighter torch at closer range |

## Results log

Copy for each build:

| Date | Build / change | T1 valid | T1 noise (°) | T1 pupil CV % | T1 drift (°) | T2 valid | T3 blinks | T4 saccades | T5 constriction % | T6 dilation % | Notes |
|---|---|---|---|---|---|---|---|---|---|---|---|
| | | | | | | | | | | | |
