# Validation session

**Current choice:** `eyetrack validate`, a guided on-screen session. After a camera alignment step it runs a
baseline, cued blinks, the light reflex, reading and a digit-span memory task, then processes and grades the
recording. Mental arithmetic is still available with `--effort arithmetic`.
`protocol.py` runs it; `answers.py` handles spoken answers; `validation.py` grades it.

## Iterations

### 1. A test plan, recorded by hand (2026-10-24)
- Separate recordings per test: fixation, 9-point, blinks, saccades, light and arithmetic. Each is processed with
  `eyetrack process` and graded with `eyetrack check` against pass/fail targets
  ([validation_protocol.md](../validation_protocol.md)).
- Too many manual steps to repeat often, so it was turned into one guided command.

### 2. Guided session modelled on real experiments
- **Baseline** on a grey screen.
- **10 cued blinks.**
- **Light reflex:** the screen alternates between black and white.
- **Reading** a passage.
- **Arithmetic:** mental arithmetic at three difficulty levels × 5 problems, after the classic task-evoked
  pupil studies.
- A short version (`--short`) and per-part runs (`--only`) exist for quick checks.

### 3. Camera alignment step
- In early tries, hair at the edge of the frame was picked as the pupil.
- **Added:** a live alignment screen that checks the pupil is found, centred and large enough before recording.
  The preview is mirrored like Photo Booth, so moving the camera feels natural. Recordings aren't mirrored.

### 4. How to collect arithmetic answers
- The original studies had people solve silently, then answer. Talking moves the face and changes pupil size.
- **Options considered:** typing, saying the answer straight away, or pressing SPACE when solved and then
  answering aloud.
- **Chosen:** SPACE when solved, then speak, recorded by the microphone and transcribed offline with
  faster-whisper.
- **First answer test: 5/6.** A quiet "19" was missed. Fixed with a voice-onset threshold relative to the
  background noise and audio normalisation, which brought it to 6/6.
- **Transcription runs in a separate process,** because OpenCV and PyAV load conflicting copies of the same
  video library on macOS. That process can't start when Python reads the script from stdin, so it must run from
  a file.

### 5. SPACE counts immediately
- In the first full run, the user blurted answers before the "say your answer" prompt, and waiting felt
  unnatural.
- **Changed:** SPACE ends the problem straight away; keys are debounced for 1 s.
- Trials where the person talked while solving are flagged, and dilation is measured only up to the speech onset.
- Epochs ignore the `solved` marker so it doesn't cut them short.

### 6. Review tools
- **Report:** a timeline plot and `validation_report.json` with verdicts.
- **Excerpts:** short clips of detection working and failing.
- **Frame-by-frame viewer:** `eyetrack viewer --part reading`, an HTML page with an H.264 video, a saccade list
  and the step-detector signal.
- 2026-11-02: the viewer now covers the whole session by default (`--part` still picks one part). The video
  sits on top, with everything else stacked below it: a readout of the current frame, the 3 s close-up, traces of
  position, pupil size and confidence for the whole recording, saccades per protocol step, and the saccade list.
  The report plot also got a tick per saccade, because on a full-session axis short reading saccades were
  narrower than a pixel and their shading didn't show.
- The viewer loads the video into memory first, because Python's simple web server can't seek. H.264 (`avc1`)
  replaced `mp4v`, which browsers don't play.

### 7. Digit span replaces arithmetic as the effort task (2026-11-01)
- Arithmetic took about 5 minutes and its load depended on strategy, with variable solve times, so the pupil
  peak landed at a different time on every trial. Effort dilation wasn't detected in the first full run.
- **Options considered:** digit span, n-back, Stroop and auditory oddball, alongside arithmetic. In Beatty's
  1982 review, seven-digit spans and hard multiplication give the largest task-evoked responses (about
  0.5 mm). N-back is usually in between; Stroop and oddball give smaller effects. N-back and Stroop also
  need a response on every item, and Stroop changes screen colour.
- **Chosen:** digit span (Kahneman & Beatty 1966). The load is fixed by the list length, every trial has the
  same timing, and the step-up with each digit is a built-in check. Lists of 3, 5 and 7 digits, 4 each,
  about 3.5 minutes. Seven is the limit because dilation stops rising once a list exceeds memory span
  (Granholm et al. 1996).
- Spoken digits (macOS `say`), so the screen doesn't change. Each digit is rendered once and lists are
  assembled into one WAV with exact 1 s spacing, so the timing doesn't depend on the speech engine.
- Measured from the last digit to the end of the 2 s hold, before any speech. All trials count, since the
  load is presented whether or not the recall is right.
- Arithmetic kept as an option, cut from 5 to 2 problems per level.

### 8. Cleaner digit-span measure (2026-11-02)
- First full digit-span run: 3 / 5 / 7 digits gave +3.1% / +4.8% / +13.3% (± 4.1 / 2.6 / 9.1), with single
  lists from −4% to +40%. Inside one 3 s window the measured size swung 7–42% of baseline, against a real
  response of a few percent.
- Causes found: frames where a lash or the lid was taken for the pupil and still passed, frame-to-frame
  jitter, a 0.5 s baseline divided into the response, and speech. In 3 of the 4 seven-digit lists you started
  saying the digits during the 2 s "silent" hold (the microphone level matched the spoken answers), including
  the +40% list.
- **Changed:** untrusted frames removed, gaps up to 250 ms filled, a 4 Hz zero-phase low-pass, a 1 s median
  baseline subtracted in px (Mathôt et al. 2018), and lists dropped when under 80% usable or with speech in
  the hold. The instructions now say to wait for the prompt. None of these settings were tuned on the session.
- Same session re-analysed: +0.2% ± 4.9 (4 lists) / +3.7% ± 1.6 (4) / +0.2% (1 list, 3 dropped). No
  reliable memory-load effect in that session; the earlier 7-digit figure came mostly from talking.

## Results of the first full run (2026-10-26)
- **Clear:** blinks 10/10; light reflex −50%.
- **Problems:**
  - effort dilation not detected;
  - reading data 58% valid because of screen flicker and the reflection;
  - pupil noise "moderate";
  - frame rate dropped late in the session because the battery was low.
- The raw video was deleted afterwards. **Keep `eye.avi` until the session has been reviewed**; only the
  `clip_*.mp4` and `viewer_*.mp4` files are safe to delete (they're regenerated).

## Next changes planned
- A 9-point calibration plus held-out validation points at the start and end, for accuracy, precision and drift.
- A "look around" step for the eye model.
- An automatic pass/fail quality check at the end.
- More digit lists per length if the 7 vs 3 gap doesn't separate from the noise (`--trials-per-level`).
