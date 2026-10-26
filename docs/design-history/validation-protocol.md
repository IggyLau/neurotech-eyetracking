# Validation session

**Current choice:** `eyetrack validate`, a guided on-screen session. After a camera alignment step it runs a
baseline, cued blinks, the light reflex, reading and arithmetic, then processes and grades the recording.
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
- The viewer loads the video into memory first, because Python's simple web server can't seek. H.264 (`avc1`)
  replaced `mp4v`, which browsers don't play.

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
- 10–20 arithmetic problems per level (`--trials-per-level`).
