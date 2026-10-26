# Blinks

**Current choice:** `blinks.method: brightness` (`metrics/blinks.py`, `classify_dropouts`).

Every run of frames without a trusted pupil is classified from image brightness, compared with the trusted
frames just before it:

| What the image does | Label |
|---|---|
| Eye area gets brighter (eyelid) | `blink` (50–500 ms), `closed` (longer) or `dropout` (shorter) |
| Whole frame gets darker | `lighting` |
| Neither | `lost` |
| Frames missing from the timestamps | `camera_gap` |

Only eyelid events hide frames from later stages.

## Iterations

### 1. Blink = a run of missing pupil frames, 50–500 ms (2026-10-23)
- The standard approach: the pupil disappears during a blink.
- On Lund2013 (27 labelled blinks), merging valid "flickers" under 80 ms back into the blink (a half-open
  eyelid) raised F1 from 0.86 to **0.96**.

### 2. Separating glare from blinks
- On the JEOresearch sample video, frequent glare produced 52 "blinks".
- **Rule:** a dropout where the detector stays fairly confident (median confidence > 0.35) is `low_confidence`,
  not a blink. That cut the 52 to 3. It was never checked against labelled blinks.

### 3. Our hardware: real blinks looked like glare (2026-10-26)
- In the cued-blink test, real blinks were labelled `low_confidence`. During a blink the detector latches onto
  the eyelid crease with confidence 0.41–0.62.
- **Quick fix:** raised the cut-off to 0.65 in `config/local.yaml`. It worked for the cued blinks, but the rule
  depends on how a particular detector behaves on a particular face.

### 4. Screen flicker broke everything downstream
- On white screens and during reading, frame brightness cycled every 10 frames (~12 Hz), probably a beat between
  the screen's light and the camera exposure, not the LEDs.
- Each dark frame lost the pupil. The blink merger joined these short dropouts into an **18.8 s "closed_or_lost"
  event**, which then hid the whole reading section from the saccade detector.

### 5. Brightness-based classifier (current)
- **Question:** "what happened to the image?", not "how confident was the detector?". An eyelid makes the eye
  area brighter; flicker makes the whole frame darker; neither depends on the pupil detector.
- **Rules:** only eyelid-type events are merged, so flicker can't grow into a long "closed" event. Only eyelid
  kinds mask surrounding frames; lighting dropouts lose only their own frames.
- **Result:** 10/10 cued blinks found, and the reading section was usable again.
- A per-frame `dropout_kind` column records why each frame lacks a pupil, so data loss can be broken down by
  cause.

### 6. Live version (real-time worker)
- Same rule, using past frames only.
- The first ~30 ms of each blink reads `dropout` until it's long enough to call.
- The final kind is known about 64–80 ms after the blink ends.
- Agreement with offline: on 30 s of replay, the same 16 blinks.

## Open questions
- Blink timing accuracy (onset and offset error) is unmeasured. Hand-labelling ~200 blinks from a recording
  would give precision, recall and timing error, and validate the brightness thresholds (`lighting_drop: 0.08`,
  `eyelid_rise: 3.0`).
- With the IR-pass filter, screen flicker should mostly disappear. Re-check how many `lighting` frames remain.
