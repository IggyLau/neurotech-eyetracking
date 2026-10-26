# Design history

How each part of the tracker got to its current version: what was tried, what the data showed, and why
the final choice was made. The user-facing docs ([README](../../README.md), [benchmarks](../benchmarks.md))
describe what the code does now; these notes keep the reasoning and the dead ends, so nobody has to rediscover
them.

| Part | Current choice | Notes |
|---|---|---|
| Pupil detection | JEO-based threshold detector, confidence ≥ 0.5 + size-jump filter | [pupil-detection.md](pupil-detection.md) |
| Blinks | Brightness-based blink vs dropout classifier | [blinks.md](blinks.md) |
| Saccades and fixations | I-VT, 30 °/s, amplitude ≥ 3× noise, short-gap filling, 1-sample duration correction | [saccades.md](saccades.md) |
| Eye model and gaze angles | Orthographic eyeball fit, once per session (known to be wrong; fix planned) | [eye-model-and-gaze.md](eye-model-and-gaze.md) |
| Pupil size | Major axis, Kret & Sjak-Shie cleaning, marker epochs | [pupil-size.md](pupil-size.md) |
| Validation session | `eyetrack validate`, SPACE-when-solved arithmetic with spoken answers | [validation-protocol.md](validation-protocol.md) |
| Real-time processing | Worker thread on every frame, causal metrics, everything saved | [realtime.md](realtime.md) |
| Hardware | OV9281 at 640×400 / 120 fps, 850 nm LEDs, wire + tape mount, 850 nm bandpass filter ordered | [hardware.md](hardware.md) |

Data behind the numbers:
- **LPW**: hand-labelled pupil centres, head-mounted IR video.
- **Lund2013**: hand-labelled saccades, fixations and blinks, resampled to 120 Hz with simulated noise.
- **Validate session** of 2026-10-26 (`data/20261026_023319_validate`). Its raw video was deleted afterwards; the
  per-frame results (`pupil.csv`) remain.

When something changes, add a dated entry to the relevant file instead of rewriting the old ones.
