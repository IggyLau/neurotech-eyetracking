# Pupil detection

**Current choice:** the JEOresearch-based threshold detector (`pupil/detector.py`, `method: threshold`,
`refine: angle`), keeping frames with confidence ≥ 0.5 that pass a size-jump filter. Three alternatives exist as
options but are off by default.

## Iterations

### 1. Port of the JEOresearch detector (2026-10-23)
- **What:** darkest block → threshold at darkest + 5 / 15 / 25 → largest roundish contour → ellipse fit, keeping
  the best of the three thresholds. Ported from JEOresearch/EyeTracker (MIT), made resolution-independent, with a
  morphological closing instead of dilation so diameters aren't inflated.
- **Why this base:** the team's repo review picked JEO as the closest open-source match to a head-mounted IR
  setup. It is simple and fast (1.3 ms/frame).

### 2. Tuning on LPW (2026-10-23)
- **Data:** LPW (Tonsen 2016) records with a Pupil Labs camera, the closest public match to our OV9281.
  It was center-cropped to 640×400. Tuning used participants 1–11; results are reported on 12–22.
- **Tried:** glint removal, median blur, other threshold offsets, search box sizes, closing kernels and contour
  refinement, 60+ configurations in all. **None beat the default** on normal footage (79.7% within 5 px, median
  error 1.6 px).
- **Kept as options:** multi-candidate search (`n_seeds: 5, score: shape`). It rescued heavy mascara (14% → 72%)
  but was slightly worse on normal footage and 10× slower.

### 3. Which frames to trust
- With no filtering, 11.4% of kept frames were wrong (>10 px), usually the iris or a shadow.
- Confidence ≥ 0.6 alone cut that to 3.1% but threw away 4.9% of correct frames.
- **Chosen:** confidence ≥ 0.5 plus dropping frames whose diameter jumps >1.5× against the 2 s rolling median.
  That gives the same 3.1% wrong while keeping 88% of frames.

### 4. Our own recordings: the screen reflection (2026-10-26)
- In the first validate session, confidence sat around 0.65 on grey and white screens but was 1.0 on a dark screen.
- **Cause:** the lit screen reflects on the cornea as a grey patch about as bright as the iris. Where it covers
  the pupil, the dark blob's outline runs along the reflection. Visual checks showed the ellipse was usually
  still right; the confidence score penalised the outline.
- The JEOresearch sample video scores the same through our detector (median 0.62), so JEO's footage is not
  "higher confidence"; JEO's tracker just doesn't display a score.

### 5. Alternatives tried against the reflection
All on LPW held-out frames, scored by centre error against the hand labels:

| Variant | Within 5 px | Median error | Outcome |
|---|---|---|---|
| Default | 79.7% | 1.64 px | **kept** |
| `refine: visible`: ignore outline points next to bright pixels | 79.9% | 1.65 px | Option. Same ellipses, higher confidence. Rejected as default because it raises the score, not the accuracy, and lets more wrong frames through at a given cut-off |
| `method: edge`: our edge + RANSAC version of the published Pupil Labs method | 53.2% | – | Option. Its coarse pupil search often fails, and it jittered more on our data |
| `method: pupil_labs`: Pupil Labs' own `pupil-detectors` 2.0.2 | 26.8% (on a consecutive-frame subset) vs 75.2% default | 173 px | Option. Finds the pupil region, then usually rejects it |

Building Pupil Labs' detector took three tries:
1. Against Homebrew's OpenCV 5 it didn't compile, because `fitEllipse` and friends had moved.
2. With an include patch it compiled but misdetected even clean synthetic pupils.
3. Against `opencv@4` (4.14) it compiled cleanly; the LPW result in the table above comes from this build.

Pupil Capture users normally set an eye region and tune darkness and size per camera; we only set the size
range, so this is an "out of the box" result. Its defaults target Pupil Labs' 192×192 and 400×400 cameras.

### Lesson: optimise accuracy, not the confidence score
We nearly made `refine: visible` the default because it raised confidence. Confidence is the detector's opinion
of itself. Choose detectors on error against labels, then pick the confidence cut-off that gives an acceptable
wrong-frame rate.

## Open questions
- **Outline accuracy:** LPW labels only centres, so we can't measure whether the ellipse outline is accurate,
  which is what pupil size depends on. We need an ellipse- or mask-labelled set: TEyeD, ElSe/ExCuSe or OpenEDS.
- **"Lighting" frames:** in reading, 42% of frames were lost as "lighting". The frames are darker (brightness
  97 vs 162), the fixed threshold grabs a blob 2.3× too big, and the size filter rejects it. Planned fixes: search
  near the previous pupil, set the threshold from local contrast, and accept only plausible sizes. The IR filter
  should remove most of the cause.
- **Jitter between frames:** about 0.7 px median at fixation. A median + 1€ temporal filter cut it ~55% in a
  quick test, but would need to run offline in both directions (forwards and backwards) to keep saccade timing.

### Edge penalty (2026-11-01)
- On the glasses, a dark glasses frame, hair or mount at the image border can be darker than the pupil. The
  darkest-block search then seeds there and the ellipse lands on the frame.
- **Chosen:** `edge_penalty: 10`, `edge_margin: 0.2`. The search adds up to 10 grey levels at the border,
  fading linearly to 0 at 20% of the width/height inward. A pupil in the middle wins unless the edge region
  is more than 10 levels darker. A pupil near the edge is still found, just with less priority.
- LPW at 5 px (every 10th frame, participant 22 excluded):

  | edge_penalty | Tune (1–11) | Held-out (12–21) | Wrong among accepted, held-out |
  |---|---|---|---|
  | 0 | 66.9% | 78.0% | 3.9% |
  | 5 | 67.6% | – | – |
  | 10 | 67.8% | 78.0% | 4.0% |
  | 20 | 67.8% | – | – |
  | 40 | 67.6% | – | – |

- LPW's crops rarely have dark frames at the edge, so neutral on held-out is the expected result. The check
  here is that it doesn't hurt.

### Search zone (2026-11-01)
- In the first good `validate` recording on the glasses, every valid frame had the pupil centre inside the
  alignment box (x 257–464, y 115–189 of 640×400). 97.6% of the wrong detections were outside it, mostly hair
  or the frame at the left edge, with diameters around 280 px against a real 70 px.
- **Chosen:** `search_zone: [0.25, 0.20, 0.75, 0.80]`, the same box the alignment screen uses. Seeds outside
  it are skipped, and a candidate is rejected if its blob or fitted ellipse is centred outside it. The image
  isn't cropped (unlike `roi`), so a pupil near the box edge keeps its whole outline.
- Re-processing that session:

  | | Valid | Mean confidence | Jumps > 60 px | Diameter p99 |
  |---|---|---|---|---|
  | No zone | 89.5% | 0.86 | 175 | 476 px |
  | Zone | 97.9% | 0.91 | 4 | 95 px |

- One session from one person, so this checks the box removes the edge detections; it isn't a held-out
  benchmark. LPW runs ignore the zone, since its videos aren't aligned to the box.
- It assumes the camera stays aligned: if the pupil drifts out of the box, it's reported as lost rather than
  found somewhere else. The alignment screen and the bottom-right inset show the box.
- **Fallback seeds (2026-11-02).** In the next recording, 26% of frames found no pupil although it was sharp and
  well inside the box. A shadow from the image edge reached just inside the box and was darker than the pupil
  (whose glint brightens it), so it was the only seed; its blob was centred outside and rejected. Now, with a
  zone, up to 3 seeds are tried in order of darkness and the first that passes is kept, so the darkest valid
  spot still wins. Re-processing:

  | Session | Valid before → after | Valid frames > 20 px off the local track |
  |---|---|---|
  | 2026-11-01 21:05 | 97.9% → 98.1% | 115 → 128 |
  | 2026-11-01 23:18 | 94.7% → 94.9% | 93 → 97 |
  | 2026-11-02 00:05 | 73.7% → 99.3% | 7 → 8 |

  Raw jumps between found frames go up, but almost all of those frames are already dropped as invalid.

### Surround check (2026-11-01)
- Idea: a pupil is dark with brighter iris on every side. Hair or the frame at the image edge is dark too, but
  it runs off the image or is bright on one side only. The existing `score: shape` contrast averages the whole
  ring, so one very bright side can hide a dark side.
- `surround_fraction` samples 64 directions just outside the fitted ellipse (three distances) and a set of points
  inside, and counts how many of 8 sectors are at least 10 grey levels brighter than the inside. The score is
  multiplied by that fraction to the power `surround`. A first version built a pixel mask per candidate: 16.6 ms
  per frame, and degenerate fits thousands of px wide ran out of memory. Point sampling costs ~0.4 ms.
- LPW at 5 px (participant 22 excluded):

  | Setting | Tune (1–11) | Held-out (12–21) |
  |---|---|---|
  | default | 67.8% | 78.0% |
  | `surround: 1` | 68.2% | – |
  | `surround: 2` | 68.3% | 78.2% |
  | `n_seeds: 3` | 62.6% | – |
  | `n_seeds: 3, surround: 2` | 66.3% | – |

- 2026-11-01 validate session (5540 frames):

  | Setting | Valid | Jumps > 60 px |
  |---|---|---|
  | no zone | 89.5% | 175 |
  | no zone, `surround: 2` | 89.5% | 131 |
  | no zone, `n_seeds: 3, surround: 2` | 97.9% | 71 |
  | zone | 97.9% | 4 |
  | zone, `surround: 2` | 97.9% | 4 |
  | zone, `n_seeds: 3, surround: 2` | 96.9% | 125 |

- **Kept as an option, off.** With the search zone it adds nothing, and on LPW the gain is within noise. Extra
  seeds inside the zone mostly find eyelid shadow and lashes, which the surround check doesn't always reject.
  Without a zone (other headsets), `n_seeds: 3, surround: 2` is the better choice.

