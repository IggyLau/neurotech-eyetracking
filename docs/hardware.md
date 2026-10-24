# Hardware guide

How to build a head-mounted eye tracker for this toolkit from two USB cameras and a glasses frame.

## Cameras

| | OV9281 (eye) | GC0308 (scene) |
|---|---|---|
| Sensor | 1 MP mono, **global shutter** | 0.3 MP colour, rolling shutter |
| Vendor max | ~120 fps @ 1280×800, ~210 fps @ 640×400 | 30 fps @ 640×480 |
| IR filter | **None** (sees 850 nm) | Module sold for eye tracking, built-in IR LEDs |
| Role | Pupil, blinks, saccades, pupil size | What the wearer sees, gaze overlay |

Vendor frame rates are upper bounds. The USB link, codec and operating system decide the real rate,
so measure it with `eyetrack probe --index N` and put the best mode in your config.

## Parts

- 850 nm IR LEDs, 3 mm or 5 mm, low power (2–4). Use 850 nm rather than 940 nm because the OV9281 is
  far more sensitive at 850. Pick a part whose datasheet gives **radiant intensity (mW/sr)**.
- Resistors, 220 Ω – 1 kΩ assortment
- Thin wire, heat-shrink, and a spare USB cable to cut for 5 V (or the board's 5V/GND pads; check the pinout first)
- Hot glue, matte black marker or paint
- Cheap glasses or sunglasses with the lenses popped out; pipe cleaners or a printed clip (MEYElens / Pupil Labs models)
- M12 lens with a short focus distance, if the stock OV9281 lens can't focus at ~3 cm

## IR LED circuit

```
USB 5V ──[ R ]──►|──►|── GND        two 850 nm LEDs in series, Vf ≈ 1.4–1.6 V each
                LED1 LED2           I ≈ (5 V − 3 V) / R
```

| R | Current | Use |
|---|---|---|
| 1 kΩ | ~2 mA | **Start here** |
| 470 Ω | ~4 mA | Only if the image is still too dark after raising exposure/gain |
| 220 Ω | ~9 mA | Upper limit for near-eye use; check the datasheet first |

### Eye safety rules

- Use the least light that gives a clear dark pupil. Raise camera exposure/gain before adding current.
- Never exceed ~20 mA, and never put a lens in front of the LEDs.
- Diffuse each LED with a dab of hot glue (the JEO fix). This also softens glints.
- Compare the LED's radiant intensity at ~3 cm against the IEC 62471 *exempt group* limit.
  This guide is not a safety certification.
- Stop if the wearer feels warmth, dryness or discomfort, and keep test sessions short.

### Avoiding red-eye (bright pupil)

Red-eye happens when the light source is close to the camera's optical axis: light bounces off the
retina straight back into the lens. Eye-tracking camera modules with LEDs right next to the lens,
such as the GC0308, are prone to it. The detector needs the **pupil to be the darkest blob**, so:

- Mount the LEDs **off-axis**: ≥ 10–15 mm from the lens center, angled toward the eye.
- If the pupil still glows at some gaze angles, move the LEDs further out.
- Put matte black on the area right around the lens to cut stray reflections.

Build on a breadboard first. Check the image in `eyetrack live`, then solder.

## Mounting

- **Eye camera:** 3–4 cm from the eye, **below and slightly to the side**, looking up. This keeps
  eyelashes out of the way and the pupil visible when the wearer looks down. Mount it rigidly: any movement breaks
  calibration and the eyeball model.
- **Scene camera:** on the bridge or temple, facing forward. **Cover or desolder its IR LEDs.**
- Run the cables along the temple arm and tie them down so a tug doesn't move the cameras.
- If the eye camera shows a large area of skin, restrict the search with `pupil.roi: [x, y, w, h]` in your config.

## Verifying a build

Record each check as its own session with `eyetrack record`, then run `eyetrack process` on it.

| Check | Procedure | Expected result |
|---|---|---|
| Detection | Look at 9 points spread across the field of view for 30 s | `valid_rate` > 0.95 in `metrics_summary.json` |
| Signal noise | Any steady fixation | Low `noise_rms_s2s_deg`. Compare with the noise levels in [benchmarks.md](benchmarks.md). |
| Pupil response | Steady light 10 s → press `1`, lights off 10 s → press `2`, lights on | Clear diameter increase after `lights_off` in `report.png` |
| Saccades | Alternate between two targets ~10° apart (~10 cm apart at 57 cm), once per second for 30 s | ~30 saccades of consistent amplitude |
| Blinks | Press `b` and blink, 10 times | `blink_count` = 10 |
| Gaze accuracy | `eyetrack calibrate` | Validation error < 2° |
