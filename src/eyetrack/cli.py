"""Command-line entry point: eyetrack probe | live | record | calibrate | process."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import yaml

from .config import load_config


def cmd_probe(args, cfg):
    from .capture.probe import probe
    probe(args.index, args.seconds)


def cmd_live(args, cfg):
    """Live pupil-detection preview for positioning the camera and tuning LEDs/exposure."""
    import cv2
    from .capture.camera import Camera
    from .pupil.detector import DetectorParams, detect_pupil, draw_result

    params = DetectorParams.from_config(cfg["pupil"])
    cam = Camera.from_config(cfg["eye_camera"], "eye").start()
    print("Live view. Keys: q=quit. Aim for a dark round pupil, small glints, confidence > 0.8.")
    t_last, n_last, fps = time.perf_counter(), 0, 0.0
    try:
        while True:
            f = cam.latest()
            if f is None:
                time.sleep(0.01)
                continue
            r = detect_pupil(f.image, params)
            vis = draw_result(f.image, r)
            now = time.perf_counter()
            if now - t_last >= 1:
                fps = (cam.frame_count - n_last) / (now - t_last)
                t_last, n_last = now, cam.frame_count
            cv2.putText(vis, f"camera {fps:.0f} fps", (10, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 200, 0), 2)
            if params.roi:
                x, y, w, h = params.roi
                cv2.rectangle(vis, (x, y), (x + w, y + h), (255, 0, 255), 1)
            cv2.imshow("eyetrack live", vis)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cam.stop()
        cv2.destroyAllWindows()


def cmd_record(args, cfg):
    from .capture.recorder import record
    record(cfg, args.name, args.duration, use_scene=not args.eye_only, preview=not args.no_preview)


def cmd_calibrate(args, cfg):
    from .calibration.targets import CALIBRATION_POINTS, VALIDATION_POINTS, TargetSequence
    from .capture.recorder import record
    from .pipeline import calibrate_from_session

    if args.session:
        session = Path(args.session)
    else:
        labels = ([f"cal_{i}" for i in range(len(CALIBRATION_POINTS))] +
                  [f"val_{i}" for i in range(len(VALIDATION_POINTS))])
        seq = TargetSequence(CALIBRATION_POINTS + VALIDATION_POINTS, labels, tuple(args.screen), args.dwell,
                             dict_name=cfg["calibration"]["aruco_dict"])
        session = record(cfg, name="calibration", display=seq)
    calibrate_from_session(session, cfg)


def cmd_process(args, cfg):
    from .pipeline import process
    aois = None
    if args.aois:
        aois = (yaml.safe_load if args.aois.endswith((".yaml", ".yml")) else json.loads)(Path(args.aois).read_text())
    process(args.path, cfg, args.calibration, aois, args.annotate, args.overlay, args.max_frames)


def main(argv=None):
    p = argparse.ArgumentParser(prog="eyetrack", description=__doc__)
    p.add_argument("--config", help="YAML file overriding config/default.yaml")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("probe", help="list cameras; with --index, measure real fps per mode")
    s.add_argument("--index", type=int)
    s.add_argument("--seconds", type=float, default=3.0)
    s.set_defaults(func=cmd_probe)

    s = sub.add_parser("live", help="live pupil-detection preview on the eye camera")
    s.set_defaults(func=cmd_live)

    s = sub.add_parser("record", help="record raw eye (+scene) video with timestamps")
    s.add_argument("--name", help="label appended to the session folder name")
    s.add_argument("--duration", type=float, help="seconds; default: until q")
    s.add_argument("--eye-only", action="store_true")
    s.add_argument("--no-preview", action="store_true")
    s.set_defaults(func=cmd_record)

    s = sub.add_parser("calibrate", help="show targets full-screen, record, fit pupil->scene mapping")
    s.add_argument("--session", help="re-fit from an existing calibration recording instead of recording")
    s.add_argument("--screen", type=int, nargs=2, default=[1440, 900], metavar=("W", "H"))
    s.add_argument("--dwell", type=float, default=2.0, help="seconds per target")
    s.set_defaults(func=cmd_calibrate)

    s = sub.add_parser("process", help="compute pupil, blink, saccade/fixation (and gaze) metrics")
    s.add_argument("path", help="session folder or an eye video file")
    s.add_argument("--calibration", help="calibration.json from `eyetrack calibrate`")
    s.add_argument("--aois", help="YAML/JSON {name: [x, y, w, h]} in scene px (needs --calibration)")
    s.add_argument("--annotate", action="store_true", help="also write eye_annotated.mp4")
    s.add_argument("--overlay", action="store_true", help="also write overlay.mp4 (gaze on scene)")
    s.add_argument("--max-frames", type=int)
    s.set_defaults(func=cmd_process)

    args = p.parse_args(argv)
    cfg = load_config(args.config)
    args.func(args, cfg)


if __name__ == "__main__":
    main()
