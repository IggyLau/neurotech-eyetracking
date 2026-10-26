"""Command-line entry point: eyetrack probe | live | record | realtime-bench | calibrate | process | check | benchmark."""

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


def _eye_model_arg(args, cfg):
    from .realtime.online import load_eye_model
    path = getattr(args, "eye_model", None) or cfg.get("realtime", {}).get("eye_model")
    model = load_eye_model(path)
    if model is not None:
        print(f"Live gaze angles from eye model {path} (re-fit if the camera has moved)")
    return model


def cmd_live(args, cfg):
    """Live pupil-detection preview for positioning the camera and tuning LEDs/exposure.

    Every camera frame is processed by the real-time worker; the window shows the newest result
    (~60 Hz). Nothing is saved. --replay plays a recorded session/video through the same path.
    """
    import cv2
    import numpy as np
    from .capture.camera import Camera
    from .capture.recorder import MARKER_KEYS, _as_result
    from .realtime.replay import ReplayCamera, draw_live, resolve_video
    from .realtime.worker import RealtimeProcessor
    from .viz.live import PupilTrace, with_trace

    params_roi = cfg["pupil"].get("roi")
    trace = PupilTrace(min_confidence=cfg["pupil"]["min_confidence"],
                       baseline_s=cfg["pupil_size"]["baseline_ms"] / 1000)
    mirror = cfg["eye_camera"].get("mirror_preview", False)
    if args.replay:
        video, ts = resolve_video(args.replay)
        cam = ReplayCamera(video, ts, speed=args.speed)
    else:
        cam = Camera.from_config(cfg["eye_camera"], "eye")
    proc = RealtimeProcessor(cfg, None, backend=args.backend, eye_model=_eye_model_arg(args, cfg),
                             fps=cfg["eye_camera"]["fps"]).start()
    cam.on_frame = proc.submit
    cam.start()
    print("Live view. Keys: q=quit, m (or 1-4, b) = mark the trace (not saved). "
          "Aim for a dark round pupil, small glints, confidence > 0.8.")
    t_last, n_last, p_last, fps, pfps = time.perf_counter(), 0, 0, 0.0, 0.0
    n_done, lat = 0, []
    try:
        while not getattr(cam, "finished", False):
            for row in proc.drain():
                trace.add(row["t_capture_ns"] / 1e9, _as_result(row))
                lat.append(row["latency_ms"])
                n_done += 1
            f, row = proc.latest()
            if f is None:
                if cv2.waitKey(5) & 0xFF == ord("q"):
                    break
                continue
            vis = draw_live(f.image, row, mirror)
            now = time.perf_counter()
            if now - t_last >= 1:
                fps = (cam.frame_count - n_last) / (now - t_last)
                pfps = (n_done - p_last) / (now - t_last)
                t_last, n_last, p_last = now, cam.frame_count, n_done
                lat = lat[-int(max(pfps, 1)):]
            p95 = float(np.percentile(lat, 95)) if lat else float("nan")
            cv2.putText(vis, f"camera {fps:.0f} fps  processed {pfps:.0f} fps  latency p95 {p95:.1f} ms  "
                             f"dropped {proc.dropped}", (10, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 200, 0), 2)
            if params_roi:
                x, y, w, h = params_roi
                x = vis.shape[1] - x - w if mirror else x
                cv2.rectangle(vis, (x, y), (x + w, y + h), (255, 0, 255), 1)
            cv2.imshow("eyetrack live", with_trace(vis, trace))
            key = cv2.waitKey(15) & 0xFF          # display ~50 Hz; processing runs at the camera rate
            if key == ord("q"):
                break
            if key in MARKER_KEYS:
                trace.mark(time.perf_counter(), MARKER_KEYS[key])
    finally:
        cam.stop()
        s = proc.stop()
        cv2.destroyAllWindows()
        print(f"Processed {s['processed']} frames at {s['processing_fps']} fps, dropped {s['dropped']}, "
              f"latency p50 {s['latency_ms']['p50']} ms p95 {s['latency_ms']['p95']} ms")


def cmd_record(args, cfg):
    from .capture.recorder import record
    record(cfg, args.name, args.duration, use_scene=not args.eye_only, preview=not args.no_preview,
           realtime=False if args.no_realtime else None, eye_model=_eye_model_arg(args, cfg),
           backend=args.backend)


def cmd_realtime_bench(args, cfg):
    """Replay a recorded eye video through record() + the real-time worker and report throughput."""
    from .capture.recorder import record
    from .realtime.replay import ReplayCamera, compare_with_offline, resolve_video

    video, ts = resolve_video(args.path)
    speed = 0.0 if args.fast else args.speed
    cam = ReplayCamera(video, ts, speed=speed, max_frames=args.max_frames, start_frame=args.start_frame)
    if args.out:
        cfg["recording"]["output_dir"] = args.out
    session = record(cfg, name="realtime_bench", cameras={"eye": cam}, preview=args.display, realtime=True,
                     eye_model=_eye_model_arg(args, cfg), backend=args.backend, block_when_full=args.fast,
                     write_video=not args.no_video)
    live = json.loads((session / "live_summary.json").read_text())
    meta = json.loads((session / "meta.json").read_text())
    report = {"source": str(video), "speed": "fast (blocking)" if args.fast else speed,
              "session": str(session), "source_frames_delivered": cam.frame_count,
              "delivery_fps": meta["fps_measured"].get("eye"), **live,
              "worker_cpu_per_frame_ms": round(1000 * live["worker_cpu_s"] / max(live["processed"], 1), 3)
              if "worker_cpu_s" in live else None,
              "worker_core_utilisation": round(live["worker_cpu_s"] / live["wall_s"], 3)
              if "worker_cpu_s" in live and live["wall_s"] else None}
    offline = Path(args.path) / "pupil.csv" if Path(args.path).is_dir() else None
    if offline is not None and offline.exists():
        report["vs_offline"] = compare_with_offline(session / "live_pupil.csv", offline, args.start_frame)
    (session / "realtime_bench.json").write_text(json.dumps(report, indent=2, default=float))
    print(json.dumps(report, indent=2, default=float))


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


def cmd_benchmark(args, cfg):
    import pandas as pd
    pd.set_option("display.width", 200)
    pd.set_option("display.precision", 3)
    if args.dataset == "lund":
        from .benchmark import lund
        root = args.root or "data/datasets/lund2013"
        print("Expert vs expert saccade agreement (ceiling):", lund.human_ceiling(Path(root)))
        df = lund.run(root, cfg, rates=args.rates, noises=args.noise)
        print(lund.summarise(df).to_string())
    else:
        from .benchmark import lpw
        from .pupil.detector import DetectorParams
        root = args.root or "data/datasets/lpw"
        df = lpw.run(root, DetectorParams.from_config(cfg["pupil"]), step=args.step)
        print(json.dumps(lpw.summarise(df, cfg["pupil"]["min_confidence"]), indent=2))


def cmd_check(args, cfg):
    from .validation import check, format_report
    test, checks = check(args.path, args.test, nominal_fps=cfg["eye_camera"]["fps"],
                         expected_saccades=args.expected_saccades)
    print(format_report(args.path, test, checks))


def cmd_validate(args, cfg):
    from .protocol import run
    run(cfg, tuple(args.screen), align=not args.skip_align, short=args.short, process_after=not args.no_process,
        parts=args.only or None, trials_per_level=args.trials_per_level, seed=args.seed, use_mic=not args.no_mic)


def cmd_answer_test(args, cfg):
    from .answers import run_answer_test, score_answer_test
    if args.rescore:
        score_answer_test(args.rescore)
    else:
        run_answer_test(args.n, cfg["recording"]["output_dir"], seed=args.seed)


def cmd_viewer(args, cfg):
    from .viz.viewer import make_viewer
    path = make_viewer(args.path, args.part, cfg=cfg)
    print(f"Open {path} (double-click it, or `open {path}`)")


def main(argv=None):
    p = argparse.ArgumentParser(prog="eyetrack", description=__doc__)
    p.add_argument("--config", help="YAML file overriding config/default.yaml")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("probe", help="list cameras; with --index, measure real fps per mode")
    s.add_argument("--index", type=int)
    s.add_argument("--seconds", type=float, default=3.0)
    s.set_defaults(func=cmd_probe)

    s = sub.add_parser("live", help="live pupil-detection preview on the eye camera (every frame processed)")
    s.add_argument("--replay", metavar="SESSION", help="play a recorded session folder / eye video instead of the camera")
    s.add_argument("--speed", type=float, default=1.0, help="--replay speed (1 = real time)")
    s.add_argument("--backend", choices=["thread", "process"], help="real-time worker (default: config realtime.backend)")
    s.add_argument("--eye-model", help="session folder / metrics_summary.json with a fitted eye model: live yaw/pitch")
    s.set_defaults(func=cmd_live)

    s = sub.add_parser("record", help="record raw eye (+scene) video with timestamps")
    s.add_argument("--name", help="label appended to the session folder name")
    s.add_argument("--duration", type=float, help="seconds; default: until q")
    s.add_argument("--eye-only", action="store_true")
    s.add_argument("--no-preview", action="store_true")
    s.add_argument("--no-realtime", action="store_true", help="don't process frames live (no live_pupil.csv)")
    s.add_argument("--backend", choices=["thread", "process"], help="real-time worker (default: config realtime.backend)")
    s.add_argument("--eye-model", help="session folder / metrics_summary.json with a fitted eye model: live yaw/pitch")
    s.set_defaults(func=cmd_record)

    s = sub.add_parser("realtime-bench", help="replay a recorded eye video through the live pipeline; report fps, "
                                              "dropped frames and latency")
    s.add_argument("path", help="session folder (eye.avi + eye_ts.csv) or an eye video")
    s.add_argument("--speed", type=float, default=1.0, help="replay speed, 1 = the original frame timing")
    s.add_argument("--fast", action="store_true", help="as fast as possible (the replay waits for the worker)")
    s.add_argument("--max-frames", type=int)
    s.add_argument("--start-frame", type=int, default=0)
    s.add_argument("--backend", choices=["thread", "process"])
    s.add_argument("--display", action="store_true", help="show the record preview window during the replay")
    s.add_argument("--no-video", action="store_true", help="don't re-encode eye.avi (measures processing alone)")
    s.add_argument("--eye-model", help="session folder / metrics_summary.json with a fitted eye model")
    s.add_argument("--out", help="output root (default: recording.output_dir)")
    s.set_defaults(func=cmd_realtime_bench)

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

    s = sub.add_parser("benchmark", help="score detectors against labelled datasets in data/datasets/")
    s.add_argument("dataset", choices=["lund", "lpw"])
    s.add_argument("--root", help="dataset folder (default data/datasets/lund2013 or data/datasets/lpw)")
    s.add_argument("--rates", type=float, nargs="+", default=[500, 120, 60, 24], help="lund: simulated Hz")
    s.add_argument("--noise", type=float, nargs="+", default=[0.0, 0.2, 0.5], help="lund: added noise, deg")
    s.add_argument("--step", type=int, default=10, help="lpw: evaluate every Nth frame")
    s.set_defaults(func=cmd_benchmark)

    s = sub.add_parser("check", help="grade a processed session against docs/validation_protocol.md")
    s.add_argument("path", help="processed session folder")
    s.add_argument("--test", choices=["t1_fixation", "t2_ninepoint", "t3_blinks", "t4_saccades", "t5_light",
                                      "t6_arithmetic"], help="default: taken from the folder name")
    s.add_argument("--expected-saccades", type=int, default=30, help="t4: number of target jumps")
    s.set_defaults(func=cmd_check)

    s = sub.add_parser("validate", help="guided session: align the camera, then an on-screen pupil/blink/reading "
                                        "protocol, then a report")
    s.add_argument("--screen", type=int, nargs=2, default=[1440, 900], metavar=("W", "H"))
    s.add_argument("--skip-align", action="store_true", help="go straight to the protocol")
    s.add_argument("--short", action="store_true", help="fewer repetitions (~2.5 min) for a quick try")
    s.add_argument("--no-process", action="store_true", help="only record")
    s.add_argument("--only", nargs="+", choices=["baseline", "blinks", "light", "reading", "arithmetic"],
                   help="run only these parts, e.g. --only arithmetic")
    s.add_argument("--trials-per-level", type=int, help="arithmetic problems per difficulty level (default 5)")
    s.add_argument("--seed", type=int, help="fix the arithmetic problems, to repeat the same set")
    s.add_argument("--no-mic", action="store_true", help="don't record or score spoken answers")
    s.set_defaults(func=cmd_validate)

    s = sub.add_parser("answer-test", help="check SPACE timing, microphone and number transcription")
    s.add_argument("-n", type=int, default=6, help="numbers to say")
    s.add_argument("--seed", type=int)
    s.add_argument("--rescore", metavar="SESSION", help="re-score a saved answer-test session")
    s.set_defaults(func=cmd_answer_test)

    s = sub.add_parser("viewer", help="frame-by-frame video viewer for one part of a processed validate session")
    s.add_argument("path", help="processed session folder")
    s.add_argument("--part", default="reading", choices=["baseline", "blinks", "light", "reading", "arithmetic"])
    s.set_defaults(func=cmd_viewer)

    args = p.parse_args(argv)
    cfg = load_config(args.config)
    args.func(args, cfg)


if __name__ == "__main__":
    main()
