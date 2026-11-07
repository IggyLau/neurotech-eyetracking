"""`eyetrack ui`: one window for choosing the camera mode and running preview, record, calibrate and process.

Each action runs the matching `eyetrack` command in a child process, with a config that has the chosen camera
indices, so the window stays responsive and its log shows the command's output. The choices are remembered in
<recording.output_dir>/.ui_state.json (data/ is not committed).

On macOS, start it from Terminal.app so the child processes may use the cameras.
"""

from __future__ import annotations

import base64
import json
import os
import queue
import subprocess
import sys
import threading
from pathlib import Path

import yaml

from .capture.recorder import MODE_NAMES, MODES
from .config import REPO_ROOT, load_config

ROLES = ("eye", "eye2", "scene")
ROLE_NAMES = {"eye": "Eye camera (left / only eye)", "eye2": "Second eye camera (right)", "scene": "Scene camera"}


# ------------------------------------------------------------------ helpers (no Tk; tested)

def build_config(cfg: dict, indices: dict[str, int]) -> dict:
    """cfg with the chosen camera index for each role."""
    out = json.loads(json.dumps(cfg))       # deep copy of plain YAML data
    for role, idx in indices.items():
        if f"{role}_camera" in out and idx is not None:
            out[f"{role}_camera"]["index"] = int(idx)
    return out


def command(action: str, mode: str, config_path: str | Path, **opt) -> list[str]:
    """The eyetrack command line for one UI action."""
    base = [sys.executable, "-u", "-m", "eyetrack.cli", "--config", str(config_path)]
    if action == "preview":
        return base + ["preview", "--mode", mode]
    if action == "record":
        args = base + ["record", "--mode", mode]
        if opt.get("name"):
            args += ["--name", opt["name"]]
        if opt.get("duration"):
            args += ["--duration", str(opt["duration"])]
        return args
    if action in ("calibrate", "check"):
        if "scene" not in MODES[mode]:
            raise ValueError("Calibration needs the scene camera: pick a mode with the scene camera.")
        args = base + ["calibrate", "--mode", "3" if "eye2" in MODES[mode] else "2"]
        if opt.get("board"):
            args += ["--board"]
            if opt.get("distance_cm"):
                args += ["--distance-cm", str(opt["distance_cm"])]
        if action == "check":
            if not opt.get("calibration"):
                raise ValueError("Choose the calibration.json to check.")
            args += ["--check-with", str(opt["calibration"])]
        return args
    if action == "board":
        args = base + ["board"]
        if opt.get("distance_cm"):
            args += ["--distance-cm", str(opt["distance_cm"])]
        return args
    if action == "process":
        if not opt.get("session"):
            raise ValueError("Choose a session folder to process.")
        args = base + ["process", str(opt["session"])]
        if opt.get("calibration"):
            args += ["--calibration", str(opt["calibration"])]
        if opt.get("overlay"):
            args += ["--overlay"]
        return args
    raise ValueError(action)


def sessions(root: str | Path) -> list[Path]:
    """Session folders, newest first (folder names start with a timestamp)."""
    root = Path(root)
    if not root.exists():
        return []
    return sorted((p for p in root.iterdir() if p.is_dir() and p.name[:8].isdigit()), reverse=True)


def latest_calibration(root: str | Path) -> Path | None:
    for s in sessions(root):
        if (s / "calibration.json").exists():
            return s / "calibration.json"
    return None


def scan_cameras(max_index: int = 6) -> list[tuple[int, bytes | None, str]]:
    """(index, PNG thumbnail or None, size text) for each camera index that opens."""
    import cv2
    from .capture.camera import default_backend
    found = []
    for i in range(max_index):
        cap = cv2.VideoCapture(i, default_backend())
        if not cap.isOpened():
            continue
        img = None
        for _ in range(15):
            ok, f = cap.read()
            if ok and f is not None and f.max() > 0:
                img = f
                break
            if ok:
                img = f
        size = f"{int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}"
        cap.release()
        png = None
        if img is not None:
            thumb = cv2.resize(img, (200, int(200 * img.shape[0] / img.shape[1])))
            png = cv2.imencode(".png", thumb)[1].tobytes()
        found.append((i, png, size))
    return found


# ------------------------------------------------------------------ window

def run_ui(config_path: str | None = None):
    import tkinter as tk
    from tkinter import filedialog, messagebox, ttk

    base_cfg = load_config(config_path)
    out_root = Path(base_cfg["recording"]["output_dir"])
    if not out_root.is_absolute():
        out_root = REPO_ROOT / out_root
    out_root.mkdir(parents=True, exist_ok=True)
    state_path = out_root / ".ui_state.json"
    try:
        state = json.loads(state_path.read_text())
    except (OSError, ValueError):
        state = {}

    root = tk.Tk()
    root.title("eyetrack")
    root.geometry("1100x760")
    log_q: queue.Queue = queue.Queue()
    running: dict = {"proc": None}

    mode = tk.StringVar(value=state.get("mode", "2"))
    idx_vars = {r: tk.IntVar(value=state.get("indices", {}).get(r, base_cfg[f"{r}_camera"]["index"])) for r in ROLES}
    rec_name = tk.StringVar(value=state.get("rec_name", ""))
    rec_dur = tk.StringVar(value=state.get("rec_dur", ""))
    target = tk.StringVar(value=state.get("target", "board"))
    dist = tk.StringVar(value=str(state.get("distance_cm", base_cfg["calibration"]["board"]["distance_cm"])))
    session_var = tk.StringVar(value=state.get("session", str(next(iter(sessions(out_root)), ""))))
    cal_var = tk.StringVar(value=state.get("calibration", str(latest_calibration(out_root) or "")))
    overlay_var = tk.BooleanVar(value=state.get("overlay", False))

    def save_state():
        state.update(mode=mode.get(), indices={r: v.get() for r, v in idx_vars.items()}, rec_name=rec_name.get(),
                     rec_dur=rec_dur.get(), target=target.get(), distance_cm=dist.get(), session=session_var.get(),
                     calibration=cal_var.get(), overlay=overlay_var.get())
        try:
            state_path.write_text(json.dumps(state, indent=2))
        except OSError:
            pass

    def config_file() -> Path:
        cfg = build_config(base_cfg, {r: v.get() for r, v in idx_vars.items()})
        p = out_root / ".ui_config.yaml"
        p.write_text(yaml.safe_dump(cfg, sort_keys=False))
        return p

    def log(text: str):
        log_box.configure(state="normal")
        log_box.insert("end", text)
        log_box.see("end")
        log_box.configure(state="disabled")

    def run(action: str, done=None, **opt):
        if running["proc"] is not None:
            messagebox.showinfo("eyetrack", "Something is already running. Stop it first.")
            return
        save_state()
        try:
            args = command(action, mode.get(), config_file(), **opt)
        except ValueError as e:
            messagebox.showerror("eyetrack", str(e))
            return
        log(f"\n$ eyetrack {' '.join(args[args.index('--config') + 2:])}\n")
        proc = subprocess.Popen(args, cwd=REPO_ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                                bufsize=1)
        running["proc"] = proc
        stop_btn.configure(state="normal")

        def reader():
            for line in proc.stdout:
                log_q.put(line)
            code = proc.wait()
            log_q.put(("done", action, code, done))
        threading.Thread(target=reader, daemon=True).start()

    def poll():
        try:
            while True:
                item = log_q.get_nowait()
                if isinstance(item, tuple):
                    _, action, code, done = item
                    running["proc"] = None
                    stop_btn.configure(state="disabled")
                    log(f"[{action} finished, exit code {code}]\n")
                    refresh_lists()
                    if done and code == 0:
                        done()
                else:
                    log(item)
        except queue.Empty:
            pass
        root.after(100, poll)

    def stop():
        if running["proc"] is not None:
            running["proc"].terminate()

    def show_image(png: bytes | None, caption: str):
        if png is None:
            image_label.configure(image="", text=caption)
            return
        img = tk.PhotoImage(data=base64.b64encode(png).decode())
        scale = max(1, (img.width() + 559) // 560)
        img = img.subsample(scale, scale)
        image_label.image = img
        image_label.configure(image=img, text=caption, compound="top")

    def show_file(path: Path, caption: str):
        if path.exists():
            show_image(path.read_bytes(), caption)

    def update_roles(*_):
        roles = MODES[mode.get()]
        for r, (spin, lab) in role_widgets.items():
            st = "normal" if r in roles else "disabled"
            spin.configure(state=st)
            lab.configure(state=st)
        has_scene = "scene" in roles
        for b in scene_buttons:
            b.configure(state="normal" if has_scene else "disabled")

    def do_scan():
        if running["proc"] is not None:
            messagebox.showinfo("eyetrack", "Stop the running command first; it holds the cameras.")
            return
        for w in thumbs.winfo_children():
            w.destroy()
        log("Scanning camera indices 0-5...\n")
        root.update()
        for i, png, size in scan_cameras():
            f = ttk.Frame(thumbs)
            f.pack(side="left", padx=4)
            if png:
                img = tk.PhotoImage(data=base64.b64encode(png).decode())
                lab = ttk.Label(f, image=img)
                lab.image = img
                lab.pack()
            ttk.Label(f, text=f"index {i}  {size}" + ("" if png else "  (no frames)")).pack()
        log("Done. Set each camera's index to the matching picture.\n")

    def refresh_lists():
        names = [str(s) for s in sessions(out_root)]
        session_box.configure(values=names)
        cals = [str(s / "calibration.json") for s in sessions(out_root) if (s / "calibration.json").exists()]
        cal_box.configure(values=[""] + cals)

    def after_process():
        p = Path(session_var.get()) / "overlay_snapshot.png"
        if p.exists():
            show_file(p, "Scene camera with gaze (overlay_snapshot.png)")
        else:
            show_file(Path(session_var.get()) / "report.png", "report.png (no calibration: no scene gaze)")

    def after_calibrate():
        s = sessions(out_root)
        if s and (s[0] / "calibration.json").exists():
            cal_var.set(str(s[0] / "calibration.json"))
            save_state()

    def after_record():
        s = sessions(out_root)
        if s:
            session_var.set(str(s[0]))

    def pick_session():
        d = filedialog.askdirectory(initialdir=out_root)
        if d:
            session_var.set(d)

    def pick_cal():
        f = filedialog.askopenfilename(initialdir=out_root, filetypes=[("calibration", "calibration.json")])
        if f:
            cal_var.set(f)

    def open_folder():
        p = Path(session_var.get())
        if p.exists():
            subprocess.Popen(["open" if sys.platform == "darwin" else "xdg-open", str(p)])

    def make_board():
        def opened():
            pdf = out_root / "calibration_board.pdf"
            if pdf.exists() and sys.platform == "darwin":
                subprocess.Popen(["open", str(pdf)])
        run("board", done=opened, distance_cm=dist.get())

    # --- layout
    left = ttk.Frame(root, padding=10)
    left.pack(side="left", fill="y")
    right = ttk.Frame(root, padding=10)
    right.pack(side="left", fill="both", expand=True)

    box = ttk.LabelFrame(left, text="1. Camera mode", padding=8)
    box.pack(fill="x")
    for key, name in MODE_NAMES.items():
        ttk.Radiobutton(box, text=name, value=key, variable=mode, command=update_roles).pack(anchor="w")

    box = ttk.LabelFrame(left, text="2. Camera indices", padding=8)
    box.pack(fill="x", pady=6)
    role_widgets = {}
    for r in ROLES:
        row = ttk.Frame(box)
        row.pack(fill="x", pady=1)
        lab = ttk.Label(row, text=ROLE_NAMES[r], width=28)
        lab.pack(side="left")
        spin = ttk.Spinbox(row, from_=0, to=9, width=4, textvariable=idx_vars[r])
        spin.pack(side="left")
        role_widgets[r] = (spin, lab)
    ttk.Button(box, text="Scan cameras (show a picture from each index)", command=do_scan).pack(fill="x", pady=4)
    ttk.Button(box, text="Live preview of this mode", command=lambda: run("preview")).pack(fill="x")

    box = ttk.LabelFrame(left, text="3. Record", padding=8)
    box.pack(fill="x", pady=6)
    row = ttk.Frame(box)
    row.pack(fill="x")
    ttk.Label(row, text="Name").pack(side="left")
    ttk.Entry(row, textvariable=rec_name, width=14).pack(side="left", padx=4)
    ttk.Label(row, text="Seconds").pack(side="left")
    ttk.Entry(row, textvariable=rec_dur, width=6).pack(side="left", padx=4)
    ttk.Button(box, text="Record (q in the preview window stops)",
               command=lambda: run("record", done=after_record, name=rec_name.get().strip(),
                                   duration=rec_dur.get().strip())).pack(fill="x", pady=4)

    box = ttk.LabelFrame(left, text="4. Calibrate (needs the scene camera)", padding=8)
    box.pack(fill="x", pady=6)
    ttk.Radiobutton(box, text="Printed board on the wall", value="board", variable=target).pack(anchor="w")
    ttk.Radiobutton(box, text="Targets on this laptop screen", value="screen", variable=target).pack(anchor="w")
    row = ttk.Frame(box)
    row.pack(fill="x", pady=2)
    ttk.Label(row, text="Board distance (cm)").pack(side="left")
    ttk.Entry(row, textvariable=dist, width=6).pack(side="left", padx=4)
    scene_buttons = [
        ttk.Button(box, text="Make printable board (PDF)", command=make_board),
        ttk.Button(box, text="Calibrate",
                   command=lambda: run("calibrate", done=after_calibrate, board=target.get() == "board",
                                       distance_cm=dist.get())),
        ttk.Button(box, text="Check selected calibration at this distance",
                   command=lambda: run("check", board=target.get() == "board", distance_cm=dist.get(),
                                       calibration=cal_var.get())),
    ]
    for b in scene_buttons:
        b.pack(fill="x", pady=1)

    box = ttk.LabelFrame(left, text="5. Process", padding=8)
    box.pack(fill="x", pady=6)
    ttk.Label(box, text="Session").pack(anchor="w")
    row = ttk.Frame(box)
    row.pack(fill="x")
    session_box = ttk.Combobox(row, textvariable=session_var, width=34)
    session_box.pack(side="left")
    ttk.Button(row, text="...", width=3, command=pick_session).pack(side="left")
    ttk.Label(box, text="Calibration (blank = none)").pack(anchor="w")
    row = ttk.Frame(box)
    row.pack(fill="x")
    cal_box = ttk.Combobox(row, textvariable=cal_var, width=34)
    cal_box.pack(side="left")
    ttk.Button(row, text="...", width=3, command=pick_cal).pack(side="left")
    ttk.Checkbutton(box, text="Also write overlay.mp4 (slow)", variable=overlay_var).pack(anchor="w")
    row = ttk.Frame(box)
    row.pack(fill="x", pady=4)
    ttk.Button(row, text="Process", command=lambda: run("process", done=after_process, session=session_var.get(),
                                                         calibration=cal_var.get().strip(),
                                                         overlay=overlay_var.get())).pack(side="left", expand=True,
                                                                                           fill="x")
    ttk.Button(row, text="Show result", command=after_process).pack(side="left")
    ttk.Button(row, text="Open folder", command=open_folder).pack(side="left")

    thumbs = ttk.Frame(right)
    thumbs.pack(fill="x")
    image_label = ttk.Label(right, text="Results appear here.", anchor="center")
    image_label.pack(fill="both", expand=True)
    row = ttk.Frame(right)
    row.pack(fill="x")
    ttk.Label(row, text="Log").pack(side="left")
    stop_btn = ttk.Button(row, text="Stop running command", command=stop, state="disabled")
    stop_btn.pack(side="right")
    log_box = tk.Text(right, height=14, state="disabled", font=("Menlo", 10))
    log_box.pack(fill="both")

    update_roles()
    refresh_lists()
    root.after(100, poll)

    def on_close():
        save_state()
        stop()
        root.destroy()
    root.protocol("WM_DELETE_WINDOW", on_close)
    if os.environ.get("EYETRACK_UI_SELFTEST"):     # smoke check: build the window, then close it
        root.after(int(os.environ["EYETRACK_UI_SELFTEST"]), on_close)
    root.mainloop()
