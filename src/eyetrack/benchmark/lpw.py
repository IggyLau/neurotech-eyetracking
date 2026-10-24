"""Benchmark the pupil detector on LPW - Labelled Pupils in the Wild (Tonsen et al. 2016).

LPW: 66 head-mounted dark-pupil IR eye videos (22 people x 3), ~95 fps, 640x480,
indoors/outdoors, glasses, make-up, sunlight - with a hand-labelled pupil center per frame.

To match the OV9281's 640x400 recording mode, each frame is center-cropped
from 640x480 to 640x400 (same pixel scale, so pupil sizes in px stay realistic) and
labels are shifted accordingly; frames whose label falls outside the crop are
skipped. Standard LPW metric: detection rate = share of frames with center error
<= 5 px.

Get the data: http://datasets.d2.mpi-inf.mpg.de/tonsen/LPW.zip (2.4 GB, non-commercial
research use) -> unzip into data/datasets/lpw
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from ..pupil.detector import DetectorParams, detect_pupil

CROP_H = 400


def find_videos(root: str | Path) -> list[tuple[Path, Path]]:
    root = Path(root)
    pairs = []
    for avi in sorted(root.rglob("*.avi")):
        txt = avi.with_suffix(".txt")
        if txt.exists():
            pairs.append((avi, txt))
    return pairs


def load_labels(txt: Path) -> np.ndarray:
    return np.loadtxt(txt, ndmin=2)[:, :2]


def crop(img: np.ndarray, crop_h: int = CROP_H) -> tuple[np.ndarray, int]:
    h = img.shape[0]
    off = max((h - crop_h) // 2, 0)
    return img[off:off + crop_h], off


def eval_video(args) -> pd.DataFrame:
    avi, txt, params_dict, step, crop_h = args
    params = DetectorParams(**params_dict)
    labels = load_labels(txt)
    cap = cv2.VideoCapture(str(avi))
    rows = []
    i = 0
    while True:
        ok = cap.grab()
        if not ok or i >= len(labels):
            break
        if i % step == 0:
            _, img = cap.retrieve()
            img, off = crop(img, crop_h)
            gx, gy = labels[i, 0], labels[i, 1] - off
            if 0 <= gy < img.shape[0] and gx > 0:
                r = detect_pupil(img, params)
                err = float(np.hypot(r.cx - gx, r.cy - gy)) if r.found else np.inf
                rows.append({"video": f"{avi.parent.name}/{avi.stem}", "frame": i, "found": r.found,
                             "confidence": r.confidence, "diameter": r.diameter_px, "error_px": err})
        i += 1
    cap.release()
    return pd.DataFrame(rows)


def run(root: str | Path, params: DetectorParams | None = None, step: int = 10, crop_h: int = CROP_H,
        workers: int = 8, videos: list | None = None) -> pd.DataFrame:
    params = params or DetectorParams()
    pairs = videos or find_videos(root)
    if not pairs:
        raise FileNotFoundError(f"No LPW videos under {root}. See the module docstring to download.")
    pd_dict = asdict(params)
    pd_dict["threshold_offsets"] = tuple(pd_dict["threshold_offsets"])
    jobs = [(a, t, pd_dict, step, crop_h) for a, t in pairs]
    with ProcessPoolExecutor(workers) as ex:
        parts = list(ex.map(eval_video, jobs))
    return pd.concat(parts, ignore_index=True)


def load_conditions(root: str | Path) -> pd.DataFrame:
    """Recording conditions per video from LPW's labels.ods (parsed directly; the file trips odfpy)."""
    import xml.etree.ElementTree as ET
    import zipfile
    t = "{urn:oasis:names:tc:opendocument:xmlns:table:1.0}"
    p = "{urn:oasis:names:tc:opendocument:xmlns:text:1.0}p"
    ods = next(Path(root).rglob("labels.ods"))
    xml = ET.fromstring(zipfile.ZipFile(ods).read("content.xml"))
    rows = []
    for r in xml.iter(f"{t}table-row"):
        cells = []
        for c in r.findall(f"{t}table-cell"):
            rep = min(int(c.get(f"{t}number-columns-repeated", 1)), 20)
            cells += [" ".join(x.text or "" for x in c.findall(p))] * rep
        if any(cells):
            rows.append(cells[:11])
    df = pd.DataFrame(rows[1:], columns=rows[0])
    df = df[df.Participant.str.isdigit()]
    out = pd.DataFrame({"video": df.Participant + "/" + df.Part})
    for col in ["Outdoor", "Natural light", "Artificial light", "Glasses", "Contact lenses"]:
        out[col.lower().replace(" ", "_")] = (df[col].str.upper() == "TRUE").to_numpy()
    return out


def subsets(df: pd.DataFrame, root: str | Path) -> dict[str, pd.DataFrame]:
    """all / no glasses / indoor without glasses (closest to a controlled lab recording)."""
    cond = load_conditions(root).set_index("video")
    g = df.video.map(cond.glasses).fillna(False).astype(bool)
    o = df.video.map(cond.outdoor).fillna(False).astype(bool)
    return {"all": df, "no_glasses": df[~g], "indoor_no_glasses": df[~g & ~o]}


def summarise(df: pd.DataFrame, min_confidence: float = 0.0) -> dict:
    accepted = df.found & (df.confidence >= min_confidence)
    err = df.error_px.where(accepted, np.inf)
    per_video = df.assign(ok=err <= 5).groupby("video").ok.mean()
    return {
        "frames": int(len(df)),
        "detection_rate_5px": float((err <= 5).mean()),      # standard LPW metric (all frames)
        "detection_rate_10px": float((err <= 10).mean()),
        "accepted_rate": float(accepted.mean()),
        "median_error_px": float(np.median(df.error_px[accepted])) if accepted.any() else np.nan,
        # of the frames we keep, how many are actually wrong (>10 px) - what poisons metrics downstream
        "wrong_among_accepted": float((df.error_px[accepted] > 10).mean()) if accepted.any() else np.nan,
        "worst_video_rate_5px": float(per_video.min()),
        "median_video_rate_5px": float(per_video.median()),
    }
