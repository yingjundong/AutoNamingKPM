"""Find the DLC files, give every recording a short ID, and pair it with its raw video.

Recording ID = "<rank>-<folders>_<video name>", e.g. "0007-2022_07_07_2022_M4102F_behavCam10". The rank is the
recording's position when the files are sorted the way Keypoint-MoSeq sorts recording names, so the model sees the
recordings in the same order as when it is given the full file paths. Short IDs keep every output path short (Windows
260-character limit) and allow any characters in folder names.

A video belongs to a DLC file when it has the same name as the DLC file up to "DLC_" (DeepLabCut's naming) and sits in
the same relative folder under video_dir. Numbers are compared without leading zeros and case is ignored, so
"07_07_2022" matches "7_7_2022". If no such folder exists, a video with that name inside a folder of the same session
name is used when that match is unique.
"""
from __future__ import annotations

import os
import re
from pathlib import Path

import pandas as pd

from .util import log, natural_key


def video_stem_of(name: str) -> str:
    m = re.match(r"(.+?)DLC[_-]", name)
    return m[1] if m else Path(name).stem


def norm(text) -> str:
    return re.sub(r"\d+", lambda m: str(int(m.group())), str(text)).casefold()


def safe(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.]+", "_", text).strip("_")


def find_dlc_files(cfg) -> list[Path]:
    root = Path(cfg.dlc_dir)
    ex = [x.casefold() for x in (cfg.exclude or [])]
    files = [p for p in root.rglob(cfg.dlc_glob)
             if p.is_file() and not any(x in part.casefold() for x in ex for part in p.relative_to(root).parts)]
    if not files:
        raise SystemExit(f"no DLC files matching '{cfg.dlc_glob}' under {cfg.dlc_dir}")
    return files


def video_index(cfg):
    by_folder, by_session = {}, {}
    exts = {e.lower() for e in cfg.video_extensions}
    root = Path(cfg.video_dir)
    for p in sorted(root.rglob("*"), key=natural_key):
        if p.suffix.lower() not in exts or not p.is_file():
            continue
        rel = p.relative_to(root)
        by_folder.setdefault((norm(rel.parent.as_posix()), norm(p.stem)), p)
        by_session.setdefault((norm(rel.parent.name), norm(p.stem)), []).append(p)
    return by_folder, by_session


def build(cfg) -> pd.DataFrame:
    root = Path(cfg.dlc_dir)
    files = find_dlc_files(cfg)
    kpm_name = {p: str(p.relative_to(root)).replace(os.sep, "-").replace("/", "-") for p in files}
    files = sorted(files, key=lambda p: kpm_name[p])
    by_folder, by_session = video_index(cfg)
    width = max(4, len(str(len(files))))
    rows, seen = [], {}
    for i, p in enumerate(files):
        rel = p.relative_to(root)
        stem = video_stem_of(p.name)
        key = (norm(rel.parent.as_posix()), norm(stem))
        if key in seen:
            raise SystemExit(f"two DLC files for the same video:\n  {seen[key]}\n  {p}\n"
                             f"narrow 'dlc_glob' / 'exclude' in the config so that each video has one DLC file")
        seen[key] = p
        parts = rel.parent.parts
        lvl = int(cfg.session_level)
        session = parts[-lvl] if len(parts) >= lvl >= 1 else (parts[-1] if parts else "")
        subject = session
        if cfg.subject_regex:
            m = re.search(cfg.subject_regex, session)
            subject = m[1] if m else session
        video = by_folder.get(key)
        if video is None:
            c = by_session.get((norm(rel.parent.name), norm(stem)), [])
            video = c[0] if len(c) == 1 else None
        row = dict(rec_id=f"{i:0{width}d}-" + safe("-".join(parts + (stem,))), dlc_file=str(p), file_mb=p.stat().st_size / 1e6,
                   rel_folder=rel.parent.as_posix(), video_stem=stem, session=session or stem, subject=subject or stem,
                   video=str(video) if video else "")
        for k in range(1, len(parts) + 1):
            row[f"level{k}"] = parts[-k]
        rows.append(row)
    rec = pd.DataFrame(rows)
    if rec.session.nunique() < 2:          # cross-validation needs groups: fall back to one group per recording
        rec["session"] = rec.rec_id
    return rec


def add_fps(cfg, rec: pd.DataFrame) -> pd.DataFrame:
    """Frame rate used for speeds and durations: camera_fps if set, else each video's own frame rate."""
    if cfg.camera_fps:
        rec["fps"] = float(cfg.camera_fps)
        return rec
    import cv2
    fps = []
    for v in rec.video:
        f = 0.0
        if v:
            cap = cv2.VideoCapture(v)
            f = float(cap.get(cv2.CAP_PROP_FPS) or 0)
            cap.release()
        fps.append(f)
    rec["fps"] = fps
    good = rec.fps[rec.fps > 0]
    if not len(good):
        raise SystemExit("could not read a frame rate from any video; set camera_fps in the config")
    rec.loc[rec.fps <= 0, "fps"] = good.median()
    return rec


def summary(rec: pd.DataFrame) -> str:
    n = len(rec)
    nv = int((rec.video != "").sum())
    lines = [f"{n} recordings, {rec.session.nunique()} sessions, {rec.subject.nunique()} subjects; "
             f"raw video found for {nv}/{n}"]
    if nv < n:
        miss = rec[rec.video == ""].dlc_file.head(5).tolist()
        lines.append("  no video for e.g.: " + "; ".join(miss))
    if "file_mb" in rec and len(rec) >= 5:
        big = rec[rec.file_mb > 8 * rec.file_mb.median()]
        if len(big):
            lines.append(f"  WARNING: {len(big)} DLC file(s) are > 8x the median size - concatenated or duplicate "
                         f"recordings? If so, add a word from their name to 'exclude'. e.g.: "
                         + "; ".join(big.dlc_file.head(3)))
    return "\n".join(lines)


def load(out_dir: Path) -> pd.DataFrame:
    p = Path(out_dir) / "recordings.csv"
    if not p.exists():
        raise SystemExit(f"{p} missing: run the 'clean' step first")
    return pd.read_csv(p, keep_default_na=False, dtype={"video": str})


def log_summary(rec):
    for line in summary(rec).splitlines():
        log(line)
