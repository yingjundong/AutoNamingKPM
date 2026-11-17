"""Step 4: one video per merged syllable.

From every recording, the `bouts_per_recording` longest bouts of each merged syllable are taken from the raw video
(exact bout frames, no context), the cleaned keypoints are drawn, and a caption shows syllable, name, recording and
frame. All bouts of a syllable are written one after another into syllable_<NNN>.mp4. bouts_in_videos.csv lists every
bout with its start time in the output video, so any bout can be traced back to its recording.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .util import bodyparts_of, bouts_of, log, read_dlc

PALETTE = [(0, 0, 255), (255, 0, 0), (255, 255, 0), (255, 0, 255), (0, 180, 0), (0, 180, 255), (180, 0, 180),
           (0, 255, 255), (255, 128, 0), (128, 0, 255), (0, 128, 128), (128, 128, 0)]


def select(syl: np.ndarray, n_per: int):
    a, b = bouts_of(syl)
    bouts = sorted(zip(syl[a], a, b - 1), key=lambda t: (t[0], -(t[2] - t[1]), t[1]))
    keep, count = [], {}
    for s, x, y in bouts:
        if count.get(s, 0) < n_per:
            keep.append((int(s), int(x), int(y))); count[s] = count.get(s, 0) + 1
    return sorted(keep, key=lambda t: t[1])


def run(cfg, rec: pd.DataFrame, merged: dict, names: dict, out: Path) -> pd.DataFrame:
    import cv2
    v = cfg.videos
    out.mkdir(parents=True, exist_ok=True)
    skeleton = [tuple(e) for e in (cfg.skeleton or [])]
    writers, size, fps_out, t_out, rows = {}, {}, {}, {}, []
    for i, r in enumerate(rec.itertuples()):
        if r.rec_id not in merged or not r.video:
            continue
        syl = merged[r.rec_id]
        bouts = select(syl, int(v.bouts_per_recording))
        if not bouts:
            continue
        cap = cv2.VideoCapture(r.video)
        if not cap.isOpened():
            log(f"  cannot open {r.video}"); continue
        src_fps = float(cap.get(cv2.CAP_PROP_FPS)) or 20.0
        d = read_dlc(r.clean_h5) if r.clean_h5 else None
        pts = {}
        if d is not None:
            for bp in bodyparts_of(d):
                cols = {c[2]: c for c in d.columns if c[1] == bp}
                pts[bp] = (d[cols["x"]].to_numpy(float), d[cols["y"]].to_numpy(float))
        colour = {bp: PALETTE[k % len(PALETTE)] for k, bp in enumerate(pts)}
        for s, start, end in bouts:
            if s not in writers:
                writers[s] = None
                t_out[s] = 0
            cap.set(cv2.CAP_PROP_POS_FRAMES, start)
            n_written = 0
            t0 = t_out[s]
            for fi in range(start, end + 1):
                ok, frame = cap.read()
                if not ok:
                    break
                for a, b in skeleton:
                    if a in pts and b in pts and fi < len(pts[a][0]):
                        xa, ya = pts[a][0][fi], pts[a][1][fi]; xb, yb = pts[b][0][fi], pts[b][1][fi]
                        if np.isfinite([xa, ya, xb, yb]).all():
                            cv2.line(frame, (int(round(xa)), int(round(ya))), (int(round(xb)), int(round(yb))),
                                     (230, 230, 230), 1, cv2.LINE_AA)
                for bp, (x, y) in pts.items():
                    if fi < len(x) and np.isfinite(x[fi]) and np.isfinite(y[fi]):
                        c = (int(round(x[fi])), int(round(y[fi])))
                        cv2.circle(frame, c, 2, (0, 0, 0), -1, cv2.LINE_AA)
                        cv2.circle(frame, c, 1, colour[bp], -1, cv2.LINE_AA)
                if v.caption_scale > 0:
                    cap_text = f"syllable {s}: {names.get(s, '')} | {r.rec_id} | frame {fi}"
                    o = (6, int(round(8 + 24 * v.caption_scale)))
                    cv2.putText(frame, cap_text, o, cv2.FONT_HERSHEY_SIMPLEX, v.caption_scale, (255, 255, 255), 2, cv2.LINE_AA)
                    cv2.putText(frame, cap_text, o, cv2.FONT_HERSHEY_SIMPLEX, v.caption_scale, (0, 0, 0), 1, cv2.LINE_AA)
                if writers[s] is None:
                    size[s] = (frame.shape[1], frame.shape[0])
                    fps_out[s] = float(v.output_fps or src_fps)
                    path = out / f"syllable_{s:03d}.mp4"
                    writers[s] = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps_out[s], size[s])
                    if not writers[s].isOpened():
                        raise SystemExit(f"cannot write {path}")
                if (frame.shape[1], frame.shape[0]) != size[s]:
                    frame = cv2.resize(frame, size[s])
                writers[s].write(frame)
                n_written += 1
            t_out[s] += n_written
            if n_written:
                rows.append(dict(syllable=s, name=names.get(s, ""), rec_id=r.rec_id, video=r.video, bout_start_frame=start,
                                 bout_end_frame=end, frames=n_written,
                                 time_in_output_s=round(t0 / fps_out.get(s, src_fps), 2)))
        cap.release()
        if i % 50 == 0:
            log(f"  videos: {i + 1}/{len(rec)} recordings")
    for w in writers.values():
        if w is not None:
            w.release()
    man = pd.DataFrame(rows)
    man.to_csv(out / "bouts_in_videos.csv", index=False)
    if len(man):
        summ = man.groupby("syllable").agg(name=("name", "first"), bouts=("frames", "size"), frames=("frames", "sum")).reset_index()
        summ["video"] = [str(out / f"syllable_{s:03d}.mp4") for s in summ.syllable]
    else:
        summ = pd.DataFrame(columns=["syllable", "name", "bouts", "frames", "video"])
    summ.to_csv(out / "videos.csv", index=False)
    return summ
