"""Per-frame and per-bout movement features of every recording.

Per frame:
  speed            KPM centroid speed, cm/s (centroid smoothed with a Gaussian, sigma 2 frames)
  dheading         change of KPM heading, degrees; + = clockwise in image coordinates (y down) = RIGHT turn when the
                   camera is above the arena and the image is not mirrored
  turn_rate        smoothed dheading x fps, deg/s
  body_length_rel  nose-tail_base distance / that recording's median while running (speed above its 75th
                   percentile); < 1 = shortened body (rearing, hunching, grooming)
  body_bend        |angle| between head->mid_body and mid_body->tail_base, degrees (needs both set in the config)
  nose_lost        nose likelihood below naming.likelihood
  area, d_wall, d_corner   from naming.location_labels, if given
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d

from .util import bouts_of, log, read_dlc, xyl


def recording_frames(r, kpm: dict, cfg) -> pd.DataFrame:
    n = len(kpm["syllable"])
    fps = float(r.fps)
    f = pd.DataFrame({"rec_id": r.rec_id, "session": r.session, "subject": r.subject, "frame": np.arange(n),
                      "syllable": kpm["syllable"].astype(int), "fps": fps})
    cx = gaussian_filter1d(kpm["centroid"][:, 0].astype(float), 2)
    cy = gaussian_filter1d(kpm["centroid"][:, 1].astype(float), 2)
    f["speed"] = np.hypot(np.gradient(cx), np.gradient(cy)) * fps * cfg.cm_per_px
    dh = np.angle(np.exp(1j * np.diff(kpm["heading"].astype(float), prepend=np.nan)))
    f["dheading"] = np.degrees(dh)
    f["turn_rate"] = gaussian_filter1d(np.nan_to_num(f.dheading.to_numpy()), 2) * fps
    f["body_length_rel"] = f["body_bend"] = f["nose_lost"] = np.nan
    if r.clean_h5 and Path(r.clean_h5).exists():
        d = read_dlc(r.clean_h5).iloc[:n]
        lik = cfg.naming.likelihood
        bp = cfg.bodyparts

        def pt(name):
            x, y, l = xyl(d, name)
            ok = l >= lik
            return np.where(ok, x, np.nan), np.where(ok, y, np.nan)
        nx, ny = pt(bp["nose"]); tx, ty = pt(bp["tail_base"])
        body = np.hypot(nx - tx, ny - ty)
        if len(body) < n:
            body = np.r_[body, np.full(n - len(body), np.nan)]
        run = f.speed.to_numpy() > np.nanpercentile(f.speed, 75)
        ref = np.nanmedian(body[run]) if np.isfinite(body[run]).any() else np.nanmedian(body)
        f["body_length_rel"] = body / ref
        if bp.get("head") and bp.get("mid_body"):
            hx, hy = pt(bp["head"]); mx, my = pt(bp["mid_body"])
            a1 = np.arctan2(my - hy, mx - hx); a2 = np.arctan2(ty - my, tx - mx)
            f["body_bend"] = np.r_[np.degrees(np.abs(np.angle(np.exp(1j * (a2 - a1))))), np.full(n - len(a1), np.nan)]
        nl = (xyl(d, bp["nose"])[2] < lik).astype(float)
        f["nose_lost"] = np.r_[nl, np.full(n - len(nl), np.nan)]
    return f


def all_frames(cfg, rec: pd.DataFrame, results: dict) -> pd.DataFrame:
    frames = []
    for i, r in enumerate(rec.itertuples()):
        if r.rec_id not in results:
            continue
        frames.append(recording_frames(r, results[r.rec_id], cfg))
        if i % 100 == 0:
            log(f"  features {i + 1}/{len(rec)} recordings")
    if not frames:
        raise SystemExit("no recording of the KPM results matches recordings.csv")
    frames = pd.concat(frames, ignore_index=True)
    return add_location(cfg, rec, frames)


def add_location(cfg, rec, frames):
    """Optional frame-level location labels (see README: 'Location labels')."""
    frames = frames.assign(area=np.nan, d_wall=np.nan, d_corner=np.nan)
    ll = cfg.naming.location_labels
    if not ll or not ll.get("file"):
        return frames
    from .recordings import norm
    cols = {"frame": "frame", "area": "area", "d_wall_px": "d_wall_px", "d_corner_px": "d_corner_px", **(ll.get("columns") or {})}
    match = ll.get("match") or {"recording": "rec_id"}
    vals = {"wall": "wall", "corner": "corner", "centre": "centre", **(ll.get("values") or {})}
    use = list(dict.fromkeys(list(match) + [c for c in cols.values() if c]))
    log(f"  reading location labels {ll['file']}")
    lab = pd.read_csv(ll["file"], usecols=lambda c: c in use)
    key = lambda df, names: df[names].astype(str).apply(lambda s: s.map(norm)).agg("|".join, axis=1)
    lab["_k"] = key(lab, list(match))
    rk = rec.set_index("rec_id")[list(match.values())].astype(str)
    rec_key = rk.apply(lambda s: s.map(norm)).agg("|".join, axis=1)
    lab = lab[lab._k.isin(set(rec_key))]
    lab = lab.drop_duplicates(["_k", cols["frame"]])
    lab["rec_id"] = lab._k.map({v: k for k, v in rec_key.items()})
    area = lab[cols["area"]].map({vals["wall"]: "wall", vals["corner"]: "corner", vals["centre"]: "centre"})
    sub = pd.DataFrame({"rec_id": lab.rec_id, "frame": lab[cols["frame"]].astype(int), "_area": area,
                        "_dw": lab[cols["d_wall_px"]] * cfg.cm_per_px if cols.get("d_wall_px") in lab else np.nan,
                        "_dc": lab[cols["d_corner_px"]] * cfg.cm_per_px if cols.get("d_corner_px") in lab else np.nan})
    m = frames.merge(sub, on=["rec_id", "frame"], how="left")
    m["area"], m["d_wall"], m["d_corner"] = m.pop("_area"), m.pop("_dw"), m.pop("_dc")
    log(f"  location labels found for {m.area.notna().mean():.0%} of frames")
    return m


def bout_table(frames: pd.DataFrame) -> pd.DataFrame:
    """One row per contiguous run of one syllable within a recording (uses frames.syllable)."""
    out = []
    for rid, g in frames.groupby("rec_id", sort=False):
        s = g.syllable.to_numpy()
        a, b = bouts_of(s)
        dh = g.dheading.to_numpy()
        net = np.array([np.nansum(dh[x + 1:y]) for x, y in zip(a, b)])
        out.append(pd.DataFrame({"rec_id": rid, "syllable": s[a], "subject": g.subject.iloc[0], "start": a, "end": b,
                                 "n_frames": b - a, "duration_s": (b - a) / g.fps.iloc[0], "net_turn_deg": net}))
    return pd.concat(out, ignore_index=True)
