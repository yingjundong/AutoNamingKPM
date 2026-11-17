"""Step 1: DLC cleaning ("strict occlusion" settings).

Per file and body part: points with likelihood < `likelihood`, (0, 0) points, frame-to-frame jumps above
median + `velocity_mad` x 1.4826 x MAD, and skeleton edges longer than min(median + `shape_mad` robust SD,
`shape_max_factor` x median) are removed; gaps up to `interpolation_limit` frames are linearly interpolated; x / y
are smoothed with a centred `moving_average`-frame mean (frames still missing after interpolation stay missing);
the skeleton-edge filter is applied once more to the smoothed points. Likelihood values are kept unchanged.
Output: one DLC-style .h5 (key df_with_missing) per recording.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .util import bodyparts_of, log, read_dlc


def _col(df, bp, coord):
    m = [c for c in df.columns if c[1] == bp and c[2] == coord]
    return m[0] if m else None


def _robust_upper(values, scale):
    values = values[np.isfinite(values)]
    if len(values) < 10:
        return None
    med = float(np.nanmedian(values))
    mad = float(np.nanmedian(np.abs(values - med)))
    if mad == 0:
        q75, q25 = np.nanpercentile(values, [75, 25])
        mad = float((q75 - q25) / 1.349) if q75 > q25 else 1.0
    return med + scale * 1.4826 * mad


def _invalidate(df, bp, bad):
    xc, yc = _col(df, bp, "x"), _col(df, bp, "y")
    if xc is None or yc is None:
        return 0
    bad = bad & (df[xc].astype(float).notna().to_numpy() | df[yc].astype(float).notna().to_numpy())
    df.loc[bad, xc] = np.nan
    df.loc[bad, yc] = np.nan
    return int(bad.sum())


def _shape_filter(df, skeleton, scale, max_factor):
    n = 0
    for a, b in skeleton:
        cols = [_col(df, a, "x"), _col(df, a, "y"), _col(df, b, "x"), _col(df, b, "y")]
        if None in cols:
            continue
        xa, ya, xb, yb = (df[c].to_numpy(float) for c in cols)
        d = np.sqrt((xa - xb) ** 2 + (ya - yb) ** 2)
        fin = d[np.isfinite(d)]
        if len(fin) < 10:
            continue
        cut = [c for c in (_robust_upper(fin, scale), float(np.nanmedian(fin)) * max_factor if max_factor else None)
               if c is not None]
        if not cut:
            continue
        bad = d > min(cut)
        n += _invalidate(df, a, bad) + _invalidate(df, b, bad)
    return n


def clean_table(df: pd.DataFrame, c, skeleton) -> tuple[pd.DataFrame, dict]:
    out = df.copy()
    bps = bodyparts_of(df)
    info = dict(frames=len(df), likelihood_removed=0, zero_removed=0, jump_removed=0, shape_removed=0)
    for bp in bps:
        lc = _col(out, bp, "likelihood")
        if lc is not None and c.likelihood is not None:
            info["likelihood_removed"] += _invalidate(out, bp, out[lc].to_numpy(float) < c.likelihood)
    if c.remove_zero_xy:
        for bp in bps:
            xc, yc = _col(out, bp, "x"), _col(out, bp, "y")
            if xc is not None and yc is not None:
                info["zero_removed"] += _invalidate(out, bp, (out[xc].to_numpy(float) == 0) & (out[yc].to_numpy(float) == 0))
    for bp in bps:
        xc, yc = _col(out, bp, "x"), _col(out, bp, "y")
        if xc is None or yc is None:
            continue
        x, y = out[xc].to_numpy(float), out[yc].to_numpy(float)
        speed = np.sqrt(np.diff(x, prepend=np.nan) ** 2 + np.diff(y, prepend=np.nan) ** 2)
        cut = _robust_upper(speed, c.velocity_mad)
        if cut is not None:
            info["jump_removed"] += _invalidate(out, bp, speed > cut)
    if skeleton:
        info["shape_removed"] += _shape_filter(out, skeleton, c.shape_mad, c.shape_max_factor)
    for bp in bps:
        for coord in ("x", "y"):
            col = _col(out, bp, coord)
            if col is None:
                continue
            s = out[col].astype(float)
            if c.interpolation_limit > 0:
                s = s.interpolate(method="linear", limit=int(c.interpolation_limit), limit_direction="both")
            ok = s.notna()
            if c.moving_average > 1:
                s = s.rolling(window=int(c.moving_average), center=True, min_periods=1).mean().where(ok)
            out[col] = s
    if skeleton:
        info["shape_removed_after_smoothing"] = _shape_filter(out, skeleton, c.shape_mad, c.shape_max_factor)
    return out, info


def run(cfg, rec: pd.DataFrame, out_dir: Path) -> pd.DataFrame:
    """Fills rec['clean_h5']. With cleaning disabled, .h5 inputs are used in place and CSV inputs converted."""
    c = cfg.cleaning
    skeleton = [tuple(e) for e in (cfg.skeleton or [])]
    if c.enabled and not skeleton:
        log("note: no 'skeleton' in the config, so the skeleton-length filter is skipped")
    rows, paths = [], []
    for i, r in enumerate(rec.itertuples()):
        src = Path(r.dlc_file)
        dst = out_dir / r.rel_folder / f"{r.video_stem}_clean.h5"
        if i == 0 or (i + 1) % 100 == 0:
            log(f"  {'cleaning' if c.enabled else 'converting'} {i + 1}/{len(rec)}: {src.name}")
        try:
            if not c.enabled and src.suffix.lower() in (".h5", ".hdf5"):
                read_dlc(src)                                   # check the format
                paths.append(str(src)); rows.append(dict(rec_id=r.rec_id, status="used as is")); continue
            df = read_dlc(src)
            info = {}
            if c.enabled:
                df, info = clean_table(df, c, skeleton)
            dst.parent.mkdir(parents=True, exist_ok=True)
            df.to_hdf(dst, key="df_with_missing", mode="w", format="fixed")
            paths.append(str(dst)); rows.append(dict(rec_id=r.rec_id, status="ok", **info))
        except Exception as e:                                   # one bad file must not stop the run
            paths.append(""); rows.append(dict(rec_id=r.rec_id, status=f"error: {e}"))
    rec = rec.copy()
    rec["clean_h5"] = paths
    man = pd.DataFrame(rows)
    out_dir.mkdir(parents=True, exist_ok=True)
    man.to_csv(out_dir / "cleaning_manifest.csv", index=False)
    bad = man[man.status.str.startswith("error")]
    if len(bad):
        log(f"WARNING: {len(bad)} DLC files failed and are left out (see {out_dir / 'cleaning_manifest.csv'})")
    return rec[rec.clean_h5 != ""].reset_index(drop=True)
