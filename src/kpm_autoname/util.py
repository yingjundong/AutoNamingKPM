from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd


def log(msg: str) -> None:
    print(f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {msg}", flush=True)


def natural_key(p) -> list:
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", str(p))]


def read_dlc(path) -> pd.DataFrame:
    """DeepLabCut table (CSV or .h5) with columns (scorer, bodypart, coord). Multi-animal files are not supported."""
    path = Path(path)
    if path.suffix.lower() in (".h5", ".hdf5"):
        df = pd.read_hdf(path)
    else:
        df = pd.read_csv(path, header=[0, 1, 2], index_col=0)
    if df.columns.nlevels != 3:
        raise SystemExit(f"{path}: expected single-animal DLC columns (scorer, bodyparts, coords); "
                         f"multi-animal DLC output is not supported")
    return df


def bodyparts_of(df: pd.DataFrame) -> list[str]:
    return list(dict.fromkeys(df.columns.get_level_values(1)))


def xyl(df: pd.DataFrame, bp: str):
    """x, y, likelihood arrays of one body part."""
    cols = {c[2]: c for c in df.columns if c[1] == bp}
    return tuple(df[cols[k]].to_numpy(float) for k in ("x", "y", "likelihood"))


def bouts_of(s: np.ndarray):
    """start, end (exclusive) of every contiguous run of one label."""
    s = np.asarray(s)
    starts = np.flatnonzero(np.r_[True, s[1:] != s[:-1]])
    return starts, np.r_[starts[1:], len(s)]
