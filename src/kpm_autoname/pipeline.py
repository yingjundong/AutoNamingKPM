"""Steps: clean -> kpm -> merge (+ name) -> videos -> report. Each writes into <output_dir> and marks itself done in
<output_dir>/.state; finished steps are skipped when the run is started again."""
from __future__ import annotations

import json
import shutil
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from . import __version__
from . import recordings as recmod
from .config import crop_px
from .util import log

STEPS = ["clean", "kpm", "merge", "videos", "report"]


class Run:
    def __init__(self, cfg):
        self.cfg = cfg
        self.dir = Path(cfg.output_dir)
        self.clean = self.dir / "01_dlc_clean"
        self.kpm = self.dir / "02_kpm"
        self.merge = self.dir / "03_merge_and_names"
        self.videos = self.dir / "04_videos"
        self.state = self.dir / ".state"

    def done(self, s):
        return (self.state / f"{s}.done").exists()

    def mark(self, s, info):
        self.state.mkdir(parents=True, exist_ok=True)
        (self.state / f"{s}.done").write_text(json.dumps({"finished": f"{datetime.now():%Y-%m-%d %H:%M:%S}", **info},
                                                         indent=2, default=str))

    def info(self, s):
        p = self.state / f"{s}.done"
        return json.loads(p.read_text()) if p.exists() else {}


# ------------------------------------------------------------------ 1
def step_clean(r: Run):
    cfg = r.cfg
    rec = recmod.build(cfg)
    rec = recmod.add_fps(cfg, rec)
    recmod.log_summary(rec)
    from . import clean
    rec = clean.run(cfg, rec, r.clean)
    rec.to_csv(r.dir / "recordings.csv", index=False)
    return {"recordings": len(rec), "with_video": int((rec.video != "").sum()), "cleaned": bool(cfg.cleaning.enabled)}


# ------------------------------------------------------------------ 2
def step_kpm(r: Run):
    from . import kpm_fit
    rec = recmod.load(r.dir)
    md = kpm_fit.fit(r.cfg, rec, r.kpm)
    n_rec, n_frames, n_syl = kpm_fit.syllable_count(md)
    return {"model": str(md), "recordings": n_rec, "frames": n_frames, "syllables_used": n_syl}


# ------------------------------------------------------------------ 3
def step_merge(r: Run):
    from . import features as feat, kpm_fit, merge, naming
    cfg, out = r.cfg, r.merge
    m = cfg.merge
    cache = out / "cache"
    cache.mkdir(parents=True, exist_ok=True)
    rec = recmod.load(r.dir)
    model = kpm_fit.model_dir(r.kpm)
    results = kpm_fit.load_results(model)
    log("per-frame features")
    frames = feat.all_frames(cfg, rec, results)
    log("sampling bouts")
    bt = merge.sample(frames, rec, results, cache, int(m.bouts_per_syllable), int(m.frames_per_bout))
    if m.features != "kinematic":
        log(f"embedding egocentric crops ({crop_px(cfg)} px) with DINOv2")
        merge.run_embed(cfg, cache, crop_px(cfg))
    log(f"pairwise separability ({m.features}); pairs with AUC < {m.max_auc} are merged")
    fps = bt.groupby("syllable").n_frames.sum()
    labels, auc, rare_tab, common = merge.run_merge(cache, fps, m.rare_pct, m.features, m.max_auc, m.rare_min_prob)
    pd.DataFrame([dict(a=k[0], b=k[1], auc=round(v, 3)) for k, v in auc.items()],
                 columns=["a", "b", "auc"]).sort_values("auc").to_csv(out / "pair_auc.csv", index=False)
    rare_tab.to_csv(out / "rare_assignment.csv", index=False)
    groups = merge.groups_from(labels)
    log(f"common syllables {common}; merge groups {groups}")
    mp = merge.build_merged(model, groups, out)

    log("naming merged syllables")
    old = frames.syllable.to_numpy()
    lut = dict(zip(mp.old_syllable, mp.new_syllable))
    naming.describe_all(frames, pd.Series(old).map(lut).to_numpy(), out, m.rare_pct, "merged syllables")
    log("describing the original syllables (merge check)")
    feats_dir = out / "unmerged_reference"
    naming.describe_all(frames, old, feats_dir, m.rare_pct, "original KPM syllables")
    naming.consistency(out, mp, m.rare_pct)

    if m.compare_to:
        manual = pd.read_csv(m.compare_to)
        f = pd.read_csv(feats_dir / "syllable_features.csv")
        rows = [dict(method="hand-made merge", **merge.score(dict(zip(manual.old_syllable, manual.new_syllable)), manual, f, m.rare_pct)),
                dict(method="no merge", **merge.score({int(s): int(s) for s in manual.old_syllable}, manual, f, m.rare_pct)),
                dict(method=f"this run ({m.features}, AUC < {m.max_auc})", **merge.score(labels, manual, f, m.rare_pct))]
        pd.DataFrame(rows).to_csv(out / "comparison_with_hand_merge.csv", index=False)
        log("agreement with the hand-made merge:\n" + pd.DataFrame(rows).to_string(index=False))
    return {"syllables_before": int((mp.old_frames > 0).sum()), "syllables_after": int(mp[mp.old_frames > 0].new_syllable.nunique()),
            "groups": groups}


# ------------------------------------------------------------------ 4
def step_videos(r: Run):
    from . import videos
    import h5py
    rec = recmod.load(r.dir)
    merged = {}
    with h5py.File(r.merge / "merged_results.h5", "r") as f:
        for k in f:
            merged[k] = f[k]["syllable"][()].astype(int)
    names = _names(r)
    if r.videos.exists():
        shutil.rmtree(r.videos)
    s = videos.run(r.cfg, rec, merged, names, r.videos)
    return {"videos": len(s), "bouts": int(s.bouts.sum()) if len(s) else 0}


def _names(r: Run) -> dict:
    from .naming import pooled_of
    desc = pd.read_csv(r.merge / "syllable_descriptions.csv")
    pooled = pooled_of(desc)
    mp = pd.read_csv(r.merge / "merge_mapping.csv")
    names = dict(zip(desc.syllable, desc.movement_name))
    return {int(s): ("rare" if s in pooled else names.get(s, "")) for s in mp.new_syllable.unique()}


# ------------------------------------------------------------------ 5
def step_report(r: Run):
    from .report import write
    return write(r, STEPS)


def run(cfg, redo_from: str | None = None, stop_after: str | None = None):
    r = Run(cfg)
    r.dir.mkdir(parents=True, exist_ok=True)
    (r.dir / "run_settings.json").write_text(json.dumps({"package_version": __version__, "started": f"{datetime.now():%Y-%m-%d %H:%M:%S}",
                                                         **cfg}, indent=2, default=str), encoding="utf-8")
    funcs = dict(clean=step_clean, kpm=step_kpm, merge=step_merge, videos=step_videos, report=step_report)
    redo = STEPS.index(redo_from) if redo_from else len(STEPS)
    for i, s in enumerate(STEPS):
        if r.done(s) and i < redo:
            log(f"step {i + 1} {s}: already done, skipped")
        else:
            if (r.state / f"{s}.done").exists():
                (r.state / f"{s}.done").unlink()
            t0 = time.time()
            log(f"===== step {i + 1}/{len(STEPS)}: {s}")
            info = funcs[s](r)
            r.mark(s, {**(info or {}), "minutes": round((time.time() - t0) / 60, 1)})
            log(f"step {s} finished in {(time.time() - t0) / 60:.1f} min")
        if stop_after == s:
            break
    log(f"DONE -> {r.dir / 'REPORT.md'}")
    return r
