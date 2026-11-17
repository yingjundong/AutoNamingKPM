"""Step 3a: merge syllables that look and move alike.

  1 sample   every bout = contiguous run of one syllable. Per bout: mean speed, still fraction (< 1 cm/s), |turn rate|,
             signed turn rate, net heading change, body length, nose lost, log duration. Up to bouts_per_syllable bouts
             per syllable (>= 3 frames where possible), frames_per_bout evenly spaced frames each.
  2 embed    embed.py: egocentric crops -> DINOv2 image features (mean and SD over a bout's frames) -> PCA 32.
  3 compare  for every pair of common syllables (>= rare_pct % of frames) a cross-validated logistic regression (folds
             grouped by session, so one session is never in both training and test data) tries to tell their bouts
             apart from video + kinematic features (each block with equal total weight). distance = 2 x AUC - 1.
             Average-linkage clustering; syllables are merged while a classifier separates them with AUC < max_auc.
  4 rare     a classifier trained on the merged common groups predicts each rare syllable's bouts; the syllable joins
             the group with the highest mean probability if that is >= rare_min_prob, otherwise it stays on its own.
  5 build    Keypoint-MoSeq's own relabelling (generate_syllable_mapping / apply_syllable_mapping): merged syllables
             are renumbered by frequency (bouts); merged_results.h5 + merge_mapping.csv.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from .util import bouts_of, log

KIN = ["speed", "still_frac", "abs_turn", "signed_turn", "net_turn", "body_length_rel", "nose_lost", "log_duration"]


def sample(frames: pd.DataFrame, rec: pd.DataFrame, results: dict, cache: Path, n_per: int, k_frames: int, seed=0):
    vid = dict(zip(rec.rec_id, rec.video))
    rows, track = [], {}
    for rid, f in frames.groupby("rec_id", sort=False):
        k = results[rid]
        track[rid] = (k["centroid"][:, 0].astype(float), k["centroid"][:, 1].astype(float), k["heading"].astype(float))
        s = f.syllable.to_numpy()
        sp, tr, dh = f.speed.to_numpy(), f.turn_rate.to_numpy(), f.dheading.to_numpy()
        bl, nl, fps = f.body_length_rel.to_numpy(), f.nose_lost.to_numpy(), f.fps.iloc[0]
        for a, b in zip(*bouts_of(s)):
            rows.append(dict(rec_id=rid, video=vid.get(rid, ""), session=f.session.iloc[0], syllable=int(s[a]),
                             start=int(a), end=int(b), n_frames=int(b - a), speed=sp[a:b].mean(),
                             still_frac=(sp[a:b] < 1.0).mean(), abs_turn=np.abs(tr[a:b]).mean(),
                             signed_turn=tr[a:b].mean(), net_turn=np.nansum(dh[a + 1:b]),
                             body_length_rel=np.nanmean(bl[a:b]) if np.isfinite(bl[a:b]).any() else np.nan,
                             nose_lost=np.nanmean(nl[a:b]) if np.isfinite(nl[a:b]).any() else np.nan,
                             log_duration=np.log((b - a) / fps)))
    bt = pd.DataFrame(rows)
    bt.to_csv(cache / "bouts.csv", index=False)
    rng = np.random.default_rng(seed)
    pick = []
    for _, g in bt[bt.video != ""].groupby("syllable"):
        cand = g[g.n_frames >= 3] if (g.n_frames >= 3).any() else g
        pick.append(cand.sample(min(n_per, len(cand)), random_state=int(rng.integers(1 << 31))))
    if not pick:
        raise SystemExit("no bout has a video: check video_dir (kpm-autoname check <config> lists missing videos)")
    sm = pd.concat(pick).reset_index().rename(columns={"index": "bout_id"})
    fr = np.full((len(sm), k_frames), -1, int)
    cx, cy, hd = (np.full((len(sm), k_frames), np.nan) for _ in range(3))
    for j, r in enumerate(sm.itertuples()):
        idx = np.unique(np.linspace(r.start, r.end - 1, min(k_frames, r.n_frames)).round().astype(int))
        x, y, h = track[r.rec_id]
        fr[j, :len(idx)] = idx; cx[j, :len(idx)] = x[idx]; cy[j, :len(idx)] = y[idx]; hd[j, :len(idx)] = h[idx]
    sm.to_csv(cache / "samples.csv", index=False)
    np.savez(cache / "samples.npz", frames=fr, cx=cx, cy=cy, heading=hd)
    log(f"  {len(bt):,} bouts; {len(sm):,} sampled from {sm.syllable.nunique()} syllables")
    return bt


def run_embed(cfg, cache: Path, crop_px: int):
    py = cfg.merge.embed_python or sys.executable
    cmd = [str(py), str(Path(__file__).with_name("embed.py")), "--cache", str(cache), "--crop-px", str(crop_px),
           "--device", str(cfg.merge.device)]
    log("$ " + " ".join(cmd))
    if subprocess.run(cmd).returncode:
        raise SystemExit("embedding failed (see above). It needs torch and timm in the Python given by "
                         "merge.embed_python (default: this environment)")


def features(cache: Path, kind: str):
    from sklearn.decomposition import PCA
    sm = pd.read_csv(cache / "samples.csv", keep_default_na=False, dtype={"video": str})
    for c in KIN:
        sm[c] = pd.to_numeric(sm[c], errors="coerce")
    keep = np.ones(len(sm), bool)
    parts = []
    if kind in ("video", "video+kinematic"):
        e = np.load(cache / "embeddings.npz")
        keep = e["n_frames"] > 0
        v = np.hstack([e["mean"], e["sd"]])
        v = (v - v[keep].mean(0)) / (v[keep].std(0) + 1e-6)
        parts.append(PCA(min(32, int(keep.sum()), v.shape[1]), random_state=0).fit(v[keep]).transform(v))
    if kind in ("kinematic", "video+kinematic"):
        k = sm[KIN].copy()
        k = k.fillna(k.median()).fillna(0)
        k = ((k - k.mean()) / k.std().replace(0, 1)).to_numpy()
        parts.append(k * (np.sqrt(32 / len(KIN)) if kind == "video+kinematic" else 1.0))
    return sm[keep].reset_index(drop=True), np.hstack(parts)[keep]


def pair_auc(sm, X, common, seed=0):
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import GroupKFold
    rng = np.random.default_rng(seed)
    out = {}
    for i, a in enumerate(common):
        for b in common[i + 1:]:
            ia, ib = np.flatnonzero(sm.syllable == a), np.flatnonzero(sm.syllable == b)
            n = min(len(ia), len(ib))
            idx = np.r_[rng.choice(ia, n, replace=False), rng.choice(ib, n, replace=False)]
            y = np.r_[np.zeros(n), np.ones(n)]
            grp = sm.session.to_numpy()[idx]
            nf = min(5, len(np.unique(grp)))
            p = np.full(len(idx), np.nan)
            if nf >= 2:
                for tr, te in GroupKFold(nf).split(idx, y, grp):
                    if len(np.unique(y[tr])) < 2:
                        continue
                    clf = LogisticRegression(C=0.1, max_iter=2000).fit(X[idx[tr]], y[tr])
                    p[te] = clf.predict_proba(X[idx[te]])[:, 1]
            ok = ~np.isnan(p)
            out[(a, b)] = roc_auc_score(y[ok], p[ok]) if len(np.unique(y[ok])) == 2 else 1.0
    return out


def cluster(common, auc, max_auc):
    from scipy.cluster.hierarchy import fcluster, linkage
    from scipy.spatial.distance import squareform
    if len(common) < 2:
        return {s: i + 1 for i, s in enumerate(common)}
    n = len(common)
    D = np.zeros((n, n))
    for (a, b), v in auc.items():
        i, j = common.index(a), common.index(b)
        D[i, j] = D[j, i] = max(0.0, 2 * v - 1)
    lab = fcluster(linkage(squareform(D), "average"), t=2 * max_auc - 1 - 1e-9, criterion="distance")
    return {s: int(l) for s, l in zip(common, lab)}


def assign_rare(sm, X, common_lab, rare, min_prob):
    from sklearn.linear_model import LogisticRegression
    rows, out = [], {}
    tr = sm.syllable.isin(common_lab).to_numpy()
    y = sm.syllable[tr].map(common_lab).to_numpy()
    clf = None
    if len(np.unique(y)) >= 2:
        clf = LogisticRegression(C=0.1, max_iter=3000, class_weight="balanced").fit(X[tr], y)
    for s in rare:
        m = (sm.syllable == s).to_numpy()
        if not m.any() or clf is None:
            rows.append(dict(syllable=s, n_bouts=int(m.sum()), nearest_group_members="", mean_prob=np.nan, assigned=False))
            continue
        P = clf.predict_proba(X[m]).mean(0)
        best = int(clf.classes_[P.argmax()])
        ok = bool(P.max() >= min_prob)
        if ok:
            out[s] = best
        members = sorted(k for k, v in common_lab.items() if v == best)
        rows.append(dict(syllable=s, n_bouts=int(m.sum()), nearest_group_members=str(members),
                         mean_prob=round(float(P.max()), 2), assigned=ok))
    return out, pd.DataFrame(rows)


def groups_from(labels: dict):
    g = {}
    for s, l in labels.items():
        g.setdefault(l, []).append(int(s))
    return sorted([sorted(v) for v in g.values() if len(v) > 1])


def run_merge(cache, frames_per_syl, rare_pct, kind, max_auc, min_prob):
    sm, X = features(cache, kind)
    pct = 100 * frames_per_syl / frames_per_syl.sum()
    common = sorted(int(s) for s in pct.index[pct >= rare_pct] if (sm.syllable == s).sum() >= 10)
    rare = sorted(int(s) for s in pct.index if int(s) not in common)
    auc = pair_auc(sm, X, common)
    lab = cluster(common, auc, max_auc)
    rare_map, rare_tab = assign_rare(sm, X, lab, rare, min_prob)
    full = dict(lab)
    nxt = max(lab.values(), default=0) + 1
    for s in rare:
        if s in rare_map:
            full[s] = rare_map[s]
        else:
            full[s] = nxt; nxt += 1
    return full, auc, rare_tab, common


def build_merged(model: Path, groups: list, out: Path) -> pd.DataFrame:
    """Relabel with Keypoint-MoSeq's own functions; write merged_results.h5 and merge_mapping.csv."""
    import keypoint_moseq as kpms
    results = kpms.load_hdf5(str(Path(model) / "results.h5"))
    mapping = kpms.generate_syllable_mapping(results, groups, runlength=True)
    new = kpms.apply_syllable_mapping(results, mapping)
    group_of = {s: i for i, g in enumerate(groups) for s in g}

    def stats(res):
        fr, runs = {}, {}
        for v in res.values():
            s = np.asarray(v["syllable"]).ravel()
            for x, c in zip(*np.unique(s, return_counts=True)):
                fr[int(x)] = fr.get(int(x), 0) + int(c)
            st = s[np.r_[True, s[1:] != s[:-1]]] if len(s) else s
            for x, c in zip(*np.unique(st, return_counts=True)):
                runs[int(x)] = runs.get(int(x), 0) + int(c)
        return fr, runs
    (bf, br), (af, ar) = stats(results), stats(new)
    total = sum(af.values())
    for name in results:                                    # verify every frame follows the mapping
        old = np.asarray(results[name]["syllable"]).ravel()
        if not np.array_equal(np.vectorize(lambda s: mapping[int(s)])(old), np.asarray(new[name]["syllable"]).ravel()):
            raise SystemExit(f"internal error: relabelling of {name} does not follow the mapping")
    rep = pd.DataFrame([dict(old_syllable=int(o), new_syllable=int(n), merged=o in group_of,
                             merge_group=group_of.get(o, ""), old_frames=bf.get(o, 0), old_runs=br.get(o, 0),
                             new_frames=af.get(n, 0), new_runs=ar.get(n, 0),
                             new_frame_frequency=af.get(n, 0) / total if total else np.nan)
                        for o, n in sorted(mapping.items())])
    p = out / "merged_results.h5"
    if p.exists():
        p.unlink()
    kpms.save_hdf5(str(p), new, exist_ok=True, overwrite=True)
    rep.to_csv(out / "merge_mapping.csv", index=False)
    (out / "merge_groups.json").write_text(json.dumps(groups), encoding="utf-8")
    return rep


def score(labels: dict, manual: pd.DataFrame, feats: pd.DataFrame, rare_pct: float):
    """Agreement with a hand-made merge_mapping.csv: frame-weighted adjusted Rand index (all / common syllables) and
    the within-group spread of z-scored syllable features (lower = groups move more alike)."""
    from sklearn.metrics import adjusted_rand_score
    m = manual.set_index("old_syllable")
    w = m.old_frames
    sy = [s for s in m.index if s in labels and w[s] > 0]
    truth = np.repeat(m.new_syllable[sy].to_numpy(), w[sy].to_numpy())
    pred = np.repeat(np.array([labels[s] for s in sy]), w[sy].to_numpy())
    com = [s for s in sy if 100 * w[s] / w.sum() >= rare_pct]
    tc = np.repeat(m.new_syllable[com].to_numpy(), w[com].to_numpy())
    pc = np.repeat(np.array([labels[s] for s in com]), w[com].to_numpy())
    F = ["speed_cm_s", "still_frac", "abs_turn_deg_s", "signed_turn_deg_s", "body_length_rel", "wall_frac", "nose_lost_frac"]
    f = feats.set_index("syllable").reindex(com)[F]
    zf = ((f - f.mean()) / f.std()).fillna(0)
    ww = w[com] / w[com].sum()
    lab = pd.Series({s: labels[s] for s in com})
    spread = 0.0
    for _, idx in lab.groupby(lab).groups.items():
        wi = ww[idx] / ww[idx].sum()
        mu = zf.loc[idx].mul(wi, axis=0).sum()
        spread += ww[idx].sum() * ((zf.loc[idx] - mu) ** 2).mul(wi, axis=0).sum().mean()
    return dict(ARI_all=round(adjusted_rand_score(truth, pred), 3), ARI_common=round(adjusted_rand_score(tc, pc), 3),
                spread=round(spread, 3), n_groups_common=int(lab.nunique()), common_groups=str(groups_from(lab.to_dict())))
