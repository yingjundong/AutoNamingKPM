"""Name syllables from measured movement, by fixed rules (no language model).

Per syllable: medians / fractions of the per-frame features (features.py), bout statistics, and the rank of each
feature among the syllables. The description uses only features that stand out relative to the other syllables, and
every phrase carries its number, e.g. "turning RIGHT in 91% of turning bouts". A bout is a turning bout if its net
heading change is >= 30 degrees.

Name rules, in priority order (movement_name):
  1 posture unreliable  body length < 0.3 x running length = keypoints collapsed -> posture not used
  2 turning             |turn rate| in the top quarter of syllables: >= 70% of turning bouts one way -> right / left
                        turn, otherwise scanning / head sweeps
  3 rearing             shortened body (bottom quarter and < 0.9x running length, or <= 0.75x) AND nose untracked
                        >= 1.5x the overall rate (and top quarter of syllables)
  4 stationary          >= 60% of frames < 1 cm/s ("pause at wall" if enriched >= 1.3x near walls / corners)
  5 locomotion          fastest syllable or speed in the top quarter
  6 slow walking        median < 1.5 cm/s or speed in the bottom third
  7 exploratory walking otherwise (+ ", veering left/right" if >= 70% of turning bouts go one way and >= 30% of
                        bouts are turning bouts)
"""
from __future__ import annotations

import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .features import bout_table

TURN_BOUT_DEG = 30.0
SHORT_BODY_ABS = 0.75
RARE = -1
RANKED = ["speed_cm_s", "abs_turn_deg_s", "turning_bout_frac", "median_bout_s", "body_length_rel", "body_bend_deg",
          "nose_lost_frac", "wall_frac", "corner_frac", "centre_frac", "still_frac"]


def summarise(frames, bt):
    rows = []
    total = len(frames)
    for s, g in frames.groupby("syllable"):
        b = bt[bt.syllable == s]
        tb = b[b.net_turn_deg.abs() >= TURN_BOUT_DEG]
        loc = g.dropna(subset=["area"])
        rows.append(dict(
            syllable=s, frame_pct=100 * len(g) / total, n_bouts=len(b), median_bout_s=b.duration_s.median(),
            speed_cm_s=g.speed.median(), speed_p90=g.speed.quantile(0.9), still_frac=(g.speed < 1.0).mean(),
            abs_turn_deg_s=g.turn_rate.abs().median(), signed_turn_deg_s=g.turn_rate.mean(),
            turning_bout_frac=len(tb) / max(len(b), 1),
            right_turn_bout_frac=(tb.net_turn_deg > 0).mean() if len(tb) else np.nan,
            median_net_turn_deg=b.net_turn_deg.median(),
            body_length_rel=g.body_length_rel.median(), body_bend_deg=g.body_bend.median(),
            nose_lost_frac=g.nose_lost.mean(),
            wall_frac=(loc.area == "wall").mean() if len(loc) else np.nan,
            corner_frac=(loc.area == "corner").mean() if len(loc) else np.nan,
            centre_frac=(loc.area == "centre").mean() if len(loc) else np.nan,
            d_wall_cm=loc.d_wall.median() if len(loc) else np.nan,
            d_corner_cm=loc.d_corner.median() if len(loc) else np.nan,
            n_location_frames=len(loc)))
    return pd.DataFrame(rows)


def movement_name(x, q, turning, rf, short, nose, overall):
    unreliable = np.isfinite(x.body_length_rel) and x.body_length_rel < 0.3
    if unreliable:
        short = nose = False
    if turning:
        if np.isfinite(rf) and rf >= 0.7:
            return "right turn", f"turn rate top quarter; {100 * rf:.0f}% of turning bouts rightward"
        if np.isfinite(rf) and rf <= 0.3:
            return "left turn", f"turn rate top quarter; {100 * (1 - rf):.0f}% of turning bouts leftward"
        return "scanning / head sweeps" + (" (posture tracking unreliable)" if unreliable else ""), \
            "turn rate top quarter, no consistent direction"
    if short and nose:
        return "rearing", f"body {x.body_length_rel:.2f}x running length and nose untracked {100 * x.nose_lost_frac:.0f}%"
    if x.still_frac >= 0.6:
        at_wall = any(np.isfinite(x[c]) and overall[c] > 0 and x[c] / overall[c] >= 1.3 for c in ("wall_frac", "corner_frac"))
        return ("pause at wall" if at_wall else "pause / stationary"), f"{100 * x.still_frac:.0f}% of frames < 1 cm/s"
    if x.speed_cm_s == x.speed_max or q.speed_cm_s >= 0.75:
        return "locomotion", f"median speed {x.speed_cm_s:.1f} cm/s (top quarter)"
    if x.speed_cm_s < 1.5 or q.speed_cm_s <= 0.34:
        return "slow walking", f"median speed {x.speed_cm_s:.1f} cm/s"
    veer = ""
    if np.isfinite(rf) and (rf >= 0.7 or rf <= 0.3) and x.turning_bout_frac >= 0.3:
        veer = ", veering " + ("right" if rf >= 0.7 else "left")
    return "exploratory walking" + veer, f"median speed {x.speed_cm_s:.1f} cm/s, no fast turning"


def describe(t, overall):
    r = t.set_index("syllable")
    r["speed_max"] = r.speed_cm_s.max()
    rank = r[RANKED].rank(pct=True)
    out = []
    for s, x in r.iterrows():
        q = rank.loc[s]; parts = []
        if x.speed_cm_s == r.speed_cm_s.max():
            parts.append(f"fastest locomotion (median {x.speed_cm_s:.1f} cm/s)")
        elif q.speed_cm_s >= 0.75:
            parts.append(f"moving fast (median {x.speed_cm_s:.1f} cm/s)")
        elif x.still_frac >= 0.6:
            parts.append(f"nearly still ({100 * x.still_frac:.0f}% of frames < 1 cm/s)")
        elif q.speed_cm_s <= 0.34 or x.speed_cm_s < 1.5:
            parts.append(f"slow (median {x.speed_cm_s:.1f} cm/s)")
        else:
            parts.append(f"moderate speed (median {x.speed_cm_s:.1f} cm/s)")
        rf = x.right_turn_bout_frac
        turning = q.abs_turn_deg_s >= 0.75
        if turning:
            if np.isfinite(rf) and rf >= 0.7:
                parts.append(f"turning RIGHT (clockwise from above) in {100 * rf:.0f}% of turning bouts, "
                             f"median {x.median_net_turn_deg:+.0f} deg/bout, |turn rate| {x.abs_turn_deg_s:.0f} deg/s")
            elif np.isfinite(rf) and rf <= 0.3:
                parts.append(f"turning LEFT (counter-clockwise from above) in {100 * (1 - rf):.0f}% of turning bouts, "
                             f"median {x.median_net_turn_deg:+.0f} deg/bout, |turn rate| {x.abs_turn_deg_s:.0f} deg/s")
            else:
                parts.append(f"turning / scanning in both directions (|turn rate| {x.abs_turn_deg_s:.0f} deg/s, "
                             f"{100 * rf:.0f}% of turning bouts rightward)")
        elif np.isfinite(rf) and (rf >= 0.7 or rf <= 0.3) and x.turning_bout_frac >= 0.3:
            side = "right" if rf >= 0.7 else "left"
            parts.append(f"no fast turning, but heading drifts {side} over a bout "
                         f"({100 * max(rf, 1 - rf):.0f}% of bouts with >= 30 deg net change go {side})")
        short = np.isfinite(x.body_length_rel) and x.body_length_rel < 0.9 and (
            q.body_length_rel <= 0.25 or x.body_length_rel <= SHORT_BODY_ABS)
        nose = np.isfinite(x.nose_lost_frac) and x.nose_lost_frac >= 1.5 * overall["nose_lost_frac"] and q.nose_lost_frac >= 0.75
        if np.isfinite(x.body_length_rel) and x.body_length_rel < 0.3:
            parts.append(f"body-length estimate {x.body_length_rel:.2f}x running length: keypoints collapsed, "
                         "posture not interpretable")
        elif short and nose:
            parts.append(f"shortened body ({x.body_length_rel:.2f} x running length) with nose often untracked "
                         f"({100 * x.nose_lost_frac:.0f}% vs {100 * overall['nose_lost_frac']:.0f}% overall): "
                         "posture consistent with rearing")
        elif short:
            parts.append(f"shortened body ({x.body_length_rel:.2f} x running length)")
        elif nose:
            parts.append(f"nose often untracked ({100 * x.nose_lost_frac:.0f}% vs {100 * overall['nose_lost_frac']:.0f}% overall)")
        if np.isfinite(x.body_bend_deg) and q.body_bend_deg >= 0.85 and x.body_bend_deg > 1.2 * overall["body_bend_deg"]:
            parts.append(f"strongly bent body ({x.body_bend_deg:.0f} deg vs {overall['body_bend_deg']:.0f} overall)")
        for col, word in (("corner_frac", "corners"), ("wall_frac", "walls"), ("centre_frac", "the centre")):
            if np.isfinite(x[col]) and overall[col] > 0 and x[col] / overall[col] >= 1.3 and x.n_location_frames >= 500:
                parts.append(f"enriched near {word} ({100 * x[col]:.0f}% of frames vs {100 * overall[col]:.0f}% overall)")
        if q.median_bout_s >= 0.85:
            parts.append(f"long bouts (median {x.median_bout_s:.1f} s)")
        elif q.median_bout_s <= 0.2:
            parts.append(f"short bouts (median {x.median_bout_s:.1f} s)")
        conf = "low (rare: < 100 bouts)" if x.n_bouts < 100 else "medium" if x.n_bouts < 500 else "high"
        name, basis = movement_name(x, q, turning, rf, short, nose, overall)
        out.append(dict(syllable=s, movement_name=name, name_basis=basis, frame_pct=round(x.frame_pct, 1),
                        n_bouts=int(x.n_bouts), confidence=conf, turning=bool(turning), description="; ".join(parts)))
    return pd.DataFrame(out)


def heatmap(t, path):
    r = t.set_index("syllable")[RANKED]
    rank = r.rank(pct=True)
    fig, ax = plt.subplots(figsize=(1.0 * len(RANKED) + 2, 0.5 * len(r) + 2))
    im = ax.imshow(rank.to_numpy(float), cmap="viridis", vmin=0, vmax=1, aspect="auto")
    for i in range(len(r)):
        for j in range(len(RANKED)):
            v = r.iloc[i, j]
            ax.text(j, i, "" if not np.isfinite(v) else (f"{v:.2f}" if abs(v) < 10 else f"{v:.0f}"),
                    ha="center", va="center", fontsize=7, color="white" if rank.iloc[i, j] < 0.6 else "black")
    ax.set_xticks(range(len(RANKED))); ax.set_xticklabels(RANKED, rotation=35, ha="right", fontsize=8)
    ax.set_yticks(range(len(r))); ax.set_yticklabels([f"syllable {s} ({p:.1f}%)" for s, p in zip(r.index, t.frame_pct)], fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.03).set_label("rank among syllables (1 = highest)")
    ax.set_title("Syllable features (numbers = values; colour = rank among syllables)", fontsize=10)
    fig.tight_layout(); fig.savefig(path, dpi=140, facecolor="white"); plt.close(fig)


def describe_all(frames: pd.DataFrame, labels: np.ndarray, out: Path, rare_pct: float, title: str) -> pd.DataFrame:
    """Describe the syllables given per frame by `labels`; syllables below rare_pct % of frames are pooled."""
    out.mkdir(parents=True, exist_ok=True)
    f = frames.assign(syllable=np.asarray(labels).astype(int))
    pct = f.syllable.value_counts(normalize=True) * 100
    pooled = sorted(int(s) for s in pct.index[pct < rare_pct]) if rare_pct > 0 else []
    if pooled:
        f.loc[f.syllable.isin(pooled), "syllable"] = RARE
    bt = bout_table(f)
    has_loc = f.area.notna().mean() > 0
    den = max(f.area.notna().mean(), 1e-9)
    overall = dict(nose_lost_frac=f.nose_lost.mean(), body_bend_deg=f.body_bend.median(),
                   wall_frac=(f.area == "wall").mean() / den if has_loc else np.nan,
                   corner_frac=(f.area == "corner").mean() / den if has_loc else np.nan,
                   centre_frac=(f.area == "centre").mean() / den if has_loc else np.nan)
    t = summarise(f, bt)
    t.to_csv(out / "syllable_features.csv", index=False)
    by_subject = pd.concat([summarise(g, bt[bt.subject == m]).assign(subject=m) for m, g in f.groupby("subject")],
                           ignore_index=True)
    by_subject.to_csv(out / "syllable_features_by_subject.csv", index=False)
    described = t[t.syllable != RARE].reset_index(drop=True)
    d = describe(described, overall)
    if (t.syllable == RARE).any():
        rr = t[t.syllable == RARE].iloc[0]
        d = pd.concat([d, pd.DataFrame([dict(
            syllable=RARE, movement_name="rare syllables (pooled, not described)",
            name_basis=f"{len(pooled)} syllables each < {rare_pct}% of frames", frame_pct=round(rr.frame_pct, 1),
            n_bouts=int(rr.n_bouts), confidence="n/a", turning=False, description=f"pooled syllables {pooled}")])],
            ignore_index=True)
    cons = []
    for s, turning in zip(d.syllable, d.turning):
        bm = by_subject[(by_subject.syllable == s) & (by_subject.n_bouts >= 20)]
        rf = bm.right_turn_bout_frac.dropna()
        if not turning or not len(rf):
            cons.append(""); continue
        side = "right" if np.median(rf) > 0.5 else "left"
        n_same = int(((rf > 0.5) if side == "right" else (rf < 0.5)).sum())
        cons.append(f"turns {side} in {n_same}/{len(rf)} subjects")
    d["turn_direction_by_subject"] = cons
    d.to_csv(out / "syllable_descriptions.csv", index=False)
    with open(out / "syllable_descriptions.md", "w", encoding="utf-8") as fh:
        fh.write(f"# Syllable descriptions ({title})\n\n{f.subject.nunique()} subjects, {f.rec_id.nunique()} recordings, "
                 f"{len(f):,} frames. Turn direction: + = right = clockwise seen from above (camera above, not mirrored).\n\n")
        for r in d.itertuples():
            fh.write(f"- **Syllable {r.syllable}: {r.movement_name}** ({r.frame_pct}% of frames, {r.n_bouts} bouts, "
                     f"confidence {r.confidence}; name rule: {r.name_basis}): {r.description}"
                     + (f" [{r.turn_direction_by_subject}]" if r.turn_direction_by_subject else "") + "\n")
    heatmap(described, out / "syllable_feature_heatmap.png")
    return d


def pooled_of(desc: pd.DataFrame) -> list[int]:
    rr = desc[desc.syllable == RARE]
    return [int(x) for x in re.findall(r"\d+", rr.description.iloc[0])] if len(rr) else []


def base_name(name: str) -> str:
    n = str(name).strip()
    return "rare" if n.casefold().startswith("rare") else n.split(",")[0].split(" (")[0].strip()


def consistency(out: Path, mapping: pd.DataFrame, rare_pct: float) -> pd.DataFrame:
    """For every merged syllable: its members, their own names, and the share of frames whose member has the same name."""
    ref_d = pd.read_csv(out / "unmerged_reference" / "syllable_descriptions.csv")
    ref_f = pd.read_csv(out / "unmerged_reference" / "syllable_features.csv").set_index("syllable")
    new_d = pd.read_csv(out / "syllable_descriptions.csv")
    pooled_new, pooled_old = pooled_of(new_d), pooled_of(ref_d)
    names_old = dict(zip(ref_d.syllable, ref_d.movement_name))
    rows = []
    for new, g in mapping[mapping.old_frames > 0].groupby("new_syllable"):
        label = RARE if new in pooled_new else new
        gname = new_d.loc[new_d.syllable == label, "movement_name"]
        gname = gname.iloc[0] if len(gname) else "?"
        members, agree, tot = [], 0, g.old_frames.sum()
        for r in g.sort_values("old_frames", ascending=False).itertuples():
            o = int(r.old_syllable)
            oname = "rare (not described)" if o in pooled_old else names_old.get(o, "?")
            agree += r.old_frames * (base_name(oname) == base_name(gname))
            fo = ref_f.loc[o] if o in ref_f.index and o not in pooled_old else None
            members.append(f"{o} ({100 * r.old_frames / tot:.0f}%: {oname}"
                           + (f"; {fo.speed_cm_s:.1f} cm/s, |turn| {fo.abs_turn_deg_s:.0f} deg/s" if fo is not None else "") + ")")
        rows.append(dict(merged_syllable=new, described_as=label, movement_name=gname, n_members=len(g),
                         frames=int(tot), name_agreement=round(agree / tot, 2), members="; ".join(members)))
    c = pd.DataFrame(rows)
    c.to_csv(out / "merge_group_consistency.csv", index=False)
    with open(out / "syllable_descriptions.md", "a", encoding="utf-8") as fh:
        fh.write("\n## Merge check\n\nFor each merged syllable: its original member syllables (share of the group's "
                 "frames, the member's own name when described separately, speed, turn rate). name_agreement = fraction "
                 "of the group's frames whose member has the same movement name as the group; < 0.7 means the merge "
                 "joined syllables that move differently.\n\n")
        for r in c[c.n_members > 1].itertuples():
            flag = "  **<- check**" if r.name_agreement < 0.7 else ""
            fh.write(f"- merged {r.merged_syllable} -> **{r.movement_name}**, agreement {r.name_agreement}{flag}: {r.members}\n")
        fh.write(f"\nRare-pool threshold: {rare_pct}% of frames.\n")
    return c
