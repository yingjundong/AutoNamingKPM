"""Step 5: REPORT.md + final_syllables.csv - for every merged syllable its name, original KPM syllables, why they were
merged, the closest syllable kept separate, why it got its name, and its video."""
from __future__ import annotations

from datetime import datetime

import numpy as np
import pandas as pd

from .config import crop_px
from .naming import pooled_of


def write(r, steps):
    cfg = r.cfg
    m = cfg.merge
    out = r.merge
    mp = pd.read_csv(out / "merge_mapping.csv")
    mp = mp[mp.old_frames > 0]
    desc = pd.read_csv(out / "syllable_descriptions.csv")
    ref = pd.read_csv(out / "unmerged_reference" / "syllable_descriptions.csv")
    feats = pd.read_csv(out / "syllable_features.csv").set_index("syllable")
    auc = pd.read_csv(out / "pair_auc.csv")
    rare = pd.read_csv(out / "rare_assignment.csv")
    cons = pd.read_csv(out / "merge_group_consistency.csv").set_index("merged_syllable")
    vfile = r.videos / "videos.csv"
    vids = pd.read_csv(vfile).set_index("syllable") if vfile.exists() else pd.DataFrame()
    pooled_new, pooled_old = pooled_of(desc), pooled_of(ref)
    ref_names = {s: n for s, n in zip(ref.syllable, ref.movement_name)}
    for s in pooled_old:
        ref_names[s] = "rare (not described separately)"
    dsc = desc.set_index("syllable")
    total = mp.old_frames.sum()
    A = {}
    for t in auc.itertuples():
        A[t.a, t.b] = A[t.b, t.a] = t.auc
    common = sorted(set(auc.a) | set(auc.b))
    rare_t = rare.set_index("syllable") if len(rare) else pd.DataFrame()

    rows, sections = [], []
    for s in mp.groupby("new_syllable").old_frames.sum().sort_values(ascending=False).index:
        g = mp[mp.new_syllable == s].sort_values("old_frames", ascending=False)
        members = g.old_syllable.astype(int).tolist()
        frames = int(g.old_frames.sum())
        pooled = s in pooled_new or s not in dsc.index
        name = "rare (too few frames to describe)" if pooled else dsc.at[s, "movement_name"]
        conf = "n/a" if pooled else dsc.at[s, "confidence"]
        why = []
        cm = [x for x in members if x in common]
        for i, x in enumerate(cm):
            for y in cm[i + 1:]:
                why.append(f"original {x} and {y}: a video + kinematics classifier separates their bouts with AUC "
                           f"{A.get((x, y), float('nan')):.2f} (< {m.max_auc}: not reliably distinguishable)")
        for x in members:
            if x in rare_t.index and bool(rare_t.at[x, "assigned"]) and cm:
                why.append(f"original {x} is rare ({100 * g.set_index('old_syllable').at[x, 'old_frames'] / total:.2f}% of "
                           f"frames, {int(rare_t.at[x, 'n_bouts'])} sampled bouts); its bouts were classified as this "
                           f"group with mean probability {rare_t.at[x, 'mean_prob']:.2f}")
        if len(members) == 1:
            why.append(f"not merged: every other syllable is distinguishable from it (AUC >= {m.max_auc})"
                       if members[0] in common else "not merged: rare syllable whose bouts did not clearly resemble any group")
        outside = [(A[(x, y)], x, y) for x in cm for y in common if y not in members and (x, y) in A]
        closest = min(outside) if outside else None
        if pooled:
            why_name, evid = f"below {m.rare_pct}% of frames: not named (too few frames for reliable features)", ""
        else:
            why_name, evid = dsc.at[s, "name_basis"], dsc.at[s, "description"]
            tds = dsc.at[s, "turn_direction_by_subject"] if "turn_direction_by_subject" in dsc.columns else ""
            if isinstance(tds, str) and tds:
                evid += f"; {tds}"
        f = feats.loc[s] if s in feats.index else None
        mem = "; ".join(f"{x} ({100 * o / frames:.0f}% of group: {ref_names.get(x, '?')})" for x, o in zip(g.old_syllable, g.old_frames))
        video = vids.at[s, "video"] if s in vids.index else ""
        nb = int(vids.at[s, "bouts"]) if s in vids.index else 0
        rows.append(dict(merged_syllable=int(s), movement_name=name, confidence=conf, frame_pct=round(100 * frames / total, 2),
                         original_syllables=str(members), original_members_detail=mem,
                         name_agreement_of_members=cons.at[s, "name_agreement"] if s in cons.index else np.nan,
                         speed_cm_s=None if f is None else round(f.speed_cm_s, 2),
                         abs_turn_deg_s=None if f is None else round(f.abs_turn_deg_s, 0),
                         body_length_rel=None if f is None else round(f.body_length_rel, 2),
                         why_name=why_name, name_evidence=evid, why_merged=" | ".join(why),
                         closest_other=f"original {closest[1]} vs {closest[2]}: AUC {closest[0]:.2f}" if closest else "",
                         video=video, bouts_in_video=nb))
        sec = [f"### Merged syllable {int(s)}: **{name}**  (confidence {conf}, {100 * frames / total:.1f}% of frames)",
               f"- **Original KPM syllables:** {mem}"]
        if s in cons.index and len(members) > 1:
            sec.append(f"- **Members described separately agree on the name for** {100 * cons.at[s, 'name_agreement']:.0f}% of the group's frames")
        sec.append("- **Why these were merged:** " + "; ".join(why))
        if closest:
            sec.append(f"- **Closest syllable kept separate:** original {closest[1]} vs {closest[2]}, AUC {closest[0]:.2f} (>= {m.max_auc})")
        sec.append(f"- **Why this name:** {why_name}")
        if evid:
            sec.append(f"- **Evidence:** {evid}")
        sec.append(f"- **Video:** `{video}` ({nb} bouts)" if video else "- **Video:** not made")
        sections.append("\n".join(sec))
    final = pd.DataFrame(rows)
    final.to_csv(r.dir / "final_syllables.csv", index=False)

    k, c = cfg.kpm, cfg.cleaning
    ki = r.info("kpm")
    head = [f"# Syllable naming report", "",
            f"Generated {datetime.now():%Y-%m-%d %H:%M} from `{cfg._config_file}`. DLC input: `{cfg.dlc_dir}`.", "",
            "## Summary", "",
            f"- Keypoint-MoSeq model: `{ki.get('model', '?')}`; {ki.get('recordings', '?')} recordings, "
            f"{ki.get('frames', 0):,} frames",
            f"- {mp.old_syllable.nunique()} KPM syllables -> **{mp.new_syllable.nunique()} merged syllables**; "
            f"{int(final.movement_name.str.startswith('rare').sum())} of them too rare to name", "",
            "| merged | name | confidence | % frames | original KPM syllables | bouts in video |", "|---|---|---|---|---|---|"]
    head += [f"| {t.merged_syllable} | {t.movement_name} | {t.confidence} | {t.frame_pct} | {t.original_syllables} | {t.bouts_in_video} |"
             for t in final.itertuples()]
    clean_txt = (f"points with likelihood < {c.likelihood}, jumps > {c.velocity_mad} robust SD and skeleton edges > "
                 f"{c.shape_mad} robust SD or > {c.shape_max_factor} x median removed; gaps <= {c.interpolation_limit} "
                 f"frames interpolated; {c.moving_average}-frame centred moving average" if c.enabled else "skipped (input already cleaned)")
    method = ["", "## How the results were produced", "",
              f"1. **DLC cleaning:** {clean_txt}.",
              f"2. **Keypoint-MoSeq** {k.ar_iters} AR-only iterations" + (f" + {k.full_iters} full-model iterations" if k.full_iters else
              " (no full-model iterations: the model is the AR-HMM)") + f"; seed {k.seed}, kappa {k.kappa:g}, latent_dim "
              f"{k.latent_dim}, outlier_scale_factor {k.outlier_scale_factor}, fps {k.fps}.",
              f"3. **Merging** without human input: for each pair of common syllables (>= {m.rare_pct}% of frames) a "
              "cross-validated classifier (folds by session) tries to tell their bouts apart from egocentric video "
              f"({crop_px(cfg)}-px crops rotated to the animal's heading, DINOv2 ViT-S/14 image features) plus speed / "
              f"turning / body length. Pairs it separates with AUC < {m.max_auc} (0.5 = identical, 1 = always separable) are "
              f"merged (average linkage). Rare syllables join the group their bouts are classified as (mean probability >= "
              f"{m.rare_min_prob}), otherwise they stay separate.",
              "4. **Naming:** speed, signed turn rate (+ = right, camera above), body length and nose visibility (DLC) and, "
              "where location labels are given, location are measured per merged syllable and ranked across syllables; "
              "fixed rules turn them into a name (no language model). The 'Evidence' lines are the measured values behind each name.",
              f"5. **Videos:** up to {cfg.videos.bouts_per_recording} bouts per merged syllable from every recording, raw "
              "video with the cleaned keypoints, written one after another per merged syllable "
              "(`04_videos/bouts_in_videos.csv` gives each bout's recording and time in the video).", "",
              "Caveats: merging decides only whether syllables are *distinguishable*; joining distinguishable syllables "
              "into coarser categories is a judgement call left to the user. Names come from kinematics and posture "
              "only; check the videos before relying on a name, especially for low-confidence and rare syllables.", "",
              "## Merged syllables: members, merge justification, name justification", ""]
    (r.dir / "REPORT.md").write_text("\n".join(head + method) + "\n" + "\n\n".join(sections) + "\n", encoding="utf-8")
    return {"report": str(r.dir / "REPORT.md"), "merged_syllables": len(final)}
