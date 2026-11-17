"""Synthetic demo: a simulated animal (walk, left / right turns, pauses, rearing) in a square arena, filmed from above,
with DLC-style keypoints; then the whole pipeline with fewer iterations. Checks the installation end to end and
compares the names with the simulated behaviour (demo_truth_check.csv)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from .util import log

BPS = ["Nose", "Head", "Left_Ear", "Right_Ear", "Center_Torso", "Back_Torso", "Tail_Base"]
SKEL = [["Nose", "Head"], ["Nose", "Left_Ear"], ["Nose", "Right_Ear"], ["Head", "Left_Ear"], ["Head", "Right_Ear"],
        ["Head", "Center_Torso"], ["Center_Torso", "Back_Torso"], ["Back_Torso", "Tail_Base"]]
STATES = ["walk", "turn_left", "turn_right", "pause", "rear"]
FPS, SIZE, ARENA, BODY, CM_PER_PX = 30, 240, (20, 220), 40.0, 0.2


def simulate(rng, n):
    x, y, th = np.empty(n), np.empty(n), np.empty(n)
    state = np.empty(n, object)
    px, py, pth, s, left = 120.0, 120.0, rng.uniform(-np.pi, np.pi), "walk", 0
    lo, hi = ARENA[0] + BODY * 0.6, ARENA[1] - BODY * 0.6
    for i in range(n):
        if left <= 0:
            s = rng.choice(STATES, p=[0.35, 0.15, 0.15, 0.2, 0.15])
            left = int(rng.uniform(0.6, 2.0) * FPS)
        left -= 1
        v = {"walk": 3.0, "turn_left": 0.6, "turn_right": 0.6, "pause": 0.0, "rear": 0.0}[s]
        nx, ny = px + v * np.cos(pth), py + v * np.sin(pth)
        if not (lo < nx < hi and lo < ny < hi):          # at a wall: turn left or right on the spot until free
            if s not in ("turn_left", "turn_right"):
                s = rng.choice(["turn_left", "turn_right"])
            left = max(left, 10)
            nx, ny = px, py
        dth = {"turn_left": -0.09, "turn_right": 0.09}.get(s, 0.0) + rng.normal(0, 0.01)
        pth = np.angle(np.exp(1j * (pth + dth)))
        px, py = nx, ny
        x[i], y[i], th[i], state[i] = px, py, pth, s
    # posture per behaviour, changing smoothly: body length factor, bend (+ = right), stride sway amplitude
    target = {"walk": (1.1, 0.0, 0.07), "turn_left": (0.9, -0.3, 0.0), "turn_right": (0.9, 0.3, 0.0),
              "pause": (0.85, 0.0, 0.0), "rear": (0.55, 0.0, 0.0)}
    post = np.array([target[s] for s in state])
    for i in range(1, n):
        post[i] = post[i - 1] + 0.25 * (post[i] - post[i - 1])
    body = BODY * post[:, 0]
    sway = post[:, 2] * np.sin(2 * np.pi * 4.0 * np.arange(n) / FPS)        # ~4 Hz stride
    return x, y, th, body, post[:, 1], sway, state


def keypoints(rng, x, y, th, body, bend, sway, state):
    u = np.stack([np.cos(th), np.sin(th)], 1)
    w = np.stack([-np.sin(th), np.cos(th)], 1)
    off = {"Nose": (0.5, 0), "Head": (0.33, 0), "Left_Ear": (0.27, -0.12), "Right_Ear": (0.27, 0.12),
           "Center_Torso": (0, 0), "Back_Torso": (-0.25, 0), "Tail_Base": (-0.5, 0)}
    c = np.stack([x, y], 1)
    out = {}
    for bp in BPS:
        a, b = off[bp]
        lateral = b * BODY + (bend * 4 * a * a + sway * np.sign(a) * abs(a) * 2) * BODY   # arc when turning, sway when walking
        p = c + (a * body)[:, None] * u + lateral[:, None] * w + rng.normal(0, 0.7, (len(x), 2))
        lik = np.full(len(x), 0.95)
        if bp == "Nose":
            lik = np.where(state == "rear", rng.uniform(0.1, 0.9, len(x)), lik)   # nose often hidden when rearing
        out[bp] = (p[:, 0], p[:, 1], lik)
    return out


def render(path, kp, state):
    import cv2
    vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (SIZE, SIZE))
    spine = ["Tail_Base", "Back_Torso", "Center_Torso", "Head"]
    for i in range(len(state)):
        img = np.full((SIZE, SIZE, 3), 40, np.uint8)
        cv2.rectangle(img, (ARENA[0], ARENA[0]), (ARENA[1], ARENA[1]), (200, 200, 200), -1)
        pts = np.array([[kp[bp][0][i], kp[bp][1][i]] for bp in spine]).round().astype(np.int32)
        cv2.polylines(img, [pts], False, (60, 60, 60), 13, cv2.LINE_AA)
        hx, hy = int(round(kp["Head"][0][i])), int(round(kp["Head"][1][i]))
        rear = state[i] == "rear"
        cv2.circle(img, (hx, hy), 9 if rear else 6, (150, 150, 220) if rear else (90, 90, 140), -1)
        for ear in ("Left_Ear", "Right_Ear"):
            cv2.circle(img, (int(round(kp[ear][0][i])), int(round(kp[ear][1][i]))), 3, (120, 120, 160), -1)
        vw.write(img)
    vw.release()


def make_data(root: Path, n_rec=8, n_frames=3000, seed=0):
    rng = np.random.default_rng(seed)
    truth = []
    for k in range(n_rec):
        sess = f"mouse{k % 2 + 1}_day{k // 2 + 1}"
        d_dlc, d_vid = root / "dlc" / sess, root / "videos" / sess
        d_dlc.mkdir(parents=True, exist_ok=True); d_vid.mkdir(parents=True, exist_ok=True)
        x, y, th, body, bend, sway, state = simulate(rng, n_frames)
        kp = keypoints(rng, x, y, th, body, bend, sway, state)
        cols = pd.MultiIndex.from_tuples([("synthetic", bp, c) for bp in BPS for c in ("x", "y", "likelihood")],
                                         names=["scorer", "bodyparts", "coords"])
        data = np.column_stack([v for bp in BPS for v in kp[bp]])
        pd.DataFrame(data, columns=cols).to_csv(d_dlc / "cam1DLC_synthetic.csv")
        render(d_vid / "cam1.mp4", kp, state)
        truth.append(pd.DataFrame({"session": sess, "frame": np.arange(n_frames), "true_behaviour": state}))
        log(f"  synthetic recording {k + 1}/{n_rec}: {sess}")
    pd.concat(truth).to_csv(root / "truth.csv", index=False)


def main(out: Path, embed_python=None):
    from . import config, pipeline
    out = Path(out)
    if not (out / "truth.csv").exists():
        log(f"making synthetic data in {out}")
        make_data(out)
    cfg_path = out / "demo_config.yml"
    cfg = dict(dlc_dir=str(out / "dlc"), video_dir=str(out / "videos"), output_dir=str(out / "run"), cm_per_px=CM_PER_PX,
               camera_fps=FPS, subject_regex="^([^_]+)", skeleton=SKEL,
               bodyparts=dict(nose="Nose", tail_base="Tail_Base", head="Head", mid_body="Center_Torso"),
               kpm=dict(fps=FPS, kappa=1e4, trajectory_plots=False),   # less data than a real study: lower kappa
               merge=dict(bouts_per_syllable=40, frames_per_bout=4, crop_px=60, embed_python=embed_python))
    cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    r = pipeline.run(config.load(cfg_path))
    # names vs simulated behaviour
    import h5py
    rec = pd.read_csv(r.dir / "recordings.csv", keep_default_na=False)
    truth = pd.read_csv(out / "truth.csv")
    rows = []
    with h5py.File(r.merge / "merged_results.h5", "r") as f:
        for t in rec.itertuples():
            if t.rec_id in f:
                tr = truth[truth.session == t.session].true_behaviour.to_numpy()
                s = f[t.rec_id]["syllable"][()]
                rows.append(pd.DataFrame({"merged_syllable": s[:len(tr)], "true_behaviour": tr[:len(s)]}))
    df = pd.concat(rows)
    final = pd.read_csv(r.dir / "final_syllables.csv").set_index("merged_syllable")
    ct = pd.crosstab(df.merged_syllable, df.true_behaviour, normalize="index").round(2)
    ct.insert(0, "name", [final.movement_name.get(s, "") for s in ct.index])
    ct.insert(1, "frame_pct", [final.frame_pct.get(s, np.nan) for s in ct.index])
    ct.sort_values("frame_pct", ascending=False).to_csv(out / "demo_truth_check.csv")
    log("names vs simulated behaviour (share of each syllable's frames):\n" + ct.sort_values("frame_pct", ascending=False).to_string())
    log(f"demo finished: {r.dir / 'REPORT.md'}")
