"""Fast tests of the parts that need no Keypoint-MoSeq fit or GPU.  Run:  pytest tests
(The end-to-end check is `kpm-autoname demo <folder>`.)"""
import numpy as np
import pandas as pd
import pytest

from kpm_autoname import config, merge, naming, recordings
from kpm_autoname.clean import clean_table
from kpm_autoname.config import Config, DEFAULTS

BPS = ["Nose", "Tail_Base"]


def dlc_table(n=200, seed=0):
    rng = np.random.default_rng(seed)
    cols = pd.MultiIndex.from_tuples([("s", bp, c) for bp in BPS for c in ("x", "y", "likelihood")],
                                     names=["scorer", "bodyparts", "coords"])
    t = np.arange(n)
    nose = np.c_[100 + 0.5 * t, 100 + rng.normal(0, 0.3, n), np.full(n, 0.95)]
    tail = np.c_[60 + 0.5 * t, 100 + rng.normal(0, 0.3, n), np.full(n, 0.95)]
    return pd.DataFrame(np.c_[nose, tail], columns=cols)


def write_cfg(tmp_path, **extra):
    (tmp_path / "dlc").mkdir(exist_ok=True); (tmp_path / "vid").mkdir(exist_ok=True)
    import yaml
    c = dict(dlc_dir=str(tmp_path / "dlc"), video_dir=str(tmp_path / "vid"), output_dir=str(tmp_path / "out"), cm_per_px=0.1)
    c.update(extra)
    p = tmp_path / "c.yml"
    p.write_text(yaml.safe_dump(c))
    return p


def test_config_defaults_and_unknown_key(tmp_path):
    cfg = config.load(write_cfg(tmp_path))
    assert cfg.kpm.seed == 42 and cfg.kpm.ar_iters == 50 and cfg.merge.max_auc == 0.75
    with pytest.raises(SystemExit):
        config.load(write_cfg(tmp_path, kpm={"sed": 1}))


def test_clean_removes_jump_and_interpolates():
    df = dlc_table()
    df.iloc[100, 0] += 80                                    # one implausible nose jump
    df.iloc[50:53, 2] = 0.1                                  # three low-likelihood frames
    c = Config(DEFAULTS["cleaning"])
    out, info = clean_table(df, c, [("Nose", "Tail_Base")])
    assert info["jump_removed"] >= 1 and info["likelihood_removed"] == 3
    assert abs(out.iloc[100, 0] - (100 + 50)) < 3            # jump replaced by interpolation + smoothing
    assert out.iloc[50:53, [0, 1]].notna().all().all()       # the 3-frame low-likelihood gap is filled


def test_recording_ids_and_video_matching(tmp_path):
    p = write_cfg(tmp_path)
    for sess in ("M1_box", "M2_box"):
        d = tmp_path / "dlc" / "2022" / "07_07_2022" / sess
        d.mkdir(parents=True)
        dlc_table().to_csv(d / "behavCam1DLC_resnet50_x.csv")
        v = tmp_path / "vid" / "2022" / "7_7_2022" / sess   # unpadded date in the video tree
        v.mkdir(parents=True)
        (v / "behavCam1.avi").write_bytes(b"")
    cfg = config.load(p, {"subject_regex": "^([^_]+)"})
    rec = recordings.build(cfg)
    assert list(rec.rec_id) == ["0000-2022_07_07_2022_M1_box_behavCam1", "0001-2022_07_07_2022_M2_box_behavCam1"]
    assert (rec.video != "").all() and list(rec.subject) == ["M1", "M2"]


def row(**kw):
    base = dict(body_length_rel=1.0, nose_lost_frac=0.05, still_frac=0.1, speed_cm_s=5.0, speed_max=10.0,
                turning_bout_frac=0.1, wall_frac=np.nan, corner_frac=np.nan)
    base.update(kw)
    return pd.Series(base)


def test_naming_rules():
    q = pd.Series(dict(speed_cm_s=0.5))
    overall = dict(wall_frac=np.nan, corner_frac=np.nan)
    assert naming.movement_name(row(), q, True, 0.9, False, False, overall)[0] == "right turn"
    assert naming.movement_name(row(), q, True, 0.1, False, False, overall)[0] == "left turn"
    assert naming.movement_name(row(body_length_rel=0.7), q, False, np.nan, True, True, overall)[0] == "rearing"
    assert naming.movement_name(row(still_frac=0.8), q, False, np.nan, False, False, overall)[0] == "pause / stationary"
    assert naming.movement_name(row(speed_cm_s=10.0), q, False, np.nan, False, False, overall)[0] == "locomotion"


def test_cluster_merges_only_indistinguishable_pairs():
    auc = {(0, 1): 0.6, (0, 2): 0.95, (1, 2): 0.97}
    lab = merge.cluster([0, 1, 2], auc, 0.75)
    assert lab[0] == lab[1] != lab[2]
    assert merge.groups_from(lab) == [[0, 1]]
