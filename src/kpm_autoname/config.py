"""Run configuration: one YAML file per dataset. Every setting has a default except the four paths/values marked
REQUIRED in config_template.yml."""
from __future__ import annotations

import copy
from pathlib import Path

import yaml

DEFAULTS = {
    # ---- required
    "dlc_dir": None,
    "video_dir": None,
    "output_dir": None,
    "cm_per_px": None,
    # ---- input files
    "dlc_glob": "*DLC*.csv",
    "exclude": ["_filtered", "_labeled", "_meta", "_full", "concatenat"],
    "video_extensions": [".avi", ".mp4", ".mov", ".mkv"],
    "camera_fps": None,
    "session_level": 1,
    "subject_regex": None,
    "bodyparts": {"nose": "Nose", "tail_base": "Tail_Base", "head": None, "mid_body": None},
    "use_bodyparts": None,
    "skeleton": None,
    # ---- 1 clean
    "cleaning": {
        "enabled": True,
        "likelihood": 0.6,
        "velocity_mad": 8.0,
        "shape_mad": 4.0,
        "shape_max_factor": 2.0,
        "remove_zero_xy": True,
        "interpolation_limit": 15,
        "moving_average": 15,
    },
    # ---- 2 kpm
    "kpm": {
        "seed": 42,
        "kappa": 1.0e6,
        "latent_dim": 4,
        "outlier_scale_factor": 6.0,
        "fps": 20,
        "ar_iters": 50,
        "full_iters": 0,
        "jitter": 1.0e-2,
        "min_frames": 5,
        "device": "cpu",
        "trajectory_plots": True,
        "export_csv": True,
        "config_overrides": {},
    },
    # ---- 3 merge
    "merge": {
        "max_auc": 0.75,
        "rare_pct": 0.5,
        "rare_min_prob": 0.5,
        "bouts_per_syllable": 300,
        "frames_per_bout": 8,
        "crop_px": None,
        "crop_cm": 16.5,
        "features": "video+kinematic",
        "embed_python": None,
        "device": "auto",
        "compare_to": None,
    },
    # ---- 3 naming
    "naming": {
        "likelihood": 0.6,
        "location_labels": None,
    },
    # ---- 4 videos
    "videos": {
        "bouts_per_recording": 3,
        "caption_scale": 0.3,
        "output_fps": None,
    },
}

REQUIRED = ["dlc_dir", "video_dir", "output_dir", "cm_per_px"]


class Config(dict):
    """dict with attribute access for nested sections (cfg.kpm.seed)."""

    def __getattr__(self, k):
        try:
            v = self[k]
        except KeyError as e:
            raise AttributeError(k) from e
        return Config(v) if isinstance(v, dict) and not isinstance(v, Config) else v


def _merge(base: dict, new: dict, where: str = ""):
    for k, v in (new or {}).items():
        if k not in base:
            raise SystemExit(f"config: unknown setting '{where}{k}' (check spelling; see config_template.yml)")
        if isinstance(base[k], dict) and isinstance(v, dict) and k not in ("bodyparts", "config_overrides"):
            _merge(base[k], v, f"{where}{k}.")
        elif isinstance(base[k], dict) and isinstance(v, dict):
            base[k].update(v)
        else:
            base[k] = v


def load(path: Path, overrides: dict | None = None) -> Config:
    path = Path(path)
    if not path.exists():
        raise SystemExit(f"config file not found: {path}  (create one with: kpm-autoname init {path})")
    user = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    cfg = copy.deepcopy(DEFAULTS)
    _merge(cfg, user)
    _merge(cfg, overrides or {})
    missing = [k for k in REQUIRED if cfg[k] in (None, "")]
    if missing:
        raise SystemExit(f"config {path}: required setting(s) missing: {', '.join(missing)}")
    for k in ("dlc_dir", "video_dir", "output_dir"):
        cfg[k] = Path(cfg[k])
    for k in ("dlc_dir", "video_dir"):
        if not cfg[k].exists():
            raise SystemExit(f"config: {k} not found: {cfg[k]}")
    if not cfg["bodyparts"].get("nose") or not cfg["bodyparts"].get("tail_base"):
        raise SystemExit("config: bodyparts.nose and bodyparts.tail_base are required (DLC body-part names)")
    if cfg["merge"]["features"] not in ("video+kinematic", "video", "kinematic"):
        raise SystemExit("config: merge.features must be video+kinematic, video or kinematic")
    cfg["_config_file"] = str(path.resolve())
    return Config(cfg)


def crop_px(cfg) -> int:
    m = cfg.merge
    return int(m.crop_px) if m.crop_px else int(round(m.crop_cm / cfg.cm_per_px))
