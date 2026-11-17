"""Step 2: Keypoint-MoSeq.

Modelling settings = kpm_defaults.yml (the configuration of the reference model) + the `kpm` section of the run
config. Fitting follows the reference procedure exactly:
  PCA -> sigmasq_loc estimate -> init (seed) -> AR-only fit for ar_iters -> load that checkpoint -> set kappa ->
  full-model fit for full_iters (0 = the exported model is the AR-HMM after ar_iters) -> reindex syllables by
  frequency -> extract results.h5.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from .util import log

HERE = Path(__file__).resolve().parent


def _deep_update(d: dict, u: dict):
    for k, v in u.items():
        if isinstance(v, dict) and isinstance(d.get(k), dict):
            _deep_update(d[k], v)
        else:
            d[k] = v


def write_config(cfg, project: Path, bodyparts: list[str]):
    import keypoint_moseq as kpms
    if not (project / "config.yml").exists():
        project.parent.mkdir(parents=True, exist_ok=True)
        kpms.setup_project(str(project), overwrite=project.exists())
    k = cfg.kpm
    conf = yaml.safe_load((project / "config.yml").read_text(encoding="utf-8"))
    base = yaml.safe_load((HERE / "kpm_defaults.yml").read_text(encoding="utf-8"))
    _deep_update(conf, base)
    use = cfg.use_bodyparts or bodyparts
    miss = [b for b in use + [cfg.bodyparts["nose"], cfg.bodyparts["tail_base"]] if b not in bodyparts]
    if miss:
        raise SystemExit(f"body part(s) {miss} not in the DLC files (they have {bodyparts})")
    _deep_update(conf, dict(
        bodyparts=list(bodyparts), use_bodyparts=list(use),
        skeleton=[list(e) for e in (cfg.skeleton or [])],
        anterior_bodyparts=[cfg.bodyparts["nose"]], posterior_bodyparts=[cfg.bodyparts["tail_base"]],
        video_dir=str(cfg.video_dir), fps=int(k.fps), outlier_scale_factor=float(k.outlier_scale_factor),
        ar_hypparams=dict(latent_dim=int(k.latent_dim)), trans_hypparams=dict(kappa=float(k.kappa))))
    _deep_update(conf, dict(k.config_overrides or {}))
    (project / "config.yml").write_text(yaml.safe_dump(conf, sort_keys=False), encoding="utf-8")


def fit(cfg, rec: pd.DataFrame, project: Path) -> Path:
    k = cfg.kpm
    if str(k.device).lower() == "cpu":
        os.environ["JAX_PLATFORM_NAME"] = "cpu"
    else:
        os.environ.setdefault("TF_GPU_ALLOCATOR", "cuda_malloc_async")
    import jax
    jax.config.update("jax_enable_x64", True)
    import keypoint_moseq as kpms
    from jax_moseq.utils.debugging import convert_data_precision
    import matplotlib
    matplotlib.use("Agg")
    log(f"JAX backend: {jax.default_backend()}")

    # ---- keypoints (recording name = short recording ID)
    coords, confs, bodyparts = {}, {}, None
    for i, r in enumerate(rec.itertuples()):
        c, cf, bps = kpms.load_keypoints(r.clean_h5, "deeplabcut")
        if bodyparts is None:
            bodyparts = list(bps)
        elif list(bps) != bodyparts:
            raise SystemExit(f"{r.clean_h5}: body parts {bps} differ from the other files ({bodyparts})")
        (key,) = c.keys()
        coords[r.rec_id], confs[r.rec_id] = c[key], cf[key]
        if i % 200 == 0:
            log(f"  loaded {i + 1}/{len(rec)} recordings")
    write_config(cfg, project, bodyparts)
    conf = lambda: kpms.load_config(str(project))
    log(f"outlier removal (outlier_scale_factor {k.outlier_scale_factor})")
    coords, confs = kpms.outlier_removal(coords, confs, str(project), overwrite=False, **conf())
    short = [n for n, c in coords.items() if len(c) < k.min_frames]
    if short:
        log(f"skipping {len(short)} recordings shorter than {k.min_frames} frames")
        coords = {n: c for n, c in coords.items() if n not in short}
        confs = {n: c for n, c in confs.items() if n not in short}
    data, metadata = kpms.format_data(coords, confs, **conf())
    data = convert_data_precision(data, x64=True)

    log("PCA")
    pca = kpms.fit_pca(**data, **conf())
    kpms.save_pca(pca, str(project))
    kpms.print_dims_to_explain_variance(pca, 0.9)
    kpms.plot_scree(pca, project_dir=str(project))
    kpms.plot_pcs(pca, project_dir=str(project), **conf())

    kpms.update_config(str(project), latent_dim=int(k.latent_dim))
    kpms.update_config(str(project), sigmasq_loc=kpms.estimate_sigmasq_loc(data["Y"], data["mask"],
                                                                           filter_size=conf()["fps"]))
    model = kpms.init_model(data, pca=pca, **conf(), seed=int(k.seed))
    log(f"AR-only fitting, {k.ar_iters} iterations")
    model, model_name = kpms.fit_model(model, data, metadata, str(project), ar_only=True, num_iters=int(k.ar_iters),
                                       jitter=float(k.jitter), seed=int(k.seed))
    try:
        model, data, metadata, it = kpms.load_checkpoint(str(project), model_name, iteration=int(k.ar_iters))
    except AssertionError as e:
        raise SystemExit("AR-only fitting stopped early (usually 'NaNs encountered'): raise kpm.jitter, lower "
                         "kpm.latent_dim, or remove noisy recordings") from e
    model = kpms.update_hypparams(model, kappa=float(k.kappa))
    log(f"full-model fitting, {k.full_iters} further iterations")
    kpms.fit_model(model, data, metadata, str(project), model_name, ar_only=False, start_iter=it,
                   num_iters=it + int(k.full_iters), jitter=float(k.jitter), seed=int(k.seed))
    kpms.reindex_syllables_in_checkpoint(str(project), model_name)

    log("extracting results")
    model, data, metadata, _ = kpms.load_checkpoint(str(project), model_name)
    results = kpms.extract_results(model, metadata, str(project), model_name)
    if k.export_csv:
        kpms.save_results_as_csv(results, str(project), model_name)
    if k.trajectory_plots:
        kpms.generate_trajectory_plots(coords, results, str(project), model_name, **conf())
    return project / model_name


def model_dir(project: Path) -> Path:
    c = [p for p in Path(project).iterdir() if p.is_dir() and (p / "results.h5").exists()] if Path(project).exists() else []
    if not c:
        raise SystemExit(f"no fitted model (results.h5) under {project}")
    return max(c, key=lambda p: p.stat().st_mtime)


def load_results(model: Path) -> dict:
    import h5py
    out = {}
    with h5py.File(Path(model) / "results.h5", "r") as f:
        for name in f:
            g = f[name]
            out[name] = {"syllable": g["syllable"][()].astype(int), "centroid": g["centroid"][()],
                         "heading": g["heading"][()]}
    return out


def syllable_count(model: Path) -> tuple[int, int, int]:
    res = load_results(model)
    s = np.concatenate([v["syllable"] for v in res.values()])
    return len(res), len(s), len(np.unique(s))
