"""Command line: kpm-autoname {init, check, run, demo}."""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

from . import __version__

HERE = Path(__file__).resolve().parent


def cmd_init(a):
    dst = Path(a.config)
    if dst.exists() and not a.force:
        raise SystemExit(f"{dst} exists (use --force to overwrite)")
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(HERE / "config_template.yml", dst)
    print(f"wrote {dst}. Fill in the REQUIRED settings, then run:  kpm-autoname check {dst}")


def cmd_check(a):
    from . import config, recordings
    from .util import bodyparts_of, read_dlc
    cfg = config.load(a.config)
    ok = True
    print(f"config {a.config}: OK")
    rec = recordings.build(cfg)
    print(recordings.summary(rec))
    df = read_dlc(rec.dlc_file.iloc[0])
    bps = bodyparts_of(df)
    print(f"body parts in the DLC files: {bps}")
    for role, name in cfg.bodyparts.items():
        if name and name not in bps:
            print(f"  PROBLEM: bodyparts.{role} = {name!r} is not one of them"); ok = False
    for e in cfg.skeleton or []:
        if any(b not in bps for b in e):
            print(f"  PROBLEM: skeleton edge {e} uses an unknown body part"); ok = False
    if not cfg.skeleton:
        print("  note: no skeleton set: the skeleton-length cleaning filter is skipped")
    rec = recordings.add_fps(cfg, rec)
    print(f"frame rate used for speeds: {sorted(set(rec.fps.round(2)))} fps"
          + ("" if cfg.camera_fps else " (from the video files; set camera_fps if the files' rate is wrong)"))
    print(f"egocentric crop: {config.crop_px(cfg)} px = {config.crop_px(cfg) * cfg.cm_per_px:.1f} cm")
    ll = cfg.naming.location_labels
    if ll and ll.get("file") and not Path(ll["file"]).exists():
        print(f"  PROBLEM: location label file not found: {ll['file']}"); ok = False
    try:
        import keypoint_moseq  # noqa: F401
        print(f"keypoint_moseq {keypoint_moseq.__version__}: OK")
    except Exception as e:
        print(f"  PROBLEM: keypoint_moseq cannot be imported: {e}"); ok = False
    if cfg.merge.features != "kinematic":
        py = cfg.merge.embed_python or sys.executable
        r = subprocess.run([str(py), "-c", "import torch, timm, cv2; print(torch.__version__, timm.__version__, "
                            "'cuda' if torch.cuda.is_available() else 'cpu')"], capture_output=True, text=True)
        if r.returncode:
            print(f"  PROBLEM: torch / timm not importable in {py}:\n{r.stderr.strip().splitlines()[-1] if r.stderr else ''}"); ok = False
        else:
            t, tm, dev = r.stdout.split()
            print(f"image model: torch {t}, timm {tm}, device {dev}"
                  + ("  (no GPU: embedding works on CPU but is slow for large datasets)" if dev == "cpu" else ""))
    print("\nready: kpm-autoname run " + str(a.config) if ok else "\nfix the PROBLEM lines above first")
    if not ok:
        raise SystemExit(1)


def cmd_run(a):
    from . import config, pipeline
    cfg = config.load(a.config)
    pipeline.run(cfg, a.redo_from, a.stop_after)


def cmd_demo(a):
    from . import demo
    demo.main(Path(a.out_dir), a.embed_python)


def main(argv=None):
    ap = argparse.ArgumentParser(prog="kpm-autoname", description="DeepLabCut output -> Keypoint-MoSeq syllables -> "
                                 "merged, named syllables with justification and one video per syllable.")
    ap.add_argument("--version", action="version", version=__version__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("init", help="write a config template to fill in")
    p.add_argument("config"); p.add_argument("--force", action="store_true"); p.set_defaults(f=cmd_init)
    p = sub.add_parser("check", help="check a config, the input files and the environment (fast, writes nothing)")
    p.add_argument("config"); p.set_defaults(f=cmd_check)
    p = sub.add_parser("run", help="run the pipeline (finished steps are skipped)")
    p.add_argument("config")
    p.add_argument("--redo-from", choices=["clean", "kpm", "merge", "videos", "report"], help="rerun this step and all later ones")
    p.add_argument("--stop-after", choices=["clean", "kpm", "merge", "videos", "report"])
    p.set_defaults(f=cmd_run)
    p = sub.add_parser("demo", help="make a small synthetic dataset and run the whole pipeline on it (~5-10 min)")
    p.add_argument("out_dir"); p.add_argument("--embed-python", help="Python with torch + timm, if not this one")
    p.set_defaults(f=cmd_demo)
    a = ap.parse_args(argv)
    a.f(a)


if __name__ == "__main__":
    main()
