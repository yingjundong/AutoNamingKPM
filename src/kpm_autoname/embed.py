"""Image features of sampled bout frames: egocentric crop -> DINOv2 ViT-S/14.

Self-contained (numpy, pandas, opencv, torch, timm only), so it can run in another Python environment than the rest
of the pipeline (merge.embed_python in the config), e.g. one with a CUDA build of PyTorch.

Every sampled frame is cropped around the KPM centroid (crop_px square), rotated so the KPM heading points up (position
and direction in the arena do not matter), resized to 224 x 224 and passed through DINOv2 ViT-S/14
(timm 'vit_small_patch14_dinov2.lvd142m': a self-supervised image model trained on 142 M natural images, no behaviour
labels). Bout embedding = mean and SD of its frames' 384-d features.

  python embed.py --cache <run>/03_merge_and_names/cache --crop-px 112 [--device auto|cuda|cpu]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

MODEL = "vit_small_patch14_dinov2.lvd142m"


def crop(frame, x, y, theta, crop_px):
    import cv2
    M = cv2.getRotationMatrix2D((float(x), float(y)), float(np.degrees(theta) + 90.0), 1.0)
    M[0, 2] += crop_px / 2 - x; M[1, 2] += crop_px / 2 - y
    return cv2.warpAffine(frame, M, (crop_px, crop_px), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)


def embed(cache: Path, crop_px: int, device: str = "auto", batch: int = 256):
    import cv2
    import timm
    import torch
    sm = pd.read_csv(cache / "samples.csv", keep_default_na=False)
    z = np.load(cache / "samples.npz")
    fr, cx, cy, hd = z["frames"], z["cx"], z["cy"], z["heading"]
    dev = ("cuda" if torch.cuda.is_available() else "cpu") if device == "auto" else device
    dt = torch.float16 if dev == "cuda" else torch.float32
    model = timm.create_model(MODEL, pretrained=True, num_classes=0, img_size=224).eval().to(dev).to(dt)
    mean = torch.tensor([0.485, 0.456, 0.406], device=dev).view(1, 3, 1, 1).to(dt)
    std = torch.tensor([0.229, 0.224, 0.225], device=dev).view(1, 3, 1, 1).to(dt)
    K = fr.shape[1]
    emb = np.full((len(sm), K, model.num_features), np.nan, np.float32)
    check, buf, where = {}, [], []

    def flush():
        if not buf:
            return
        t = torch.from_numpy(np.stack(buf)).to(dev).to(dt).permute(0, 3, 1, 2) / 255.0
        t = torch.nn.functional.interpolate(t, size=224, mode="bilinear", align_corners=False)
        with torch.no_grad():
            e = model((t - mean) / std).float().cpu().numpy()
        for (j, k), v in zip(where, e):
            emb[j, k] = v
        buf.clear(); where.clear()

    groups = sm.groupby("video").indices
    print(f"embedding {len(sm)} bouts from {len(groups)} videos on {dev}", flush=True)
    for n, (video, rows) in enumerate(groups.items()):
        need = {}
        for j in rows:
            for k in range(K):
                if fr[j, k] >= 0:
                    need.setdefault(int(fr[j, k]), []).append((j, k))
        if not need:
            continue
        cap = cv2.VideoCapture(video)
        fi, last = 0, max(need)
        while fi <= last:
            ok, img = cap.read()
            if not ok:
                break
            if fi in need:
                rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                for j, k in need[fi]:
                    c = crop(rgb, cx[j, k], cy[j, k], hd[j, k], crop_px)
                    buf.append(c); where.append((j, k))
                    s = int(sm.syllable.iloc[j])
                    if k == 0 and len(check.setdefault(s, [])) < 6:
                        check[s].append(c)
                if len(buf) >= batch:
                    flush()
            fi += 1
        cap.release()
        if n % 50 == 0:
            print(f"  embed {n}/{len(groups)} videos", flush=True)
    flush()
    ok = ~np.isnan(emb[:, :, 0])
    mean_e = np.where(ok[..., None], emb, 0).sum(1) / np.maximum(ok.sum(1), 1)[:, None]
    sd_e = np.sqrt(np.where(ok[..., None], (emb - mean_e[:, None]) ** 2, 0).sum(1) / np.maximum(ok.sum(1), 1)[:, None])
    np.savez(cache / "embeddings.npz", mean=mean_e, sd=sd_e, n_frames=ok.sum(1))
    # visual check: 6 crops per syllable; the animal's head should point UP in every crop
    syls = sorted(check)
    tile = 96
    sheet = np.full((max(len(syls), 1) * tile, 6 * tile + 60, 3), 255, np.uint8)
    for r, s in enumerate(syls):
        cv2.putText(sheet, str(s), (5, r * tile + 55), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
        for c, im in enumerate(check[s]):
            sheet[r * tile:(r + 1) * tile, 60 + c * tile:60 + (c + 1) * tile] = cv2.resize(im, (tile, tile))
    cv2.imwrite(str(cache / "crops_check.png"), cv2.cvtColor(sheet, cv2.COLOR_RGB2BGR))
    print(f"embed: {int(ok.any(1).sum())}/{len(sm)} bouts embedded on {dev}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cache", type=Path, required=True)
    ap.add_argument("--crop-px", type=int, required=True)
    ap.add_argument("--device", default="auto")
    a = ap.parse_args()
    embed(a.cache, a.crop_px, a.device)
