# kpm-autoname

This tool goes from DeepLabCut keypoints to **named behaviour syllables**:
- it cleans the keypoints;
- fits [Keypoint-MoSeq](https://github.com/dattalab/keypoint-moseq) (KPM);
- merges syllables that look and move alike;
- names each merged syllable from measured movement;
- writes a report that justifies every merge and every name, plus one video per merged syllable to check it by eye.

```
DLC files ──► 1 clean ──► 2 Keypoint-MoSeq ──► 3 merge + name ──► 4 videos ──► 5 report
```

Example of a report entry:

> **Merged syllable 3: rearing** (confidence high, 7.1% of frames)
> - **Original KPM syllables:** 3 (100% of group: rearing); 19 and 23 (rare)
> - **Why these were merged:** original 19 is rare (0.01% of frames); its bouts were classified as this group with mean probability 0.71; …
> - **Closest syllable kept separate:** original 3 vs 1, AUC 0.88 (>= 0.75)
> - **Why this name:** body 0.68x running length and nose untracked 16%
> - **Evidence:** slow (median 1.3 cm/s); shortened body (0.68 x running length) with nose often untracked (16% vs 3% overall)
> - **Video:** `04_videos/syllable_003.mp4` (1060 bouts)

## Install

You need [conda](https://docs.conda.io/en/latest/miniconda.html). Then, from this folder:

```
conda env create -f environment.yml
conda activate kpm-autoname
kpm-autoname demo demo_output        # ~5 min: simulated data through the whole pipeline
```

The demo ends with a table of each named syllable against the simulated behaviour. On the default settings the main syllables come out as left turn, right turn, pause, locomotion and rearing.

**If you already have a Keypoint-MoSeq environment,** you can use it instead: `pip install -e ".[video]"` there.

**GPU:**
- Keypoint-MoSeq runs on the CPU by default. That takes about 15 minutes for 650 recordings (6 M frames).
- The image model in step 3 uses a GPU if PyTorch has CUDA. On the CPU it works but is slow for large datasets.
- To use the GPU anyway, set `merge.embed_python` to the Python of an environment that has CUDA PyTorch and `timm`, for example a DeepLabCut environment. That one Python is then used only for the image features.

## Use

```
kpm-autoname init my_study.yml     # writes a config template
# edit my_study.yml: dlc_dir, video_dir, output_dir, cm_per_px, body-part names
kpm-autoname check my_study.yml    # fast: finds files, pairs videos, checks body parts and the environment
kpm-autoname run my_study.yml      # the whole pipeline
```

**Stopping and restarting:**
- Finished steps are skipped when you start `run` again, so an interrupted run can simply be restarted.
- `--redo-from merge` (or `clean`, `kpm`, `videos`, `report`) reruns that step and all later ones, for example after changing `merge.max_auc`.

**Inputs:**
- **DLC output files** (single animal), `.csv` or `.h5`, anywhere under `dlc_dir`. Which files count is set by `dlc_glob` (default `*DLC*.csv`) and `exclude`.
- **The raw videos.** Don't use copies with keypoints drawn on them, because the image model would see the drawn dots.

A video belongs to a DLC file when both of these hold:
- **Same name:** the video has the DLC file's name up to `DLC_` (DeepLabCut's own naming).
- **Same relative folder:** it sits in the same folder relative to `video_dir`. Numbers match without leading zeros, so `07_07_2022` matches `7_7_2022`.

`check` lists any recording without a video.

**Groups:**
- **Session:** the folder holding each DLC file. `session_level: 2` uses its parent instead.
- **Animal:** `subject_regex`, for example `"^([^_]+)"` turns `M4102F_largesq` into `M4102F`.
- **Where they're used:** sessions keep cross-validation honest, since a session's bouts are never in both training and test data. Animals are used to check that turn directions hold in every animal.

## Outputs (`output_dir`)

| | |
|---|---|
| `REPORT.md` | Summary table, then for every merged syllable: name, original KPM syllables (with their own names), why they were merged, the closest syllable kept separate, why it got its name (measured values) and its video |
| `final_syllables.csv` | The same as a table |
| `recordings.csv` | Every recording: short ID, DLC file, video, session, animal, frame rate |
| `01_dlc_clean/` | Cleaned keypoints (.h5) and `cleaning_manifest.csv` (points removed per file) |
| `02_kpm/` | The Keypoint-MoSeq project; `<model>/results.h5` holds the original syllables |
| `03_merge_and_names/` | `merged_results.h5` (KPM format, merged syllables), `merge_mapping.csv` (original → merged), `pair_auc.csv`, `rare_assignment.csv`, `syllable_descriptions.md/.csv`, `syllable_features.csv`, `syllable_feature_heatmap.png`, `merge_group_consistency.csv`, `unmerged_reference/` (the same description of the original syllables), `cache/crops_check.png` (sample crops: the head should point up) |
| `04_videos/` | `syllable_NNN.mp4` per merged syllable; `bouts_in_videos.csv` gives every bout's recording, frames and time in the video |
| `run_settings.json`, `.state/` | All settings used; when each step finished and its key numbers |

## How it works

**1. Cleaning.** Per file and body part, these points are removed:
- points with likelihood < 0.6;
- `(0, 0)` points;
- frame-to-frame jumps above 8 robust SD;
- skeleton edges longer than 4 robust SD or 2 × their median.

Then gaps of up to 15 frames are interpolated and x/y are smoothed with a centred 15-frame moving average. All thresholds are in the config. Set `cleaning.enabled: false` for keypoints you have already cleaned.

**2. Keypoint-MoSeq.** These defaults reproduce the model the method was developed on: 50 AR-only iterations, kappa 1e6, latent_dim 4, seed 42. `full_iters: 0` means the exported model is the AR-HMM. All Keypoint-MoSeq settings are in `src/kpm_autoname/kpm_defaults.yml` and `kpm.config_overrides`. Recordings get short IDs, so output paths stay short and any folder names work.

**3a. Merging: are two syllables distinguishable?**
- **Bouts:** for every syllable, up to 300 bouts are sampled with 8 frames each.
- **Crops:** each frame is cropped around the animal (about 16.5 cm) and rotated so the head points up.
- **Image features:** each crop is described by [DINOv2](https://github.com/facebookresearch/dinov2) ViT-S/14, loaded through `timm` as `vit_small_patch14_dinov2.lvd142m`. DINOv2 is a self-supervised *image* model trained on 142 M natural images, without behaviour labels. A bout's video features are the mean and SD of its frames' features, reduced to 32 by PCA.
- **Kinematics:** 8 kinematic features are added with equal total weight: speed, still fraction, turn rate (absolute and signed), net turn, body length, nose lost, and duration.
- **Comparing pairs:** for each pair of common syllables (≥ 0.5% of frames), a logistic regression cross-validated across sessions tries to tell their bouts apart. Pairs it separates with **AUC < 0.75** are merged (average linkage).
- **Rare syllables:** they join the group their bouts are classified as (mean probability ≥ 0.5), otherwise they stay separate.

This decides only whether syllables are *distinguishable*. Grouping distinguishable syllables into coarser categories is left to you; the report gives the evidence.

**3b. Naming.** These features are measured per merged syllable and ranked among the syllables:
- speed (cm/s) and signed turn rate (+ = right, with the camera above and not mirrored);
- body length relative to running, and nose visibility;
- location, if labels are given.

Fixed rules, in priority order, turn the measurements into a name:
1. **posture unreliable:** body length < 0.3 × running length, so posture isn't used;
2. **right turn / left turn / scanning:** fastest-turning quarter; a direction needs ≥ 70% of turning bouts going that way;
3. **rearing:** shortened body and nose often untracked;
4. **pause / stationary or pause at wall:** ≥ 60% of frames below 1 cm/s;
5. **locomotion:** fastest quarter;
6. **slow walking;**
7. **exploratory walking,** optionally veering left or right.

The full rules are in [`naming.py`](src/kpm_autoname/naming.py). No language model is involved; every name comes with the numbers that triggered it.

**4. Videos.** From every recording, the 3 longest bouts of each merged syllable are taken from the raw video, with the cleaned keypoints and a caption drawn on.

## Location labels (optional)

To get names like "pause at wall", give a CSV with one row per frame and a column for the area (wall / corner / centre). The `columns` mapping names those columns. Under `match`, each key is a column of your file and each value is the recording field it must equal. The recording fields are:
- `rec_id`;
- `session`, `subject` and `video_stem` (the video name, e.g. `behavCam1`);
- `level1`, `level2`, …: folder names above the DLC file, where 1 is the folder holding it.

`values` gives the strings your file uses for wall, corner and centre.

```yaml
naming:
  location_labels:
    file: area_labels.csv
    columns: {frame: frame, area: area, d_wall_px: dist_wall, d_corner_px: dist_corner}
    match: {date: level2, session: session, camera: video_stem}
    values: {wall: boundary, corner: corner, centre: center}
```

## Good to know

- **Refits differ.** Keypoint-MoSeq is stochastic. Refitting the same data with the same seed on another machine or operating system gives a different, equally valid segmentation. In our data, frame-level agreement between such refits matched the agreement between seeds (adjusted Rand index ~0.3–0.5). Syllable *numbers* are never comparable between fits; the names and the report are.
- **Check `camera_fps`.** It is the true camera frame rate. Some acquisition software writes a wrong rate into the video files, and speeds and durations depend on it.
- **Check the turn direction.** Turn direction assumes the camera looks down on the arena and the video is not mirrored.
- **Watch the videos** before relying on a name, especially for low-confidence and rare syllables.

## Tests

```
pytest tests             # fast unit tests
kpm-autoname demo out    # end-to-end on simulated data
```

## Credits

Built on [Keypoint-MoSeq](https://github.com/dattalab/keypoint-moseq) (Weinreb et al., 2024, *Nature Methods*), [DeepLabCut](https://github.com/DeepLabCut/DeepLabCut) and [DINOv2](https://github.com/facebookresearch/dinov2) (Oquab et al., 2023) via [timm](https://github.com/huggingface/pytorch-image-models). Please cite them if you use this tool.

## License

MIT, see [LICENSE](LICENSE).
