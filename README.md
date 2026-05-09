# Assignment 5: CNN Seed Counting + Neural Style Transfer Video

Two deep learning systems built for AgriVision Technologies:
- **Task 1**: CNN regression model for seed counting (replaces clustering/edge detection from A2/A3)
- **Task 2**: Human matting U-Net + VGG19 Neural Style Transfer applied to a self-recorded video

---

## Requirements

All code runs on **Kaggle GPU notebooks** (T4). Dependencies are pre-installed on Kaggle; for local use:

```bash
pip install torch torchvision opencv-python matplotlib pyyaml
```

---

## Kaggle Setup

```python
# In a Kaggle notebook cell:
!git clone https://github.com/itbaans/icv-hw-5.git
%cd /kaggle/working/icv-hw-5
```

Required Kaggle datasets (add via the notebook sidebar → Add Data):

| Dataset slug | Used by |
|---|---|
| `abdullahahmedani/seeds-data` | Task 1 training |
| `laurentmih/aisegmentcom-matting-human-datasets` | Task 2 matting |

Required Kaggle model (add via sidebar → Add Model):

| Model slug | Used by |
|---|---|
| `abdullahahmedani/matting-model` | Task 2 inference / eval |

---

## Task 1 — CNN Seed Counter

All scripts run from `task1_cnn/`.

### 1. Train Model A and Model B

```python
%cd /kaggle/working/icv-hw-5/task1_cnn
!python train_final.py --epochs 100 --patience 15
```

Outputs in `task1_cnn/cnn_outputs/`:
- `model_a_best.pth`, `model_b_best.pth` — checkpoints
- `model_a_curves.png`, `model_b_curves.png` — training curves
- `model_a_gradcam.png`, `model_b_gradcam.png` — GradCAM success/failure grids
- `final_results.csv` — best val MAE and epoch

### 2. Ablation Study (optional — results already in `cnn_outputs/ablation_results.csv`)

```python
%cd /kaggle/working/icv-hw-5/task1_cnn
!python ablation.py
```

### 3. Failure Case Comparison vs. A2/A3

```python
%cd /kaggle/working/icv-hw-5/task1_cnn
!python failure_inference.py \
    --img-dir /kaggle/input/datasets/abdullahahmedani/seeds-data/filtered \
    --ckpt-a  cnn_outputs/model_a_best.pth \
    --ckpt-b  cnn_outputs/model_b_best.pth
```

Outputs: `failure_comparison.csv`, `failure_comparison.png`, `failure_summary.txt`

---

## Task 2 — Human Matting + NST Video Pipeline

All scripts run from `task2_nst_video/`. Config is `task2_nst_video/config.yaml`.

### 1. Train Matting Model (skip if using saved checkpoint)

```python
%cd /kaggle/working/icv-hw-5/task2_nst_video
!python matting/train.py --config config.yaml
```

Checkpoint saved to `outputs/matting_best.pth`. Training log at `outputs/matting_train_log.csv`.

### 2. Evaluate Matting Model (IoU / MAD on 500 test images)

```python
%cd /kaggle/working/icv-hw-5/task2_nst_video
!python eval_matting.py --config config.yaml
```

Outputs: `matting_overlay.png`, `matting_error_grid.png`, `matting_training_curves.png`, `matting_eval_metrics.txt`

### 3. Run NST Video Pipeline

Before running, set your 3 style images in `config.yaml`:

```yaml
style_images:
  - "style/your_style_1.jpg"   # applied to first third of video
  - "style/your_style_2.jpg"   # applied to second third
  - "style/your_style_3.jpg"   # applied to final third
```

Then:

```python
%cd /kaggle/working/icv-hw-5/task2_nst_video
!python video_pipeline.py --config config.yaml
```

Outputs in `outputs/`:
- `stylized_background.mp4` — background stylized, subject natural
- `stylized_subject.mp4` — subject stylized, background natural
- `stylized_full.mp4` — entire frame stylized

### 4. Generate Visualisation Figures (for report)

```python
%cd /kaggle/working/icv-hw-5/task2_nst_video
!python visualize.py --config config.yaml
```

Outputs: `grid.png`, `beta_alpha_ablation.png`, `layer_ablation.png`, `feature_maps.png`

---

## Config Reference

| File | Controls |
|---|---|
| `task1_cnn/config.yaml` | data paths, image size, batch size |
| `task2_nst_video/config.yaml` | matting, NST weights, video paths, style images |

Key NST parameters in `config.yaml`:

```yaml
nst:
  optimizer:        "lbfgs"   # or "adam" (much faster, lower quality)
  iterations:       100       # steps per video frame (frames 1+)
  iterations_first: 500       # steps for frame 0 and static NST figures
  content_weight:   100000.0  # α
  style_weight:     10000000.0 # β  (β/α = 100 → strong stylisation)
  height:           512       # resolution; use 360 to speed up
  max_frames:       0         # 0 = all frames; set e.g. 30 for a quick test
```

---

## Reproduced Results

| Method | MAE | RMSE | Notes |
|---|---|---|---|
| Clustering (A2) | 39.04 | 59.82 | Full 141-image dataset |
| Canny Edge Det. (A3) | 7.67 | 12.77 | Full 141-image dataset |
| CNN Model A | 2.19 | 3.49 | On 20 hard failure cases (GT 50–135) |
| CNN Model B | 2.40 | 3.38 | On 20 hard failure cases (GT 50–135) |
| Matting U-Net | IoU 0.972 | MAD 0.017 | 500 test images, AISegment |

---

## Project Structure

```
icv-hw-5/
├── task1_cnn/
│   ├── config.yaml            # data + training config
│   ├── models.py              # AblationModel, ModelA, ModelB
│   ├── data.py                # SeedDataset + dataloaders
│   ├── train_final.py         # train Model A & B, save curves + GradCAM
│   ├── ablation.py            # one-factor-at-a-time ablation study
│   ├── gradcam.py             # standalone GradCAM visualiser
│   ├── failure_inference.py   # inference on A2/A3 failure cases
│   └── cnn_outputs/           # checkpoints, plots, CSVs
│
├── task2_nst_video/
│   ├── config.yaml            # all Task 2 hyperparameters
│   ├── nst.py                 # VGG19 NST core (Adam / L-BFGS)
│   ├── video_pipeline.py      # matting + NST + compositing pipeline
│   ├── visualize.py           # ablation figures + feature maps
│   ├── eval_matting.py        # matting evaluation on test split
│   ├── matting/
│   │   ├── model.py           # U-Net architecture
│   │   ├── dataset.py         # AISegment dataset loader
│   │   ├── train.py           # matting training loop
│   │   ├── losses.py          # L1 + Dice combined loss
│   │   └── metrics.py         # IoU + MAD
│   ├── style/                 # style paintings (add your own)
│   └── outputs/               # generated videos and figures
│
└── report/
    └── report.tex             # LaTeX report source
```
