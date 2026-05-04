**Assignment 5**

# 1\. Business Context

AgriVision Technologies has grown from a hand-tuned image-processing prototype (Assignment 2) into a feature-engineered system using edge detection (Assignment 3). For the final phase, the company wants to (a) replace the hand-coded pipeline with a deep learning solution, and (b) generate branded, on-brand video content for the Series A pitch using neural style transfer applied to founder-recorded footage. Both goals are connected by one technical idea: convolutional feature learning.

# 2\. Prerequisites

This assignment requires verified outputs from Assignments 2 and 3. You must:

- Have your Assignment 2 intermediate_outputs/ folder available and unmodified.
- Have your Assignment 3 edge_outputs/ folder available and unmodified.
- Use the evaluation script from Assignment 2 (baseline_code/evaluate.py) so all three methods (clustering, edge detection, CNN) are scored identically.
- Submit your verification codes from Assignments 2 and 3.

# 3\. Datasets

## 3.1 Seeds Dataset (from Assignment 2)

Reuse the same seed images from Assignment 2. Train Task 1 on the preprocessed images (assignment2/intermediate_outputs/preprocessed_images/filtered/) with labels from segmentation/labeled_components.pkl. Do not regenerate or alter these files.

## 3.2 AISegment Human Matting Dataset (new, for Task 2)

AISegment.com Matting Human Dataset is a large collection of half-length human portraits paired with high-quality alpha mattes (~34,000 image/matte pairs at 600×800). You will use it to train a human matting model that produces a per-pixel alpha matte for any frame of your own video. That matte is what makes selective style transfer possible - stylize the background while leaving you natural, or vice versa.

**Source:** <https://www.kaggle.com/datasets/laurentmih/aisegmentcom-matting-human-datasets>

Working subset: 5,000 image/matte pairs for training, 500 for validation, 500 for test (~6,000 pairs total). Document any subsampling strategy in your config.

## 3.3 Style Images for Task 2

Use public-domain artwork only - not copyrighted or AI-generated images. Recommended sources:

- WikiArt (filter for public-domain works): <https://www.wikiart.org/>
- Met Museum Open Access: [https://www.metmuseum.org/art/collection/search](https://www.metmuseum.org/art/collection/search#!?showOnly=openAccess)
- Art Institute of Chicago (CC0): <https://www.artic.edu/collection?is_public_domain=1>

## 3.4 Self-Recorded Video (Task 2)

You must record an original video of yourself. Stock footage, AI-generated video, video downloaded from the internet, or video of someone else is not acceptable and will be treated as a missing deliverable.

- Length: at least 10 seconds, no longer than 30 seconds.

# 4\. Task 1 - CNN From Scratch (Seeds Dataset)

## Goal

Train a CNN from scratch (no pretrained weights) to solve the same seed task you solved with clustering in Assignment 1 and edge detection in Assignment 2. The network must be trained end-to-end, evaluated using Assignment 1's evaluator, and compared head-to-head against your two earlier methods.

## Required Architectures

Build and compare two CNN architectures:

- **Model A - Baseline CNN:** 3 conv blocks (Conv2D → BatchNorm → ReLU → MaxPool), filter progression 32 → 64 → 128, Global Average Pooling, Dense → Softmax. Total parameters ≤ 1.5M.
- **Model B - Deeper/Regularized CNN:** at least 4 conv blocks with dropout (0.2-0.5), L2 weight decay, optional residual connection. Show that Model B improves over Model A or explain why it overfits.

## Training Requirements

- Loss: categorical cross-entropy.
- Optimizer: compare Adam vs SGD-with-momentum on Model A; pick the winner for Model B.
- Augmentation: rotation ±30°, horizontal/vertical flip, brightness ±20%, zoom ±10%.
- Schedule: minimum 30 epochs with early stopping (patience ≥ 5 on validation loss).
- Reproducibility: fix all random seeds and record them in config.yaml.

## Required Comparisons

Use calculate_metrics from Assignment 2 to produce a unified comparison table:

| **Method**          | **Accuracy** | **MAE** | **RMSE** | **Failure Cases Fixed** |
| ------------------- | ------------ | ------- | -------- | ----------------------- |
| Clustering (A1)     | -            | -       | -        | reference               |
| Edge Detection (A2) | -            | -       | -        | X / N                   |
| CNN Model A         | -            | -       | -        | X / N                   |
| CNN Model B         | -            | -       | -        | X / N                   |

_"Failure cases fixed" = number of images from Assignment 2's failure_cases.json that your CNN now handles correctly. This is the key business metric._

## Deliverables

- Source code: train.py, models.py, data.py, evaluate.py
- Saved weights for Model A and Model B
- Training logs (TensorBoard or CSV)
- Comparison table CSV
- Plots: loss curves, accuracy curves, confusion matrix, predictions on failure cases
- config.yaml with all hyperparameters and the random seed

# 5\. Task 2 - Neural Style Transfer on a Video You Record

## Goal

Implement Neural Style Transfer (Gatys, Ecker, Bethge, 2015) using a pretrained VGG19, train a human matting model on the AISegment dataset, and use the two together to produce a stylized video starring you. The matting model is what makes the result interesting: instead of stylizing every pixel uniformly, you can stylize only the background (subject stays natural) or only the subject (background stays natural) - exactly the kind of branded marketing content AgriVision wants for the Series A deck.

## Method Specification

### A. Human Matting Model

- Architecture: a simple U-Net (encoder-decoder with skip connections) or a MobileNet-V2 backbone with a lightweight decoder. You may NOT use a pre-trained matting/segmentation model - train your own on AISegment.
- Input: RGB frame, resized to 256×256 or 320×320.
- Output: single-channel alpha matte ∈ \[0, 1\].
- Loss: a combination of L1 on the alpha matte and a binary cross-entropy or Dice term. Document your weighting choice.
- Training: minimum 20 epochs, Adam optimizer, augmentation (flip, color jitter, random crop).
- Target: IoU ≥ 0.85 on the AISegment validation split. If you fall short, explain why and what you tried.

### B. Neural Style Transfer (Gatys et al.)

- Backbone: pretrained VGG19, frozen, eval mode.
- Content layer: a mid-level layer such as conv4_2 (relu4_2).
- Style layers: five layers across depths, e.g. relu1_1, relu2_1, relu3_1, relu4_1, relu5_1.
- Style representation: Gram matrix of feature activations, normalized.
- Optimization: optimize the pixels of the generated image (initialized from the content frame) using L-BFGS or Adam.
- Loss: L_total = α·L_content + β·L_style. Sweep at least three β/α ratios (e.g. 1e3, 1e5, 1e7).
- Temporal consistency (recommended): when stylizing frame t, initialize the optimizer from the stylized frame t−1 instead of from the raw content frame. This dramatically reduces flicker between frames at zero extra parameters.

### C. Video Compositing Pipeline

- Decode your input video into frames (e.g. with ffmpeg or OpenCV).
- For each frame: run the matting model to obtain alpha α_t, run NST to obtain the stylized version S_t.
- Composite per pixel: O_t = α_t · F_t + (1 − α_t) · S_t for "background-stylized", or O_t = α_t · S_t + (1 − α_t) · F_t for "subject-stylized", where F_t is the original frame.
- Re-encode the composited frames back into a video at the original frame rate.

## Required Outputs

- **Stylized image grid (NST sanity check):** 5 content images (extract them from your video) × 3 style images = 15 results. This validates your NST implementation before you commit to running it on every frame.
- **β/α ablation:** the same content+style pair rendered at three different style weights, side by side.
- **Layer ablation:** render once using only shallow style layers, once using only deep ones. Discuss what each captures.
- **Matting visualization:** 5 sample frames from your video shown alongside their predicted alpha mattes and the resulting cutout. Report your matting model's IoU on the AISegment test split.
- **Final stylized videos (3 variants), each at least 10 seconds:**
  - Variant 1 - Background stylized, subject natural.
  - Variant 2 - Subject stylized, background natural.
  - Variant 3 - Whole frame stylized (no matting). Used as a baseline to show why matting matters.
- **Feature-map visualization:** plot 8 channels each from one shallow and one deep VGG19 layer, applied to one frame of your video and to one seed image from Task 1. Connect this to what your Task 1 CNN was learning.
- **Branded poster:** one final 1024×1024 stylized still - ideally a cherry-picked frame from your stylized video - suitable for marketing.

## Deliverables

- Source code: matting/train.py, matting/model.py, nst.py, video_pipeline.py, config.yaml
- Saved weights for the matting model
- input_video.mp4 - your raw recording, unmodified
- content/ folder with 5 frames extracted from your video
- style/ folder with 3 paintings + a README.md citing the source and license of each
- outputs/ folder containing:
  - grid.png, beta_alpha_ablation.png, layer_ablation.png, matting_overlay.png, feature_maps.png, branded_poster.png
  - stylized_background.mp4, stylized_subject.mp4, stylized_full.mp4

# 6\. Submission

Submit a single zip file: assignment5\_&lt;erp&gt;.zip with the following structure:

- README.md (how to run, environment, hardware used)
- environment.yml or requirements.txt
- submission_metadata.json (with verification codes from Assignments 1 and 2)
- task1_cnn/ - code, weights, logs, cnn_outputs/
- task2_nst_video/ - matting/, nst code, input_video.mp4, content/, style/, outputs/
- report/report.pdf - 10 to 14 pages

# 7\. Report Structure

PDF, 10-14 pages, single column, 11pt, 1.5 spacing. Required sections:

- **Executive Summary** - half a page; which method wins on the seed task, by how much, what it costs.
- **Background & Linkage** - recap A1 and A2 results; state the gap you intend to close.
- **Task 1 results** - architecture diagrams, training curves, comparison table, failure-case analysis, ablations.
- **Task 2 results** - matting model performance (IoU, qualitative samples), NST β/α and layer ablations, video pipeline description, sample frames from each of the three stylized variants, feature-map figures connecting back to Task 1.
- **Cross-method comparison** - one unified table for the seed task: Clustering → Edge → CNN-from-scratch, with accuracy, latency, model size, training cost, failure cases fixed.
- **Cost & deployment discussion** - honest take on GPU/CPU latency for the matting + NST video pipeline, memory, real-time feasibility. Recommend what AgriVision should ship.
- **Limitations & future work.**
- **Reflection** - what surprised you, what you'd redo.