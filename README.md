# R3D-AD: Reconstruction via Diffusion for 3D Anomaly Detection

### [Website](https://zhouzheyuan.github.io/r3d-ad) | [Paper](https://arxiv.org/abs/2407.10862)

> [**R3D-AD: Reconstruction via Diffusion for 3D Anomaly Detection**](https://doi.org/10.1007/978-3-031-72764-1_6),
> Zheyuan Zhou∗, Le Wang∗, Naiyu Fang, Zili Wang, Lemiao Qiu, Shuyou Zhang
> **ECCV 2024**

---

## Table of Contents

- [Overview](#overview)
- [Method Overview](#method-overview)
  - [High-Level Pipeline](#high-level-pipeline)
  - [Encoder: PointNet](#encoder-pointnet)
  - [Decoder: Diffusion Model](#decoder-diffusion-model)
  - [Training Objective](#training-objective)
  - [Anomaly Detection at Inference](#anomaly-detection-at-inference)
  - [Anomaly Scoring](#anomaly-scoring)
- [Evaluation Metrics](#evaluation-metrics)
  - [Training Log Output](#training-log-output)
  - [Metric Definitions](#metric-definitions)
  - [Scoring Methods](#scoring-methods)
  - [Interpreting Results](#interpreting-results)
- [Pseudo Anomaly Generation](#pseudo-anomaly-generation)
  - [Overview](#pseudo-anomaly-overview)
  - [Method 1: Random Patch (Localized Bulge/Dent)](#method-1-random-patch-localized-bulgedent)
  - [Method 2: Random Translate (Scattered Micro-Displacements)](#method-2-random-translate-scattered-micro-displacements)
  - [How Pseudo Anomalies Connect to Training](#how-pseudo-anomalies-connect-to-training)
- [Integrating Your Own Pseudo Anomaly Synthesis Module](#integrating-your-own-pseudo-anomaly-synthesis-module)
  - [Step 1: Implement Your Anomaly Function](#step-1-implement-your-anomaly-function)
  - [Step 2: Register in the Dataset Loader](#step-2-register-in-the-dataset-loader)
  - [Step 3: Connect to the Training Loop](#step-3-connect-to-the-training-loop)
  - [Design Guidelines](#design-guidelines)
- [Repository Structure](#repository-structure)
- [Installation](#installation)
- [Datasets](#datasets)
  - [Anomaly-ShapeNet](#anomaly-shapenet)
- [Training and Testing](#training-and-testing)
- [Visualization](#visualization)
- [Configuration](#configuration)
- [Acknowledgement](#acknowledgement)
- [Citation](#citation)

---

## Overview

R3D-AD is a **reconstruction-based 3D anomaly detection** method that leverages **diffusion models** to reconstruct 3D point clouds. The core idea is straightforward: a model trained exclusively on *normal* (defect-free) point clouds will reconstruct normal samples faithfully but will fail to reconstruct anomalous regions accurately. By measuring the reconstruction error between the input and the output, both object-level (is this object anomalous?) and point-level (which points are anomalous?) anomaly detection can be performed.

The method is evaluated on the **Anomaly-ShapeNet** benchmark comprising 40 object categories.

---

## Method Overview

### High-Level Pipeline

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                         TRAINING (Normal Data Only)                          │
│                                                                              │
│   Point Cloud ──► PointNet Encoder ──► Latent Code z ──► Diffusion Decoder   │
│   (B, 2048, 3)       (B, 256)              │              (B, 2048, 3)       │
│                                             │                                │
│                                      MSE Loss(ε̂, ε)                         │
└──────────────────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────────────────┐
│                             INFERENCE                                        │
│                                                                              │
│   Input PC ──► Encoder ──► z ──► Diffusion Decoder ──► Reconstructed PC      │
│       │                                                       │              │
│       └────────────── Distance Comparison ────────────────────┘              │
│                              │                                               │
│                 ┌────────────┴────────────┐                                  │
│                 │                         │                                  │
│         Image-Level Score         Point-Level Score                          │
│        (Is object anomalous?)    (Which points are                           │
│                                   anomalous?)                                │
└──────────────────────────────────────────────────────────────────────────────┘
```

### Encoder: PointNet

The encoder (`models/encoders/pointnet.py`) is a **PointNet** architecture that maps a raw 3D point cloud to a compact latent representation:

| Layer | Operation | Output Shape |
|-------|-----------|-------------|
| Input | Point cloud | (B, N, 3) |
| Conv1d + BN + ReLU | 3 → 128 | (B, 128, N) |
| Conv1d + BN + ReLU | 128 → 128 | (B, 128, N) |
| Conv1d + BN + ReLU | 128 → 256 | (B, 256, N) |
| Conv1d + BN | 256 → 512 | (B, 512, N) |
| Max Pooling | Global | (B, 512) |
| FC + BN + ReLU → FC + BN + ReLU → FC | 512 → 256 → 128 → **z_dim** | (B, 256) |

The encoder produces both a **mean** and **logvariance** vector (VAE-style), but in practice only the mean is used as the deterministic latent code `z` (256-dimensional).

An alternative `PointNetEncoderTNet` variant adds **Spatial Transformer Networks** (STN3d, STNkd) for rotation invariance.

### Decoder: Diffusion Model

The decoder (`models/diffusion.py`) is a **Denoising Diffusion Probabilistic Model (DDPM)** that reconstructs the point cloud from the latent code:

**Variance Schedule:**
- Type: Linear
- Steps: 200
- β₁ = 1e-4, β_T = 0.05
- Produces noise schedule: `α_t = 1 - β_t`, `ᾱ_t = Π α_i`

**PointwiseNet** (the denoising network):
- 6 layers of `ConcatSquashLinear` blocks
- Each layer receives: noisy point coordinates, time embedding `[β, sin(β), cos(β)]`, and latent code `z`
- Architecture: 3 → 128 → 256 → 512 → 256 → 128 → 3
- Residual connection from input to output
- `ConcatSquashLinear` uses a gating mechanism: `output = Linear(x) * σ(gate(ctx)) + bias(ctx)`

**Forward Diffusion (Training):**
```
x_t = √(ᾱ_t) · x_0 + √(1 - ᾱ_t) · ε      where ε ~ N(0, I)
```

**Reverse Diffusion (Sampling):**
```
x_{t-1} = (1/√α_t) · (x_t - (1-α_t)/√(1-ᾱ_t) · ε_θ(x_t, t, z)) + σ_t · z
```
Starting from pure Gaussian noise `x_T ~ N(0, I)`, the model iteratively denoises for 200 steps conditioned on the latent code to produce the final reconstruction.

### Training Objective

The standard training loss is the **noise prediction MSE**:

```
L = E_{t, x_0, ε} [ ||ε - ε_θ(x_t, t, z)||² ]
```

Where `ε_θ` is the PointwiseNet predicting the noise added at timestep `t`, and `z = Encoder(x_0)`.

When training with pseudo anomalies (using the `--rel` flag and `x_raw`), a **reconstruction loss** variant is used instead:

```
L = E_{t, x_0, ε} [ ||x̂_0 - x_raw||² ]
```

Where `x̂_0` is the estimated clean point cloud derived from the diffusion prediction, and `x_raw` is the original clean point cloud before augmentation. This trains the model to explicitly map augmented (pseudo-anomalous) inputs back to their clean versions.

### Anomaly Detection at Inference

1. **Encode** the input point cloud to get latent code `z`
2. **Decode** `z` via the full reverse diffusion process (200 steps) to get the reconstructed point cloud
3. **Compare** input vs. reconstruction to compute anomaly scores

### Anomaly Scoring

Two complementary scoring methods are used (`evaluation/evaluation_metrics.py`):

**1. Chamfer Distance with PatchCore-Style Reweighting:**
- For each input point, find the nearest reconstructed point (and vice versa) using `torch.cdist`
- The point with maximum nearest-neighbor distance is the most anomalous point
- **Reweighting** (inspired by PatchCore): the image-level score is modulated by the distance to k-nearest neighbors of the most anomalous matching pair, with softmax normalization for numerical stability
- Point-level scores are the per-point minimum distances (segmentation map)

**2. KNN-based Scoring (Nearest Neighbour):**
- Each point is represented by its k=64 nearest neighbor coordinates (flattened)
- FAISS is used for efficient nearest-neighbor search between input and reconstruction feature spaces
- Provides both image-level (max score) and point-level anomaly scores

**Metrics:** Image-level AUROC, Image-level AP, Point-level AUROC, Point-level AP

---

## Evaluation Metrics

During training, the model is periodically validated on the test set and outputs two lines of metrics per validation iteration. This section explains what each metric means and how to interpret them.

### Training Log Output

A typical validation log looks like this:

```
[Val] Iter 40000 | ROC_i_cdist 0.942857 | ROC_p_cdist 0.714458 | AP_i_cdist 0.951158 | AP_p_cdist 0.307707
[Val] Iter 40000 | ROC_i_nn 0.685714 | ROC_p_nn 0.591540 | AP_i_nn 0.726107 | AP_p_nn 0.058300
```

The first line reports metrics computed using the **Chamfer distance (cdist)** scoring method. The second line reports metrics computed using the **k-nearest neighbor (nn)** scoring method. Each line contains four metrics covering both detection granularities (image-level and point-level) and two evaluation protocols (AUROC and AP).

Here is the full mapping of all 8 logged metrics:

| Log Name | Full Name | Granularity | Protocol | Scoring Method |
|----------|-----------|-------------|----------|----------------|
| `ROC_i_cdist` | Image-level AUROC | Object-level | AUROC | Chamfer Distance |
| `ROC_p_cdist` | Point-level AUROC | Point-level | AUROC | Chamfer Distance |
| `AP_i_cdist` | Image-level Average Precision | Object-level | AP | Chamfer Distance |
| `AP_p_cdist` | Point-level Average Precision | Point-level | AP | Chamfer Distance |
| `ROC_i_nn` | Image-level AUROC | Object-level | AUROC | k-Nearest Neighbor |
| `ROC_p_nn` | Point-level AUROC | Point-level | AUROC | k-Nearest Neighbor |
| `AP_i_nn` | Image-level Average Precision | Object-level | AP | k-Nearest Neighbor |
| `AP_p_nn` | Point-level Average Precision | Point-level | AP | k-Nearest Neighbor |

### Metric Definitions

**AUROC (Area Under the Receiver Operating Characteristic Curve):**
- Measures the model's ability to **rank** anomalous samples higher than normal ones across all possible thresholds
- Computed using `sklearn.metrics.roc_auc_score`
- Range: 0.0 to 1.0 (1.0 = perfect separation, 0.5 = random chance)
- **Threshold-independent** — evaluates ranking quality, not a single decision boundary

**AP (Average Precision):**
- Summarizes the **precision-recall curve** as the weighted mean of precisions at each recall threshold
- Computed using `sklearn.metrics.average_precision_score`
- Range: 0.0 to 1.0 (1.0 = perfect precision at all recall levels)
- More sensitive to **class imbalance** than AUROC — particularly informative for point-level evaluation where anomalous points are rare

**Image-level (`_i`) vs. Point-level (`_p`):**
- **Image-level** (suffix `_i`): One score per object — answers *"Is this object anomalous?"*. Each test sample gets a single scalar anomaly score, compared against the binary label (0 = normal, 1 = anomalous)
- **Point-level** (suffix `_p`): One score per point — answers *"Which points are anomalous?"*. Each of the 2048 points gets an anomaly score, compared against the per-point ground truth mask

### Scoring Methods

Both scoring methods compute anomaly scores from the same input/reconstruction pair, but use different distance computation strategies:

**Chamfer Distance (`_cdist`):**
1. Computes pairwise distances between all input and reconstructed points using `torch.cdist`
2. For each input point, finds the minimum distance to any reconstructed point
3. The image-level score is the maximum per-point distance, refined with **PatchCore-style reweighting** that modulates the score based on k-nearest neighbor distances in the reconstruction space (see `evaluation/evaluation_metrics.py`, lines 62–84)
4. Point-level scores are the per-point minimum distances (segmentation map)

**k-Nearest Neighbor (`_nn`):**
1. For each point, computes a local geometric descriptor by concatenating the coordinates of its k=64 nearest neighbors (flattened to a 192-dimensional vector)
2. Uses **FAISS** (via the `NearestNeighbourScorer` in `evaluation/patchcore.py`) to find the nearest neighbor of each input point's descriptor in the reconstruction descriptor space
3. The image-level score is the maximum point-level score
4. Point-level scores are the per-point nearest neighbor distances

### Interpreting Results

Using the example log output:

```
ROC_i_cdist 0.942857  →  94.3% image-level AUROC (Chamfer) — strong object-level detection
ROC_p_cdist 0.714458  →  71.4% point-level AUROC (Chamfer) — moderate point localization
AP_i_cdist  0.951158  →  95.1% image-level AP (Chamfer)    — excellent detection precision-recall
AP_p_cdist  0.307707  →  30.8% point-level AP (Chamfer)    — low, due to class imbalance at point level

ROC_i_nn    0.685714  →  68.6% image-level AUROC (KNN)     — moderate object-level detection
ROC_p_nn    0.591540  →  59.2% point-level AUROC (KNN)     — weak point localization
AP_i_nn     0.726107  →  72.6% image-level AP (KNN)        — moderate detection precision-recall
AP_p_nn     0.058300  →   5.8% point-level AP (KNN)        — low, typical for sparse anomalies
```

**Key observations:**
- **Point-level AP is typically much lower** than other metrics because anomalous points are a small minority of all points (severe class imbalance). This is expected behavior, not a model failure
- **Chamfer distance (`_cdist`) metrics tend to be higher** than KNN (`_nn`) metrics in this codebase, as the PatchCore-style reweighting improves discrimination
- **Image-level metrics are generally higher** than point-level metrics because object-level detection is an easier task than precise point-level localization
- The model is validated every `--val_freq` iterations (default: 1000). The final metrics at the last iteration represent the model's end-of-training performance

---

## Pseudo Anomaly Generation

<a id="pseudo-anomaly-overview"></a>

### Overview

Pseudo anomaly generation creates **synthetic defects** on normal point clouds to produce training pairs of (anomalous input, clean target). This is critical for training the model to *actively* reconstruct anomalous regions back to normality, rather than simply memorizing normal shapes.

The codebase provides two pseudo anomaly generation strategies in `utils/util.py`. Both functions take a normal point cloud and return a modified point cloud along with a binary mask indicating which points were altered.

### Method 1: Random Patch (Localized Bulge/Dent)

**Function:** `random_patch(points, patch_num, scale)` in `utils/util.py`

This method simulates **localized surface deformations** (bulges or dents) — the most common real-world 3D defect type.

**Algorithm:**

```
Input: points (N, 3), patch_num (number of points to deform), scale (deformation magnitude)

1. DEFINE 26 viewpoints arranged on a grid around the origin:
   view = [(-2,-2,-2), (-2,-2,0), ..., (2,2,2)]

2. RANDOMLY SELECT one viewpoint v from the 26 options

3. COMPUTE distances from all N points to viewpoint v:
   distances[i] = ||points[i] - v||²

4. SELECT the closest `patch_num` points to v (these form the patch)
   → This naturally selects a contiguous surface region visible from v

5. FOR each selected point p_i:
   a. Compute outward normal direction: n_i = normalize(p_i - v)
   b. Randomly decide direction: inward (dent) or outward (bulge)
      with 50/50 probability, multiply normal by +scale or -scale
   c. Generate random magnitude: r_i ~ Uniform(0, 1), sorted descending
      → Center points get larger displacement, edges get smaller (smooth falloff)
   d. Displace: p_i' = p_i + n_i * scale * r_i

6. RETURN modified point cloud and binary mask of affected points
```

**Key Properties:**
- Creates **spatially contiguous** deformations (points close to the viewpoint)
- **Smooth falloff**: central points are displaced more than edge points (via sorted random magnitudes)
- **Bidirectional**: randomly creates either bumps (outward) or dents (inward)
- The 26 viewpoints ensure coverage of all surface orientations

**Visual Intuition:**
```
  Normal Surface          Bulge (outward)         Dent (inward)
  ─────────────          ───────╱╲───────        ─────╲  ╱─────
                                ╱  ╲                    ╲╱
```

### Method 2: Random Translate (Scattered Micro-Displacements)

**Function:** `random_translate(points, radius=0.08, translate=0.02, part=16)` in `utils/util.py`

This method simulates **scattered noise/roughness** — small random displacements across multiple local regions.

**Algorithm:**

```
Input: points (N, 3), radius (neighborhood size), translate (max displacement), part (number of regions)

1. FOR each of `part` (16) iterations:
   a. RANDOMLY SELECT a seed point from the point cloud
   b. FIND all points within `radius` (0.08) of the seed point
   c. FOR each neighboring point:
      - Generate random translation in each dimension: Δ ~ Uniform(-translate, +translate)
      - Apply: p_i' = p_i + (Δx, Δy, Δz)
   d. Mark affected points in the mask

2. RETURN modified point cloud and binary mask
```

**Key Properties:**
- Creates **multiple small patches** of noise (16 regions by default)
- Each region is a **sphere** of radius 0.08 around a random point
- Displacements are **small** (±0.02 units) — simulating surface roughness
- Regions can **overlap**, creating varying displacement intensities

### How Pseudo Anomalies Connect to Training

The pseudo anomaly functions are available in `utils/util.py` and imported by the dataset class (`utils/dataset.py`). The training pipeline supports them through the `x_raw` mechanism:

1. **Dataset provides two versions**: When augmentation is applied, the dataset can provide both the augmented point cloud (`pointcloud`) and the original clean version (`pointcloud_raw`)
2. **Training flag `--rel`**: When enabled, `train_ae.py` passes both `x` (augmented) and `x_raw` (clean) to `model.get_loss(x, x_raw)`
3. **Modified loss**: In `models/diffusion.py`, when `x_raw` is provided, the loss becomes reconstruction-to-clean rather than noise prediction:
   ```python
   # Standard: predict noise
   loss = MSE(ε_θ, ε)

   # With x_raw: reconstruct clean from augmented
   x̂_0 = (x_t - √(1-ᾱ) · ε_θ) / √(ᾱ)
   loss = MSE(x̂_0, x_raw)
   ```

This trains the diffusion model to not just denoise, but to **actively remove anomalies** and reconstruct the clean underlying surface.

---

## Integrating Your Own Pseudo Anomaly Synthesis Module

This section explains how to add a custom pseudo anomaly generation method to the pipeline.

### Step 1: Implement Your Anomaly Function

Add your function to `utils/util.py`. It must follow this signature:

```python
def your_custom_anomaly(points, **kwargs):
    """
    Generate pseudo anomalies on a 3D point cloud.

    Args:
        points: numpy array of shape (N, 3) — the input point cloud

    Returns:
        new_points: numpy array of shape (N, 3) — the modified point cloud
        mask: numpy array of shape (N,) — binary mask (1 = anomalous, 0 = normal)
    """
    new_points = points.copy()
    mask = np.zeros(points.shape[0], np.float32)

    # --- Your anomaly synthesis logic here ---
    # Example: randomly remove and re-scatter points in a local region
    # Example: cut a hole and fill with points from another object
    # Example: apply non-rigid deformation to a surface patch
    # Example: add structured noise patterns (scratches, cracks)

    return new_points, mask
```

**Important constraints:**
- Input and output must have the **same number of points** (N)
- The mask must be a float32 array with 1.0 for modified points, 0.0 for unchanged
- Operate on numpy arrays (the dataset handles torch conversion)

### Step 2: Register in the Dataset Loader

In `utils/dataset.py`, import and integrate your function:

```python
# At the top of utils/dataset.py, add your import:
from utils.util import normalize, random_rorate, random_patch, random_translate, your_custom_anomaly
```

Then modify the `load()` method of the `ShapeNetAD` class to apply your anomaly during training data loading:

```python
# Inside ShapeNetAD.load(), in the train split section:
for pc_id in tqdm(range(self.num_aug), 'Augment'):
    if self.num_aug == len(tpls):
        pointcloud = tpls[pc_id]
    else:
        pointcloud = random.choice(tpls)
    pointcloud = random_rorate(pointcloud)
    choice = np.random.choice(len(pointcloud), self.num_points, False)
    pointcloud_clean = pointcloud[choice].copy()

    # Apply your pseudo anomaly
    pointcloud_aug, aug_mask = your_custom_anomaly(pointcloud_clean)

    pc = torch.from_numpy(pointcloud_clean)         # Clean version (reconstruction target)
    pc_aug = torch.from_numpy(pointcloud_aug)        # Augmented version (encoder input)
    mask = torch.from_numpy(aug_mask)
    label = 0
    # Store both versions
    self.append_with_raw(pc_aug, pc, cate, pc_id, mask, label)
```

You will also need to add a helper method to store both the augmented and clean versions:

```python
def append_with_raw(self, pc, pc_raw, cate, pc_id, mask, label):
    pc, shift, scale = self.scale(pc)
    pc_raw = (pc_raw - shift) / scale  # Apply same normalization
    self.pointclouds.append({
        'pointcloud': pc,
        'pointcloud_raw': pc_raw,
        'cate': cate,
        'id': pc_id,
        'shift': shift,
        'scale': scale,
        'mask': mask,
        'label': label,
    })
```

### Step 3: Connect to the Training Loop

Enable the reconstruction loss mode by passing the `--rel` flag when training:

```bash
python train_ae.py --category ashtray0 --rel True --dataset_path ./data/shapenet-ad
```

The training loop in `train_ae.py` already supports this:

```python
# In train_ae.py, the train() function:
if args.rel:
    x_raw = batch['pointcloud_raw'].to(args.device)
    loss = model.get_loss(x, x_raw)   # Reconstruction loss: map augmented → clean
else:
    loss = model.get_loss(x)           # Standard noise prediction loss
```

### Design Guidelines

When designing your pseudo anomaly function, consider:

| Aspect | Recommendation |
|--------|---------------|
| **Locality** | Real defects are usually localized — modify a contiguous region rather than random scattered points |
| **Magnitude** | Keep deformations within a realistic range (the existing methods use 0.02–0.1 scale relative to normalized point clouds) |
| **Diversity** | Randomize the location, size, and type of anomaly each time the function is called |
| **Mask accuracy** | Ensure the mask precisely marks all and only the modified points — this affects evaluation |
| **Point count** | Maintain the same number of points (N) in the output — do not add or remove points |
| **Smoothness** | For surface deformations, use smooth falloff at patch boundaries to avoid unrealistic sharp edges |
| **Types to consider** | Bulges, dents, scratches (linear deformations), holes (inward collapse), roughness (high-frequency noise), missing parts, foreign object addition |

---

## Repository Structure

```
r3d-ad/
├── configs/
│   └── shapenet-ad/
│       └── base.yaml              # Base configuration (dataset, max_iters)
├── evaluation/
│   ├── __init__.py
│   ├── evaluation_metrics.py      # ROC_AP: AUROC & AP computation with PatchCore-style scoring
│   └── patchcore.py               # FAISS-based nearest neighbour scorer
├── models/
│   ├── autoencoder.py             # AutoEncoder, AutoEncoderTNet, DenoisingAutoEncoder
│   ├── common.py                  # ConcatSquashLinear, linear scheduler, math utilities
│   ├── diffusion.py               # DiffusionPoint, VarianceSchedule, PointwiseNet
│   └── encoders/
│       ├── __init__.py
│       └── pointnet.py            # PointNetEncoder, PointNetEncoderTNet, STN3d, STNkd
├── utils/
│   ├── config.py                  # YAML config loading
│   ├── dataset.py                 # ShapeNetAD dataset class
│   ├── transform.py               # Data transforms (RandomRotate, AddNoise, RandomScale, etc.)
│   └── util.py                    # Pseudo anomaly generation (random_patch, random_translate)
├── train_ae.py                    # Main training script
├── train_test.py                  # Wrapper: trains all categories sequentially
├── ensemble.py                    # Aggregates results across categories
├── vis_result.py                  # Open3D visualization of input/reconstruction/anomaly
└── README.md
```

---

## Installation

```sh
pip install easydict faiss-gpu ninja numpy open3d==0.16.0 opencv-python-headless pyyaml scikit-learn scipy tensorboard timm torch tqdm
pip install "git+https://github.com/unlimblue/KNN_CUDA.git#egg=knn_cuda&subdirectory=."
pip install "git+https://github.com/erikwijmans/Pointnet2_PyTorch.git#egg=pointnet2_ops&subdirectory=pointnet2_ops_lib"
```

**Requirements:**
- Python 3.8+
- PyTorch 1.10+
- CUDA-capable GPU

---

## Datasets

### Anomaly-ShapeNet

Download the dataset from [Google Drive](https://drive.google.com/file/d/16R8b39Os97XJOenB4bytxlV4vd_5dn0-/view?usp=sharing) and extract the `pcd` folder into `./data/shapenet-ad/`.

**40 object categories:** ashtray, bag, bottle (×3), bowl (×6), bucket (×2), cap (×4), cup (×2), eraser, headset (×2), helmet (×4), jar, microphone, shelf, tap (×2), vase (×8)

```
data/shapenet-ad/
├── ashtray0/
│   ├── train/                     # Normal samples only (.pcd files)
│   │   ├── ashtray0_template0.pcd
│   │   └── ...
│   ├── test/                      # Normal ("positive") + anomalous samples
│   │   ├── ashtray0_positive0.pcd
│   │   ├── ashtray0_bulge0.pcd
│   │   └── ...
│   └── GT/                        # Ground truth masks (per-point labels)
│       ├── ashtray0_bulge0.txt    # CSV: x, y, z, label (1=anomalous)
│       └── ...
├── bag0/
│   └── ...
└── vase9/
```

Each point cloud contains ~2048 points sampled during loading.

---

## Training and Testing

**Train on all categories:**
```bash
python train_test.py configs/shapenet-ad/base.yaml --tag experiment_name
```

**Train on a single category:**
```bash
python train_ae.py --category ashtray0 --dataset_path ./data/shapenet-ad --max_iters 40000
```

**Train with pseudo anomaly reconstruction mode:**
```bash
python train_ae.py --category ashtray0 --dataset_path ./data/shapenet-ad --rel True
```

**Key training arguments:**

| Argument | Default | Description |
|----------|---------|-------------|
| `--model` | `AutoEncoder` | Model class (`AutoEncoder`, `AutoEncoderTNet`, `DenoisingAutoEncoder`) |
| `--latent_dim` | 256 | Dimensionality of the latent code |
| `--num_steps` | 200 | Number of diffusion steps |
| `--beta_1` | 1e-4 | Starting noise schedule value |
| `--beta_T` | 0.05 | Ending noise schedule value |
| `--flexibility` | 0.0 | Sampling flexibility (0 = DDPM, 1 = DDIM-like) |
| `--residual` | True | Residual connection in PointwiseNet |
| `--num_points` | 2048 | Points per cloud |
| `--train_batch_size` | 128 | Training batch size |
| `--lr` | 1e-3 | Learning rate |
| `--max_iters` | ∞ (set in config) | Maximum training iterations |
| `--val_freq` | 1000 | Validation frequency (iterations) |
| `--rotate` | False | Enable random rotation augmentation |
| `--rel` | False | Enable reconstruction loss with `x_raw` |

**Aggregate results across categories:**
```bash
python ensemble.py PATH_TO_LOG_DIR
```

---

## Visualization

```bash
python vis_result.py PATH_TO_LOGS
```

Opens an interactive Open3D window showing:
- **Yellow:** Input point cloud
- **Blue:** Reconstructed point cloud
- **Red:** Ground truth anomalous points

---

## Configuration

Base configuration is in `configs/shapenet-ad/base.yaml`:

```yaml
dataset: ShapeNetAD
dataset_path: data/shapenet-ad/
max_iters: 40000
```

Additional parameters can be specified via command-line arguments (see [Training and Testing](#training-and-testing)).

---

## Acknowledgement

Thanks to previous open-sourced repos:

- [PVD](https://github.com/alexzhou907/PVD)
- [diffusion-point-cloud](https://github.com/luost26/diffusion-point-cloud)

---

## Citation

If you find this project useful in your research, please consider citing:

```bibtex
@inproceedings{zhou2024r3dad,
  title={R3D-AD: Reconstruction via Diffusion for 3D Anomaly Detection},
  author={Zhou, Zheyuan and Wang, Le and Fang, Naiyu and Wang, Zili and Qiu, Lemiao and Zhang, Shuyou},
  booktitle={European Conference on Computer Vision (ECCV)},
  year={2024}
}
```