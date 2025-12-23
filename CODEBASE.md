# R3D-AD Codebase Documentation

This document provides a comprehensive technical explanation of the R3D-AD (Reconstruction via Diffusion for 3D Anomaly Detection) repository, an implementation of the ECCV 2024 paper.

---

## High-Level Overview

### Problem Statement
This repository implements a 3D point cloud anomaly detection system. Given a 3D point cloud of an object, the system identifies whether the object is normal or anomalous, and localizes which points are anomalous (defective regions).

### Core Methodology
The system uses a **reconstruction-based anomaly detection** approach:
1. An autoencoder with a diffusion-based decoder learns to reconstruct normal point clouds
2. At test time, anomalous regions are harder to reconstruct accurately
3. The reconstruction error (distance between input and reconstructed points) serves as the anomaly score

The key innovation is using a **denoising diffusion probabilistic model (DDPM)** as the decoder, which iteratively refines random noise into a point cloud conditioned on a latent code from the encoder.

### Expected Inputs and Outputs
- **Input**: 3D point clouds in `.pcd` format (2048 points by default)
- **Output**: 
  - Image-level anomaly score (whether the object is defective)
  - Point-level anomaly scores (segmentation mask indicating defective regions)
  - Metrics: I-AUROC, P-AUROC, I-AP, P-AP

---

## Repository Structure

```
r3d-ad/
├── configs/
│   └── shapenet-ad/
│       └── base.yaml          # Default training configuration
├── evaluation/
│   ├── __init__.py
│   ├── evaluation_metrics.py  # ROC-AUC, AP metrics calculation
│   └── patchcore.py           # FAISS-based nearest neighbor scoring
├── models/
│   ├── autoencoder.py         # Main AutoEncoder model class
│   ├── common.py              # Shared utilities (ConcatSquashLinear, scheduler)
│   ├── diffusion.py           # Diffusion model and variance schedule
│   └── encoders/
│       ├── __init__.py
│       └── pointnet.py        # PointNet encoder implementations
├── utils/
│   ├── config.py              # YAML configuration parser
│   ├── data.py                # Data iterator utilities
│   ├── dataset.py             # ShapeNetAD dataset class
│   ├── misc.py                # Logging, checkpointing, seeding
│   ├── transform.py           # Point cloud augmentation transforms
│   └── util.py                # Geometric utilities (rotation, normalization)
├── train_ae.py                # Main training script (single category)
├── train_test.py              # Orchestrator script (all categories)
├── ensemble.py                # Results aggregation across categories
├── vis_result.py              # 3D visualization of results
└── README.md
```

### Entry Points
- **`train_test.py`**: Primary entry point that launches training for all object categories
- **`train_ae.py`**: Core training/evaluation script for a single category
- **`vis_result.py`**: Post-hoc visualization of reconstruction results
- **`ensemble.py`**: Aggregates metrics across all categories

---

## Core Components & Logic

### 1. AutoEncoder (`models/autoencoder.py`)

The main model class that combines:
- **Encoder**: `PointNetEncoder` – extracts a global latent code from input point cloud
- **Decoder**: `DiffusionPoint` – reconstructs point cloud via iterative denoising

```
Input PC (B, N, 3) → PointNet Encoder → Latent Code (B, 256)
                                              ↓
                                        Diffusion Decoder
                                              ↓
                                    Reconstructed PC (B, N, 3)
```

Key methods:
- `encode(x)`: Maps point cloud to latent representation
- `decode(code, num_points)`: Generates point cloud from latent code via diffusion sampling
- `get_loss(x)`: Computes diffusion training loss

### 2. PointNet Encoder (`models/encoders/pointnet.py`)

Standard PointNet architecture with:
- 4 shared 1D convolution layers (3 → 128 → 128 → 256 → 512)
- Global max pooling over points
- FC layers mapping to latent mean and variance vectors

Two variants:
- `PointNetEncoder`: Basic encoder returning mean/variance
- `PointNetEncoderTNet`: Includes spatial transformer networks for rotation invariance

### 3. Diffusion Model (`models/diffusion.py`)

Implements denoising diffusion for point clouds:

**`VarianceSchedule`**: Manages the noise schedule
- Linear beta schedule from `beta_1` to `beta_T`
- Precomputes alpha bars and sigma values for efficient sampling

**`PointwiseNet`**: Noise prediction network
- 6-layer MLP with `ConcatSquashLinear` layers
- Each layer receives: point coordinates, timestep embedding, latent context
- Residual connection optionally adds input to output

**`DiffusionPoint`**: Main diffusion process
- `get_loss()`: Forward diffusion (add noise) + predict noise
- `sample()`: Reverse diffusion (iterative denoising from x_T to x_0)

### 4. Evaluation Metrics (`evaluation/evaluation_metrics.py`)

**`ROC_AP()`** function computes:
- **Image-level scores**: Maximum point-wise distance with PatchCore reweighting
- **Pixel-level scores**: Per-point reconstruction error

Two scoring methods:
1. **Chamfer-based**: Nearest neighbor distance between input/reconstruction
2. **NN-based**: Uses 64-NN neighborhoods with FAISS for efficient scoring

### 5. Dataset (`utils/dataset.py`)

**`ShapeNetAD`** class handles:
- Loading `.pcd` files from train/test splits
- Sampling fixed number of points (default 2048)
- Data augmentation for training (random rotation)
- Loading ground truth anomaly masks from `.txt` files

---

## Data Flow & Execution Pipeline

### Training Flow

```
1. train_test.py: Parse config, iterate over categories
       ↓
2. train_ae.py: For each category:
       ↓
3. Load Dataset: ShapeNetAD(split='train') with augmentation
       ↓
4. Create Model: AutoEncoder with PointNet + Diffusion
       ↓
5. Training Loop (40k iterations default):
   a. Sample batch of point clouds
   b. Encode to latent code
   c. Compute diffusion loss (noise prediction)
   d. Backprop and update
       ↓
6. Validation (every 1k iterations):
   a. Encode test point clouds
   b. Decode via full diffusion sampling
   c. Compute ROC/AP metrics
   d. Save checkpoint if improved
```

### Inference/Evaluation Flow

```
1. Load trained model checkpoint
2. For each test point cloud:
   a. Normalize and subsample
   b. Encode to latent code
   c. Run reverse diffusion (200 steps)
   d. Compare input vs reconstruction
3. Compute anomaly scores:
   a. Point-level: per-point distances
   b. Image-level: aggregated with reweighting
4. Evaluate against ground truth masks
```

### Key Tensors and Shapes
- Point cloud: `(B, N, 3)` where N=2048
- Latent code: `(B, 256)`
- Beta schedule: `(T+1,)` where T=200
- Noise prediction: `(B, N, 3)`

---

## Configuration & Extensibility

### Configuration System

YAML-based configuration in `configs/`:
```yaml
dataset: ShapeNetAD
dataset_path: data/shapenet-ad/
max_iters: 40000
```

Command-line arguments (in `train_ae.py`) override/extend config:
- Model: `--model`, `--latent_dim`, `--num_steps`, `--beta_1`, `--beta_T`
- Data: `--dataset`, `--category`, `--num_points`
- Training: `--lr`, `--max_iters`, `--val_freq`

`utils/config.py` provides:
- `cfg_from_yaml_file()`: Load YAML with inheritance support (`_BASE_CONFIG_`)
- `cmd_from_config()`: Convert config to command-line string

### Extension Points

**Adding new datasets**:
1. Create dataset class in `utils/dataset.py` following `ShapeNetAD` interface
2. Implement `load()`, `__len__()`, `__getitem__()`
3. Register category list (like `all_shapenetad_cates`)

**Adding new encoders**:
1. Create encoder in `models/encoders/`
2. Must return latent code `(B, D)` from point cloud `(B, N, 3)`
3. Import in `models/encoders/__init__.py`

**Adding new decoders**:
1. Subclass or modify `DiffusionPoint` in `models/diffusion.py`
2. Ensure `sample()` returns `(B, N, 3)` point cloud

**Modifying evaluation**:
1. Add scoring methods in `evaluation/evaluation_metrics.py`
2. Update `ROC_AP()` to include new metrics in returned dict

---

## Key Assumptions & Constraints

### Assumptions
- Point clouds are centered and normalized (handled by `scale()` in dataset)
- All point clouds have same number of points (fixed sampling)
- Training data contains only normal (non-anomalous) samples
- Anomalies manifest as geometric deviations detectable via reconstruction error

### Design Constraints
- Requires GPU for efficient training (CUDA tensors, KNN_CUDA, pointnet2_ops)
- Memory scales with `batch_size × num_points × num_diffusion_steps`
- Single-category training: each category trains separate model

### Known Limitations
- No support for multi-category joint training
- Diffusion sampling is slow (200 iterative steps per sample)
- Evaluation requires test-time computation of full diffusion reverse process
- `break` statement in `train_test.py` limits to single category (likely debug artifact)

### Trade-offs
- Diffusion decoder provides high-quality reconstruction but slower than feedforward
- PointNet encoder is simple/fast but lacks local geometric features
- FAISS-based scoring trades exactness for speed on large point clouds

---

## Typical Usage Pattern

### Training on All Categories
```bash
python train_test.py configs/shapenet-ad/base.yaml --tag experiment_v1
```

### Training on Single Category
```bash
python train_ae.py --category vase0 --dataset ShapeNetAD --dataset_path ./data/shapenet-ad --max_iters 40000 --log_root ./logs
```

### Aggregating Results
```bash
python ensemble.py logs_shapenet-ad/experiment_v1_20240101-120000/
```

### Visualizing Reconstructions
```bash
python vis_result.py logs_shapenet-ad/experiment_v1_20240101-120000/vase0_*/
```

### Expected Output
```
[Train] Iter 1000 | Loss 0.045123 | Grad 2.3456
[Val] Iter 1000 | ROC_i_nn 0.823456 | ROC_p_nn 0.912345 | AP_i_nn 0.789012 | AP_p_nn 0.856789
```

---

## Dependencies

Core requirements (from README):
- PyTorch with CUDA
- Open3D 0.16.0 (point cloud I/O and visualization)
- KNN_CUDA (GPU-accelerated k-nearest neighbors)
- pointnet2_ops (furthest point sampling, ball query)
- FAISS (efficient nearest neighbor search)
- Standard ML stack: numpy, scipy, scikit-learn, tensorboard

---

## Summary

R3D-AD implements 3D anomaly detection by training a diffusion-based autoencoder on normal point clouds. The encoder-decoder architecture learns to reconstruct normal geometry, while anomalous regions produce higher reconstruction errors at test time. The codebase is modular, with clear separation between data handling, model definition, and evaluation, making it straightforward to adapt for new datasets or architectural modifications.
