# Pseudo Anomaly Synthesis Integration Guide

## Overview

This document explains how the pseudo anomaly synthesis module has been integrated into `utils/util.py` to support custom anomaly generation for training R3D-AD models.

## What Was Changed

### 1. Enhanced `utils/util.py`

The following components have been added to `utils/util.py`:

- **`SmartAnomaly_Cfg`**: Dataclass for configuring anomaly parameters
- **`AnomalyPreset`**: Factory class for 11 different anomaly presets
- **`pseudo_anomaly_synthesis()`**: Core function for generating pseudo anomalies
- Helper functions: `_kernel()`, `_local_frame()`, `_one_side_gate()`, `_normal_alignment_gate()`

### 2. Updated `utils/dataset.py`

The `ShapeNetAD` class now supports:

- New parameter `use_pseudo_anomaly` (default: `False`) to enable pseudo anomaly synthesis
- New parameter `anomaly_preset_config` to provide configuration for anomaly presets
- Automatic application of pseudo anomaly synthesis during training data augmentation

## Backward Compatibility

**All existing code continues to work without modification.** The integration is fully backward compatible:

- Existing training scripts (e.g., `train_ae.py`, `train_test.py`) work without changes
- The original `random_rorate()`, `random_patch()`, and `random_translate()` functions remain unchanged
- Pseudo anomaly synthesis is opt-in via the `use_pseudo_anomaly` parameter

## Usage

### Method 1: Use with ShapeNetAD Dataset (Recommended)

```python
from utils.dataset import ShapeNetAD

# Create configuration for anomaly presets
class AnomalyConfig:
    R_low_bound = 0.10      # Min anomaly radius (fraction of diameter)
    R_up_bound = 0.30       # Max anomaly radius
    R_alpha = 2.0           # Beta distribution shape
    R_beta = 5.0            # Beta distribution shape
    B_low_bound = 0.02      # Min displacement magnitude
    B_up_bound = 0.15       # Max displacement magnitude
    B_alpha = 2.0           # Beta distribution shape
    B_beta = 5.0            # Beta distribution shape

# Create dataset with pseudo anomaly synthesis
train_dset = ShapeNetAD(
    path='./data/shapenet-ad',
    cates=['ashtray0'],
    split='train',
    num_points=2048,
    num_aug=2048,
    use_pseudo_anomaly=True,                  # Enable pseudo anomaly
    anomaly_preset_config=AnomalyConfig()     # Provide config
)

# Use in DataLoader as normal
train_loader = DataLoader(train_dset, batch_size=128, shuffle=True)
for batch in train_loader:
    # Training code here
    pass
```

### Method 2: Direct Function Usage

```python
import numpy as np
from utils.util import pseudo_anomaly_synthesis, SmartAnomaly_Cfg, AnomalyPreset

# Load your point cloud
points = ...  # (N, 3) numpy array
normals = ... # (N, 3) numpy array  
center = ...  # (3,) numpy array

# Option A: Use a preset
class AnomalyConfig:
    R_low_bound = 0.10
    R_up_bound = 0.30
    R_alpha = 2.0
    R_beta = 5.0
    B_low_bound = 0.02
    B_up_bound = 0.15
    B_alpha = 2.0
    B_beta = 5.0

presets = AnomalyPreset(AnomalyConfig())
cfg = presets.type_1_basic_bulge()  # Choose from 11 presets
deformed = pseudo_anomaly_synthesis(points, normals, center, cfg)

# Option B: Use custom configuration
custom_cfg = SmartAnomaly_Cfg(
    R=0.3,              # Radius
    beta=0.1,           # Magnitude
    alpha=+1,           # +1 for bulge, -1 for dent
    kernel="cosine",    # "cosine", "gaussian", "poly", "hard"
    radii=(1.0, 1.0, 1.0),  # Anisotropic shape
    one_sided=True      # One-sided vs double-sided deformation
)
deformed = pseudo_anomaly_synthesis(points, normals, center, custom_cfg)
```

### Method 3: Modify Existing Training Script

To integrate with an existing training script like `train_ae.py`:

```python
# 1. Add imports at the top
from utils.util import AnomalyPreset

# 2. Add command line arguments (or config file parameters)
parser.add_argument('--use_pseudo_anomaly', type=eval, default=False, choices=[True, False])
parser.add_argument('--R_low_bound', type=float, default=0.10)
parser.add_argument('--R_up_bound', type=float, default=0.30)
parser.add_argument('--R_alpha', type=float, default=2.0)
parser.add_argument('--R_beta', type=float, default=5.0)
parser.add_argument('--B_low_bound', type=float, default=0.02)
parser.add_argument('--B_up_bound', type=float, default=0.15)
parser.add_argument('--B_alpha', type=float, default=2.0)
parser.add_argument('--B_beta', type=float, default=5.0)

# 3. Create anomaly config if enabled
anomaly_config = None
if args.use_pseudo_anomaly:
    class AnomalyConfig:
        R_low_bound = args.R_low_bound
        R_up_bound = args.R_up_bound
        R_alpha = args.R_alpha
        R_beta = args.R_beta
        B_low_bound = args.B_low_bound
        B_up_bound = args.B_up_bound
        B_alpha = args.B_alpha
        B_beta = args.B_beta
    anomaly_config = AnomalyConfig()

# 4. Pass to dataset constructor
train_dset = ShapeNetAD(
    path=args.dataset_path,
    cates=[args.category],
    split='train',
    num_points=args.num_points,
    num_aug=args.num_aug,
    use_pseudo_anomaly=args.use_pseudo_anomaly,
    anomaly_preset_config=anomaly_config
)
```

## Anomaly Presets

The module includes 11 different anomaly presets:

1. **Basic Bulge**: Isotropic outward deformation
2. **Basic Dent**: Isotropic inward deformation
3. **Ridge**: Elongated anisotropic bulge
4. **Trench**: Elongated anisotropic dent
5. **Elliptic Patch/Flat Spot**: Pressed/flattened region
6. **Skewed Impact Crater**: Oblique one-sided dent
7. **Shear along U**: Tangential slip deformation
8. **Shear along V**: Tangential slip in perpendicular direction
9. **Double-Sided Ripple**: Alternating bulge/dent pattern
10. **Micro Dimple Field**: Tiny corrosion-like dimples
11. **Directional Drag/Stretch**: Anisotropic plastic deformation

Each preset is randomly selected during training to provide diverse anomaly types.

## Configuration Parameters

### Anomaly Size & Strength

- `R_low_bound`, `R_up_bound`: Anomaly radius as fraction of diameter [0.10, 0.30]
- `B_low_bound`, `B_up_bound`: Anomaly magnitude (displacement) [0.02, 0.15]
- `R_alpha`, `R_beta`: Beta distribution shape for radius sampling [2.0, 5.0]
- `B_alpha`, `B_beta`: Beta distribution shape for magnitude sampling [2.0, 5.0]

### SmartAnomaly_Cfg Parameters

- `R`: Support radius (None = 0.2 * object diameter)
- `beta`: Magnitude of displacement
- `alpha`: +1 for bulge, -1 for dent, None for random
- `kernel`: Falloff kernel type ("cosine", "gaussian", "poly", "hard")
- `radii`: Anisotropic shape (ru, rv, rn)
- `dir_mode`: Displacement direction ("normal_point", "normal_mean", "tangent_u", "tangent_v")
- `one_sided`: One-sided vs double-sided deformation
- `gate_mode`: Gating mode ("global", "normals")

## Examples

### Train with Pseudo Anomaly Synthesis

```bash
python train_ae.py \
    --category ashtray0 \
    --use_pseudo_anomaly True \
    --R_low_bound 0.10 \
    --R_up_bound 0.30 \
    --B_low_bound 0.02 \
    --B_up_bound 0.15 \
    --num_aug 2048 \
    --train_batch_size 128
```

### Train without Pseudo Anomaly (Original Behavior)

```bash
python train_ae.py \
    --category ashtray0 \
    --use_pseudo_anomaly False \
    --num_aug 2048 \
    --train_batch_size 128
```

## Comparison with preprocessing.py

The `preprocessing.py` module remains available for advanced use cases with:

- Full integration with MinkowskiEngine for sparse voxelization
- Support for reinforcement learning-based anomaly selection
- Batch processing with rollout modes
- Advanced caching and prefetching mechanisms

The `utils/util.py` integration provides:

- Simpler API for basic use cases
- Direct integration with existing R3D-AD code
- Backward compatibility with minimal code changes
- Easier to understand and customize

## Performance Considerations

- **Normal Estimation**: Point cloud normals are estimated using Open3D when not available
- **Anomaly Region**: By default, 30% of points are selected for anomaly application
- **Computational Cost**: Pseudo anomaly synthesis adds ~10-20% overhead during training data loading
- **Memory**: No significant memory overhead as anomalies are generated on-the-fly

## Testing

Three test scripts are provided:

1. **`test_pseudo_anomaly_util.py`**: Tests the core pseudo anomaly synthesis functions
2. **`test_backward_compat.py`**: Verifies backward compatibility with existing code
3. **`example_pseudo_anomaly_usage.py`**: Demonstrates usage patterns

Run tests:

```bash
python test_pseudo_anomaly_util.py
python test_backward_compat.py
python example_pseudo_anomaly_usage.py
```

## Troubleshooting

### Issue: Dataset creation fails with missing data

**Solution**: Ensure the dataset is available at `./data/shapenet-ad/` with the correct structure.

### Issue: Pseudo anomaly has no visible effect

**Solution**: Increase `B_up_bound` parameter or check that normals are correctly estimated.

### Issue: Training slower with pseudo anomaly

**Solution**: This is expected due to normal estimation and anomaly synthesis. Consider:
- Reducing `num_aug` parameter
- Caching point cloud data if memory allows
- Using fewer workers in DataLoader

### Issue: Want to use original simple anomaly synthesis

**Solution**: Set `use_pseudo_anomaly=False` or simply don't specify the parameter (defaults to False).

## Future Enhancements

Possible future improvements:

1. Add caching of estimated normals to avoid recomputation
2. GPU acceleration for anomaly synthesis using PyTorch
3. Additional anomaly presets (e.g., cracks, holes, surface roughness)
4. Automatic tuning of anomaly parameters based on validation performance
5. Visualization tools for inspecting generated anomalies

## References

- Original R3D-AD paper: [arxiv.org/abs/2407.10862](https://arxiv.org/abs/2407.10862)
- Full preprocessing module: `preprocessing.py`
- Integration documentation: `PSEUDO_ANOMALY_INTEGRATION.md`
