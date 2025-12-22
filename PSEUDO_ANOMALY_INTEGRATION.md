# Pseudo Anomaly Synthesis Integration Guide

This document explains how the pseudo anomaly synthesis module has been integrated into the R3DAD framework.

## Overview

The pseudo anomaly synthesis module enables training 3D anomaly detection models using synthetically generated anomalies on normal point clouds. This eliminates the need for real anomalous samples during training.

## Architecture

### Key Components

1. **`data/AnomalyShapeNet/transform.py`**: Data augmentation transforms
   - `Compose`: Chains multiple transforms together
   - `NormalizeCoord`: Normalizes coordinates to [-1, 1]
   - `CenterShift`: Centers point clouds at origin
   - `RandomRotate`: Random rotation around x, y, or z axis
   - `SphereCropMask`: Divides point clouds into spherical patches

2. **`preprocessing.py`**: Core preprocessing module
   - `AnomalyPreset`: Factory for 10+ different anomaly configurations
   - `SmartAnomaly_Cfg`: Dataclass for anomaly configuration parameters
   - `Dataset`: Custom dataset class with pseudo anomaly generation
   - `make_collate`: Collate function factory for DataLoader

3. **`train_with_pseudo_anomaly.py`**: Training script
   - Integrates preprocessing.Dataset with R3DAD models
   - Supports AutoEncoder and other reconstruction models
   - Comprehensive configuration via command-line or YAML

4. **`train_pseudo_anomaly_batch.py`**: Batch training wrapper
   - Trains multiple categories sequentially
   - Supports config files from `configs/shapenet-ad/`

## Anomaly Types

The framework includes 10 preset anomaly types:

1. **Basic Local Bulge**: Isotropic outward deformation
2. **Basic Local Dent**: Isotropic inward deformation
3. **Ridge**: Elongated anisotropic bulge
4. **Trench**: Elongated anisotropic dent
5. **Elliptic Patch/Flat Spot**: Pressed/flattened region
6. **Skewed Impact Crater**: Oblique one-sided dent
7. **Shear along U**: Tangential slip deformation
8. **Shear along V**: Tangential slip in perpendicular direction
9. **Double-Sided Ripple**: Alternating bulge/dent pattern
10. **Micro Dimple Field**: Tiny corrosion-like dimples
11. **Directional Drag/Stretch**: Anisotropic plastic deformation

## Configuration Parameters

### Anomaly Size & Strength
- `R_low_bound`, `R_up_bound`: Anomaly radius as fraction of diameter [0.10, 0.30]
- `B_low_bound`, `B_up_bound`: Anomaly magnitude (displacement) [0.02, 0.15]
- `R_alpha`, `R_beta`: Beta distribution shape for radius sampling [2.0, 5.0]
- `B_alpha`, `B_beta`: Beta distribution shape for magnitude sampling [2.0, 5.0]

### Anomaly Characteristics
- `one_sided_prob`: Probability of one-sided vs double-sided [0.7]
- `cosine_kernel_prob`: Probability of cosine kernel [0.4]
- `gaussian_kernel_prob`: Probability of gaussian kernel [0.3]
- `poly_kernel_prob`: Probability of polynomial kernel [0.2]
- `hard_kernel_prob`: Probability of hard cutoff kernel [0.1]
- `poly_q`: Exponent for polynomial kernel [2.0]

### Dataset
- `mask_num`: Number of spherical patches [32]
- `voxel_size`: Voxel size for sparse quantization [0.05]
- `batch_size`: Training batch size [4]
- `data_repeat`: Times to repeat training data [1]
- `cache_dataset`: Cache training data in memory [false]

## Usage Examples

### Single Category Training
```bash
python train_with_pseudo_anomaly.py \
    --category ashtray0 \
    --batch_size 4 \
    --max_iters 40000 \
    --smart_anomaly True \
    --R_low_bound 0.10 \
    --R_up_bound 0.30 \
    --B_low_bound 0.02 \
    --B_up_bound 0.15
```

### Batch Training with Config
```bash
python train_pseudo_anomaly_batch.py \
    configs/shapenet-ad/pseudo_anomaly.yaml \
    --tag experiment1
```

### Single Category Testing
```bash
python train_pseudo_anomaly_batch.py \
    configs/shapenet-ad/pseudo_anomaly.yaml \
    --category ashtray0 \
    --single
```

## Dataset Structure

The integration expects the standard R3DAD dataset structure:

```
data/shapenet-ad/
├── ashtray0/
│   ├── train/
│   │   ├── ashtray0_template0.pcd
│   │   └── ...
│   ├── test/
│   │   ├── ashtray0_bulge0.pcd
│   │   ├── ashtray0_positive0.pcd
│   │   └── ...
│   └── GT/
│       ├── ashtray0_bulge0.txt
│       └── ...
├── bag0/
└── ...
```

## Integration Notes

1. **Dataset Compatibility**: The preprocessing module has been updated to work with R3DAD's shapenet-ad dataset structure (`.pcd` files instead of `.obj` files)

2. **Transform Module**: Created `data/AnomalyShapeNet/transform.py` to provide required augmentation classes

3. **Path Updates**: All dataset paths now point to `data/shapenet-ad/` instead of `data/AnomalyShapeNet/dataset/`

4. **Normal Computation**: Training point clouds now use Open3D's normal estimation instead of mesh vertex normals

5. **Modular Design**: The integration maintains backward compatibility - existing R3DAD training scripts continue to work

## Extending the Framework

### Adding New Anomaly Types

To add a custom anomaly type, extend the `AnomalyPreset` class in `preprocessing.py`:

```python
def type_N_custom_anomaly(self):
    R, B = self.get_R_B()
    return SmartAnomaly_Cfg(
        R=R,
        radii=(1.5, 0.8, 0.6),  # Anisotropic shape
        kernel="gaussian",
        dir_mode="normal_mean",
        one_sided=True,
        gate_mode="normals",
        alpha=+1,
        beta=B,
        sigma=0.4
    )
```

Then add it to the `presets` list in `__init__`.

### Custom Transforms

Add new transforms to `data/AnomalyShapeNet/transform.py`:

```python
class CustomTransform:
    def __init__(self, param1, param2):
        self.param1 = param1
        self.param2 = param2
    
    def __call__(self, data):
        # Modify data['coord'], data['normal'], data['mask']
        return data
```

## Performance Considerations

- **Caching**: Enable `cache_dataset=true` for faster training with sufficient RAM
- **Workers**: Adjust `num_works` based on CPU cores (default: 4)
- **Batch Size**: Reduce if GPU memory is insufficient (default: 4)
- **Data Repeat**: Increase for smaller datasets to avoid overfitting

## Troubleshooting

### Issue: ImportError for data.AnomalyShapeNet.transform
**Solution**: Ensure `data/AnomalyShapeNet/__init__.py` and `data/AnomalyShapeNet/transform.py` exist

### Issue: Dataset files not found
**Solution**: Verify dataset is in `data/shapenet-ad/` and follows the correct structure

### Issue: Out of memory during training
**Solution**: Reduce `batch_size`, disable `cache_dataset`, or reduce `mask_num`

### Issue: Training loss not decreasing
**Solution**: Adjust anomaly parameters (reduce `B_up_bound`), increase `max_iters`, or tune learning rate

## References

- Original R3DAD paper: [arxiv.org/abs/2407.10862](https://arxiv.org/abs/2407.10862)
- Pseudo Anomaly Synthesis: Inspired by industrial anomaly detection techniques
