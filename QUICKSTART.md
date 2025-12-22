# Quick Start Guide: Pseudo Anomaly Synthesis

This guide helps you quickly get started with the pseudo anomaly synthesis module integrated into R3DAD.

## Prerequisites

1. Install required dependencies:
```bash
pip install easydict faiss-gpu ninja numpy open3d==0.16.0 opencv-python-headless pyyaml scikit-learn scipy tensorboard timm torch tqdm
pip install "git+https://github.com/unlimblue/KNN_CUDA.git#egg=knn_cuda&subdirectory=."
pip install "git+https://github.com/erikwijmans/Pointnet2_PyTorch.git#egg=pointnet2_ops&subdirectory=pointnet2_ops_lib"
```

2. Download and prepare the dataset:
```bash
# Download shapenet-ad dataset from Google Drive
# Extract to ./data/shapenet-ad/
```

## Verify Integration

Test that the integration is working:
```bash
python test_integration.py
```

This will verify:
- Transform module imports
- Preprocessing module imports
- Point cloud generation
- Transform pipeline
- Anomaly preset configurations

## Basic Usage

### Train a Single Category

Train with default settings:
```bash
python train_with_pseudo_anomaly.py --category ashtray0
```

Train with custom parameters:
```bash
python train_with_pseudo_anomaly.py \
    --category ashtray0 \
    --batch_size 4 \
    --max_iters 40000 \
    --R_low_bound 0.10 \
    --R_up_bound 0.30 \
    --B_low_bound 0.02 \
    --B_up_bound 0.15
```

### Train Multiple Categories

Using configuration file:
```bash
python train_pseudo_anomaly_batch.py configs/shapenet-ad/pseudo_anomaly.yaml
```

With custom tag:
```bash
python train_pseudo_anomaly_batch.py configs/shapenet-ad/pseudo_anomaly.yaml --tag my_experiment
```

Test with single category:
```bash
python train_pseudo_anomaly_batch.py configs/shapenet-ad/pseudo_anomaly.yaml --category ashtray0 --single
```

## Key Parameters

### Quick Parameter Guide

**For larger, more visible anomalies:**
```bash
--R_up_bound 0.40 --B_up_bound 0.20
```

**For subtle, realistic anomalies:**
```bash
--R_up_bound 0.20 --B_up_bound 0.08
```

**For diverse anomaly types:**
```bash
--cosine_kernel_prob 0.3 --gaussian_kernel_prob 0.3 --poly_kernel_prob 0.3
```

**For one-sided anomalies only:**
```bash
--one_sided_prob 1.0
```

## Configuration File

Edit `configs/shapenet-ad/pseudo_anomaly.yaml`:

```yaml
# Anomaly size (fraction of object diameter)
R_low_bound: 0.10
R_up_bound: 0.30

# Anomaly magnitude (displacement amount)
B_low_bound: 0.02
B_up_bound: 0.15

# Training settings
batch_size: 4
max_iters: 40000
```

## Output

Training outputs are saved to:
```
logs_pseudo_anomaly/
└── {category}_{timestamp}_{tag}/
    ├── checkpoints/
    ├── tensorboard logs/
    └── training logs
```

View with TensorBoard:
```bash
tensorboard --logdir logs_pseudo_anomaly/
```

## Common Issues

**Out of memory:**
- Reduce `batch_size` to 2 or 1
- Set `cache_dataset: false`

**Training not converging:**
- Reduce `B_up_bound` to 0.10
- Increase `max_iters` to 60000

**Dataset not found:**
- Verify dataset is in `data/shapenet-ad/`
- Check category name matches folder name

## Next Steps

1. **Visualize Results**: Use `vis_result.py` to visualize trained models
2. **Tune Parameters**: Adjust anomaly parameters for your use case
3. **Custom Anomalies**: Add new anomaly types (see PSEUDO_ANOMALY_INTEGRATION.md)
4. **Evaluation**: Test on real anomalies in the test set

## Getting Help

- See `README.md` for detailed documentation
- See `PSEUDO_ANOMALY_INTEGRATION.md` for integration details
- Check issue tracker for known issues

## Example Workflow

```bash
# 1. Verify integration
python test_integration.py

# 2. Quick test with one category
python train_pseudo_anomaly_batch.py \
    configs/shapenet-ad/pseudo_anomaly.yaml \
    --category ashtray0 \
    --single

# 3. Train all categories
python train_pseudo_anomaly_batch.py \
    configs/shapenet-ad/pseudo_anomaly.yaml \
    --tag full_training

# 4. View results
tensorboard --logdir logs_pseudo_anomaly/
```

That's it! You're now ready to train 3D anomaly detection models with synthetic anomalies.
