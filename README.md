# R3D-AD: Reconstruction via Diffusion for 3D Anomaly Detection
### [Website](https://zhouzheyuan.github.io/r3d-ad) | [Paper](https://arxiv.org/abs/2407.10862)
> [**R3D-AD: Reconstruction via Diffusion for 3D Anomaly Detection**](https://doi.org/10.1007/978-3-031-72764-1_6),  
> Zheyuan Zhou∗, Le Wang∗, Naiyu Fang, Zili Wang, Lemiao Qiu, Shuyou Zhang
> **ECCV 2024**

## Installation
```sh
pip install easydict faiss-gpu ninja numpy open3d==0.16.0 opencv-python-headless pyyaml scikit-learn scipy tensorboard timm torch tqdm 
pip install "git+https://github.com/unlimblue/KNN_CUDA.git#egg=knn_cuda&subdirectory=."
pip install "git+https://github.com/erikwijmans/Pointnet2_PyTorch.git#egg=pointnet2_ops&subdirectory=pointnet2_ops_lib"
```

## Datasets

### Anomaly-ShapeNet
Download dataset from [google drive](https://drive.google.com/file/d/16R8b39Os97XJOenB4bytxlV4vd_5dn0-/view?usp=sharing) and extract `pcd` folder into `./data/shapenet-ad/`
```
shapenet-ad
├── ashtray0
    ├── train
        ├── ashtray0_template0.pcd
        ...
    ├── test
        ├── ashtray0_bulge0.pcd
        ...
    ├── GT
        ├── ashtray0_bulge0.txt
        ... 
├── bag0
...
...
├── vase9
```

## Training and Testing

### Standard Training
```bash
python train_test.py PATH_TO_CONFIG
```

### Training with Pseudo Anomaly Synthesis

The repository now includes a pseudo anomaly synthesis module that can generate synthetic anomalies on normal point clouds during training. This allows training 3D anomaly detection models without requiring real anomalous samples.

#### Features
- **Multiple Anomaly Presets**: 10+ different types of pseudo anomalies including bulges, dents, ridges, trenches, shear deformations, ripples, and more
- **Configurable Parameters**: Control anomaly size, magnitude, kernel types, and probability distributions
- **Smart Anomaly Generation**: Anisotropic deformations with one-sided and double-sided variants
- **Integrated Pipeline**: Seamlessly works with R3DAD's training infrastructure

#### Quick Start
Train a single category with pseudo anomalies:
```bash
python train_with_pseudo_anomaly.py \
    --category ashtray0 \
    --batch_size 4 \
    --max_iters 40000 \
    --smart_anomaly True
```

Train all categories using a config file:
```bash
python train_pseudo_anomaly_batch.py configs/shapenet-ad/pseudo_anomaly.yaml --tag experiment1
```

Train a single category for testing:
```bash
python train_pseudo_anomaly_batch.py configs/shapenet-ad/pseudo_anomaly.yaml --category ashtray0 --single
```

#### Configuration

The pseudo anomaly synthesis can be configured through command-line arguments or YAML config files. Key parameters:

**Anomaly Size & Strength:**
- `--R_low_bound`, `--R_up_bound`: Anomaly radius bounds (fraction of diameter)
- `--B_low_bound`, `--B_up_bound`: Anomaly magnitude bounds (displacement)
- `--R_alpha`, `--R_beta`: Beta distribution parameters for radius
- `--B_alpha`, `--B_beta`: Beta distribution parameters for magnitude

**Anomaly Types:**
- `--one_sided_prob`: Probability of one-sided vs double-sided anomalies
- `--cosine_kernel_prob`, `--gaussian_kernel_prob`, etc.: Kernel type probabilities

**Dataset:**
- `--mask_num`: Number of spherical patches for anomaly synthesis (default: 32)
- `--voxel_size`: Voxel size for sparse quantization (default: 0.05)
- `--data_repeat`: Number of times to repeat training data (default: 1)

See `configs/shapenet-ad/pseudo_anomaly.yaml` for a complete configuration example.

## Visualization
```bash
python vis_result.py PATH_TO_LOGS
```

## Acknowledgement
Thanks to previous open-sourced repo:

[PVD](https://github.com/alexzhou907/PVD)

[diffusion-point-cloud](https://github.com/luost26/diffusion-point-cloud)

## Citation 
If you find this project useful in your research, please consider cite:

```bibtex
@inproceedings{zhou2024r3dad,
  title={R3D-AD: Reconstruction via Diffusion for 3D Anomaly Detection},
  author={Zhou, Zheyuan and Wang, Le and Fang, Naiyu and Wang, Zili and Qiu, Lemiao and Zhang, Shuyou},
  booktitle={European Conference on Computer Vision (ECCV)},
  year={2024}
}
```