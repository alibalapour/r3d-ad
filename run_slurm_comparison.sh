#!/bin/bash
#SBATCH --job-name=R3D_AD_15Class_Comparison
#SBATCH --account=aip-fhach
#SBATCH --mail-user=alibalapour93.ab@gmail.com
#SBATCH --mail-type=ALL
#SBATCH --time=0-36:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --gres=gpu:l40s:1
#SBATCH --array=0-44                  # 3 configs × 15 categories = 45 jobs
#SBATCH --output=logs/r3d_ad_15class_%A_%a.out
#SBATCH --error=logs/r3d_ad_15class_%A_%a.err

#
# run_slurm_comparison.sh
#
# SLURM array job that compares three training modes across 15 AnomalyShapeNet
# classes (index-0 variants):
#   Config 0 (jobs  0-14) – Base:         no pseudo anomalies
#   Config 1 (jobs 15-29) – Patch-Gen:    random_patch pseudo anomalies
#   Config 2 (jobs 30-44) – AF3AD:        AF3AD pseudo anomaly synthesis
#
# Array mapping:  CONFIG_IDX = TASK_ID / 15,  CAT_IDX = TASK_ID % 15
#

set -euo pipefail

echo "=== Job started at $(date) on $(hostname) ==="
echo "SLURM_JOB_ID=$SLURM_JOB_ID"
echo "SLURM_ARRAY_TASK_ID=$SLURM_ARRAY_TASK_ID"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-not set}"
nvidia-smi || true

# -------------------------
# Modules (adjust as needed for your cluster)
# -------------------------
module purge
module load StdEnv/2020
module load gcc/11.3.0
module load cuda/11.8.0
module load python/3.9
module load flexiblas

# -------------------------
# Activate virtualenv
# -------------------------
VENV_PATH="${SLURM_VENV_PATH:-$HOME/envs/me-cu118}"
source "$VENV_PATH/bin/activate"

echo "Python: $(which python)"
python --version

# -------------------------
# Configure threading for deterministic performance
# -------------------------
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

# ── Categories ───────────────────────────────────────────────────────
CATEGORIES=(
    "ashtray0"
    "bag0"
    "bottle0"
    "bowl0"
    "bucket0"
    "cap0"
    "cup0"
    "eraser0"
    "headset0"
    "helmet0"
    "jar0"
    "microphone0"
    "shelf0"
    "tap0"
    "vase0"
)
NUM_CATEGORIES=${#CATEGORIES[@]}   # 15

# ── Decode array task ID ─────────────────────────────────────────────
CONFIG_IDX=$(( SLURM_ARRAY_TASK_ID / NUM_CATEGORIES ))
CAT_IDX=$(( SLURM_ARRAY_TASK_ID % NUM_CATEGORIES ))
CATEGORY="${CATEGORIES[$CAT_IDX]}"

# ── Shared hyper-parameters ──────────────────────────────────────────
NUM_POINTS=15000
TRAIN_BATCH_SIZE=32
VAL_BATCH_SIZE=32
MAX_ITERS=40000
VAL_FREQ=10000
SEED=42

DATASET="ShapeNetAD"
DATASET_PATH="./data/dataset/pcd"

# Patch-gen parameters (used only in Config 1)
PATCH_NUM=128
PATCH_SCALE=0.05

# ── Per-config settings ──────────────────────────────────────────────
case $CONFIG_IDX in
    0)
        MODE="base"
        NUM_AUG=2048
        LOG_ROOT="./logs_ae/comparison_base"
        TAG="base"
        EXTRA_ARGS=(--use_patch False)
        ;;
    1)
        MODE="patch"
        NUM_AUG=2048
        LOG_ROOT="./logs_ae/comparison_patch"
        TAG="patch"
        EXTRA_ARGS=(--use_patch True --patch_num "$PATCH_NUM" --patch_scale "$PATCH_SCALE")
        ;;
    2)
        MODE="af3ad"
        NUM_AUG=4096
        LOG_ROOT="./logs_ae/comparison_af3ad"
        TAG="af3ad"
        EXTRA_ARGS=(--use_af3ad True)
        ;;
    *)
        echo "ERROR: unexpected CONFIG_IDX=$CONFIG_IDX"
        exit 1
        ;;
esac

echo "============================================================"
echo " Config : $MODE  (CONFIG_IDX=$CONFIG_IDX)"
echo " Category : $CATEGORY  (CAT_IDX=$CAT_IDX)"
echo " NUM_POINTS : $NUM_POINTS"
echo " NUM_AUG    : $NUM_AUG"
echo " Batch size : $TRAIN_BATCH_SIZE"
echo " Max iters  : $MAX_ITERS"
echo " Seed       : $SEED"
echo "============================================================"

# ── Launch training ──────────────────────────────────────────────────
python train_ae.py \
    --category         "$CATEGORY" \
    --log_root         "$LOG_ROOT" \
    --tag              "$TAG" \
    --dataset          "$DATASET" \
    --dataset_path     "$DATASET_PATH" \
    --num_points       "$NUM_POINTS" \
    --num_aug          "$NUM_AUG" \
    --train_batch_size "$TRAIN_BATCH_SIZE" \
    --val_batch_size   "$VAL_BATCH_SIZE" \
    --max_iters        "$MAX_ITERS" \
    --val_freq         "$VAL_FREQ" \
    --seed             "$SEED" \
    --save_ply         True \
    --no_tensorboard   True \
    "${EXTRA_ARGS[@]}"

echo "=== Job finished at $(date) ==="
