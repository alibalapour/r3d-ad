#!/usr/bin/env bash
#
# run_sequential_comparison.sh
#
# Runs the same 15-class × 3-config comparison as run_slurm_comparison.sh,
# but sequentially on a single machine (no SLURM required).
#
# Three training modes across 15 AnomalyShapeNet classes (index-0 variants):
#   1. Base mode    – no pseudo anomaly patches  (--use_patch False)
#   2. Patch-Gen    – random_patch pseudo anomalies (--use_patch True)
#   3. AF3AD mode   – AF3AD pseudo anomaly synthesis (--use_af3ad True)
#
# AF3AD uses NUM_AUG=4096 (more presets → more training samples).
# Base and Patch-Gen use NUM_AUG=2048.
#
# All 45 runs execute sequentially so the script is safe to leave
# running unattended.

set -euo pipefail

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

# ── Shared hyper-parameters ──────────────────────────────────────────
NUM_POINTS=15000
TRAIN_BATCH_SIZE=32
VAL_BATCH_SIZE=32
MAX_ITERS=40000
VAL_FREQ=10000
SEED=42

DATASET="ShapeNetAD"
DATASET_PATH="./data/dataset/pcd"

# Patch-gen parameters (used only in Patch-Gen mode)
PATCH_NUM=128
PATCH_SCALE=0.05

# Logging
LOG_ROOT_BASE="./logs_ae/comparison_base"
LOG_ROOT_PATCH="./logs_ae/comparison_patch"
LOG_ROOT_AF3AD="./logs_ae/comparison_af3ad"
# ─────────────────────────────────────────────────────────────────────

# Common arguments shared by base and patch modes (NUM_AUG=2048)
COMMON_ARGS_BASE=(
    --dataset      "$DATASET"
    --dataset_path "$DATASET_PATH"
    --num_points   "$NUM_POINTS"
    --num_aug      2048
    --train_batch_size "$TRAIN_BATCH_SIZE"
    --val_batch_size   "$VAL_BATCH_SIZE"
    --max_iters    "$MAX_ITERS"
    --val_freq     "$VAL_FREQ"
    --seed         "$SEED"
    --save_ply     True
)

# Common arguments for AF3AD mode (NUM_AUG=4096)
COMMON_ARGS_AF3AD=(
    --dataset      "$DATASET"
    --dataset_path "$DATASET_PATH"
    --num_points   "$NUM_POINTS"
    --num_aug      4096
    --train_batch_size "$TRAIN_BATCH_SIZE"
    --val_batch_size   "$VAL_BATCH_SIZE"
    --max_iters    "$MAX_ITERS"
    --val_freq     "$VAL_FREQ"
    --seed         "$SEED"
    --save_ply     True
)

echo "============================================================"
echo " R3D-AD  –  Base vs. Patch-Gen vs. AF3AD Comparison"
echo "============================================================"
echo " Categories : ${CATEGORIES[*]}"
echo " Points     : $NUM_POINTS"
echo " Batch size : $TRAIN_BATCH_SIZE"
echo " Max iters  : $MAX_ITERS"
echo " Seed       : $SEED"
echo " Num AUG    : base/patch=2048, af3ad=4096"
echo "============================================================"
echo ""

# ── Mode 1: Base (no pseudo anomalies) ──────────────────────────────
echo ">>> MODE 1 – Base (use_patch=False)"
echo "------------------------------------------------------------"
for CATEGORY in "${CATEGORIES[@]}"; do
    echo ""
    echo "  ▸ Training category: $CATEGORY"
    python train_ae.py \
        --category   "$CATEGORY" \
        --log_root   "$LOG_ROOT_BASE" \
        --tag        "base" \
        --use_patch  False \
        "${COMMON_ARGS_BASE[@]}"
    echo "  ✓ Finished base training for $CATEGORY"
done

# ── Mode 2: Patch-Gen (with random_patch pseudo anomalies) ──────────
echo ""
echo ">>> MODE 2 – Patch-Gen (use_patch=True)"
echo "------------------------------------------------------------"
for CATEGORY in "${CATEGORIES[@]}"; do
    echo ""
    echo "  ▸ Training category: $CATEGORY"
    python train_ae.py \
        --category    "$CATEGORY" \
        --log_root    "$LOG_ROOT_PATCH" \
        --tag         "patch" \
        --use_patch   True \
        --patch_num   "$PATCH_NUM" \
        --patch_scale "$PATCH_SCALE" \
        "${COMMON_ARGS_BASE[@]}"
    echo "  ✓ Finished patch-gen training for $CATEGORY"
done

# ── Mode 3: AF3AD (pseudo anomalies via AF3AD synthesiser) ──────────
echo ""
echo ">>> MODE 3 – AF3AD (use_af3ad=True)"
echo "------------------------------------------------------------"
for CATEGORY in "${CATEGORIES[@]}"; do
    echo ""
    echo "  ▸ Training category: $CATEGORY"
    python train_ae.py \
        --category    "$CATEGORY" \
        --log_root    "$LOG_ROOT_AF3AD" \
        --tag         "af3ad" \
        --use_af3ad   True \
        "${COMMON_ARGS_AF3AD[@]}"
    echo "  ✓ Finished AF3AD training for $CATEGORY"
done

echo ""
echo "============================================================"
echo " All runs complete."
echo " Base    logs → $LOG_ROOT_BASE"
echo " Patch   logs → $LOG_ROOT_PATCH"
echo " AF3AD   logs → $LOG_ROOT_AF3AD"
echo "============================================================"
