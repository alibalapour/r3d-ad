#!/usr/bin/env bash
#
# run_comparison.sh
#
# Compares two training modes on three AnomalyShapeNet classes (ashtray0, bottle0, vase0):
#   1. Base mode   – no pseudo anomaly patches  (--use_patch False)
#   2. Modified mode – pseudo anomaly patches enabled (--use_patch True)
#
# Both modes share the same hyper-parameters.  Differences from the
# defaults in train_ae.py:
#   • num_points increased from 2048 → 4096
#   • train / val batch size reduced from 128 → 32
#   • PLY saving enabled (--save_ply True)
#
# All six runs execute sequentially so the script is safe to leave
# running unattended.

set -euo pipefail

# ── Configurable parameters ──────────────────────────────────────────
CATEGORIES=("ashtray0" "bottle0" "vase0")

NUM_POINTS=4096
NUM_AUG=2048
TRAIN_BATCH_SIZE=32
VAL_BATCH_SIZE=32
MAX_ITERS=40000
VAL_FREQ=1000
SEED=2020

DATASET="ShapeNetAD"
DATASET_PATH="./data/shapenet-ad"

# Patch-gen parameters (used only in modified mode)
PATCH_NUM=128
PATCH_SCALE=0.05

# Logging
LOG_ROOT_BASE="./logs_ae/comparison_base"
LOG_ROOT_PATCH="./logs_ae/comparison_patch"
# ─────────────────────────────────────────────────────────────────────

# Common arguments shared by both modes
COMMON_ARGS=(
    --dataset      "$DATASET"
    --dataset_path "$DATASET_PATH"
    --num_points   "$NUM_POINTS"
    --num_aug      "$NUM_AUG"
    --train_batch_size "$TRAIN_BATCH_SIZE"
    --val_batch_size   "$VAL_BATCH_SIZE"
    --max_iters    "$MAX_ITERS"
    --val_freq     "$VAL_FREQ"
    --seed         "$SEED"
    --save_ply     True
)

echo "============================================================"
echo " R3D-AD  –  Base vs. Patch-Gen Comparison"
echo "============================================================"
echo " Categories : ${CATEGORIES[*]}"
echo " Points     : $NUM_POINTS"
echo " Batch size : $TRAIN_BATCH_SIZE"
echo " Max iters  : $MAX_ITERS"
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
        "${COMMON_ARGS[@]}"
    echo "  ✓ Finished base training for $CATEGORY"
done

# ── Mode 2: Modified (with pseudo anomaly patches) ──────────────────
echo ""
echo ">>> MODE 2 – Modified (use_patch=True)"
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
        "${COMMON_ARGS[@]}"
    echo "  ✓ Finished patch-gen training for $CATEGORY"
done

echo ""
echo "============================================================"
echo " All runs complete."
echo " Base    logs → $LOG_ROOT_BASE"
echo " Patch   logs → $LOG_ROOT_PATCH"
echo "============================================================"
