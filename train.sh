#!/bin/bash
# GTR: Gated Token Recurrence for Efficient Dense Prediction
# Copyright (c) 2026 The GTR Authors. All Rights Reserved.

# Multi-GPU training launcher with auto-resume and session logging.
# Usage: [CONFIG=...] [OUTPUT_DIR=...] [CUDA_VISIBLE_DEVICES=...] ./train.sh [extra train.py args]
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

CONFIG="${CONFIG:-configs/det/coco_finetune/gtr_s.yml}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/$(basename "$(dirname "$CONFIG")")/$(basename "$CONFIG" .yml)}"
SEED="${SEED:-0}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
NP=$(awk -F',' 'NF {print NF}' <<< "$CUDA_VISIBLE_DEVICES")

export OMP_NUM_THREADS=1
export OPENCV_NUM_THREADS=1

# Numerical regime of the released checkpoints: pure FP32 (use_amp: False) with TF32 matmul/conv.
export GTR_TF32="${GTR_TF32:-1}"
export GTR_FAST_MAL_LOSS="${GTR_FAST_MAL_LOSS:-1}"
export GTR_SANITIZE_LOSSES="${GTR_SANITIZE_LOSSES:-0}"

mkdir -p "$OUTPUT_DIR"
RUN_LOG="$OUTPUT_DIR/train_$(date +%Y%m%d_%H%M%S).log"
echo "===== $(date '+%F %T') config=$CONFIG gpus=$NP output=$OUTPUT_DIR =====" | tee -a "$RUN_LOG"

# train.py auto-resumes from the latest checkpoint in output_dir by default.
torchrun --standalone --nproc_per_node="$NP" train.py \
    -c "$CONFIG" \
    --seed="$SEED" \
    -u "output_dir=$OUTPUT_DIR" \
    "$@" \
    2>&1 | tee -a "$RUN_LOG"
