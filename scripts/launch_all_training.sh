#!/bin/bash
# Launch all 11 experiments across 3 H100 GPUs with nohup logging
# Each experiment runs on a dedicated GPU to maximize throughput

set -e

# Create logs directory
mkdir -p logs
mkdir -p models

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
echo "=========================================="
echo "Launching all experiments at $TIMESTAMP"
echo "Distributing across GPUs 0, 1, 2"
echo "=========================================="

# Training parameters
EPOCHS=8
LR=2e-4

# GPU 0: 4 experiments
echo ""
echo "GPU 0: Launching 4 experiments..."

CUDA_VISIBLE_DEVICES=0 nohup python -u scripts/train_experiment.py \
    --experiment exp1_baseline --epochs $EPOCHS --lr $LR \
    > logs/exp1_baseline_${TIMESTAMP}.log 2>&1 &
echo "  [GPU 0] exp1_baseline (PID: $!)"

CUDA_VISIBLE_DEVICES=0 nohup python -u scripts/train_experiment.py \
    --experiment exp1_full --epochs $EPOCHS --lr $LR \
    > logs/exp1_full_${TIMESTAMP}.log 2>&1 &
echo "  [GPU 0] exp1_full (PID: $!)"

CUDA_VISIBLE_DEVICES=0 nohup python -u scripts/train_experiment.py \
    --experiment exp2_mixed --epochs $EPOCHS --lr $LR \
    > logs/exp2_mixed_${TIMESTAMP}.log 2>&1 &
echo "  [GPU 0] exp2_mixed (PID: $!)"

CUDA_VISIBLE_DEVICES=0 nohup python -u scripts/train_experiment.py \
    --experiment exp3_no_warmup --epochs $EPOCHS --lr $LR \
    > logs/exp3_no_warmup_${TIMESTAMP}.log 2>&1 &
GPU0_LAST_PID=$!
echo "  [GPU 0] exp3_no_warmup (PID: $GPU0_LAST_PID)"

# GPU 1: 4 experiments
echo ""
echo "GPU 1: Launching 4 experiments..."

CUDA_VISIBLE_DEVICES=1 nohup python -u scripts/train_experiment.py \
    --experiment exp1_permutation --epochs $EPOCHS --lr $LR \
    > logs/exp1_permutation_${TIMESTAMP}.log 2>&1 &
echo "  [GPU 1] exp1_permutation (PID: $!)"

CUDA_VISIBLE_DEVICES=1 nohup python -u scripts/train_experiment.py \
    --experiment exp2_structured --epochs $EPOCHS --lr $LR \
    > logs/exp2_structured_${TIMESTAMP}.log 2>&1 &
echo "  [GPU 1] exp2_structured (PID: $!)"

CUDA_VISIBLE_DEVICES=1 nohup python -u scripts/train_experiment.py \
    --experiment exp2_sequential --epochs $EPOCHS --lr $LR \
    > logs/exp2_sequential_${TIMESTAMP}.log 2>&1 &
echo "  [GPU 1] exp2_sequential (PID: $!)"

CUDA_VISIBLE_DEVICES=1 nohup python -u scripts/train_experiment.py \
    --experiment exp3_warmup --epochs $EPOCHS --lr $LR \
    > logs/exp3_warmup_${TIMESTAMP}.log 2>&1 &
GPU1_LAST_PID=$!
echo "  [GPU 1] exp3_warmup (PID: $GPU1_LAST_PID)"

# GPU 2: 3 experiments
echo ""
echo "GPU 2: Launching 3 experiments..."

CUDA_VISIBLE_DEVICES=2 nohup python -u scripts/train_experiment.py \
    --experiment exp1_synthetic --epochs $EPOCHS --lr $LR \
    > logs/exp1_synthetic_${TIMESTAMP}.log 2>&1 &
echo "  [GPU 2] exp1_synthetic (PID: $!)"

CUDA_VISIBLE_DEVICES=2 nohup python -u scripts/train_experiment.py \
    --experiment exp2_unstructured --epochs $EPOCHS --lr $LR \
    > logs/exp2_unstructured_${TIMESTAMP}.log 2>&1 &
echo "  [GPU 2] exp2_unstructured (PID: $!)"

CUDA_VISIBLE_DEVICES=2 nohup python -u scripts/train_experiment.py \
    --experiment exp3_staged --epochs $EPOCHS --lr $LR \
    > logs/exp3_staged_${TIMESTAMP}.log 2>&1 &
GPU2_LAST_PID=$!
echo "  [GPU 2] exp3_staged (PID: $GPU2_LAST_PID)"

echo ""
echo "=========================================="
echo "All 11 experiments launched!"
echo "=========================================="
echo ""
echo "Monitor progress with:"
echo "  watch -n 10 'nvidia-smi'"
echo "  tail -f logs/exp*_${TIMESTAMP}.log"
echo "  bash scripts/monitor_all_experiments.sh"
echo ""
echo "Check completion:"
echo "  ls -lh models/exp*/    # Check if model directories exist"
echo ""
echo "Sample PIDs to check:"
echo "  ps -p $GPU0_LAST_PID,$GPU1_LAST_PID,$GPU2_LAST_PID"
echo ""
echo "All logs saved with timestamp: ${TIMESTAMP}"
echo "=========================================="
