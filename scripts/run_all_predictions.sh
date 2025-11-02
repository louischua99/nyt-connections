#!/bin/bash
# Generate predictions for all 11 trained models concurrently across 3 H100 GPUs
# Each model runs on a dedicated GPU to maximize throughput

set -e

# Create output directory
mkdir -p data/predictions
mkdir -p logs

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
echo "=========================================="
echo "Launching all predictions at $TIMESTAMP"
echo "Distributing across GPUs 0, 1, 2"
echo "=========================================="

# GPU 0: 4 predictions
echo ""
echo "GPU 0: Launching 4 predictions..."

CUDA_VISIBLE_DEVICES=0 nohup python -u scripts/generate_predictions.py \
    --model models/exp1_baseline --output data/predictions/exp1_baseline.json --gpu 0 \
    > logs/pred_exp1_baseline_${TIMESTAMP}.log 2>&1 &
echo "  [GPU 0] exp1_baseline (PID: $!)"

CUDA_VISIBLE_DEVICES=0 nohup python -u scripts/generate_predictions.py \
    --model models/exp1_full --output data/predictions/exp1_full.json --gpu 0 \
    > logs/pred_exp1_full_${TIMESTAMP}.log 2>&1 &
echo "  [GPU 0] exp1_full (PID: $!)"

CUDA_VISIBLE_DEVICES=0 nohup python -u scripts/generate_predictions.py \
    --model models/exp2_mixed --output data/predictions/exp2_mixed.json --gpu 0 \
    > logs/pred_exp2_mixed_${TIMESTAMP}.log 2>&1 &
echo "  [GPU 0] exp2_mixed (PID: $!)"

CUDA_VISIBLE_DEVICES=0 nohup python -u scripts/generate_predictions.py \
    --model models/exp3_warmup/phase2_full_final --output data/predictions/exp3_warmup.json --gpu 0 \
    > logs/pred_exp3_warmup_${TIMESTAMP}.log 2>&1 &
GPU0_LAST_PID=$!
echo "  [GPU 0] exp3_warmup (PID: $GPU0_LAST_PID)"

# GPU 1: 4 predictions
echo ""
echo "GPU 1: Launching 4 predictions..."

CUDA_VISIBLE_DEVICES=1 nohup python -u scripts/generate_predictions.py \
    --model models/exp1_permutation --output data/predictions/exp1_permutation.json --gpu 1 \
    > logs/pred_exp1_permutation_${TIMESTAMP}.log 2>&1 &
echo "  [GPU 1] exp1_permutation (PID: $!)"

CUDA_VISIBLE_DEVICES=1 nohup python -u scripts/generate_predictions.py \
    --model models/exp2_structured --output data/predictions/exp2_structured.json --gpu 1 \
    > logs/pred_exp2_structured_${TIMESTAMP}.log 2>&1 &
echo "  [GPU 1] exp2_structured (PID: $!)"

CUDA_VISIBLE_DEVICES=1 nohup python -u scripts/generate_predictions.py \
    --model models/exp2_sequential/phase2_structured_final --output data/predictions/exp2_sequential.json --gpu 1 \
    > logs/pred_exp2_sequential_${TIMESTAMP}.log 2>&1 &
echo "  [GPU 1] exp2_sequential (PID: $!)"

CUDA_VISIBLE_DEVICES=1 nohup python -u scripts/generate_predictions.py \
    --model models/exp3_staged/phase3_nyt_final --output data/predictions/exp3_staged.json --gpu 1 \
    > logs/pred_exp3_staged_${TIMESTAMP}.log 2>&1 &
GPU1_LAST_PID=$!
echo "  [GPU 1] exp3_staged (PID: $GPU1_LAST_PID)"

# GPU 2: 3 predictions
echo ""
echo "GPU 2: Launching 3 predictions..."

CUDA_VISIBLE_DEVICES=2 nohup python -u scripts/generate_predictions.py \
    --model models/exp1_synthetic --output data/predictions/exp1_synthetic.json --gpu 2 \
    > logs/pred_exp1_synthetic_${TIMESTAMP}.log 2>&1 &
echo "  [GPU 2] exp1_synthetic (PID: $!)"

CUDA_VISIBLE_DEVICES=2 nohup python -u scripts/generate_predictions.py \
    --model models/exp2_unstructured --output data/predictions/exp2_unstructured.json --gpu 2 \
    > logs/pred_exp2_unstructured_${TIMESTAMP}.log 2>&1 &
echo "  [GPU 2] exp2_unstructured (PID: $!)"

CUDA_VISIBLE_DEVICES=2 nohup python -u scripts/generate_predictions.py \
    --model models/exp3_no_warmup --output data/predictions/exp3_no_warmup.json --gpu 2 \
    > logs/pred_exp3_no_warmup_${TIMESTAMP}.log 2>&1 &
GPU2_LAST_PID=$!
echo "  [GPU 2] exp3_no_warmup (PID: $GPU2_LAST_PID)"

echo ""
echo "=========================================="
echo "All 11 predictions launched!"
echo "=========================================="
echo ""
echo "Monitor progress with:"
echo "  watch -n 10 'nvidia-smi'"
echo "  tail -f logs/pred_*_${TIMESTAMP}.log"
echo ""
echo "Check completion:"
echo "  ls -lh data/predictions/    # Check prediction files"
echo ""
echo "Sample PIDs to check:"
echo "  ps -p $GPU0_LAST_PID,$GPU1_LAST_PID,$GPU2_LAST_PID"
echo ""
echo "All logs saved with timestamp: ${TIMESTAMP}"
echo "=========================================="
