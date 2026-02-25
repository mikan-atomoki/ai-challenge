#!/bin/bash
# QAT Seed Sweep — 8 GPU × 3 runs each = 24 runs
# Usage: bash run_sweep.sh

set -e
mkdir -p results

SEEDS=(1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24)
NUM_GPUS=8

echo "=== QAT Sweep: ${#SEEDS[@]} runs on ${NUM_GPUS} GPUs ==="

# Launch all runs, distributing across GPUs
PIDS=()
for i in "${!SEEDS[@]}"; do
    GPU_ID=$((i % NUM_GPUS))
    SEED=${SEEDS[$i]}
    echo "Launching seed=${SEED} on GPU ${GPU_ID}"
    CUDA_VISIBLE_DEVICES=${GPU_ID} python qat_sweep.py \
        --seed ${SEED} --out results/run_${SEED}.pt \
        > results/log_${SEED}.txt 2>&1 &
    PIDS+=($!)

    # Wait for a batch to finish before launching more (3 per GPU)
    if (( (i + 1) % NUM_GPUS == 0 )); then
        echo "  Waiting for batch $((i / NUM_GPUS + 1))..."
        for pid in "${PIDS[@]}"; do
            wait $pid
        done
        PIDS=()
        echo "  Batch done."
    fi
done

# Wait for remaining
for pid in "${PIDS[@]}"; do
    wait $pid
done

echo ""
echo "=== All runs complete. Finding best... ==="
python -c "
import os, torch
best_acc, best_file = 0, ''
for f in sorted(os.listdir('results')):
    if not f.endswith('.pt'): continue
    path = os.path.join('results', f)
    ckpt = torch.load(path, map_location='cpu', weights_only=False)
    acc = ckpt['accuracy']
    seed = ckpt['seed']
    print(f'  seed={seed:3d}  acc={acc:.2f}%  ({f})')
    if acc > best_acc:
        best_acc = acc
        best_file = path
print()
print(f'  BEST: {best_acc:.2f}% ({best_file})')
# Copy best to checkpoints
import shutil
shutil.copy(best_file, 'checkpoints/student_bitnet.pt')
print(f'  -> Saved to checkpoints/student_bitnet.pt')
"
