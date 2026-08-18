#!/usr/bin/env bash
# End-to-end pipeline for one C-MAPSS subset.
# Usage: bash scripts/run_pipeline.sh FD001
set -e

SUBSET=${1:-FD001}
DEVICE=${2:-cuda}   # pass "cuda" if you have a GPU

echo "=== [1/4] Preprocessing $SUBSET ==="
python -m src.data.preprocessing --subset "$SUBSET"

echo "=== [2/4] Training teacher on $SUBSET ==="
python -m src.train_teacher --subset "$SUBSET" --device "$DEVICE"

echo "=== [3/4] Training student via KD on $SUBSET ==="
python -m src.train_student_kd --subset "$SUBSET" --device "$DEVICE"

echo "=== [4/4] Evaluating teacher vs student vs LSTM baseline on $SUBSET ==="
python -m src.evaluate --subset "$SUBSET" --device "$DEVICE"

echo "Done. See results/comparison_${SUBSET}.csv"
