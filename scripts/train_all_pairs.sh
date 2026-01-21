#!/bin/bash
# Train models for multiple currency pairs
# Usage: ./scripts/train_all_pairs.sh

set -e

# Activate virtual environment
source .venv/bin/activate

echo "======================================================================"
echo "TRAINING MODELS FOR MULTIPLE PAIRS"
echo "======================================================================"
echo ""

# Define pairs to train
PAIRS=("EURUSD" "GBPUSD" "USDJPY" "AUDUSD" "USDCAD")
EPOCHS=150  # Reduced for faster training

# Train each pair
for PAIR in "${PAIRS[@]}"; do
    echo "----------------------------------------------------------------------"
    echo "Training: $PAIR"
    echo "----------------------------------------------------------------------"

    poe train --symbol "$PAIR" --epochs $EPOCHS

    echo ""
    echo "✓ $PAIR model saved"
    echo ""
done

echo "======================================================================"
echo "ALL MODELS TRAINED SUCCESSFULLY"
echo "======================================================================"
echo ""
echo "Models saved in ./models/"
ls -lh ./models/*.onnx | awk '{print "  " $9, "(" $5 ")"}'
echo ""
