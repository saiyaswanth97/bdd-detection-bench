#!/bin/bash

WEIGHTS_DIR="output"
EVAL_DIR="output/eval"

# Create eval base directory if it doesn't exist
mkdir -p "$EVAL_DIR"

for weight_path in "$WEIGHTS_DIR"/*.pth; do
    # Skip if no files found
    [ -e "$weight_path" ] || continue

    # Extract filename without extension
    filename=$(basename "$weight_path" .pth)

    # Create matching output folder
    out_dir="$EVAL_DIR/$filename"
    mkdir -p "$out_dir"

    echo "======================================"
    echo "Evaluating: $filename"
    echo "Output Dir: $out_dir"
    echo "======================================"

    python3 -m src.task2.eval_models \
        --weights "$weight_path" \
        --output-dir "$out_dir"

done

echo "All evaluations completed."
