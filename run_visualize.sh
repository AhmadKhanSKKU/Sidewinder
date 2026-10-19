#!/bin/bash
# run_visualize.sh

WORLD_TYPE="closed_world" # Set to "closed_world" or "open_world"
FEEDBACK_MODE="hard_labeled"
ARCH="mamba"

DATASETS=("cic2023")
DEFENSES=("vib")
ATTACKS=("ddos")

echo "=========================================================="
echo "[★] STARTING DECISION BOUNDARY VISUALIZATION"
echo "=========================================================="

for dataset in "${DATASETS[@]}"; do
    for defense in "${DEFENSES[@]}"; do
        for attack in "${ATTACKS[@]}"; do
            
            if [ ! -f "data/${dataset}/${attack}.pkl" ]; then
                echo "[!] Dataset file data/${dataset}/${attack}.pkl not found. Skipping..."
                continue
            fi

            echo ""
            echo "----------------------------------------------------------"
            echo "▶ RUNNING: Dataset=${dataset} | Defense=${defense} | Attack=${attack} | World=${WORLD_TYPE}"
            echo "----------------------------------------------------------"
            
            python3 visualize_boundary.py \
                --world_type "$WORLD_TYPE" \
                --dataset "$dataset" \
                --attack "$attack" \
                --defense "$defense" \
                --feedback_mode "$FEEDBACK_MODE" \
                --arch "$ARCH" \
                --num_samples 39
                
        done
    done
done

echo "=========================================================="
echo "[★] VISUALIZATION PIPELINE COMPLETE!"
echo "=========================================================="