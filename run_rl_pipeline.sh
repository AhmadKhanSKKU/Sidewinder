#!/bin/bash
# run_rl_pipeline.sh

WORLD_TYPE="open_world"
FEEDBACK_MODE="hard_labeled"
ARCH="mamba" # Options: mamba | lstm | gru | netmasquerade
OUTPUT_FILE="mamba_cnn_dos.md"

DATASETS=("cic2025" "cic2023" "TTDF")
DEFENSES=("transformer" "cnn" "dl" "vib" "pandora")
ATTACKS=("spoofing" "web" "ddos" "dos")

# Initialize / Overwrite markdown summary file
echo "# Multi-Dataset Evaluation Results (${ARCH^^})" > "$OUTPUT_FILE"
echo "**World Type:** \`$WORLD_TYPE\` | **Feedback Mode:** \`$FEEDBACK_MODE\`" >> "$OUTPUT_FILE"
echo "" >> "$OUTPUT_FILE"

echo "=========================================================="
echo "[★] STARTING RL PIPELINE EVALUATION (${ARCH^^})"
echo "[*] Datasets: ${DATASETS[*]}"
echo "[*] Defenses: ${DEFENSES[*]}"
echo "[*] Attacks:  ${ATTACKS[*]}"
echo "=========================================================="

for dataset in "${DATASETS[@]}"; do
    for defense in "${DEFENSES[@]}"; do
        for attack in "${ATTACKS[@]}"; do
            
            # Skip execution if dataset file does not exist
            if [ ! -f "data/${dataset}/${attack}.pkl" ]; then
                echo "[!] Dataset file data/${dataset}/${attack}.pkl not found. Skipping..."
                continue
            fi

            echo ""
            echo "----------------------------------------------------------"
            echo "▶ RUNNING: Dataset=${dataset} | Defense=${defense} \vert{} Attack=${attack}"
            echo "----------------------------------------------------------"
            
            python3 -m advGenerate.train \
                --world_type "$WORLD_TYPE" \
                --dataset "$dataset" \
                --attack "$attack" \
                --defense "$defense" \
                --feedback_mode "$FEEDBACK_MODE" \
                --arch "$ARCH" \
                --run_mode "train" \
                --disable_per \
                --out_md "$OUTPUT_FILE"
                
        done
    done
done

echo "=========================================================="
echo "[★] ALL EXPERIMENTS COMPLETE!"
echo "[★] Full Markdown summary generated at: $OUTPUT_FILE"
echo "=========================================================="