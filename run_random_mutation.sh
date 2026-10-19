#!/bin/bash
# run_random_mutation.sh

WORLD_TYPE="closed_world"
MAX_STEPS=10
OUTPUT_FILE="random_mutation.md"

DATASETS=("cic2025" "cic2023" "TTDF")
DEFENSES=("pandora_seq")
# DEFENSES=("transformer" "cnn" "dl" "vib" "pandora")
ATTACKS=("spoofing" "web" "ddos" "dos")

echo "=========================================================="
echo "[★] STARTING MULTI-DATASET RANDOM MUTATION EVALUATION"
echo "[*] Datasets: ${DATASETS[*]}"
echo "[*] Defenses: ${DEFENSES[*]}"
echo "[*] Attacks:  ${ATTACKS[*]}"
echo "=========================================================="

python3 -m advGenerate.random_mutation \
    --world_type "$WORLD_TYPE" \
    --datasets "${DATASETS[@]}" \
    --defenses "${DEFENSES[@]}" \
    --attacks "${ATTACKS[@]}" \
    --max_steps "$MAX_STEPS" \
    --out_md "$OUTPUT_FILE"

echo "=========================================================="
echo "[★] COMPLETED! Results saved to $OUTPUT_FILE"
echo "=========================================================="