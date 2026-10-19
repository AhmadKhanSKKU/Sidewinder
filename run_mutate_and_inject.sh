#!/bin/bash
# run_mutate_and_inject.sh

WORLD_TYPE="closed_world"
MAX_STEPS=10
OUTPUT_FILE="mutate_and_inject.md"

DATASETS=("cic2025" "cic2023" "TTDF")
DEFENSES=("transformer" "cnn" "dl" "vib" "pandora_seq")
ATTACKS=("spoofing" "web" "ddos" "dos")

echo "=========================================================="
echo "[★] STARTING MULTI-DATASET MUTATE-AND-INJECT EVALUATION"
echo "[*] Datasets: ${DATASETS[*]}"
echo "[*] Defenses: ${DEFENSES[*]}"
echo "[*] Attacks:  ${ATTACKS[*]}"
echo "=========================================================="

python3 -m advGenerate.mutate_and_inject \
    --world_type "$WORLD_TYPE" \
    --datasets "${DATASETS[@]}" \
    --defenses "${DEFENSES[@]}" \
    --attacks "${ATTACKS[@]}" \
    --max_steps "$MAX_STEPS" \
    --out_md "$OUTPUT_FILE"

echo "=========================================================="
echo "[★] COMPLETED! Results saved to $OUTPUT_FILE"
echo "=========================================================="