#!/bin/bash
# run_netmasquerade.sh

WORLD_TYPE="closed_world"
MAX_STEPS=10
OUTPUT_FILE="netmasquerade.md"

DATASETS=("cic2025" "cic2023" "TTDF")
DEFENSES=("transformer" "cnn" "dl" "vib" "pandora_seq")
ATTACKS=("ddos", "dos", "spoofing", "web")

echo "=========================================================="
echo "[★] STARTING MULTI-DATASET NETMASQUERADE (SAC-GRU) EVALUATION"
echo "[*] Datasets: ${DATASETS[*]}"
echo "[*] Defenses: ${DEFENSES[*]}"
echo "[*] Attacks:  ${ATTACKS[*]}"
echo "=========================================================="

python3 -m advGenerate.netmasquerade_eval \
    --world_type "$WORLD_TYPE" \
    --datasets "${DATASETS[@]}" \
    --defenses "${DEFENSES[@]}" \
    --attacks "${ATTACKS[@]}" \
    --max_steps "$MAX_STEPS" \
    --out_md "$OUTPUT_FILE"

echo "=========================================================="
echo "[★] COMPLETED! Results saved to $OUTPUT_FILE"
echo "=========================================================="