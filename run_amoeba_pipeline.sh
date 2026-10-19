#!/bin/bash
# run_amoeba_pipeline.sh

WORLD_TYPE="closed_world"
OUTPUT_FILE="amoeba_evaluation_results.md"
MAX_STEPS=10

# Configure datasets, defenses, and attack vectors
DATASETS=("cic2023" "cic2025" "TTDF")
DEFENSES=("dl" "cnn" "vib" "pandora_seq" transformer)
ATTACKS=("dos" "ddos" "web" "spoofing")

echo "=========================================================="
echo "[★] STARTING AMOEBA BASELINE EVALUATION (GRU-SAC)"
echo "[*] Datasets: ${DATASETS[*]}"
echo "[*] Defenses: ${DEFENSES[*]}"
echo "[*] Attacks:  ${ATTACKS[*]}"
echo "=========================================================="

python3 -m advGenerate.amoeba_eval \
    --world_type "$WORLD_TYPE" \
    --datasets "${DATASETS[@]}" \
    --defenses "${DEFENSES[@]}" \
    --attacks "${ATTACKS[@]}" \
    --max_steps "$MAX_STEPS" \
    --out_md "$OUTPUT_FILE"

echo "=========================================================="
echo "[★] ALL AMOEBA EXPERIMENTS COMPLETE!"
echo "[★] Full Markdown summary generated at: $OUTPUT_FILE"
echo "=========================================================="