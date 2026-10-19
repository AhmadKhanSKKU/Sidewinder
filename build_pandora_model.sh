#!/bin/bash
# build_pandora_models.sh
# Usage: ./build_pandora_models.sh [WORLD_TYPE]
# Example: ./build_pandora_models.sh open_world

WORLD_TYPE=${1:-closed_world} # Options: closed_world | open_world
DATASETS=("cic2023" "cic2025" "TTDF")

echo "=========================================================="
echo "[★] STARTING PANDORA-SEQ AUTOMATION FOR ALL DATASETS [$WORLD_TYPE]"
echo "=========================================================="

for DATASET_NAME in "${DATASETS[@]}"; do
    PKL_DIR="data/$DATASET_NAME"
    BENIGN_PKL="$PKL_DIR/BenignTraffic.pkl"
    OUT_DIR="models/$WORLD_TYPE/$DATASET_NAME/pandora_seq"
    REPORT_MD="pandora_seq_training_report.md"

    mkdir -p "$OUT_DIR"

    echo "----------------------------------------------------------"
    echo "[*] Processing Dataset: $DATASET_NAME [$WORLD_TYPE]"
    echo "[*] Output Directory: $OUT_DIR"
    echo "[*] Markdown Report: $REPORT_MD"
    echo "----------------------------------------------------------"

    if [ ! -f "$BENIGN_PKL" ]; then
        echo "[!] WARNING: Benign baseline not found at $BENIGN_PKL. Skipping $DATASET_NAME..."
        continue
    fi

    for attack_pkl in "$PKL_DIR"/*.pkl; do
        [ -e "$attack_pkl" ] || continue
        filename=$(basename -- "$attack_pkl")
        attack_name="${filename%.*}"

        if [ "$attack_name" == "BenignTraffic" ] || [ "$attack_name" == "Recon-PortScan" ]; then
            continue
        fi

        echo "[*] Training Pandora-Seq IDS for Attack: $attack_name ($DATASET_NAME) [$WORLD_TYPE]"

        python3 -m advGenerate.train_pandora_seq \
            --world_type "$WORLD_TYPE" \
            --benign "$BENIGN_PKL" \
            --malicious "$attack_pkl" \
            --data_dir "$PKL_DIR" \
            --save_dir "$OUT_DIR" \
            --report_md "$REPORT_MD"
    done
done

echo "=========================================================="
echo "[★] ALL DATASETS COMPLETE: $WORLD_TYPE Pandora-Seq Models Trained!"
echo "=========================================================="