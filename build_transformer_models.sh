#!/bin/bash
# build_transformer_models.sh
# Usage: ./build_transformer_models.sh [WORLD_TYPE]
# Example: ./build_transformer_models.sh open_world

WORLD_TYPE=${1:-closed_world} # Options: closed_world | open_world
DATASETS=("cic2023" "cic2025" "TTDF")

echo "=========================================================="
echo "[★] STARTING TRANSFORMER IDS AUTOMATION FOR ALL DATASETS [$WORLD_TYPE]"
echo "=========================================================="

for DATASET_NAME in "${DATASETS[@]}"; do
    PKL_DIR="data/$DATASET_NAME"
    BENIGN_PKL="$PKL_DIR/BenignTraffic.pkl"
    OUT_DIR="models/$WORLD_TYPE/$DATASET_NAME/transformer"
    REPORT_MD="transformer_training_report.md"

    mkdir -p "$OUT_DIR"

    echo "----------------------------------------------------------"
    echo "[*] Processing Dataset: $DATASET_NAME [$WORLD_TYPE]"
    echo "[*] Benign Baseline: $BENIGN_PKL"
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

        echo "[*] Training Transformer IDS for Attack: $attack_name ($DATASET_NAME) [$WORLD_TYPE]"

        MODEL_OUT="$OUT_DIR/${attack_name}_ids.pth"

        python3 -m advGenerate.train_transformer_ids \
            --world_type "$WORLD_TYPE" \
            --benign "$BENIGN_PKL" \
            --malicious "$attack_pkl" \
            --data_dir "$PKL_DIR" \
            --model_out "$MODEL_OUT" \
            --report_md "$REPORT_MD"
    done
done

echo "=========================================================="
echo "[★] ALL DATASETS COMPLETE: $WORLD_TYPE Transformer Models Trained!"
echo "=========================================================="