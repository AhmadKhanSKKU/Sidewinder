#!/bin/bash

# Accept dataset folder name as argument (Defaults to cic2023)
DATASET=${1:-cic2023}
PKL_DIR="data/$DATASET"

echo "======================================================"
echo "      STARTING BATCH ML MODEL TRAINING (RF, SVM, XGB)"
echo "      Working Directory: $(pwd)"
echo "      Found Data in: $PKL_DIR"
echo "======================================================"

if [ ! -d "$PKL_DIR" ]; then
    echo "[!] ERROR: Dataset directory $PKL_DIR not found!"
    exit 1
fi

# Dynamically loop through all .pkl files
for attack_pkl in "$PKL_DIR"/*.pkl; do
    
    filename=$(basename -- "$attack_pkl")
    attack_name="${filename%.*}"

    # Skip the benign baseline
    if [[ "$attack_name" == *"Benign"* ]]; then
        continue
    fi

    echo "------------------------------------------------------"
    echo ">> Compiling ML models for: $attack_name"
    echo "------------------------------------------------------"
    
    # Add the current directory (src/) to the Python path
    export PYTHONPATH=".:$PYTHONPATH"
    
    # Execute the training script
    python3 advGenerate/train_ml_ids.py \
        --dataset "$DATASET" \
        --attack "$attack_name"
        
done

echo "======================================================"
echo "[+] ALL ML MODELS COMPILED AND SAVED SUCCESSFULLY."
echo "======================================================"