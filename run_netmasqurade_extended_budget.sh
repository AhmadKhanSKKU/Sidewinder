#!/bin/bash
# run_netmasquerade_extended_budget.sh

# Ensure python finds local modules when executed from src/
export PYTHONPATH="${PYTHONPATH}:."

WORLD_TYPE="closed_world"
FEEDBACK_MODE="hard_labeled"
ARCH="netmasquerade"
MAX_STEPS=3000
OUTPUT_FILE="netmasquerade_extended_budget.md"

# Initialize / Overwrite markdown summary file
echo "# Extended Budget Evaluation Results (${ARCH^^} - ${MAX_STEPS} Steps)" > "$OUTPUT_FILE"
echo "**World Type:** \`$WORLD_TYPE\` | **Feedback Mode:** \`$FEEDBACK_MODE\` | **Max Steps:** \`$MAX_STEPS\`" >> "$OUTPUT_FILE"
echo "" >> "$OUTPUT_FILE"

echo "=========================================================="
echo "[★] STARTING EXTENDED BUDGET EVALUATION (${ARCH^^})"
echo "[*] Target Steps: $MAX_STEPS"
echo "[*] Output: $OUTPUT_FILE"
echo "=========================================================="

# Define target failed combinations (dataset:defense:attack)
TARGETS=(
    # --- CIC-2023 Failures ---
    # "cic2023:cnn:web"
    # "cic2023:cnn:ddos"
    "cic2023:cnn:spoofing"
    "cic2023:transformer:spoofing"
    "cic2023:vib:spoofing"

    # --- CIC-2025 Failures ---
    # "cic2025:dl:ddos"
    "cic2025:pandora:ddos"
    # "cic2025:vib:ddos"
    # "cic2025:cnn:dos"
    # "cic2025:dl:dos"
    # "cic2025:vib:dos"
    # "cic2025:transformer:dos"

    # --- TTDF Failures ---
    # "TTDF:dl:dos"
    "TTDF:transformer:dos"
)

TOTAL=${#TARGETS[@]}
COUNT=1

for ITEM in "${TARGETS[@]}"; do
    IFS=":" read -r dataset defense attack <<< "$ITEM"

    # Skip execution if dataset file does not exist
    if [ ! -f "data/${dataset}/${attack}.pkl" ]; then
        echo "[!] Dataset file data/${dataset}/${attack}.pkl not found. Skipping..."
        continue
    fi

    echo ""
    echo "----------------------------------------------------------"
    echo "▶ RUNNING ($COUNT/$TOTAL): Dataset=${dataset} | Defense=${defense} \vert{} Attack=${attack}"
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

    ((COUNT++))
done

echo ""
echo "=========================================================="
echo "[★] ALL EXTENDED BUDGET RUNS COMPLETE!"
echo "[★] Results saved to: $OUTPUT_FILE"
echo "=========================================================="