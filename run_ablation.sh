#!/bin/bash
# run_ablation.sh

# Default CLI Fallbacks
DATASET=${1:-"cic2023"}
DEFENSE=${2:-"cnn"}
ATTACK=${3:-"dos"}
WORLD_TYPE=${4:-"closed_world"}
FEEDBACK_MODE=${5:-"hard_labeled"}

OUTPUT_FILE="ablation_results_${DATASET}_${DEFENSE}_${ATTACK}.md"

echo "=========================================================="
echo "[★] STARTING ABLATION STUDY"
echo "[*] Dataset:       $DATASET"
echo "[*] Defense:       $DEFENSE"
echo "[*] Attack:        $ATTACK"
echo "[*] World Type:    $WORLD_TYPE"
echo "[*] Feedback Mode: $FEEDBACK_MODE"
echo "=========================================================="

# Re-initialize summary file
echo "# Ablation Study Results: $DATASET \vert{}$DEFENSE | $ATTACK" > "$OUTPUT_FILE"
echo "**World Type:** \`$WORLD_TYPE\` | **Feedback Mode:** \`$FEEDBACK_MODE\`" >> "$OUTPUT_FILE"
echo "" >> "$OUTPUT_FILE"

# Data check
if [ ! -f "data/${DATASET}/${ATTACK}.pkl" ]; then
    echo "[!] Error: Dataset file data/${DATASET}/${ATTACK}.pkl not found!"
    exit 1
fi

# ---------------------------------------------------------
# Experiment 2: Only BERT predictions at random places (No RL)
# ---------------------------------------------------------
echo ""
echo "----------------------------------------------------------"
echo "▶ Running Exp 2: BERT Predictions at Random Places (No RL)"
echo "----------------------------------------------------------"
python3 ablation_runner.py \
    --exp_type "bert_only" \
    --world_type "$WORLD_TYPE" \
    --dataset "$DATASET" \
    --defense "$DEFENSE" \
    --attack "$ATTACK" \
    --feedback_mode "$FEEDBACK_MODE" \
    --max_steps 10 \
    --out_md "$OUTPUT_FILE"

# ---------------------------------------------------------
# Experiment 3: RL Agent selecting Index AND Value (Amoeba-like)
# ---------------------------------------------------------
echo ""
echo "----------------------------------------------------------"
echo "▶ Running Exp 3: RL Agent selecting both Index and Value (Amoeba-like)"
echo "----------------------------------------------------------"
python3 ablation_runner.py \
    --exp_type "amoeba_rl" \
    --world_type "$WORLD_TYPE" \
    --dataset "$DATASET" \
    --defense "$DEFENSE" \
    --attack "$ATTACK" \
    --feedback_mode "$FEEDBACK_MODE" \
    --max_steps 10 \
    --out_md "$OUTPUT_FILE"

echo ""
echo "=========================================================="
echo "[★] ABLATION EXPERIMENTS COMPLETE!"
echo "[★] Markdown report generated at: $OUTPUT_FILE"
echo "=========================================================="