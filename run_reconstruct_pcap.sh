#!/usr/bin/env bash
set -e

# Arguments or Defaults
MODEL_DIR="${1:-.}"
INPUT_PKL="${2:-./data/cic2025/ddos.pkl}"
OUTPUT_PCAP="${3:-out.pcap}"
MAX_STEPS="${4:-10}"

echo "=========================================================="
echo "    Sidewinder: Feature-to-PCAP Reconstruction Pipeline"
echo "=========================================================="
echo " Model Directory : ${MODEL_DIR}"
echo " Input Feature PKL: ${INPUT_PKL}"
echo " Output PCAP Path : ${OUTPUT_PCAP}"
echo " Max Step Budget  : ${MAX_STEPS}"
echo "=========================================================="

python reconstruct_pcap.py \
    --model-dir "${MODEL_DIR}" \
    --input-pkl "${INPUT_PKL}" \
    --output-pcap "${OUTPUT_PCAP}" \
    --max-steps "${MAX_STEPS}"

echo "[+] PCAP Reconstruction Complete."