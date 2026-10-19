#!/bin/bash
# build_datasets.sh
# PLACE THIS FILE IN: project/my_attacker/src/

# 1. Accept dataset folder name as argument (Defaults to cic2023 if none provided)
DATASET_NAME=${1:-cic2023}

# 2. Track the PCAP path (Exactly two steps up from src)
DATASET_SRC_DIR="../../$DATASET_NAME"

# 3. Create an organized folder to hold your extracted .pkl files
OUT_DIR="data/$DATASET_NAME"
mkdir -p "$OUT_DIR"

echo "=========================================================="
echo "[★] STARTING DATASET PROCESSING FOR: $DATASET_NAME"
echo "[*] Source PCAPs: $DATASET_SRC_DIR"
echo "[*] Output PKLs:  $OUT_DIR"
echo "=========================================================="

# Check if the target dataset directory actually exists
if [ ! -d "$DATASET_SRC_DIR" ]; then
    echo "[!] ERROR: Directory '$DATASET_SRC_DIR' not found."
    echo "[!] Please verify you are running this from inside 'project/my_attacker/src/'"
    exit 1
fi

# 4. Loop strictly through files ending with '.pcap'
# This glob pattern naturally ignores the '.pcap:Zone.Identifier' files!
for pcap_path in "$DATASET_SRC_DIR"/*.pcap; do
    
    # Safety check in case no matching pcap files are found
    [ -f "$pcap_path" ] || continue

    filename=$(basename -- "$pcap_path")
    name_without_ext="${filename%.*}"

    echo "----------------------------------------------------------"
    echo "[*] Extracting flows from: $filename"
    echo "----------------------------------------------------------"

    # Run your dynamic vocabulary builder python script
    python3 -m trafficMimic.dataset.build_real_vocab \
        --pcap "$pcap_path" \
        --out "$OUT_DIR/${name_without_ext}.pkl"

done

echo "=========================================================="
echo "[★] STEP 1 COMPLETE: All PCAPs successfully processed."
echo "[*] Check your tokens here: $OUT_DIR/"
echo "=========================================================="