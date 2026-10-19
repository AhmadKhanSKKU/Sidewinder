# src/advGenerate/compile_dataset.py

import pickle

from typing import TypedDict, List


class FlowSample(TypedDict):
    ipds: List[float]
    sizes: List[int]
    label: int


def create_ml_dataset(benign_pkl: str, attack_pkl: str, output_pkl: str) -> None:
    """
    Combines parsed sliding window tuples into a labeled dictionary format
    suitable for training classic ML Classifiers.
    """
    ml_dataset: list[FlowSample] = []

    # 1. Load Benign Samples
    print(f"[*] Loading Benign data from: {benign_pkl}")
    with open(benign_pkl, "rb") as f:
        benign_data = pickle.load(f)  # List of (ipd_list, size_list)

    for ipds, sizes in benign_data:
        ml_dataset.append(
            {"ipds": ipds, "sizes": sizes, "label": 0}  # 0 denotes Benign/Normal
        )
    print(f"[+] Added {len(benign_data)} Benign samples.")

    # 2. Load Attack Samples
    print(f"[*] Loading Attack data from: {attack_pkl}")
    with open(attack_pkl, "rb") as f:
        attack_data = pickle.load(f)

    for ipds, sizes in attack_data:
        ml_dataset.append(
            {"ipds": ipds, "sizes": sizes, "label": 1}  # 1 denotes Malicious/Attack
        )
    print(f"[+] Added {len(attack_data)} Attack samples.")

    # 3. Save Combined Dataset
    with open(output_pkl, "wb") as f:
        pickle.dump(ml_dataset, f)

    print(f"\n[SUCCESS] Unified dataset saved to: {output_pkl}")
    print(f"Total dataset size: {len(ml_dataset)} samples.")


if __name__ == "__main__":
    # 1. Generate these two paths using your build_real_vocab.py script!
    BENIGN_INPUT = "/media/ahmad/Work/Experiments/Pandora/my_attacker/src/sliding_window_benign.pkl"
    ATTACK_INPUT = "/media/ahmad/Work/Experiments/Pandora/my_attacker/src/sliding_window_attack_bruteforce_dictionary_ssh_ap.pkl"

    # 2. This will be the master file you hand to your ML models
    OUTPUT_FILE = (
        "/media/ahmad/Work/Experiments/Pandora/my_attacker/src/master_ids_dataset.pkl"
    )

    create_ml_dataset(BENIGN_INPUT, ATTACK_INPUT, OUTPUT_FILE)
