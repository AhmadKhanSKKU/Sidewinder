import os
import random
import pickle
import argparse
import numpy as np
import torch
from tqdm import tqdm
import sklearn.preprocessing._data

from advGenerate.sac_gru import GRUSAC
from advGenerate.environments.dl_environment_eval import DL_IDS_Evaluator
from advGenerate.environments.vib_environment_eval import VIB_IDS_Evaluator
from advGenerate.environments.transformer_environment_eval import (
    Transformer_IDS_Evaluator,
)
from advGenerate.environments.cnn_environment_eval import CNN_IDS_Evaluator
from advGenerate.environments.pandora_seq_environment_eval import (
    PandoraSeq_IDS_Evaluator,
)


from typing import Union

allowed_globals = [sklearn.preprocessing._data.StandardScaler]

try:
    import numpy._core.multiarray as _np_multiarray

    allowed_globals.append(_np_multiarray.scalar)
except ImportError:
    try:
        import numpy.core.multiarray as _np_multiarray

        allowed_globals.append(_np_multiarray.scalar)
    except ImportError:
        pass

torch.serialization.add_safe_globals(allowed_globals)


def load_raw_data(pkl_path: str):
    with open(pkl_path, "rb") as f:
        return pickle.load(f)


def get_evaluator(defense: str, world_type: str, dataset: str, attack: str):
    num_classes = 3 if world_type == "open_world" else 2

    if defense == "dl":
        model_path = f"models/{world_type}/{dataset}/dl/{attack}_ids.pth"
        if not os.path.exists(model_path):
            return None
        return DL_IDS_Evaluator(
            models_dir=model_path, mode="hard_labeled", threshold=0.3
        )

    elif defense == "vib":
        model_path = f"models/{world_type}/{dataset}/vib/{attack}_vib.pth"
        if not os.path.exists(model_path):
            return None
        return VIB_IDS_Evaluator(
            models_dir=model_path,
            mode="hard_labeled",
            threshold=0.5,
            num_classes=num_classes,
        )

    elif defense == "transformer":
        model_path = f"models/{world_type}/{dataset}/transformer/{attack}_ids.pth"
        if not os.path.exists(model_path):
            return None
        return Transformer_IDS_Evaluator(
            models_dir=model_path,
            mode="hard_labeled",
            threshold=0.5,
            num_classes=num_classes,
        )

    elif defense == "cnn":
        model_path = f"models/{world_type}/{dataset}/cnn/{attack}_ids.pth"
        if not os.path.exists(model_path):
            return None
        return CNN_IDS_Evaluator(
            models_dir=model_path, mode="hard_labeled", threshold=0.5
        )

    elif defense in ["pandora", "pandora_seq"]:
        suffix = "_ids_open_world.pth" if world_type == "open_world" else "_ids.pth"
        model_path = f"models/{world_type}/{dataset}/pandora_seq/{attack}{suffix}"

        if not os.path.exists(model_path):
            model_path_base = f"models/{world_type}/{dataset}/pandora_seq/{attack}"
            if os.path.exists(model_path_base):
                model_path = model_path_base
            else:
                return None

        return PandoraSeq_IDS_Evaluator(
            models_dir=model_path, mode="hard_labeled", threshold=0.5
        )

    return None


def format_state(
    ipds: list[float],
    sizes: list[float],
    forbidden_mask: list[bool],
    max_seq_len: int = 128,
):
    timevocab_size = 56
    sizevocab_size = 1606

    # Discretize raw features to vocabulary indices
    ipd_indices = np.clip(
        np.array(ipds, dtype=np.float32) * 100.0, 0, timevocab_size - 1
    ).astype(np.int64)
    size_indices = np.clip(
        np.array(sizes, dtype=np.float32) / 6.0, 0, sizevocab_size - 1
    ).astype(np.int64)

    real_len = min(len(ipds), max_seq_len)

    ipd_padded = np.zeros(max_seq_len, dtype=np.int64)
    size_padded = np.zeros(max_seq_len, dtype=np.int64)
    forb_padded = np.ones(max_seq_len, dtype=bool)

    ipd_padded[:real_len] = ipd_indices[:real_len]
    size_padded[:real_len] = size_indices[:real_len]
    forb_padded[:real_len] = forbidden_mask[:real_len]

    return {
        "flow_ipd": torch.tensor(ipd_padded, dtype=torch.long),
        "flow_size": torch.tensor(size_padded, dtype=torch.long),
        "real_length": torch.tensor(real_len, dtype=torch.long),
        "forbidden_indices": torch.tensor(forb_padded, dtype=torch.bool),
    }


def run_netmasquerade_eval(
    dataset_pkl: str,
    evaluator: Union[
        Transformer_IDS_Evaluator,
        CNN_IDS_Evaluator,
        PandoraSeq_IDS_Evaluator,
        VIB_IDS_Evaluator,
        DL_IDS_Evaluator,
    ],
    agent: GRUSAC,
    max_steps: int,
):
    data = load_raw_data(dataset_pkl)

    total_valid = 0
    success_count = 0
    total_success_steps = 0
    counter = 0

    ipds: list[float]
    sizes: list[float]

    for item in tqdm(data, desc="Attacking Flows (NetMasquerade)", leave=False):
        if isinstance(item, tuple) and len(item) == 2:
            ipds, sizes = list(item[0]), list(item[1])
        elif isinstance(item, dict):
            ipds, sizes = list(item.get("ipds", [])), list(item.get("sizes", []))

        else:
            continue

        if len(ipds) <= 1:
            continue

        # Initial baseline check
        _, _, initial_class = evaluator.predict(ipds, sizes, max_seq_len=128)
        if initial_class == 0:
            continue

        total_valid += 1

        flow_mean:float = np.mean(ipds)
        flow_std:float = np.std(ipds)
        size_min = min(sizes) if sizes else 40
        size_max = max(sizes) if sizes else 1500

        forbidden_mask = [False] * len(ipds)
        evaded = False
        steps_taken = 0

        for step in range(1, max_steps + 1):
            state = format_state(ipds, sizes, forbidden_mask, max_seq_len=128)
            action_idx = agent.take_action(state, eval_mode=True)

            if action_idx < len(ipds):
                action_type = random.choice(["mutate", "inject"])
                if action_type == "mutate":
                    new_ipd = max(0.0, random.gauss(flow_mean, flow_std))
                    ipds[action_idx] = new_ipd
                    forbidden_mask[action_idx] = True
                elif action_type == "inject":
                    if len(ipds) < 128:
                        chaff_ipd = max(0.0, random.gauss(flow_mean, flow_std))
                        chaff_size = random.randint(int(size_min), int(size_max))
                        ipds.insert(action_idx, chaff_ipd)
                        sizes.insert(action_idx, chaff_size)
                        forbidden_mask.insert(action_idx, True)

            _, _, pred_class = evaluator.predict(ipds, sizes, max_seq_len=128)

            if pred_class == 0:
                evaded = True
                steps_taken = step
                break

        if evaded:
            success_count += 1
            total_success_steps += steps_taken

        counter += 1
        if counter >= 1000:
            break

    asr = (success_count / total_valid * 100) if total_valid > 0 else 0.0
    avg_steps = (total_success_steps / success_count) if success_count > 0 else 0.0

    return asr, avg_steps


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--world_type", type=str, default="closed_world")
    parser.add_argument("--datasets", nargs="+", default=["cic2025", "cic2023", "TTDF"])
    parser.add_argument(
        "--defenses",
        nargs="+",
        default=["transformer", "cnn", "dl", "vib", "pandora_seq"],
    )
    parser.add_argument(
        "--attacks", nargs="+", default=["spoofing", "web", "ddos", "dos"]
    )
    parser.add_argument("--max_steps", type=int, default=10)
    parser.add_argument("--out_md", type=str, default="netmasquerade.md")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize NetMasquerade SAC-GRU Agent
    agent = GRUSAC(
        timevocab_size=56,
        sizevocab_size=1606,
        embedding_dim=128,
        seq_len=128,
        actor_lr=0.0001,
        critic_lr=0.0001,
        alpha_lr=0.0001,
        target_entropy=-5,
        tau=0.005,
        gamma=0.95,
        device=device,
    )

    with open(args.out_md, "w") as f:
        f.write(f"# {args.world_type}\n\n")

        for dataset in args.datasets:
            f.write(f"## {dataset}\n\n")
            f.flush()

            for defense in args.defenses:
                f.write(f"### {dataset} {defense}\n\n")
                f.flush()

                for attack in args.attacks:
                    dataset_pkl = f"data/{dataset}/{attack}.pkl"

                    if not os.path.exists(dataset_pkl):
                        print(
                            f"[!] Dataset pickle not found: {dataset_pkl}. Skipping..."
                        )
                        continue

                    evaluator = get_evaluator(defense, args.world_type, dataset, attack)
                    if evaluator is None:
                        print(
                            f"[!] Model for {defense}/{args.world_type}/{dataset}/{attack} not found. Skipping..."
                        )
                        continue

                    # Load trained NetMasquerade agent weights if present
                    model_dir = (
                        f"models/{args.world_type}/{dataset}/netmasquerade/{attack}"
                    )
                    if os.path.exists(model_dir):
                        agent.load_model(model_dir)

                    print(
                        f"[*] [NetMasquerade] Running: Dataset={dataset} | Defense={defense} | Attack={attack}"
                    )
                    asr, avg_steps = run_netmasquerade_eval(
                        dataset_pkl, evaluator, agent, args.max_steps
                    )

                    f.write(f"#### {dataset} {defense} {attack}\n\n")
                    f.write(f"ASR = {asr:.2f}%\n")
                    f.write(f"avg steps = {avg_steps:.2f}\n\n")
                    f.flush()

    print(f"\n[★] Finished NetMasquerade Evaluation! Output written to: {args.out_md}")


if __name__ == "__main__":
    main()
