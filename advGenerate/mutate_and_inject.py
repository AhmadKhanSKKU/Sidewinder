import os
import random
import pickle
import argparse
import numpy as np
from tqdm import tqdm
import torch
import sklearn.preprocessing._data

from advGenerate.environments.dl_environment_eval import DL_IDS_Evaluator
from advGenerate.environments.vib_environment_eval import VIB_IDS_Evaluator
from advGenerate.environments.transformer_environment_eval import (
    Transformer_IDS_Evaluator,
)
from advGenerate.environments.cnn_environment_eval import CNN_IDS_Evaluator
from advGenerate.environments.pandora_seq_environment_eval import (
    PandoraSeq_IDS_Evaluator,
)

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


def run_mutate_and_inject_eval(dataset_pkl: str, evaluator, max_steps: int):
    data = load_raw_data(dataset_pkl)

    total_valid = 0
    success_count = 0
    total_success_steps = 0
    counter = 0

    for item in tqdm(data, desc="Attacking Flows", leave=False):
        if isinstance(item, tuple) and len(item) == 2:
            ipds, sizes = list(item[0]), list(item[1])
        else:
            ipds, sizes = list(item.get("ipds", [])), list(item.get("sizes", []))

        if len(ipds) <= 1:
            continue

        # Baseline check
        _, _, initial_class = evaluator.predict(ipds, sizes, max_seq_len=128)
        if initial_class == 0:
            continue

        total_valid += 1

        flow_mean = np.mean(ipds)
        flow_std = np.std(ipds)
        size_min = min(sizes) if sizes else 40
        size_max = max(sizes) if sizes else 1500

        evaded = False
        steps_taken = 0

        for step in range(1, max_steps + 1):
            action = random.choice(["mutate", "inject"])

            if action == "mutate":
                idx = random.randint(0, len(ipds) - 1)
                new_ipd = random.gauss(flow_mean, flow_std)
                ipds[idx] = max(0.0, new_ipd)

            elif action == "inject":
                if len(ipds) < 128:
                    idx = random.randint(0, len(ipds))
                    chaff_ipd = max(0.0, random.gauss(flow_mean, flow_std))
                    chaff_size = random.randint(int(size_min), int(size_max))

                    ipds.insert(idx, chaff_ipd)
                    sizes.insert(idx, chaff_size)

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
    parser.add_argument("--out_md", type=str, default="mutate_and_inject.md")
    args = parser.parse_args()

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

                    print(
                        f"[*] [Mutate & Inject] Running: Dataset={dataset} | Defense={defense} | Attack={attack}"
                    )
                    asr, avg_steps = run_mutate_and_inject_eval(
                        dataset_pkl, evaluator, args.max_steps
                    )

                    f.write(f"#### {dataset} {defense} {attack}\n\n")
                    f.write(f"ASR = {asr:.2f}%\n")
                    f.write(f"avg steps = {avg_steps:.2f}\n\n")
                    f.flush()

    print(
        f"\n[★] Finished Mutate-and-Inject Evaluation! Output written to: {args.out_md}"
    )


if __name__ == "__main__":
    main()
