# my_attacker/src/advGenerate/eval_metrices.py

import os
import argparse
import logging
import numpy as np
import torch
from tqdm import tqdm
from typing import Any, Optional, Dict, List, Tuple

from advGenerate.config_parser import load_rl_config, load_bert_config
from advGenerate.sac_mamba import MambaSAC
from advGenerate.sac_lstm import LSTMSAC
from advGenerate.sac_gru import GRUSAC
from advGenerate.environments.environment import IntelligentPaddingEnv

from advGenerate.environments.dl_environment_eval import DL_IDS_Evaluator
from advGenerate.environments.vib_environment_eval import VIB_IDS_Evaluator
from advGenerate.environments.transformer_environment_eval import (
    Transformer_IDS_Evaluator,
)
from advGenerate.environments.cnn_environment_eval import CNN_IDS_Evaluator
from advGenerate.environments.pandora_seq_environment_eval import (
    PandoraSeq_IDS_Evaluator,
)
from advGenerate.train import TokenizedFlowDataset

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)

# ==========================================
# EDIT & PERTURBATION METRIC HELPERS
# ==========================================


def compute_levenshtein_edit_distance(
    seq_a: List[Tuple[float, float]], seq_b: List[Tuple[float, float]]
) -> int:
    """
    Computes exact sequence edit distance (Levenshtein) on categorical packet sequence pairs.
    """
    n, m = len(seq_a), len(seq_b)
    if n == 0:
        return m
    if m == 0:
        return n

    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0 if seq_a[i - 1] == seq_b[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,  # Deletion
                dp[i][j - 1] + 1,  # Insertion
                dp[i - 1][j - 1] + cost,  # Substitution
            )
    return dp[n][m]


def compute_pointwise_edit_stats(
    orig_ipds: np.ndarray,
    orig_sizes: np.ndarray,
    morphed_ipds: np.ndarray,
    morphed_sizes: np.ndarray,
) -> Dict[str, float]:
    """
    Calculates sparse edit metrics on categorical bin indices.
    """
    min_len = min(len(orig_ipds), len(morphed_ipds))
    if min_len == 0:
        return {
            "overall_sparsity_pct": 0.0,
            "ipd_sparsity_pct": 0.0,
            "size_sparsity_pct": 0.0,
            "norm_edit_distance": 0.0,
            "avg_ipd_edit_mag": 0.0,
            "avg_size_edit_mag": 0.0,
            "total_ipd_overhead": 0.0,
            "total_size_overhead": 0.0,
        }

    orig_ipd_sub, morphed_ipd_sub = orig_ipds[:min_len], morphed_ipds[:min_len]
    orig_size_sub, morphed_size_sub = orig_sizes[:min_len], morphed_sizes[:min_len]

    ipd_diffs = morphed_ipd_sub - orig_ipd_sub
    size_diffs = morphed_size_sub - orig_size_sub

    # Mask for modified indices (Bin differences)
    ipd_modified_mask = np.abs(ipd_diffs) > 1e-6
    size_modified_mask = np.abs(size_diffs) > 1e-6
    any_modified_mask = ipd_modified_mask | size_modified_mask

    overall_sparsity_pct = (int(np.sum(any_modified_mask)) / float(min_len)) * 100.0
    ipd_sparsity_pct = (int(np.sum(ipd_modified_mask)) / float(min_len)) * 100.0
    size_sparsity_pct = (int(np.sum(size_modified_mask)) / float(min_len)) * 100.0

    # Conditional edit magnitudes (Calculated ONLY on modified packets)
    avg_ipd_edit_mag = (
        float(np.mean(np.abs(ipd_diffs[ipd_modified_mask])))
        if np.any(ipd_modified_mask)
        else 0.0
    )
    avg_size_edit_mag = (
        float(np.mean(np.abs(size_diffs[size_modified_mask])))
        if np.any(size_modified_mask)
        else 0.0
    )

    # Total accumulated bin shift overhead
    total_ipd_overhead = float(np.sum(np.abs(ipd_diffs)))
    total_size_overhead = float(np.sum(np.abs(size_diffs)))

    # Sequence Edit Distance
    seq_orig = list(zip(orig_ipd_sub.tolist(), orig_size_sub.tolist()))
    seq_morphed = list(zip(morphed_ipd_sub.tolist(), morphed_size_sub.tolist()))
    raw_edit_dist = compute_levenshtein_edit_distance(seq_orig, seq_morphed)
    norm_edit_dist = raw_edit_dist / float(max(len(seq_orig), len(seq_morphed), 1))

    return {
        "overall_sparsity_pct": overall_sparsity_pct,
        "ipd_sparsity_pct": ipd_sparsity_pct,
        "size_sparsity_pct": size_sparsity_pct,
        "norm_edit_distance": norm_edit_dist,
        "avg_ipd_edit_mag": avg_ipd_edit_mag,
        "avg_size_edit_mag": avg_size_edit_mag,
        "total_ipd_overhead": total_ipd_overhead,
        "total_size_overhead": total_size_overhead,
    }


# ==========================================
# EVALUATION PIPELINE
# ==========================================


class RobustnessMetricEvaluator:

    def __init__(
        self,
        rl_yaml_path: str,
        bert_yaml_path: str,
        evaluator: Any,
        arch: str = "mamba",
        pretrained_model_path: str = None,
        test_dataset_path: str = None,
    ):
        self.rl_args = load_rl_config(rl_yaml_path)
        self.bert_args = load_bert_config(bert_yaml_path)
        self.device = torch.device(
            f"cuda:{self.rl_args.trainer.device}"
            if torch.cuda.is_available()
            else "cpu"
        )

        self.env = IntelligentPaddingEnv(
            self.rl_args, self.bert_args, mode="eval", evaluator=evaluator
        )
        self.seq_len = getattr(self.rl_args.model, "seq_len", 128)

        agent_kwargs = dict(
            timevocab_size=int(self.rl_args.model.timevocab_size),
            sizevocab_size=int(self.rl_args.model.sizevocab_size),
            embedding_dim=int(self.rl_args.model.embedding_dim),
            seq_len=self.seq_len,
            actor_lr=self.rl_args.model.actor_lr,
            critic_lr=self.rl_args.model.critic_lr,
            alpha_lr=self.rl_args.model.alpha_lr,
            target_entropy=self.rl_args.model.target_entropy,
            tau=self.rl_args.model.tau,
            gamma=self.rl_args.model.gamma,
            device=self.device,
            n_steps=3,
        )

        if arch == "mamba":
            self.agent = MambaSAC(**agent_kwargs)
        elif arch == "lstm":
            self.agent = LSTMSAC(**agent_kwargs)
        elif arch in ["gru", "netmasquerade"]:
            self.agent = GRUSAC(**agent_kwargs)
        else:
            raise ValueError(f"Unsupported architecture choice: {arch}")

        if pretrained_model_path and os.path.exists(pretrained_model_path):
            logging.info(f"[*] Loading pretrained agent from: {pretrained_model_path}")
            self.agent.load_model(pretrained_model_path)
        else:
            raise FileNotFoundError(
                f"[-] Specified model path does not exist: {pretrained_model_path}"
            )

        data_path = test_dataset_path or self.rl_args.trainer.test_data_pth
        self.test_dataset = TokenizedFlowDataset(
            data_path,
            self.device,
            split="test",
            time_vocab=self.env.time_vocab,
            size_vocab=self.env.size_vocab,
        )

    @torch.no_grad()
    def evaluate_morphed_similarity(
        self, max_samples: Optional[int] = None
    ) -> Dict[str, Any]:
        logging.info(
            "[+] Starting Action-Aware Flow Perturbation Evaluation for Research..."
        )

        num_samples = min(
            len(self.test_dataset), max_samples or self.rl_args.trainer.test_episodes
        )
        total_evaluated, evaded_count = 0, 0

        # Pointwise Edit & Sparsity Accumulators
        overall_sparsity_list: List[float] = []
        ipd_sparsity_list: List[float] = []
        size_sparsity_list: List[float] = []
        norm_edit_dist_list: List[float] = []
        avg_ipd_edit_mag_list: List[float] = []
        avg_size_edit_mag_list: List[float] = []
        total_ipd_overhead_list: List[float] = []
        total_size_overhead_list: List[float] = []

        for idx in tqdm(range(num_samples), desc="Evaluating Flows"):
            state_data = self.test_dataset[idx]
            real_len = int(state_data["real_length"].item())
            if real_len <= 1:
                continue

            self.env.state = {
                k: v.clone().detach().to(self.device) for k, v in state_data.items()
            }
            self.env.step_num = 0
            self.env.forbidden_indices = torch.zeros(
                self.seq_len, dtype=torch.bool, device=self.device
            )
            self.env.forbidden_indices[0] = True
            self.env.state["forbidden_indices"] = self.env.forbidden_indices.clone()

            # Baseline evaluation
            orig_ipds, orig_sizes = self.env._get_raw_traffic_dimensions()
            orig_ipds_np = np.array(orig_ipds[:real_len], dtype=np.float64)
            orig_sizes_np = np.array(orig_sizes[:real_len], dtype=np.float64)

            eval_res = self.env.evaluator.predict(
                orig_ipds, orig_sizes, max_seq_len=128
            )
            self.env.baseline_score = self.env.prev_score = eval_res[1]

            done, state = False, self.env.state
            last_pred_class = eval_res[2] if len(eval_res) == 3 else 1

            while not done:
                action = self.agent.take_action(state, eval_mode=True)
                next_state, _, done, info = self.env.step(action)
                state = next_state
                last_pred_class = info.get("pred_class", 1)
                if done and info.get("is_success", False):
                    break

            total_evaluated += 1

            if last_pred_class == 0:
                evaded_count += 1
                morphed_ipds, morphed_sizes = self.env._get_raw_traffic_dimensions()
                morphed_real_len = int(self.env.state["real_length"].item())

                morphed_ipds_np = np.array(
                    morphed_ipds[:morphed_real_len], dtype=np.float64
                )
                morphed_sizes_np = np.array(
                    morphed_sizes[:morphed_real_len], dtype=np.float64
                )

                # Compute point-wise sparse edit stats based on categorical tokens
                stats = compute_pointwise_edit_stats(
                    orig_ipds_np, orig_sizes_np, morphed_ipds_np, morphed_sizes_np
                )

                overall_sparsity_list.append(stats["overall_sparsity_pct"])
                ipd_sparsity_list.append(stats["ipd_sparsity_pct"])
                size_sparsity_list.append(stats["size_sparsity_pct"])
                norm_edit_dist_list.append(stats["norm_edit_distance"])
                avg_ipd_edit_mag_list.append(stats["avg_ipd_edit_mag"])
                avg_size_edit_mag_list.append(stats["avg_size_edit_mag"])
                total_ipd_overhead_list.append(stats["total_ipd_overhead"])
                total_size_overhead_list.append(stats["total_size_overhead"])

        ter = (evaded_count / total_evaluated * 100) if total_evaluated > 0 else 0.0

        results = {
            "ter": ter,
            "evaded_count": evaded_count,
            "overall_sparsity_pct": (
                float(np.mean(overall_sparsity_list)) if overall_sparsity_list else 0.0
            ),
            "ipd_sparsity_pct": (
                float(np.mean(ipd_sparsity_list)) if ipd_sparsity_list else 0.0
            ),
            "size_sparsity_pct": (
                float(np.mean(size_sparsity_list)) if size_sparsity_list else 0.0
            ),
            "norm_edit_distance": (
                float(np.mean(norm_edit_dist_list)) if norm_edit_dist_list else 0.0
            ),
            "avg_ipd_edit_mag": (
                float(np.mean(avg_ipd_edit_mag_list)) if avg_ipd_edit_mag_list else 0.0
            ),
            "avg_size_edit_mag": (
                float(np.mean(avg_size_edit_mag_list))
                if avg_size_edit_mag_list
                else 0.0
            ),
            "total_ipd_overhead": (
                float(np.mean(total_ipd_overhead_list))
                if total_ipd_overhead_list
                else 0.0
            ),
            "total_size_overhead": (
                float(np.mean(total_size_overhead_list))
                if total_size_overhead_list
                else 0.0
            ),
        }

        logging.info(
            "\n================ CATEGORICAL PERTURBATION (RESEARCH SUMMARY) ================"
        )
        logging.info(f"Total Evaluated Flows:                 {total_evaluated}")
        logging.info(
            f"Evasion Success Rate (TER):            {evaded_count} ({ter:.2f}%)"
        )
        logging.info(
            "------------------------------------------------------------------------"
        )
        logging.info("MODIFICATION SPARSITY (Tokens Edited):")
        logging.info(
            f"  - Overall Flow Sparsity:             {results['overall_sparsity_pct']:.2f}% of sequence modified"
        )
        logging.info(
            f"  - Timing (IPD) Sparsity:             {results['ipd_sparsity_pct']:.2f}% of IPD tokens modified"
        )
        logging.info(
            f"  - Payload (Size) Sparsity:           {results['size_sparsity_pct']:.2f}% of Size tokens modified"
        )
        logging.info(
            f"  - Normalized Edit Distance:          {results['norm_edit_distance']:.4f} (Levenshtein alignment)"
        )
        logging.info(
            "------------------------------------------------------------------------"
        )
        logging.info(
            "CONDITIONAL TOKEN SHIFT MAGNITUDE (Calculated ONLY on edited packets):"
        )
        logging.info(
            f"  - Avg IPD Token Shift per Edit:      {results['avg_ipd_edit_mag']:.2f} bins"
        )
        logging.info(
            f"  - Avg Size Token Shift per Edit:     {results['avg_size_edit_mag']:.2f} bins"
        )
        logging.info(
            "------------------------------------------------------------------------"
        )
        logging.info("CUMULATIVE CATEGORICAL OVERHEAD:")
        logging.info(
            f"  - Total IPD Bin Shift:               {results['total_ipd_overhead']:.2f} bins / session"
        )
        logging.info(
            f"  - Total Size Bin Shift:              {results['total_size_overhead']:.2f} bins / session"
        )
        logging.info(
            "========================================================================\n"
        )

        return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate Pretrained RL Agent Perturbation Minimality (Sparse Edit Metrics)"
    )
    parser.add_argument(
        "--world_type",
        type=str,
        default="closed_world",
        choices=["closed_world", "open_world"],
    )
    parser.add_argument("--dataset", type=str, default="cic2023")
    parser.add_argument("--attack", type=str, required=True)
    parser.add_argument(
        "--defense",
        type=str,
        default="dl",
        choices=["dl", "vib", "ml", "cnn", "pandora", "transformer"],
    )
    parser.add_argument(
        "--ml_model", type=str, default="rf", choices=["rf", "svm", "xgb"]
    )
    parser.add_argument(
        "--feedback_mode",
        type=str,
        default="score_based",
        choices=["score_based", "hard_labeled"],
    )
    parser.add_argument(
        "--arch",
        type=str,
        default="mamba",
        choices=["lstm", "mamba", "gru", "netmasquerade"],
    )
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to pretrained agent checkpoint",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=500,
        help="Maximum number of test flows to evaluate",
    )
    args = parser.parse_args()

    if args.defense == "dl":
        evaluator = DL_IDS_Evaluator(
            models_dir=f"models/{args.world_type}/{args.dataset}/dl/{args.attack}_ids.pth",
            mode=args.feedback_mode,
            threshold=0.3,
        )
    elif args.defense == "vib":
        evaluator = VIB_IDS_Evaluator(
            models_dir=f"models/{args.world_type}/{args.dataset}/vib/{args.attack}_vib.pth",
            mode=args.feedback_mode,
            threshold=0.5,
            num_classes=3 if args.world_type == "open_world" else 2,
        )
    elif args.defense == "transformer":
        evaluator = Transformer_IDS_Evaluator(
            models_dir=f"models/{args.world_type}/{args.dataset}/transformer/{args.attack}_ids.pth",
            mode=args.feedback_mode,
            threshold=0.5,
            num_classes=3 if args.world_type == "open_world" else 2,
        )
    elif args.defense == "cnn":
        evaluator = CNN_IDS_Evaluator(
            models_dir=f"models/{args.world_type}/{args.dataset}/cnn/{args.attack}_ids.pth",
            mode=args.feedback_mode,
            threshold=0.5,
        )
    elif args.defense == "pandora":
        evaluator = PandoraSeq_IDS_Evaluator(
            models_dir=f"models/{args.world_type}/{args.dataset}/pandora_seq/{args.attack}",
            mode=args.feedback_mode,
            threshold=0.5,
        )
    else:
        raise ValueError(f"Unsupported defense type: {args.defense}")

    evaluator_runner = RobustnessMetricEvaluator(
        rl_yaml_path="advGenerate/config/sac.yaml",
        bert_yaml_path="trafficMimic/config/bert.yaml",
        evaluator=evaluator,
        arch=args.arch,
        pretrained_model_path=args.model_path,
        test_dataset_path=f"data/{args.dataset}/{args.attack}.pkl",
    )

    logging.info("Evaluating morphed similarity...")
    evaluator_runner.evaluate_morphed_similarity(max_samples=args.max_samples)
