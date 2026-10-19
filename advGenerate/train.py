# my_attacker/src/advGenerate/train.py

import os
import random
import torch
import logging
import pickle
import argparse
import sys
import time
import matplotlib
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset
from typing import Literal, Union, Dict, Any

from advGenerate.config_parser import load_rl_config, load_bert_config
from advGenerate.sac_mamba import MambaSAC
from advGenerate.sac_lstm import LSTMSAC
from advGenerate.sac_gru import GRUSAC
from advGenerate.environments.environment import IntelligentPaddingEnv
from advGenerate.buffer import PrioritizedReplayBuffer

from advGenerate.environments.dl_environment_eval import DL_IDS_Evaluator
from advGenerate.environments.vib_environment_eval import VIB_IDS_Evaluator
from advGenerate.environments.transformer_environment_eval import (
    Transformer_IDS_Evaluator,
)
from advGenerate.environments.cnn_environment_eval import CNN_IDS_Evaluator
from advGenerate.environments.pandora_seq_environment_eval import (
    PandoraSeq_IDS_Evaluator,
)
from advGenerate.rl_types import FlowState

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)


class TokenizedFlowDataset(Dataset[Dict[str, torch.Tensor]]):
    def __init__(
        self,
        data_path: str,
        device: torch.device,
        time_vocab: list[float],
        size_vocab: list[int],
        split: Literal["train", "test"] = "train",
        split_ratio: float = 0.8,
    ):
        self.device = device
        self.time_vocab = time_vocab
        self.size_vocab = size_vocab

        with open(data_path, "rb") as f:
            full_data = pickle.load(f)

        random.seed(42)
        random.shuffle(full_data)

        split_idx = int(len(full_data) * split_ratio)
        if split == "train":
            self.data = full_data[:split_idx]
        else:
            self.data = full_data[split_idx:]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int):
        flow = self.data[idx]

        ipd_list = list(flow[0])
        size_list = list(flow[1])

        real_len = len(ipd_list)
        max_len = 128

        if real_len > max_len:
            ipd_list = ipd_list[:max_len]
            size_list = size_list[:max_len]
            real_len = max_len
        else:
            padding_needed = max_len - real_len
            ipd_list = ipd_list + [0] * padding_needed
            size_list = size_list + [0] * padding_needed

        return {
            "flow_ipd": torch.tensor(ipd_list, dtype=torch.long),
            "flow_size": torch.tensor(size_list, dtype=torch.long),
            "real_length": torch.tensor(real_len, dtype=torch.long),
        }


class AdversarialTrainer:

    def __init__(
        self,
        rl_yaml_path: str,
        bert_yaml_path: str,
        evaluator: Union[
            DL_IDS_Evaluator,
            VIB_IDS_Evaluator,
            Transformer_IDS_Evaluator,
            CNN_IDS_Evaluator,
            PandoraSeq_IDS_Evaluator,
        ],
        disable_per: bool = False,
        arch: str = "mamba",
        max_steps: int = 3000,
    ):
        self.rl_args = load_rl_config(rl_yaml_path)
        self.bert_args = load_bert_config(bert_yaml_path)

        self.device = torch.device(
            f"cuda:{self.rl_args.trainer.device}"
            if torch.cuda.is_available()
            else "cpu"
        )

        self.env = IntelligentPaddingEnv(
            self.rl_args,
            self.bert_args,
            mode="train",
            evaluator=evaluator,
        )
        self.seq_len = (
            self.rl_args.model.seq_len
            if hasattr(self.rl_args.model, "seq_len")
            else 128
        )

        target_entropy = self.rl_args.model.target_entropy
        use_per_val = not disable_per
        n_step_val = 3

        agent_kwargs = dict(
            timevocab_size=int(self.rl_args.model.timevocab_size),
            sizevocab_size=int(self.rl_args.model.sizevocab_size),
            embedding_dim=int(self.rl_args.model.embedding_dim),
            seq_len=self.seq_len,
            actor_lr=self.rl_args.model.actor_lr,
            critic_lr=self.rl_args.model.critic_lr,
            alpha_lr=self.rl_args.model.alpha_lr,
            target_entropy=target_entropy,
            tau=self.rl_args.model.tau,
            gamma=self.rl_args.model.gamma,
            device=self.device,
            n_steps=n_step_val,
        )

        if arch == "mamba":
            self.agent = MambaSAC(**agent_kwargs)
        elif arch == "lstm":
            self.agent = LSTMSAC(**agent_kwargs)
        elif arch in ["gru", "netmasquerade"]:
            self.agent = GRUSAC(**agent_kwargs)
        else:
            raise ValueError(f"Unsupported architecture choice: {arch}")

        self.buffer = PrioritizedReplayBuffer(
            capacity=self.rl_args.trainer.buffer_size,
            device=self.device,
            n_step=n_step_val,
            gamma=self.rl_args.model.gamma,
            user_per=use_per_val,
            success_ratio=0.1,
        )

        self.train_dataset = TokenizedFlowDataset(
            self.rl_args.trainer.train_data_pth,
            self.device,
            split="train",
            time_vocab=self.env.time_vocab,
            size_vocab=self.env.size_vocab,
        )
        self.test_dataset = TokenizedFlowDataset(
            self.rl_args.trainer.test_data_pth,
            self.device,
            split="test",
            time_vocab=self.env.time_vocab,
            size_vocab=self.env.size_vocab,
        )
        self.beta_start = 0.4
        self.beta_end = 1.0
        self.total_episodes = max_steps

    def train(self) -> tuple[list[float], float]:
        logging.info("[+] Starting Decoupled RL Training with Episodic Updates")
        batch_size = self.rl_args.trainer.batch_size

        best_avg_reward = -float("inf")
        recent_rewards: list[float] = []
        all_episode_rewards: list[float] = []

        base_pth = str(self.rl_args.trainer.model_save_pth)
        latest_model_pth = os.path.join(base_pth, "latest_model")
        best_model_pth = os.path.join(base_pth, "best_model")
        os.makedirs(latest_model_pth, exist_ok=True)
        os.makedirs(best_model_pth, exist_ok=True)

        start_train_time = time.perf_counter()

        pbar = tqdm(range(self.total_episodes), desc="Training Episodes")
        for episode in pbar:
            valid_malicious_flow = False
            best_score = -float("inf")
            best_state = None
            attempts = 0
            max_attempts = 100

            while not valid_malicious_flow and attempts < max_attempts:
                attempts += 1
                idx = random.randint(0, len(self.train_dataset) - 1)
                state_data = self.train_dataset[idx]

                real_len = int(state_data["real_length"].item())
                if real_len <= 1:
                    continue

                self.env.state = {k: v.clone() for k, v in state_data.items()}
                raw_ipds, raw_sizes = self.env._get_raw_traffic_dimensions()
                res = self.env.evaluator.predict(raw_ipds, raw_sizes, max_seq_len=128)
                initial_score: float = res[1]

                if initial_score > best_score:
                    best_score = initial_score
                    best_state = {k: v.clone() for k, v in state_data.items()}

                if initial_score >= 0.50:
                    valid_malicious_flow = True
                    self.env.prev_score = initial_score
                    self.env.baseline_score = initial_score

            if not valid_malicious_flow:
                if best_state is None:
                    raise RuntimeError("No valid flows found in training data.")
                self.env.state = best_state
                self.env.prev_score = best_score
                self.env.baseline_score = best_score

            self.env.step_num = 0
            self.env.forbidden_indices = torch.zeros(
                self.seq_len, dtype=torch.bool, device=self.device
            )
            self.env.forbidden_indices[0] = True
            self.env.state["forbidden_indices"] = self.env.forbidden_indices.clone()
            state: FlowState = self.env.state

            episode_reward = 0
            episode_steps = 0
            done = False

            while not done:
                action = self.agent.take_action(state, eval_mode=False)
                next_state, reward, done, info = self.env.step(action)

                is_timeout = info.get("timeout", False)
                terminal_state = done and not is_timeout
                action_tensor = torch.tensor([action], dtype=torch.float32)

                state_buffered = {
                    k: v.detach().clone() if isinstance(v, torch.Tensor) else v
                    for k, v in state.items()
                }
                next_state_buffered = {
                    k: v.detach().clone() if isinstance(v, torch.Tensor) else v
                    for k, v in next_state.items()
                }

                self.buffer.push(
                    state_buffered,
                    action_tensor,
                    reward,
                    next_state_buffered,
                    terminal=terminal_state,
                    flush=done,
                )

                state = next_state
                episode_reward += reward
                episode_steps += 1

            recent_rewards.append(episode_reward)
            all_episode_rewards.append(episode_reward)
            if len(recent_rewards) > 50:
                recent_rewards.pop(0)

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            if len(self.buffer.buffer) >= batch_size:
                current_beta = self.beta_start + (self.beta_end - self.beta_start) * (
                    episode / self.total_episodes
                )

                for _ in range(1):
                    (
                        b_states,
                        b_actions,
                        b_rewards,
                        b_next_states,
                        b_dones,
                        indices,
                        weights,
                    ) = self.buffer.sample(batch_size, beta=current_beta)
                    transition_dict = {
                        "states": b_states,
                        "actions": b_actions,
                        "rewards": b_rewards,
                        "next_states": b_next_states,
                        "dones": b_dones,
                    }
                    td_errors = self.agent.update(
                        transition_dict=transition_dict, weights=weights
                    )
                    new_priorities = td_errors + 1e-6
                    self.buffer.update_priorities(indices, new_priorities)

            pbar.set_postfix({"reward": f"{episode_reward:.2f}"})

            if len(recent_rewards) >= 20:
                current_avg = sum(recent_rewards) / len(recent_rewards)
                if current_avg > best_avg_reward:
                    best_avg_reward = current_avg
                    self.agent.save_model(best_model_pth)

        self.agent.save_model(latest_model_pth)
        total_train_time = time.perf_counter() - start_train_time
        logging.info(
            f"[+] Total Training Time ({self.total_episodes} steps): {total_train_time:.2f}s"
        )
        return all_episode_rewards, total_train_time

    @torch.no_grad()
    def evaluate(self, model_path: str = None):
        if model_path is not None and os.path.exists(model_path):
            logging.info(f"[*] Loading agent weights for Evaluation from: {model_path}")
            if hasattr(self.agent, "load_model"):
                self.agent.load_model(model_path)

        logging.info("[+] Commencing Evaluation Phase")

        ter_count = 0  # Class 0 (Benign)
        acr_count = 0  # Class 2 (UNK)
        mal_count = 0  # Class 1 (Malicious)
        total_episodes = 0
        total_steps = 0

        flow_modification_times: list[float] = []

        eval_runs = min(len(self.test_dataset), self.rl_args.trainer.test_episodes)

        for idx in range(eval_runs):
            state_data = self.test_dataset[idx]

            real_len = int(state_data["real_length"].item())
            if real_len <= 1:
                continue

            self.env.state = {
                k: v.clone().detach().to(self.env.device) for k, v in state_data.items()
            }
            self.env.step_num = 0
            self.env.forbidden_indices = torch.zeros(
                self.seq_len, dtype=torch.bool, device=self.env.device
            )
            self.env.forbidden_indices[0] = True
            self.env.state["forbidden_indices"] = self.env.forbidden_indices.clone()

            state = self.env.state
            raw_ipds, raw_sizes = self.env._get_raw_traffic_dimensions()
            eval_res = self.env.evaluator.predict(raw_ipds, raw_sizes, max_seq_len=128)

            self.env.baseline_score = eval_res[1]
            self.env.prev_score = eval_res[1]
            done = False
            last_pred_class = eval_res[2] if len(eval_res) == 3 else 1

            start_flow_time = time.perf_counter()

            while not done:
                action = self.agent.take_action(state, eval_mode=True)
                next_state, _, done, info = self.env.step(action)
                state = next_state
                total_steps += 1
                last_pred_class = info.get("pred_class", 1)

                if done and info.get("is_success", False):
                    break

            flow_duration = time.perf_counter() - start_flow_time
            flow_modification_times.append(flow_duration)

            if last_pred_class == 0:
                ter_count += 1
            elif last_pred_class == 2:
                acr_count += 1
            else:
                mal_count += 1

            total_episodes += 1

        ter = (ter_count / total_episodes) * 100 if total_episodes > 0 else 0.0
        acr = (acr_count / total_episodes) * 100 if total_episodes > 0 else 0.0
        mal_rate = (mal_count / total_episodes) * 100 if total_episodes > 0 else 0.0
        avg_step_count = total_steps / total_episodes if total_episodes > 0 else 0.0
        avg_flow_mod_time = (
            (sum(flow_modification_times) / len(flow_modification_times)) * 1000
            if flow_modification_times
            else 0.0
        )  # in milliseconds

        logging.info("\n================ EVALUATION SUMMARY ================")
        logging.info(f"Total Sequences Evaluated:            {total_episodes}")
        logging.info(
            f"Targeted Evasion Rate (TER -> Benign): {ter:.2f}% ({ter_count}/{total_episodes})"
        )
        logging.info(
            f"Anomaly Catch Rate (ACR -> UNK):      {acr:.2f}% ({acr_count}/{total_episodes})"
        )
        logging.info(
            f"Remains Malicious Rate (Failed):      {mal_rate:.2f}% ({mal_count}/{total_episodes})"
        )
        logging.info(f"Average Perturbation Steps:           {avg_step_count:.2f}")
        logging.info(f"Avg Flow Modification Time (TER):    {avg_flow_mod_time:.2f} ms")
        logging.info("====================================================\n")

        return ter, acr, mal_rate, avg_step_count, avg_flow_mod_time


def append_to_markdown(
    out_md_path: str,
    dataset: str,
    defense: str,
    attack: str,
    ter: float,
    acr: float,
    mal_rate: float,
    avg_steps: float,
    train_time: float = 0.0,
    avg_mod_time_ms: float = 0.0,
):
    file_exists = os.path.exists(out_md_path)
    with open(out_md_path, "a") as f:
        if not file_exists:
            f.write("# Adversarial RL Evaluation Results\n\n")

        f.write(f"### {dataset} | {defense.upper()} | {attack}\n\n")
        f.write(f"- **TER (Targeted Evasion Rate)**: `{ter:.2f}%`  \n")
        f.write(f"- **ACR (Anomaly Catch Rate)**: `{acr:.2f}%`  \n")
        f.write(f"- **Remains Malicious Rate**: `{mal_rate:.2f}%`  \n")
        f.write(f"- **Average Perturbation Steps**: `{avg_steps:.2f}`  \n")
        f.write(f"- **Total Training Time**: `{train_time:.2f}s`  \n")
        f.write(f"- **Avg Flow Modification Time**: `{avg_mod_time_ms:.2f} ms`  \n\n")
        f.write("---\n\n")


if __name__ == "__main__":
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    parser = argparse.ArgumentParser(description="Adversarial RL Auto-Runner")
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
        "--run_mode", type=str, default="train", choices=["train", "eval"]
    )
    parser.add_argument("--disable_per", action="store_true")
    parser.add_argument(
        "--arch",
        type=str,
        default="mamba",
        choices=["lstm", "mamba", "gru", "netmasquerade"],
    )
    parser.add_argument(
        "--max_steps", type=int, default=3000, help="Number of training steps/episodes"
    )
    parser.add_argument(
        "--out_md", type=str, default=None, help="Path to markdown output file"
    )

    args = parser.parse_args()

    dataset_pkl = f"data/{args.dataset}/{args.attack}.pkl"
    defense_path = (
        f"{args.defense}/{args.ml_model}" if args.defense == "ml" else args.defense
    )
    defense_display_name = (
        f"ML-{args.ml_model.upper()}" if args.defense == "ml" else args.defense.upper()
    )

    ablation_tag = f"PER-{'False' if args.disable_per else 'True'}_{args.arch}"
    logs_dir = f"training_logs/{args.world_type}/{args.feedback_mode}/{args.dataset}/{defense_path}/{ablation_tag}"
    results_dir = f"results/{args.world_type}/{args.feedback_mode}/{args.dataset}/{defense_path}/{args.attack}/{ablation_tag}"

    os.makedirs(logs_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    log_file = os.path.join(logs_dir, f"{args.attack}_training_log.md")

    with open(log_file, "w") as f:
        f.write(f"# {args.arch.upper()}-SAC Training Log ({args.world_type.upper()})\n")
        f.write(
            f"**Attack:** `{args.attack}` | **Defense:** `{defense_display_name}`\n\n"
        )

    logging.getLogger().handlers.clear()
    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    file_handler = logging.FileHandler(log_file, mode="a")
    file_handler.setFormatter(formatter)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logging.basicConfig(level=logging.INFO, handlers=[file_handler, console_handler])

    if args.defense == "dl":
        ids_model_path = (
            f"models/{args.world_type}/{args.dataset}/dl/{args.attack}_ids.pth"
        )
        evaluator = DL_IDS_Evaluator(
            models_dir=ids_model_path, mode=args.feedback_mode, threshold=0.3
        )

    elif args.defense == "vib":
        ids_model_path = (
            f"models/{args.world_type}/{args.dataset}/vib/{args.attack}_vib.pth"
        )
        num_classes = 3 if args.world_type == "open_world" else 2
        evaluator = VIB_IDS_Evaluator(
            models_dir=ids_model_path,
            mode=args.feedback_mode,
            threshold=0.5,
            num_classes=num_classes,
        )

    elif args.defense == "transformer":
        ids_model_path = (
            f"models/{args.world_type}/{args.dataset}/transformer/{args.attack}_ids.pth"
        )
        num_classes = 3 if args.world_type == "open_world" else 2
        evaluator = Transformer_IDS_Evaluator(
            models_dir=ids_model_path,
            mode=args.feedback_mode,
            threshold=0.5,
            num_classes=num_classes,
        )

    elif args.defense == "cnn":
        ids_model_path = (
            f"models/{args.world_type}/{args.dataset}/cnn/{args.attack}_ids.pth"
        )
        evaluator = CNN_IDS_Evaluator(
            models_dir=ids_model_path, mode=args.feedback_mode, threshold=0.5
        )

    elif args.defense == "pandora":
        ids_model_path = (
            f"models/{args.world_type}/{args.dataset}/pandora_seq/{args.attack}"
        )
        evaluator = PandoraSeq_IDS_Evaluator(
            models_dir=ids_model_path, mode=args.feedback_mode, threshold=0.5
        )

    else:
        raise ValueError(f"Unsupported defense type: {args.defense}")

    torch.manual_seed(seed=42)
    np.random.seed(seed=42)

    trainer = AdversarialTrainer(
        rl_yaml_path="advGenerate/config/sac.yaml",
        bert_yaml_path="trafficMimic/config/bert.yaml",
        evaluator=evaluator,
        arch=args.arch,
        disable_per=args.disable_per,
        max_steps=args.max_steps,
    )

    trainer.rl_args.trainer.train_data_pth = dataset_pkl
    trainer.rl_args.trainer.test_data_pth = dataset_pkl
    trainer.rl_args.trainer.model_save_pth = os.path.join(results_dir, "checkpoints")
    os.makedirs(trainer.rl_args.trainer.model_save_pth, exist_ok=True)

    trainer.env.evaluator = evaluator

    best_checkpoint_pth = os.path.join(
        trainer.rl_args.trainer.model_save_pth, "best_model"
    )
    latest_checkpoint_pth = os.path.join(
        trainer.rl_args.trainer.model_save_pth, "latest_model"
    )
    eval_model_pth = (
        best_checkpoint_pth
        if os.path.exists(os.path.join(best_checkpoint_pth, "actor.pth"))
        else latest_checkpoint_pth
    )

    train_time = 0.0
    if args.run_mode == "train":
        logging.info("[*] Starting Training Loop...")
        history, train_time = trainer.train()

        ter, acr, mal_rate, avg_steps, avg_mod_time_ms = trainer.evaluate(
            model_path=eval_model_pth
        )

        plt.style.use(
            "seaborn-v0_8-whitegrid"
            if "seaborn-v0_8-whitegrid" in plt.style.available
            else "default"
        )
        fig, ax = plt.subplots(figsize=(11, 5.5), dpi=300)

        RAW_COLOR = "#94A3B8"
        AVG_COLOR = "#1D4ED8"

        ax.plot(
            history, color=RAW_COLOR, alpha=0.35, linewidth=0.9, label="Episode Reward"
        )

        if len(history) >= 20:
            moving_avg = np.convolve(history, np.ones(20) / 20, mode="valid")
            x_range = range(19, len(history))
            ax.plot(
                x_range,
                moving_avg,
                color=AVG_COLOR,
                linewidth=2.4,
                label="20-Epi Moving Avg",
            )
            ax.fill_between(x_range, moving_avg, y2=-3.0, color=AVG_COLOR, alpha=0.08)

        ax.set_title(
            f"Adversarial SAC Training Trajectory — {args.attack} ({args.world_type.upper()})",
            fontsize=13,
            fontweight="bold",
            pad=14,
            color="#0F172A",
        )
        ax.set_xlabel(
            "Episode", fontsize=10, fontweight="bold", color="#334155", labelpad=8
        )
        ax.set_ylabel(
            "Reward", fontsize=10, fontweight="bold", color="#334155", labelpad=8
        )

        stats_text = (
            f"  EVALUATION SUMMARY  \n"
            f"──────────────────────\n"
            f"  TER (Evasion):  {ter:6.2f}%\n"
            f"  ACR (Anomaly):  {acr:6.2f}%\n"
            f"  Malicious:      {mal_rate:6.2f}%\n"
            f"  Avg Steps:      {avg_steps:6.2f}\n"
            f"  Train Time:     {train_time:6.1f}s\n"
            f"  Mod Time/Flow:  {avg_mod_time_ms:6.2f}ms"
        )

        bbox_props = dict(
            boxstyle="round,pad=0.7",
            facecolor="#F8FAFC",
            edgecolor="#CBD5E1",
            linewidth=1.2,
            alpha=0.92,
        )
        ax.text(
            x=0.03,
            y=0.45,
            s=stats_text,
            transform=ax.transAxes,
            fontsize=9.5,
            family="monospace",
            verticalalignment="center",
            bbox=bbox_props,
            color="#0F172A",
        )

        ax.grid(True, linestyle=":", alpha=0.6, color="#94A3B8")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color("#CBD5E1")
        ax.spines["bottom"].set_color("#CBD5E1")
        ax.legend(
            loc="upper left", frameon=True, facecolor="white", edgecolor="#E2E8F0"
        )

        plt.tight_layout()
        plot_path = os.path.join(results_dir, f"{args.attack}_training_reward.png")
        plt.savefig(plot_path, dpi=300, bbox_inches="tight")
        plt.close()
        logging.info(f"[+] Enhanced plot with eval metrics saved to: {plot_path}")

    elif args.run_mode == "eval":
        ter, acr, mal_rate, avg_steps, avg_mod_time_ms = trainer.evaluate(
            model_path=eval_model_pth
        )

    if args.out_md:
        append_to_markdown(
            args.out_md,
            dataset=args.dataset,
            defense=args.defense,
            attack=args.attack,
            ter=ter,
            acr=acr,
            mal_rate=mal_rate,
            avg_steps=avg_steps,
            train_time=train_time,
            avg_mod_time_ms=avg_mod_time_ms,
        )
