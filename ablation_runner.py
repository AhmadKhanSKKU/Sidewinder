import os
import sys
import time
import random
import pickle
import argparse
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from typing import Dict, Tuple, List

# Project imports
from advGenerate.config_parser import load_rl_config, load_bert_config
from advGenerate.environments.environment import IntelligentPaddingEnv
from advGenerate.environments.dl_environment_eval import DL_IDS_Evaluator
from advGenerate.environments.vib_environment_eval import VIB_IDS_Evaluator
from advGenerate.environments.transformer_environment_eval import Transformer_IDS_Evaluator
from advGenerate.environments.cnn_environment_eval import CNN_IDS_Evaluator
from advGenerate.environments.pandora_seq_environment_eval import PandoraSeq_IDS_Evaluator

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


# Helpers for Safe State Tokenization
def extract_vocab_list(vocab_obj) -> List[float]:
    """Safely extracts numerical vocabulary bin values, gracefully defaulting special tokens (e.g. '[pad]') to 0.0."""
    raw_list = []
    for attr in ["idx2token", "idx2val", "vocab", "tokens", "values", "itos", "vocab_list"]:
        if hasattr(vocab_obj, attr):
            val = getattr(vocab_obj, attr)
            if isinstance(val, (list, tuple, np.ndarray)):
                raw_list = list(val)
                break
            elif isinstance(val, dict):
                raw_list = [val[k] for k in sorted(val.keys())]
                break

    if not raw_list and isinstance(vocab_obj, (list, tuple, np.ndarray)):
        raw_list = list(vocab_obj)

    if not raw_list and hasattr(vocab_obj, "__dict__"):
        for val in vocab_obj.__dict__.values():
            if isinstance(val, (list, tuple, np.ndarray)):
                raw_list = list(val)
                break
            elif isinstance(val, dict):
                raw_list = [val[k] for k in sorted(val.keys())]
                break

    if not raw_list:
        try:
            raw_list = [vocab_obj[i] for i in range(len(vocab_obj))]
        except Exception:
            try:
                raw_list = list(vocab_obj)
            except Exception:
                raw_list = []

    res = []
    for item in raw_list:
        try:
            res.append(float(item))
        except (ValueError, TypeError):
            res.append(0.0)
    return res


def get_nearest_token_idx(val: float, vals_array: np.ndarray) -> int:
    if len(vals_array) == 0:
        return 0
    return int(np.argmin(np.abs(vals_array - val)))


# Dataset Loader
class SimpleFlowDataset(Dataset):
    def __init__(self, data_path: str, max_len: int = 128):
        with open(data_path, "rb") as f:
            self.data = pickle.load(f)
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        flow = self.data[idx]
        if isinstance(flow, tuple):
            ipd_list, size_list = list(flow[0]), list(flow[1])
        elif isinstance(flow, dict):
            ipd_list, size_list = list(flow.get("ipds", [])), list(flow.get("sizes", []))
        else:
            ipd_list, size_list = [], []

        real_len = min(len(ipd_list), self.max_len)
        ipd_list = ipd_list[:real_len] + [0] * (self.max_len - real_len)
        size_list = size_list[:real_len] + [0] * (self.max_len - real_len)

        return {
            "flow_ipd": torch.tensor(ipd_list, dtype=torch.long),
            "flow_size": torch.tensor(size_list, dtype=torch.long),
            "real_length": torch.tensor(real_len, dtype=torch.long),
        }


# Evaluator Factory
def load_evaluator(defense: str, world_type: str, dataset: str, attack: str, feedback_mode: str):
    num_classes = 3 if world_type == "open_world" else 2

    if defense == "dl":
        path = f"models/{world_type}/{dataset}/dl/{attack}_ids.pth"
        return DL_IDS_Evaluator(models_dir=path, mode=feedback_mode, threshold=0.3)
    elif defense == "vib":
        path = f"models/{world_type}/{dataset}/vib/{attack}_vib.pth"
        return VIB_IDS_Evaluator(models_dir=path, mode=feedback_mode, threshold=0.5, num_classes=num_classes)
    elif defense == "transformer":
        path = f"models/{world_type}/{dataset}/transformer/{attack}_ids.pth"
        return Transformer_IDS_Evaluator(models_dir=path, mode=feedback_mode, threshold=0.5, num_classes=num_classes)
    elif defense == "cnn":
        path = f"models/{world_type}/{dataset}/cnn/{attack}_ids.pth"
        return CNN_IDS_Evaluator(models_dir=path, mode=feedback_mode, threshold=0.5)
    elif defense in ["pandora", "pandora_seq"]:
        path = f"models/{world_type}/{dataset}/pandora_seq/{attack}"
        return PandoraSeq_IDS_Evaluator(models_dir=path, mode=feedback_mode, threshold=0.5)
    else:
        raise ValueError(f"Unsupported defense architecture: {defense}")


# Experiment 2: BERT Predictions at Random Places (No RL)
def run_exp2_bert_random(env: IntelligentPaddingEnv, dataset: SimpleFlowDataset, max_steps: int = 10):
    ter_count, acr_count, mal_count = 0, 0, 0
    total_steps, total_evals = 0, 0
    mod_times = []

    for idx in range(min(len(dataset), 199)):
        sample = dataset[idx]
        real_len = int(sample["real_length"].item())
        if real_len <= 1:
            continue

        env.state = {k: v.clone().detach().to(env.device) for k, v in sample.items()}
        env.reset()

        raw_ipds, raw_sizes = env._get_raw_traffic_dimensions()
        eval_res = env.evaluator.predict(raw_ipds, raw_sizes, max_seq_len=128)

        if eval_res[2] == 0:
            continue

        total_evals += 1
        done = False
        steps = 0
        t0 = time.perf_counter()

        while not done and steps < max_steps:
            steps += 1
            rand_idx = random.randint(1, max(1, real_len - 1))
            action = torch.tensor(rand_idx, device=env.device)

            _, _, done, info = env.step(action)
            last_pred = info.get("pred_class", 1)

        mod_times.append(time.perf_counter() - t0)
        total_steps += steps

        if last_pred == 0:
            ter_count += 1
        elif last_pred == 2:
            acr_count += 1
        else:
            mal_count += 1

    return format_results(ter_count, acr_count, mal_count, total_evals, total_steps, mod_times)


# Experiment 3: Continuous Amoeba-style Mamba RL Agent
class ContinuousMambaAmoebaPolicy(nn.Module):
    """
    Continuous Mamba Actor-Critic for Amoeba RL (Experiment 3).
    Predicts:
      1. Location Index
      2. Size Ratio Action in [0.0, 0.5] (+0% to +50% padding)
      3. Delay Action in [0.0, 0.1] (0 to +100ms inserted delay)
    """
    def __init__(self, timevocab_size: int, sizevocab_size: int, embedding_dim: int = 64, seq_len: int = 128):
        super().__init__()
        self.seq_len = seq_len
        self.ipd_embedding = nn.Embedding(timevocab_size, embedding_dim)
        self.size_embedding = nn.Embedding(sizevocab_size, embedding_dim)

        try:
            from advGenerate.mamba_pytorch import Mamba
            self.mamba = Mamba(d_model=embedding_dim * 2, d_state=32, d_conv=4, expand=2)
        except Exception as e:
            logging.warning(f"[!] Mamba SSM CUDA fallback to GRU: {e}")
            self.mamba = nn.GRU(embedding_dim * 2, embedding_dim * 2, num_layers=2, batch_first=True)

        self.flat_dim = embedding_dim * 2 * seq_len
        self.loc_head = nn.Linear(self.flat_dim, seq_len)
        self.cont_head = nn.Linear(self.flat_dim, 2)
        self.val_head = nn.Linear(self.flat_dim, 1)

        nn.init.orthogonal_(self.loc_head.weight, gain=0.01)
        nn.init.orthogonal_(self.cont_head.weight, gain=0.01)

    def forward(self, state: Dict[str, torch.Tensor]):
        ipd = state["flow_ipd"]
        size = state["flow_size"]
        real_len = state["real_length"]

        if ipd.dim() == 1:
            ipd = ipd.unsqueeze(0)
        if size.dim() == 1:
            size = size.unsqueeze(0)
        if real_len.dim() == 0:
            real_len = real_len.unsqueeze(0)

        ipd = torch.clamp(ipd, min=0, max=self.ipd_embedding.num_embeddings - 1)
        size = torch.clamp(size, min=0, max=self.size_embedding.num_embeddings - 1)

        fused = torch.cat([self.ipd_embedding(ipd), self.size_embedding(size)], dim=-1)

        if isinstance(self.mamba, nn.GRU):
            mamba_out, _ = self.mamba(fused)
        else:
            mamba_out = self.mamba(fused)

        mamba_flat = mamba_out.contiguous().view(mamba_out.size(0), -1)

        loc_logits = self.loc_head(mamba_flat)
        cont_raw = torch.sigmoid(self.cont_head(mamba_flat))
        state_val = self.val_head(mamba_flat)

        # Apply numerically stable sequence length mask
        batch_size = loc_logits.size(0)
        device = loc_logits.device
        seq_indices = torch.arange(self.seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
        mask = seq_indices < real_len.view(-1, 1)
        
        loc_logits = loc_logits.masked_fill(~mask, -1e4)
        loc_logits = torch.nan_to_num(loc_logits, nan=-1e4, posinf=100.0, neginf=-1e4)

        # Scale continuous outputs to physical Amoeba bounds
        size_ratio = cont_raw[:, 0] * 0.5   # Max +50% padding
        delay_sec = cont_raw[:, 1] * 0.1    # Max +100ms inserted delay

        return loc_logits, size_ratio, delay_sec, state_val

    def select_action(self, state: Dict[str, torch.Tensor], eval_mode: bool = False):
        loc_logits, size_ratio, delay_sec, state_val = self.forward(state)

        real_len = int(state["real_length"].item()) if state["real_length"].dim() == 0 else int(state["real_length"][0].item())

        if eval_mode:
            loc_idx = torch.argmax(loc_logits, dim=-1).item()
            loc_idx = min(loc_idx, real_len - 1) if real_len > 1 else 0
            return loc_idx, size_ratio.item(), delay_sec.item(), None, state_val
        else:
            loc_dist = torch.distributions.Categorical(logits=loc_logits)
            loc_idx_tensor = loc_dist.sample()
            loc_idx = min(loc_idx_tensor.item(), real_len - 1) if real_len > 1 else 0
            log_prob = loc_dist.log_prob(loc_idx_tensor)
            return loc_idx, size_ratio.item(), delay_sec.item(), log_prob, state_val


def run_exp3_amoeba(
    env: IntelligentPaddingEnv,
    dataset: SimpleFlowDataset,
    max_steps: int = 10,
    train_steps: int = 5000
):
    time_vals = np.array(extract_vocab_list(env.time_vocab))
    size_vals = np.array(extract_vocab_list(env.size_vocab))

    time_vocab_size = len(env.time_vocab)
    size_vocab_size = len(env.size_vocab)

    agent = ContinuousMambaAmoebaPolicy(
        timevocab_size=time_vocab_size,
        sizevocab_size=size_vocab_size,
        embedding_dim=64,
        seq_len=128
    ).to(env.device)

    optimizer = torch.optim.Adam(agent.parameters(), lr=3e-4)

    # Phase 1: RL Policy Training Loop
    logging.info(f"[*] Training Amoeba Continuous Mamba Agent for {train_steps} steps...")
    agent.train()

    train_counter = 0
    dataset_len = len(dataset)
    recent_losses, recent_evasions = [], []

    while train_counter < train_steps:
        sample = dataset[train_counter % dataset_len]
        real_len = int(sample["real_length"].item())
        if real_len <= 1:
            train_counter += 1
            continue

        env.state = {k: v.clone().detach().to(env.device) for k, v in sample.items()}
        env.reset()
        raw_ipds, raw_sizes = env._get_raw_traffic_dimensions()

        eval_res = env.evaluator.predict(raw_ipds, raw_sizes, max_seq_len=128)
        if eval_res[2] == 0:  # Skip benign flows
            train_counter += 1
            continue

        step_in_ep = 0
        ep_evaded = False

        while step_in_ep < max_steps and train_counter < train_steps:
            step_in_ep += 1
            train_counter += 1

            loc_idx, size_ratio, delay_sec, log_prob, state_val = agent.select_action(env.state, eval_mode=False)

            # Continuous Amoeba Physical Mutation
            added_bytes = int(raw_sizes[loc_idx] * size_ratio)
            raw_sizes[loc_idx] += added_bytes
            raw_ipds[loc_idx] += delay_sec

            # Update env.state for next sequence step in episode
            if len(time_vals) > 0:
                env.state["flow_ipd"][loc_idx] = get_nearest_token_idx(raw_ipds[loc_idx], time_vals)
            if len(size_vals) > 0:
                env.state["flow_size"][loc_idx] = get_nearest_token_idx(raw_sizes[loc_idx], size_vals)

            eval_res = env.evaluator.predict(raw_ipds, raw_sizes, max_seq_len=128)
            pred_class = eval_res[2]

            # Reward formulation matching Amoeba
            reward = (10.0 if pred_class == 0 else -1.0) - (0.001 * added_bytes) - (2.0 * delay_sec)

            target_val = torch.tensor([[reward]], device=env.device)
            advantage = torch.clamp(torch.tensor(reward - state_val.detach().item(), device=env.device), -5.0, 5.0)

            actor_loss = -log_prob * advantage
            critic_loss = F.mse_loss(state_val, target_val)
            total_loss = actor_loss + 0.5 * critic_loss

            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(agent.parameters(), 0.5)
            optimizer.step()

            recent_losses.append(total_loss.item())

            if pred_class == 0:
                ep_evaded = True
                break

        recent_evasions.append(1.0 if ep_evaded else 0.0)

        if train_counter % 500 == 0 or train_counter >= train_steps:
            avg_loss = sum(recent_losses[-500:]) / max(len(recent_losses[-500:]), 1)
            avg_eva = (sum(recent_evasions[-100:]) / max(len(recent_evasions[-100:]), 1)) * 100.0
            logging.info(f"    [Mamba Continuous RL Training] Step {train_counter}/{train_steps} | Loss: {avg_loss:.4f} | Evasion Rate: {avg_eva:.1f}%")

    logging.info("[+] Training Complete! Proceeding to Evaluation Pass...")

    # Phase 2: Evaluation Pass
    agent.eval()
    ter_count, acr_count, mal_count = 0, 0, 0
    total_steps, total_evals = 0, 0
    mod_times = []

    with torch.no_grad():
        for idx in range(min(len(dataset), 199)):
            sample = dataset[idx]
            real_len = int(sample["real_length"].item())
            if real_len <= 1:
                continue

            env.state = {k: v.clone().detach().to(env.device) for k, v in sample.items()}
            env.reset()
            raw_ipds, raw_sizes = env._get_raw_traffic_dimensions()

            eval_res = env.evaluator.predict(raw_ipds, raw_sizes, max_seq_len=128)
            if eval_res[2] == 0:
                continue

            total_evals += 1
            done = False
            steps = 0
            t0 = time.perf_counter()

            while not done and steps < max_steps:
                steps += 1

                loc_idx, size_ratio, delay_sec, _, _ = agent.select_action(env.state, eval_mode=True)

                raw_sizes[loc_idx] += int(raw_sizes[loc_idx] * size_ratio)
                raw_ipds[loc_idx] += delay_sec

                # Update env.state for multi-step eval
                if len(time_vals) > 0:
                    env.state["flow_ipd"][loc_idx] = get_nearest_token_idx(raw_ipds[loc_idx], time_vals)
                if len(size_vals) > 0:
                    env.state["flow_size"][loc_idx] = get_nearest_token_idx(raw_sizes[loc_idx], size_vals)

                eval_res = env.evaluator.predict(raw_ipds, raw_sizes, max_seq_len=128)
                last_pred = eval_res[2]

                if last_pred == 0:
                    done = True

            mod_times.append(time.perf_counter() - t0)
            total_steps += steps

            if last_pred == 0:
                ter_count += 1
            elif last_pred == 2:
                acr_count += 1
            else:
                mal_count += 1

    return format_results(ter_count, acr_count, mal_count, total_evals, total_steps, mod_times)


# Helpers & Reporter
def format_results(ter_c, acr_c, mal_c, total, steps, times):
    if total == 0:
        return 0.0, 0.0, 0.0, 0.0, 0.0
    ter = (ter_c / total) * 100
    acr = (acr_c / total) * 100
    mal = (mal_c / total) * 100
    avg_steps = steps / total
    avg_time_ms = (sum(times) / len(times)) * 1000 if times else 0.0
    return ter, acr, mal, avg_steps, avg_time_ms


def append_to_markdown(out_md: str, dataset: str, defense: str, attack: str, exp_name: str, metrics: Tuple[float, ...]):
    ter, acr, mal, avg_steps, avg_time_ms = metrics
    file_exists = os.path.exists(out_md)
    with open(out_md, "a") as f:
        if not file_exists:
            f.write("# Ablation Study Evaluation Results\n\n")
        f.write(f"### {dataset} | {defense.upper()} | {attack} | Exp: `{exp_name}`\n\n")
        f.write(f"- **TER (Targeted Evasion Rate)**: `{ter:.2f}%`  \n")
        f.write(f"- **ACR (Anomaly Catch Rate)**: `{acr:.2f}%`  \n")
        f.write(f"- **Remains Malicious Rate**: `{mal:.2f}%`  \n")
        f.write(f"- **Average Perturbation Steps**: `{avg_steps:.2f}`  \n")
        f.write(f"- **Avg Flow Mod Time**: `{avg_time_ms:.2f} ms`  \n\n")
        f.write("---\n\n")


# Main CLI Entrypoint
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ablation Study Runner (Exp 2 & 3)")
    parser.add_argument("--exp_type", type=str, required=True, choices=["bert_only", "amoeba_rl"])
    parser.add_argument("--world_type", type=str, default="open_world")
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--attack", type=str, required=True)
    parser.add_argument("--defense", type=str, required=True)
    parser.add_argument("--feedback_mode", type=str, default="hard_labeled")
    parser.add_argument("--max_steps", type=int, default=10)
    parser.add_argument("--train_steps", type=int, default=10000)
    parser.add_argument("--out_md", type=str, required=True)

    args = parser.parse_args()

    data_path = f"data/{args.dataset}/{args.attack}.pkl"
    if not os.path.exists(data_path):
        logging.error(f"Dataset path non-existent: {data_path}")
        sys.exit(1)

    evaluator = load_evaluator(args.defense, args.world_type, args.dataset, args.attack, args.feedback_mode)
    rl_args = load_rl_config("advGenerate/config/sac.yaml")
    bert_args = load_bert_config("trafficMimic/config/bert.yaml")

    env = IntelligentPaddingEnv(rl_args, bert_args, mode="test", evaluator=evaluator)
    dataset = SimpleFlowDataset(data_path)

    logging.info(f"[*] Running Ablation Experiment [{args.exp_type}] on {args.dataset}/{args.defense}/{args.attack}")

    if args.exp_type == "bert_only":
        metrics = run_exp2_bert_random(env, dataset, max_steps=args.max_steps)
    elif args.exp_type == "amoeba_rl":
        metrics = run_exp3_amoeba(env, dataset, max_steps=args.max_steps, train_steps=args.train_steps)

    append_to_markdown(args.out_md, args.dataset, args.defense, args.attack, args.exp_type, metrics)
    logging.info(f"[+] Experiment complete. Metrics saved to {args.out_md}")
