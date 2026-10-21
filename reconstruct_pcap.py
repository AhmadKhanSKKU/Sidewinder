import os
import sys
import argparse
import pickle
import torch
import numpy as np
from scapy.all import IP, TCP, Raw, wrpcap
import torch.nn.functional as F

# Ensure parent path imports work
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from advGenerate.sac_mamba import MambaSAC
from advGenerate.environments.environment import IntelligentPaddingEnv
from advGenerate.config_parser import load_rl_config, load_bert_config


class DummyEvaluator:
    """Bypasses IDS model loading during offline PCAP generation."""

    def __init__(self):
        self.mode = "hard_labeled"

    def predict(self, raw_ipds, raw_sizes, max_seq_len=128):
        return 1, 0.99, 1


def reconstruct_packets_from_features(
    sizes: list,
    ipds: list,
    src_ip="192.168.1.100",
    dst_ip="10.0.0.1",
    src_port=54321,
    dst_port=80,
):
    """Reconstructs Scapy packet objects from packet size and IPD sequences."""
    packets = []
    base_time = 1000.0
    current_time = base_time

    for size, ipd in zip(sizes, ipds):
        if size <= 0:
            continue

        current_time += float(ipd)

        header_len = 40
        payload_len = max(0, int(size) - header_len)
        payload_data = b"\x00" * payload_len

        pkt = (
            IP(src=src_ip, dst=dst_ip)
            / TCP(sport=src_port, dport=dst_port, flags="PA")
            / Raw(load=payload_data)
        )
        pkt.time = current_time
        packets.append(pkt)

    return packets


def main():
    parser = argparse.ArgumentParser(
        description="Reconstruct PCAP traces from Morphed Sequences"
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        required=True,
        help="Path to directory or file containing actor.pth",
    )
    parser.add_argument(
        "--input-pkl",
        type=str,
        required=True,
        help="Path to input features pickle file (.pkl)",
    )
    parser.add_argument(
        "--output-pcap",
        type=str,
        required=True,
        help="Destination path for output .pcap",
    )
    parser.add_argument(
        "--max-steps", type=int, default=10, help="Maximum perturbation horizon budget"
    )
    parser.add_argument(
        "--rl-config",
        type=str,
        default="advGenerate/config/sac.yaml",
        help="Path to RL config YAML",
    )
    parser.add_argument(
        "--bert-config",
        type=str,
        default="trafficMimic/config/bert.yaml",
        help="Path to BERT config YAML",
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    rl_args = load_rl_config(args.rl_config)
    bert_args = load_bert_config(args.bert_config)
    rl_args.trainer.max_stop_step = args.max_steps

    print("[*] Initializing IntelligentPaddingEnv (using Dummy Evaluator)...")

    env = IntelligentPaddingEnv(
        rl_args=rl_args,
        bert_args=bert_args,
        mode="test",
        evaluator=DummyEvaluator(),
    )

    with open(args.input_pkl, "rb") as f:
        data = pickle.load(f)

    if isinstance(data, (list, tuple)):
        flow_data = data[0]
    else:
        flow_data = data

    seq_len = 128

    if isinstance(flow_data, dict) and "flow_ipd" in flow_data:
        flow_ipd = flow_data["flow_ipd"].to(device)
        flow_size = flow_data["flow_size"].to(device)
        real_length = flow_data["real_length"].to(device)

        raw_ipd = flow_data["raw_ipd"]
        raw_size = flow_data["raw_size"]

        if not isinstance(raw_ipd, torch.Tensor):
            raw_ipd = torch.tensor(raw_ipd, dtype=torch.float32).to(device)
        if not isinstance(raw_size, torch.Tensor):
            raw_size = torch.tensor(raw_size, dtype=torch.float32).to(device)

        # Ensure batch dimension exists
        if flow_ipd.dim() == 1:
            flow_ipd = flow_ipd.unsqueeze(0)
            flow_size = flow_size.unsqueeze(0)
            raw_ipd = raw_ipd.unsqueeze(0)
            raw_size = raw_size.unsqueeze(0)

        # Apply padding or truncation
        current_len = flow_ipd.size(-1)
        if current_len < seq_len:
            pad_len = seq_len - current_len
            flow_ipd = F.pad(flow_ipd, (0, pad_len), value=0)
            flow_size = F.pad(flow_size, (0, pad_len), value=0)
            raw_ipd = F.pad(raw_ipd, (0, pad_len), value=0.0)
            raw_size = F.pad(raw_size, (0, pad_len), value=0.0)
        elif current_len > seq_len:
            flow_ipd = flow_ipd[:, :seq_len]
            flow_size = flow_size[:, :seq_len]
            raw_ipd = raw_ipd[:, :seq_len]
            raw_size = raw_size[:, :seq_len]
            real_length = torch.clamp(real_length, max=seq_len)

    else:
        ipd_list = list(flow_data[0]) if isinstance(flow_data, (tuple, list)) else []
        size_list = list(flow_data[1]) if isinstance(flow_data, (tuple, list)) else []
        real_len = min(len(ipd_list), seq_len)

        # Apply padding or truncation
        if len(ipd_list) < seq_len:
            pad_len = seq_len - len(ipd_list)
            ipd_list.extend([0] * pad_len)
            size_list.extend([0] * pad_len)

        flow_ipd = torch.tensor([ipd_list[:seq_len]], dtype=torch.long).to(device)
        flow_size = torch.tensor([size_list[:seq_len]], dtype=torch.long).to(device)
        real_length = torch.tensor([real_len], dtype=torch.long).to(device)

        raw_ipd = torch.tensor([ipd_list[:seq_len]], dtype=torch.float32).to(device)
        raw_size = torch.tensor([size_list[:seq_len]], dtype=torch.float32).to(device)

    state, _ = env.load_flows(flow_ipd, flow_size, real_length, raw_ipd, raw_size)

    if os.path.isdir(args.model_dir):
        actor_pth = os.path.join(args.model_dir, "actor.pth")
    else:
        actor_pth = args.model_dir

    timevocab_size = len(env.time_vocab)
    sizevocab_size = len(env.size_vocab)
    embedding_dim = int(getattr(rl_args.model, "embedding_dim", 128))
    seq_len = 128

    if os.path.exists(actor_pth):
        checkpoint = torch.load(actor_pth, map_location=device, weights_only=True)
        if "ipd_embedding.weight" in checkpoint:
            timevocab_size = checkpoint["ipd_embedding.weight"].shape[0]
            print(f"[*] Detected actor timevocab_size: {timevocab_size}")
        if "size_embedding.weight" in checkpoint:
            sizevocab_size = checkpoint["size_embedding.weight"].shape[0]
            print(f"[*] Detected actor sizevocab_size: {sizevocab_size}")
        if "fc_logits.weight" in checkpoint:
            # mamba_output_dim = embedding_dim * 2 * seq_len
            in_features = checkpoint["fc_logits.weight"].shape[1]
            embedding_dim = in_features // (2 * seq_len)
            print(f"[*] Detected actor embedding_dim: {embedding_dim}")

    agent = MambaSAC(
        timevocab_size=timevocab_size,
        sizevocab_size=sizevocab_size,
        embedding_dim=embedding_dim,
        seq_len=seq_len,
        actor_lr=rl_args.model.actor_lr,
        critic_lr=rl_args.model.critic_lr,
        alpha_lr=rl_args.model.alpha_lr,
        target_entropy=rl_args.model.target_entropy,
        tau=rl_args.model.tau,
        gamma=rl_args.model.gamma,
        device=device,
    )

    if os.path.exists(actor_pth):
        agent.actor.load_state_dict(checkpoint)
        print(f"[+] Successfully loaded Actor model directly from: {actor_pth}")
    else:
        print(
            f"[!] Warning: '{actor_pth}' not found. Running with default policy state."
        )

    step_count = 0
    for _ in range(args.max_steps):
        action = agent.take_action(state, eval_mode=True)
        next_state, reward, done, info = env.step(action)
        state = next_state
        step_count += 1

    morphed_ipds, morphed_sizes = env._get_raw_traffic_dimensions()
    print(
        f"[+] Morphed Sequence Generated in {step_count} steps. Total Packets: {len(morphed_sizes)}"
    )

    packets = reconstruct_packets_from_features(morphed_sizes, morphed_ipds)

    os.makedirs(os.path.dirname(os.path.abspath(args.output_pcap)), exist_ok=True)
    wrpcap(args.output_pcap, packets)
    print(f"[+] Successfully exported reconstructed PCAP trace to: {args.output_pcap}")


if __name__ == "__main__":
    main()
