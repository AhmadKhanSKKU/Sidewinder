# my-attacker/src/advGenerate/rl_types.py

# Library imports


import torch
from typing import TypedDict, Optional
from dataclasses import dataclass


class FlowState(TypedDict):
    """Strict typing for the Network Flow State"""

    flow_ipd: torch.Tensor  # Shape: [batch_size, seq_len] - Tokenized IPDs
    flow_size: torch.Tensor  # Shape: [batch_size, seq_len] - Tokenized Sizes
    real_length: (
        torch.Tensor
    )  # Shape: [batch_size] - Actual unpadded length of the flow


class Transition(TypedDict):
    """Strict typing for the Replay Buffer transitions"""

    state: FlowState
    action: torch.Tensor  # Shape: [batch_size, seq_len * 2] (Padding & Delays)
    reward: torch.Tensor  # Shape: [batch_size, 1]
    next_state: FlowState
    done: torch.Tensor  # Shape: [batch_size, 1]


def compute_batched_1d_wasserstein(
    mutated_features: torch.Tensor, baseline_features: torch.Tensor
) -> torch.Tensor:
    """
    Computes the exact 1D Wasserstein distance between two PyTorch tensors in a batched manner.
    Extremely fast: O(N log N) due to sorting, fully executed on the GPU.

    Args:
        mutated_features: Shape [batch_size, seq_len]
        baseline_features: Shape [batch_size, seq_len]

    Returns:
        torch.Tensor: Shape [batch_size, 1] containing the Wasserstein distance per flow.
    """

    sorted_mutated, _ = torch.sort(mutated_features, dim=1)
    sorted_baseline, _ = torch.sort(baseline_features, dim=1)

    # Calculate absolute difference and mean across the sequence length
    wasserstein_dist = torch.mean(
        torch.abs(sorted_mutated - sorted_baseline), dim=1, keepdim=True
    )
    return wasserstein_dist


@dataclass
class RLTrainerConfig:
    device: int
    train_data_pth: str
    test_data_pth: str
    episodes: int
    test_episodes: int
    batch_size: int
    buffer_size: int
    minimal_size: int
    model_save_pth: str
    stop_threshold: float
    max_stop_step: int
    env_name: str
    flow_save_pth: str
    bert_pth: str
    target_model_pth: Optional[str] = None
    fig_save_pth: Optional[str] = None
    figure_path: Optional[str] = None


@dataclass
class RLEnvConfig:
    bad_ip: str


@dataclass
class RLModelConfig:
    timevocab_size: int
    sizevocab_size: int
    embedding_dim: int
    state_dim: int
    hidden_dim: int
    actor_lr: float
    critic_lr: float
    alpha_lr: float
    target_entropy: float
    tau: float
    gamma: float
    seq_len: int


@dataclass
class RLArgs:
    trainer: RLTrainerConfig
    env: RLEnvConfig
    model: RLModelConfig


# ==========================================
# STRICT TYPING FOR BERT CONFIGURATION
# ==========================================


@dataclass
class BERTTrainerConfig:
    device: int
    timevocab_pth: str
    sizevocab_pth: str
    # Add other BERT trainer arguments here if they exist in bert.yaml
    train_data_pth: Optional[str] = None
    test_data_pth: Optional[str] = None


@dataclass
class BERTModelConfig:
    # Standard BERT hyperparams based on their original implementation
    seq_len: int = 512
    hidden: int = 768
    d_ff: int = 3072
    n_layers: int = 12
    attn_heads: int = 8
    dropout: float = 0.1


@dataclass
class BERTArgs:
    trainer: BERTTrainerConfig
    model: BERTModelConfig
