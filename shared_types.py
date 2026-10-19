# NetMasquerade-main/src/shared_types.py

import torch

from typing import (
    Any,
    Dict,
    TypedDict,
    List,
    Tuple,
    Union,
    Optional,
    Protocol,
    runtime_checkable,
)


# --- Configuration Shapes ---
class ModelConfig(TypedDict):
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


class TrainerConfig(TypedDict):
    device: str
    buffer_size: int
    batch_size: int
    train_data_pth: str
    test_data_pth: str
    max_stop_step: int
    stop_threshold: float
    flow_save_pth: Optional[str]
    env_name: str
    episodes: int


class RLArgs(Protocol):
    model: ModelConfig
    trainer: TrainerConfig


# --- Data Shapes ---
class StateDict(TypedDict):
    flow_ipd: torch.Tensor
    flow_size: torch.Tensor
    real_length: torch.Tensor
    src_index: torch.Tensor


# NEW: Strictly defines the dictionary returned for original features
class RealFeatDict(TypedDict):
    ipd: List[float]
    size: List[int]


class TransitionBatch(TypedDict):
    states: StateDict
    actions: torch.Tensor
    rewards: torch.Tensor
    next_states: StateDict
    dones: torch.Tensor


@runtime_checkable
class EvaluatorProtocol(Protocol):
    def evaluate(
        self, ipds: List[float], sizes: List[int]
    ) -> Union[bool, Tuple[bool, float]]: ...


class OnPolicyTransitionDict(TypedDict):
    states: List[StateDict]
    actions: List[int]
    next_states: List[StateDict]
    rewards: List[float]
    dones: List[bool]


# Use this for Off-Policy (Batched Tensors)
class OffPolicyTransitionDict(TypedDict):
    states: StateDict
    actions: torch.Tensor
    next_states: StateDict
    rewards: torch.Tensor
    dones: torch.Tensor


# Define the Environment Protocol so we don't use 'Any' for env
class EnvProtocol(Protocol):
    def reset(self) -> StateDict: ...
    def step(self, action: int) -> Tuple[StateDict, float, bool, Dict[str, Any]]: ...


# Define the Agent Protocol
class AgentProtocol(Protocol):
    def take_action(self, state: StateDict) -> int: ...
    def update(
        self, transition_dict: Union[OnPolicyTransitionDict, OffPolicyTransitionDict]
    ) -> None: ...
