# NetMasquerade-main/src/advGenerate/rl_utils.py

import torch
import collections
import random


import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm

from typing import TypedDict, List, Tuple, Deque, Any, Iterable


class StateDict(TypedDict):
    flow_ipd: torch.Tensor
    flow_size: torch.Tensor
    real_length: torch.Tensor
    src_index: torch.Tensor


class TransitionDict(TypedDict):
    states: StateDict
    actions: torch.Tensor
    next_states: StateDict
    rewards: torch.Tensor
    dones: torch.Tensor


class ReplayBuffer:
    def __init__(self, capacity: int) -> None:
        self.buffer: Deque[Tuple[StateDict, int, float, StateDict, bool]] = (
            collections.deque(maxlen=capacity)
        )

    def add(
        self,
        state: StateDict,
        action: int,
        reward: float,
        next_state: StateDict,
        done: bool,
    ) -> None:
        self.buffer.append((state, action, reward, next_state, done))

    def sample(
        self, batch_size: int
    ) -> Tuple[StateDict, List[int], List[float], StateDict, List[bool]]:
        transitions = random.sample(self.buffer, batch_size)
        state_list, action, reward, next_state_list, done = zip(*transitions)
        # if isinstance(state[0], dict):
        #     keys = state[0].keys()

        #     state = {
        #         key: torch.cat([s[key].unsqueeze(0) for s in state], dim=0)
        #         for key in keys
        #     }

        # if isinstance(next_state[0], dict):
        #     keys = next_state[0].keys()
        #     next_state = {
        #         key: torch.cat([s[key].unsqueeze(0) for s in next_state])
        #         for key in keys
        #     }

        # return state, action, reward, next_state, done
        def _collate_states(states: Iterable[StateDict]) -> StateDict:
            first = list(states)[0]
            keys = first.keys()
            return {
                key: torch.cat([s[key].unsqueeze(0) for s in states], dim=0)  # type: ignore
                for key in keys
            }  # type: ignore

        return (
            _collate_states(state_list),
            list(action),
            list(reward),
            _collate_states(next_state_list),
            list(done),
        )

    def size(self) -> int:
        return len(self.buffer)


def moving_average(a: np.ndarray, window_size: int) -> np.ndarray:
    cumulative_sum = np.cumsum(np.insert(a, 0, 0))
    middle = (
        cumulative_sum[window_size:] - cumulative_sum[:-window_size]
    ) / window_size
    r = np.arange(1, window_size - 1, 2)
    begin = np.cumsum(a[: window_size - 1])[::2] / r
    end = (np.cumsum(a[:-window_size:-1])[::2] / r)[::-1]
    return np.concatenate((begin, middle, end))


def train_on_policy_agent(env, agent, num_episodes: int) -> List[float]:
    return_list = []
    for i in range(10):
        with tqdm(total=int(num_episodes / 10), desc="Iteration %d" % i) as pbar:
            for i_episode in range(int(num_episodes / 10)):
                episode_return = 0
                transition_dict = {
                    "states": [],
                    "actions": [],
                    "next_states": [],
                    "rewards": [],
                    "dones": [],
                }
                state = env.reset()
                done = False
                while not done:
                    action = agent.take_action(state)
                    next_state, reward, done, _ = env.step(action)
                    transition_dict["states"].append(state)
                    transition_dict["actions"].append(action)
                    transition_dict["next_states"].append(next_state)
                    transition_dict["rewards"].append(reward)
                    transition_dict["dones"].append(done)
                    state = next_state
                    episode_return += reward
                return_list.append(episode_return)
                agent.update(transition_dict)
                if (i_episode + 1) % 10 == 0:
                    pbar.set_postfix(
                        {
                            "episode": "%d" % (num_episodes / 10 * i + i_episode + 1),
                            "return": "%.3f" % np.mean(return_list[-10:]),
                        }
                    )
                pbar.update(1)
    return return_list


def train_off_policy_agent(
    env, agent, num_episodes, replay_buffer, minimal_size, batch_size
):
    return_list = []
    for i in range(10):
        with tqdm(total=int(num_episodes / 10), desc="Iteration %d" % i) as pbar:
            for i_episode in range(int(num_episodes / 10)):
                episode_return = 0
                state = env.reset()
                done = False
                while not done:
                    action = agent.take_action(state)
                    next_state, reward, done, _ = env.step(action)
                    replay_buffer.add(state, action, reward, next_state, done)
                    state = next_state
                    episode_return += reward
                    if replay_buffer.size() > minimal_size:
                        b_s, b_a, b_r, b_ns, b_d = replay_buffer.sample(batch_size)
                        transition_dict = {
                            "states": b_s,
                            "actions": b_a,
                            "next_states": b_ns,
                            "rewards": b_r,
                            "dones": b_d,
                        }
                        agent.update(transition_dict)
                return_list.append(episode_return)
                if (i_episode + 1) % 10 == 0:
                    pbar.set_postfix(
                        {
                            "episode": "%d" % (num_episodes / 10 * i + i_episode + 1),
                            "return": "%.3f" % np.mean(return_list[-10:]),
                        }
                    )
                pbar.update(1)
    return return_list


def compute_advantage(
    gamma: float, lmbda: float, td_delta: torch.Tensor
) -> torch.Tensor:
    delta_np: np.ndarray[Any, np.dtype[np.float32]] = td_delta.detach().cpu().numpy()
    advantage_list: List[float] = []
    advantage: float = 0.0
    for delta in delta_np[::-1]:
        advantage = gamma * lmbda * advantage + delta
        advantage_list.append(advantage)
    advantage_list.reverse()
    return torch.tensor(advantage_list, dtype=torch.float)


def generate_mask(
    real_length: torch.Tensor, pad_length: int, mask_index: torch.Tensor
) -> torch.Tensor:
    if real_length.dim() == 0:
        real_length = real_length.unsqueeze(0)

    # Ensure mask_index is 2D [batch_size, seq_len]
    if mask_index.dim() == 1:
        mask_index = mask_index.unsqueeze(0)

    batch_size = mask_index.shape[0]

    expand_mask_index = torch.zeros(
        batch_size,
        pad_length,
        dtype=mask_index.dtype,
        device=mask_index.device,
    )

    # ---> FAST INTEGER MATH (NO GPU SYNC DELAY) <---
    col_indices = torch.arange(mask_index.shape[1], device=mask_index.device) * 2 + 1
    max_valid = min(mask_index.shape[1], pad_length // 2)
    valid_cols = col_indices[:max_valid]

    expand_mask_index[:, valid_cols] = mask_index[:, :max_valid]

    # Set even positions to 1
    even_indices = torch.arange(pad_length, device=mask_index.device)
    mask_even = even_indices % 2 == 0
    expand_mask_index[:, mask_even] = 1

    seq_idx = (
        torch.arange(pad_length, device=real_length.device)
        .unsqueeze(0)
        .repeat(batch_size, 1)
    )

    if real_length.dim() == 1:
        rl_comp = real_length.unsqueeze(1)
    else:
        rl_comp = real_length

    mask = (seq_idx > 3) & (seq_idx < rl_comp) & (expand_mask_index == 0)

    if mask.dim() == 1:
        mask = mask.unsqueeze(0)

    return mask


def plot_train_rewards(
    rewards: List[float], save_path: str, ylabel: str = "Reward"
) -> None:
    plt.style.use("seaborn-darkgrid")

    palette = plt.get_cmap("Set2")

    plt.figure(figsize=(10, 5))
    plt.plot(rewards, color=palette(0))
    plt.title("Training Reward Over Episodes")
    plt.xlabel("Episodes")
    plt.ylabel(ylabel)

    plt.savefig(save_path)
    plt.close()


if __name__ == "__main__":
    pass
