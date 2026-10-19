"""New buffer.py"""

# my-attacker/src/advGenerate/buffer.py

import torch
import numpy as np
from typing import Tuple, List
from collections import deque

from advGenerate.rl_types import FlowState, Transition


class PrioritizedReplayBuffer:

    def __init__(
        self,
        capacity: int,
        device: torch.device,
        alpha: float = 0.6,
        n_step: int = 1,  # Set to 1-step Bellman updates by default
        gamma: float = 0.99,
        user_per: bool = True,
        success_ratio: float = 0.0,  # 10% of batch drawn from successful evasions
        max_success_capacity: int = 5000,
    ):
        self.capacity = capacity
        self.device = device
        self.alpha = alpha
        self.n_step = n_step
        self.gamma = gamma
        self.user_per = user_per
        self.success_ratio = success_ratio

        # Main Replay Buffer (PER)
        self.buffer: List[Transition] = []
        self.pos = 0
        self.priorities = np.zeros((capacity,), dtype=np.float32)

        # Dedicated Success Buffer (Stores transitions where evasion succeeded: reward > 0)
        self.success_buffer: List[Transition] = []
        self.max_success_capacity = max_success_capacity

        # Temporary buffer to hold N-step trajectories (Active if n_step > 1)
        self.n_step_buffer: deque[
            Tuple[FlowState, torch.Tensor, float, FlowState, bool]
        ] = deque(maxlen=n_step)

    def _get_n_step_info(self) -> Tuple[float, FlowState, bool]:
        """Calculates the discounted reward over N steps."""
        reward, next_state, done = (
            self.n_step_buffer[-1][2],
            self.n_step_buffer[-1][3],
            self.n_step_buffer[-1][4],
        )
        for transition in reversed(list(self.n_step_buffer)[:-1]):
            r, n_s, d = transition[2], transition[3], transition[4]
            reward = r + self.gamma * reward * (1 - float(d))
            if d:
                next_state, done = n_s, d
        return reward, next_state, done

    def push(
        self,
        state: FlowState,
        action: torch.Tensor,
        reward: float,
        next_state: FlowState,
        terminal: bool,
        flush: bool = False,
    ):
        """Queues the transition, processes N-step, and stores in buffers."""

        if self.n_step == 1:
            # Fast-path for 1-step returns
            self._store(state, action, reward, next_state, terminal)
        else:
            self.n_step_buffer.append((state, action, reward, next_state, terminal))

            # Once we reach N steps, store bundled transition
            if len(self.n_step_buffer) == self.n_step:
                n_reward, n_next_state, n_done = self._get_n_step_info()
                n_state, n_action = self.n_step_buffer[0][0], self.n_step_buffer[0][1]
                self._store(n_state, n_action, n_reward, n_next_state, n_done)

            # Flush remaining transitions upon episode termination
            if flush:
                if len(self.n_step_buffer) == self.n_step:
                    self.n_step_buffer.popleft()
                while len(self.n_step_buffer) > 0:
                    n_reward, n_next_state, n_term = self._get_n_step_info()
                    n_state, n_action = (
                        self.n_step_buffer[0][0],
                        self.n_step_buffer[0][1],
                    )
                    self._store(n_state, n_action, n_reward, n_next_state, n_term)
                    self.n_step_buffer.popleft()

    def _store(
        self,
        state: FlowState,
        action: torch.Tensor,
        reward: float,
        next_state: FlowState,
        done: bool,
    ) -> None:
        """Stores transition in Main Buffer, and copies to Success Buffer if reward > 0."""

        def detach_state(s: FlowState) -> FlowState:
            detached: FlowState = {}
            for k, v in s.items():
                if isinstance(v, torch.Tensor):
                    detached[k] = v.detach().cpu()
                else:
                    detached[k] = v
            return detached

        transition: Transition = {
            "state": detach_state(state),
            "action": action.detach().cpu(),
            "reward": torch.tensor([[reward]], dtype=torch.float32),
            "next_state": detach_state(next_state),
            "done": torch.tensor([[float(done)]], dtype=torch.float32),
        }

        # 1. Store in Main Buffer
        max_prio = self.priorities.max() if self.buffer else 1.0

        if len(self.buffer) < self.capacity:
            self.buffer.append(transition)
        else:
            self.buffer[self.pos] = transition

        self.priorities[self.pos] = max_prio
        self.pos = (self.pos + 1) % self.capacity

        # 2. Store in Dedicated Success Buffer if positive reward achieved
        if reward > 0.0:
            if len(self.success_buffer) >= self.max_success_capacity:
                self.success_buffer.pop(0)  # Evict oldest success
            self.success_buffer.append(transition)

    def sample(self, batch_size: int, beta: float = 0.4):
        total_main = len(self.buffer)
        total_success = len(self.success_buffer)

        # Determine split sizes between success buffer and main buffer
        if total_success > 0 and self.success_ratio > 0.0:
            n_success = min(int(batch_size * self.success_ratio), total_success)
            n_main = batch_size - n_success
        else:
            n_success = 0
            n_main = batch_size

        # --- SAMPLE MAIN BUFFER ---
        if self.user_per:
            prios = (
                self.priorities
                if total_main == self.capacity
                else self.priorities[: self.pos]
            )
            probs = prios**self.alpha
            probs /= probs.sum()

            main_indices = np.random.choice(total_main, n_main, p=probs)
            weights = (total_main * probs[main_indices]) ** (-beta)
            weights /= weights.max()
            weights = np.array(weights, dtype=np.float32)
        else:
            main_indices = np.random.choice(total_main, n_main, replace=False)
            weights = np.ones((n_main,), dtype=np.float32)

        batch: list[Transition] = [self.buffer[idx] for idx in main_indices]

        # --- SAMPLE SUCCESS BUFFER ---
        if n_success > 0:
            success_indices = np.random.choice(
                total_success, n_success, replace=(total_success < n_success)
            )
            success_batch: list[Transition] = [
                self.success_buffer[idx] for idx in success_indices
            ]
            batch.extend(success_batch)

            # Extend PER weights with 1.0 for success samples
            success_weights = np.ones((n_success,), dtype=np.float32)
            weights = np.concatenate([weights, success_weights], axis=0)

        # --- STACK TENSORS MATCHING TRAINER FORMAT ---
        state_batch: FlowState = {}
        next_state_batch: FlowState = {}

        for k in batch[0]["state"].keys():
            state_batch[k] = torch.stack(
                [t["state"][k].squeeze() for t in batch], dim=0
            ).to(self.device)

        for k in batch[0]["next_state"].keys():
            next_state_batch[k] = torch.stack(
                [t["next_state"][k].squeeze() for t in batch], dim=0
            ).to(self.device)

        action_batch = torch.cat([t["action"] for t in batch], dim=0).to(self.device)
        reward_batch = torch.cat([t["reward"] for t in batch], dim=0).to(self.device)
        done_batch = torch.cat([t["done"] for t in batch], dim=0).to(self.device)

        return (
            state_batch,
            action_batch,
            reward_batch,
            next_state_batch,
            done_batch,
            main_indices,  # Indices returned correspond to main PER updates
            weights,
        )

    def update_priorities(
        self, batch_indices: np.ndarray, batch_priorities: np.ndarray
    ) -> None:
        if not self.user_per:
            return

        # Safely update priorities for the portion sampled from main buffer
        for idx, prio in zip(batch_indices, batch_priorities[: len(batch_indices)]):
            if idx < len(self.priorities):
                self.priorities[idx] = prio

    def __len__(self) -> int:
        return len(self.buffer)
