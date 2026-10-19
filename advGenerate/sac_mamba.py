# src/advGenerate/sac_mamba.py

import os
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from typing import Tuple, Dict, Optional

from advGenerate.mamba_pytorch import Mamba

LOG_SIG_MAX = 1
LOG_SIG_MIN = -20
EPSILON = 1e-6


class MambaPolicyNet(nn.Module):
    def __init__(
        self, timevocab_size: int, sizevocab_size: int, embedding_dim: int, seq_len: int
    ) -> None:
        super(MambaPolicyNet, self).__init__()
        self.seq_len = seq_len
        self.action_dim = seq_len

        self.ipd_embedding = nn.Embedding(timevocab_size, embedding_dim)
        self.size_embedding = nn.Embedding(sizevocab_size, embedding_dim)

        self.mamba = Mamba(d_model=embedding_dim * 2, d_state=32, d_conv=4, expand=2)
        self.mamba_output_dim = embedding_dim * 2 * seq_len
        self.fc_logits = nn.Linear(self.mamba_output_dim, self.action_dim)

        nn.init.orthogonal_(self.fc_logits.weight, gain=0.01)
        self.fc_logits.bias.data.zero_()

    def forward(self, state: Dict[str, torch.Tensor]) -> torch.Tensor:
        flow_ipd = state["flow_ipd"]
        flow_size = state["flow_size"]
        real_length = state["real_length"]

        if flow_ipd.dim() == 1:
            flow_ipd = flow_ipd.unsqueeze(0)
        if flow_size.dim() == 1:
            flow_size = flow_size.unsqueeze(0)
        if real_length.dim() == 0:
            real_length = real_length.unsqueeze(0)

        flow_ipd = torch.clamp(
            flow_ipd, min=0, max=self.ipd_embedding.num_embeddings - 1
        )
        flow_size = torch.clamp(
            flow_size, min=0, max=self.size_embedding.num_embeddings - 1
        )

        ipd_emb = self.ipd_embedding(flow_ipd)
        size_emb = self.size_embedding(flow_size)

        fused_features = torch.cat([ipd_emb, size_emb], dim=-1)
        mamba_out = self.mamba(fused_features)

        mamba_out_flat = mamba_out.contiguous().view(mamba_out.size(0), -1)
        logits = self.fc_logits(mamba_out_flat)

        batch_size = logits.size(0)
        device = logits.device

        # --- MASK 1: PADDING TOKENS ---
        # Prevent modifying any indices beyond the real packet length
        seq_indices = (
            torch.arange(self.seq_len, device=device)
            .unsqueeze(0)
            .expand(batch_size, -1)
        )
        real_len_broadcast = real_length.view(-1, 1)
        action_mask = seq_indices < real_len_broadcast
        masked_logits = logits.masked_fill(~action_mask, -1e9)

        # --- MASK 2: ALREADY MODIFIED / FORBIDDEN TOKENS ---
        if "forbidden_indices" in state:
            forbidden_mask = state["forbidden_indices"].bool()

            if forbidden_mask.dim() == 1:
                # Expand a single environment mask to match batch size
                forbidden_mask = forbidden_mask.unsqueeze(0).expand(batch_size, -1)
            elif forbidden_mask.dim() == 3:
                # Clean up dimensions from replay buffer (e.g., [batch, 1, seq_len])
                forbidden_mask = forbidden_mask.squeeze(1)

            # Apply the mask: -1e9 means 0.0% probability of being chosen
            masked_logits = masked_logits.masked_fill(forbidden_mask, -1e9)

        return masked_logits

    def sample(
        self, state: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        masked_logits = self.forward(state)
        action_probs = F.softmax(masked_logits, dim=-1)
        dist = Categorical(logits=masked_logits)

        actions: torch.Tensor = dist.sample()
        log_probs: torch.Tensor = dist.log_prob(actions)

        return actions.unsqueeze(-1), log_probs.unsqueeze(-1), action_probs


class MambaCriticNet(nn.Module):
    def __init__(
        self, timevocab_size: int, sizevocab_size: int, embedding_dim: int, seq_len: int
    ) -> None:
        super(MambaCriticNet, self).__init__()
        self.ipd_embedding = nn.Embedding(timevocab_size, embedding_dim)
        self.size_embedding = nn.Embedding(sizevocab_size, embedding_dim)
        self.mamba = Mamba(d_model=embedding_dim * 2, d_state=32, d_conv=4, expand=2)
        self.fc = nn.Linear(embedding_dim * 2 * seq_len, seq_len)

    def forward(self, state: Dict[str, torch.Tensor]) -> torch.Tensor:
        safe_ipd = torch.clamp(
            state["flow_ipd"], 0, self.ipd_embedding.num_embeddings - 1
        )
        safe_size = torch.clamp(
            state["flow_size"], 0, self.size_embedding.num_embeddings - 1
        )

        ipd_embed = self.ipd_embedding(safe_ipd)
        size_embed = self.size_embedding(safe_size)

        combined_seq = torch.cat((ipd_embed, size_embed), dim=-1)
        out = self.mamba(combined_seq)
        out = out.view(out.size(0), -1)
        return self.fc(out)


class MambaSAC:
    def __init__(
        self,
        timevocab_size: int,
        sizevocab_size: int,
        embedding_dim: int,
        seq_len: int,
        actor_lr: float,
        critic_lr: float,
        alpha_lr: float,
        target_entropy: float,
        tau: float,
        gamma: float,
        device: torch.device,
        n_steps: int = 3,
    ):
        self.device = device
        self.gamma = gamma
        self.tau = tau
        self.seq_len = seq_len
        self.n_steps = n_steps
        self.gamma_n = gamma**n_steps

        self.actor = MambaPolicyNet(
            timevocab_size, sizevocab_size, embedding_dim, seq_len
        ).to(device)
        self.critic_1 = MambaCriticNet(
            timevocab_size, sizevocab_size, embedding_dim, seq_len
        ).to(device)
        self.critic_2 = MambaCriticNet(
            timevocab_size, sizevocab_size, embedding_dim, seq_len
        ).to(device)

        self.target_critic_1 = MambaCriticNet(
            timevocab_size, sizevocab_size, embedding_dim, seq_len
        ).to(device)
        self.target_critic_2 = MambaCriticNet(
            timevocab_size, sizevocab_size, embedding_dim, seq_len
        ).to(device)
        self.target_critic_1.load_state_dict(self.critic_1.state_dict())
        self.target_critic_2.load_state_dict(self.critic_2.state_dict())

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_1_optimizer = torch.optim.Adam(
            self.critic_1.parameters(), lr=critic_lr
        )
        self.critic_2_optimizer = torch.optim.Adam(
            self.critic_2.parameters(), lr=critic_lr
        )

        """New"""
        self.target_entropy = target_entropy
        self.log_alpha = torch.tensor(0.0, requires_grad=True, device=self.device)
        self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=alpha_lr)
        self.alpha = self.log_alpha.exp()

    def take_action(
        self, state: Dict[str, torch.Tensor], eval_mode: bool = False
    ) -> int:
        state_tensor = {k: v.to(self.device) for k, v in state.items()}
        with torch.no_grad():
            masked_logits = self.actor(state_tensor)

            if eval_mode:
                action = torch.argmax(masked_logits, dim=-1)

                """Temperature Scaling for Evaluation"""

            else:
                dist = Categorical(logits=masked_logits)
                action = dist.sample()
        return int(action.item())

    def update(
        self,
        transition_dict: Dict[str, torch.Tensor],
        weights: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        states = transition_dict["states"]
        actions = transition_dict["actions"].view(-1, 1).long()
        rewards = transition_dict["rewards"].view(-1, 1)
        next_states = transition_dict["next_states"]
        dones = transition_dict["dones"].view(-1, 1)

        # ------------------- 1. CRITIC UPDATE -------------------
        with torch.no_grad():
            next_logits = self.actor(next_states)
            next_probs = torch.softmax(next_logits, dim=-1)
            next_log_probs = F.log_softmax(next_logits, dim=-1)

            target_q1 = self.target_critic_1(next_states)
            target_q2 = self.target_critic_2(next_states)
            min_target_q: torch.Tensor = torch.min(target_q1, target_q2)

            # Discrete SAC Target Value: E_a ~ pi [ Q(s', a) - alpha * log_pi(a|s') ]
            target_v: torch.Tensor = (
                next_probs * (min_target_q - self.alpha * next_log_probs)
            ).sum(dim=1, keepdim=True)
            y = rewards + (1.0 - dones) * self.gamma_n * target_v

        q1 = self.critic_1(states).gather(1, actions)
        q2 = self.critic_2(states).gather(1, actions)

        td_error_1 = torch.abs(q1 - y)
        td_error_2 = torch.abs(q2 - y)
        td_errors = ((td_error_1 + td_error_2) / 2.0).squeeze(-1).detach().cpu().numpy()

        if weights is not None:
            weights_tensor = torch.FloatTensor(weights).to(self.device).unsqueeze(1)
            critic_1_loss = (
                weights_tensor * F.mse_loss(q1, y, reduction="none")
            ).mean()
            critic_2_loss = (
                weights_tensor * F.mse_loss(q2, y, reduction="none")
            ).mean()
        else:
            critic_1_loss = F.mse_loss(q1, y)
            critic_2_loss = F.mse_loss(q2, y)

        self.critic_1_optimizer.zero_grad()
        critic_1_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic_1.parameters(), 1.0)
        self.critic_1_optimizer.step()

        self.critic_2_optimizer.zero_grad()
        critic_2_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic_2.parameters(), 1.0)
        self.critic_2_optimizer.step()

        # ------------------- 2. ACTOR UPDATE -------------------
        logits = self.actor(states)
        probs = torch.softmax(logits, dim=-1)
        log_probs = F.log_softmax(logits, dim=-1)

        with torch.no_grad():
            q1_pi: torch.Tensor = self.critic_1(states)
            q2_pi: torch.Tensor = self.critic_2(states)
            min_q_pi: torch.Tensor = torch.min(q1_pi, q2_pi)

        actor_loss = (
            (probs * (self.alpha.detach() * log_probs - min_q_pi)).sum(dim=1).mean()
        )

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
        self.actor_optimizer.step()

        alpha_loss = (
            (
                probs.detach()
                * (-self.log_alpha.exp() * (log_probs.detach() + self.target_entropy))
            )
            .sum(dim=-1)
            .mean()
        )

        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()

        # Update tensor value used in next step's critic target computation
        self.alpha = self.log_alpha.exp().detach()

        # ------------------- 4. TARGET NETWORK SOFT UPDATE -------------------
        self.soft_update()

        return td_errors

    def soft_update(self):
        for param, target_param in zip(
            self.critic_1.parameters(), self.target_critic_1.parameters()
        ):
            target_param.data.copy_(
                self.tau * param.data + (1.0 - self.tau) * target_param.data
            )
        for param, target_param in zip(
            self.critic_2.parameters(), self.target_critic_2.parameters()
        ):
            target_param.data.copy_(
                self.tau * param.data + (1.0 - self.tau) * target_param.data
            )

    def save_model(self, path: str):
        os.makedirs(path, exist_ok=True)
        torch.save(self.actor.state_dict(), os.path.join(path, "actor.pth"))
        torch.save(self.critic_1.state_dict(), os.path.join(path, "critic_1.pth"))
        torch.save(self.critic_2.state_dict(), os.path.join(path, "critic_2.pth"))

    def load_model(self, path: str):
        self.actor.load_state_dict(
            torch.load(os.path.join(path, "actor.pth"), map_location=self.device)
        )
        self.critic_1.load_state_dict(
            torch.load(os.path.join(path, "critic_1.pth"), map_location=self.device)
        )
        self.critic_2.load_state_dict(
            torch.load(os.path.join(path, "critic_2.pth"), map_location=self.device)
        )
        self.target_critic_1.load_state_dict(self.critic_1.state_dict())
        self.target_critic_2.load_state_dict(self.critic_2.state_dict())
