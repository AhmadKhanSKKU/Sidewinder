import bisect
import os
import sys
from typing import Literal, Optional, Union

import numpy as np
import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from advGenerate.environments.cnn_environment_eval import CNN_IDS_Evaluator
from advGenerate.environments.dl_environment_eval import DL_IDS_Evaluator
from advGenerate.environments.pandora_seq_environment_eval import (
    PandoraSeq_IDS_Evaluator,
)
from advGenerate.environments.prob_environment_eval import (
    Probabilistic_IDS_Evaluator,
)
from advGenerate.environments.transformer_environment_eval import (
    Transformer_IDS_Evaluator,
)
from advGenerate.environments.vib_environment_eval import VIB_IDS_Evaluator
from advGenerate.rl_types import BERTArgs, FlowState, RLArgs
from trafficMimic.dataset.vocab import SizeVocab, TimeVocab
from trafficMimic.model.architecture import BERTLM
from trafficMimic.model.architecture.bert import BERT


class IntelligentPaddingEnv:
    """NetMasquerade-parity 256-action environment (even = insert chaff, odd = delay-only)."""

    def __init__(
        self,
        rl_args: RLArgs,
        bert_args: BERTArgs,
        mode: Union[Literal["train"], Literal["test"]] = "train",
        evaluator: Optional[
            Union[
                Probabilistic_IDS_Evaluator,
                CNN_IDS_Evaluator,
                DL_IDS_Evaluator,
                Transformer_IDS_Evaluator,
                PandoraSeq_IDS_Evaluator,
                VIB_IDS_Evaluator,
            ]
        ] = None,
    ) -> None:
        self.rl_args = rl_args
        self.bert_args = bert_args
        self.mode = mode
        self.device = torch.device(
            f"cuda:{self.rl_args.trainer.device}"
            if torch.cuda.is_available()
            else "cpu"
        )

        self.time_vocab = TimeVocab.load_vocab(self.bert_args.trainer.timevocab_pth)
        self.size_vocab = SizeVocab.load_vocab(self.bert_args.trainer.sizevocab_pth)

        self.MAX_PADDING_BYTES = 500
        self.MAX_DELAY_SECONDS = 0.5
        self.max_steps = self.rl_args.trainer.max_stop_step
        self.seq_len = 128
        self.action_dim = (
            self.seq_len * 2
        )  # 256: (position * 2 + op), 0=insert, 1=delay

        self.evaluator = (
            evaluator if evaluator is not None else Probabilistic_IDS_Evaluator()
        )

        self.step_penalty = 0.1
        self.success_reward = 15.0

        time_vocab_len = len(self.time_vocab)
        size_vocab_len = len(self.size_vocab)

        bert_base = BERT(
            time_vocab_len,
            size_vocab_len,
            hidden=self.bert_args.model.hidden,
            d_ff=self.bert_args.model.d_ff,
            n_layers=self.bert_args.model.n_layers,
            attn_heads=self.bert_args.model.attn_heads,
            dropout=self.bert_args.model.dropout,
        )

        self.bert_model = BERTLM(bert_base, time_vocab_len, size_vocab_len).to(
            self.device
        )

        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.abspath(os.path.join(current_dir, "../../../"))
        model_path = os.path.join(
            project_root, "pretrained_models", "pandora_iot_bert.pth_sizebest.pth"
        )

        print(f"[*] Loading BERT weights from: {model_path}")
        self.bert_model.load_state_dict(
            torch.load(model_path, map_location=self.device, weights_only=False)
        )
        self.bert_model.eval()
        print("[+] BERT Model Loaded Successfully!")

    @staticmethod
    def _insert_tensor_1d(
        sequence: torch.Tensor, value: Union[int, float], position: int
    ) -> torch.Tensor:
        """Inserts `value` at `position`, shifting subsequent packets right and truncating to seq_len."""
        val = torch.as_tensor([value], dtype=sequence.dtype, device=sequence.device)
        return torch.cat((sequence[:position], val, sequence[position:]))[
            : sequence.shape[0]
        ]

    def _insert_into_state_tensor(
        self, tensor: torch.Tensor, value: Union[int, float], position: int
    ) -> torch.Tensor:
        if tensor.dim() == 2:
            return self._insert_tensor_1d(tensor[0], value, position).unsqueeze(0)
        return self._insert_tensor_1d(tensor, value, position)

    def get_legal_action_mask(self) -> torch.Tensor:
        """Generates a 256-dim boolean mask of valid (insert, delay) actions."""
        real_len = (
            int(self.state["real_length"].item())
            if self.state and "real_length" in self.state
            else self.seq_len
        )
        legal = torch.zeros(self.action_dim, dtype=torch.bool, device=self.device)
        can_insert = real_len < self.seq_len

        for pos in range(1, min(real_len + 1, self.seq_len)):
            if hasattr(self, "forbidden_indices") and pos < len(self.forbidden_indices):
                if bool(self.forbidden_indices[pos]):
                    continue
            # Even index (2 * pos): insert chaff packet before `pos`
            if can_insert:
                legal[2 * pos] = True
            # Odd index (2 * pos + 1): delay existing packet at `pos`
            if pos < real_len:
                legal[2 * pos + 1] = True

        # Fallback if all positions were masked
        if not torch.any(legal):
            legal[2 * min(1, self.seq_len - 1) + 1] = True
        return legal

    def load_flows(
        self,
        flow_ipd: torch.Tensor,
        flow_size: torch.Tensor,
        real_length: torch.Tensor,
        raw_ipd: torch.Tensor,
        raw_size: torch.Tensor,
    ):
        self.step_num = 0
        self.chaff_indices: list[int] = []
        real_len_int = int(real_length.item())

        raw_ipd_t = (
            raw_ipd.clone()
            if isinstance(raw_ipd, torch.Tensor)
            else torch.tensor(raw_ipd, dtype=torch.float32, device=self.device)
        )
        raw_size_t = (
            raw_size.clone()
            if isinstance(raw_size, torch.Tensor)
            else torch.tensor(raw_size, dtype=torch.float32, device=self.device)
        )

        self.state = {
            "flow_ipd": flow_ipd.clone(),
            "flow_size": flow_size.clone(),
            "real_length": real_length.clone(),
            "raw_ipd": raw_ipd_t,
            "raw_size": raw_size_t,
        }

        raw_ipds, raw_sizes = self._decode_tokens(flow_ipd, flow_size, real_len_int)
        self.original_raw_ipds = list(raw_ipds)
        self.original_raw_sizes = list(raw_sizes)

        res = self.evaluator.predict(
            self.original_raw_ipds, self.original_raw_sizes, max_seq_len=128
        )
        self.baseline_score = res[1]
        self.prev_score = self.baseline_score
        is_valid = self.baseline_score >= 0.50

        self.forbidden_indices = torch.zeros(
            self.seq_len, dtype=torch.bool, device=self.device
        )
        self.forbidden_indices[0] = True
        self.state["forbidden_indices"] = self.forbidden_indices.clone()
        self.state["legal_action_mask"] = self.get_legal_action_mask()

        return self.state, is_valid

    def _decode_tokens(
        self, flow_ipd: torch.Tensor, flow_size: torch.Tensor, real_length: int
    ):
        raw_ipds = self.state["raw_ipd"].squeeze().tolist()
        raw_sizes = self.state["raw_size"].squeeze().tolist()
        return raw_ipds[:real_length], raw_sizes[:real_length]

    def _predict_tokens(
        self, masked_ipd: torch.Tensor, masked_size: torch.Tensor, position: int
    ) -> tuple[int, int]:
        embedding_layers = [
            m for m in self.bert_model.modules() if isinstance(m, torch.nn.Embedding)
        ]
        if len(embedding_layers) >= 2:
            max_ipd_vocab = embedding_layers[0].num_embeddings
            max_size_vocab = embedding_layers[1].num_embeddings
        else:
            max_ipd_vocab = 56
            max_size_vocab = 1606

        with (
            torch.no_grad(),
            torch.autocast(device_type="cuda" if "cuda" in str(self.device) else "cpu"),
        ):
            masked_ipd = torch.clamp(masked_ipd, 0, max_ipd_vocab - 1)
            masked_size = torch.clamp(masked_size, 0, max_size_vocab - 1)

            ipd_logits, size_logits = self.bert_model(masked_ipd, masked_size)

            target_ipd_logits = ipd_logits[0, position].clone()
            target_size_logits = size_logits[0, position].clone()

            if hasattr(self.time_vocab, "special_size"):
                target_ipd_logits[: self.time_vocab.special_size] = -1e9
            if hasattr(self.size_vocab, "special_size"):
                # Enforce minimum 40-byte IPv4+TCP header size for valid reconstructed chaff packets
                min_size_tok = max(
                    self.size_vocab.special_size,
                    (
                        self.size_vocab.stoi(40)
                        if hasattr(self.size_vocab, "stoi")
                        else self.size_vocab.special_size
                    ),
                )
                target_size_logits[:min_size_tok] = -1e9

            predicted_ipd_token = int(target_ipd_logits.argmax(dim=-1).item())
            predicted_size_token = int(target_size_logits.argmax(dim=-1).item())

        return predicted_ipd_token, predicted_size_token

    def apply_action(self, action: Union[int, torch.Tensor]) -> None:
        """Applies one NetMasquerade edit: even = insert chaff packet, odd = delay-only."""
        raw_action = int(action.item()) if hasattr(action, "item") else int(action)
        raw_action = max(0, min(raw_action, self.action_dim - 1))

        position = raw_action // 2
        is_insert = (raw_action % 2) == 0

        if not hasattr(self, "forbidden_indices"):
            self.forbidden_indices = torch.zeros(
                self.seq_len, dtype=torch.bool, device=self.device
            )
        if not hasattr(self, "chaff_indices"):
            self.chaff_indices = []

        MASK_TOKEN_ID = self.time_vocab.mask_index

        if is_insert:
            # 1. Insert [MASK] at `position`, shifting all subsequent packets forward by 1
            masked_ipd = (
                self._insert_into_state_tensor(
                    self.state["flow_ipd"], MASK_TOKEN_ID, position
                )
                .reshape(1, -1)
                .to(self.device)
            )
            masked_size = (
                self._insert_into_state_tensor(
                    self.state["flow_size"], MASK_TOKEN_ID, position
                )
                .reshape(1, -1)
                .to(self.device)
            )

            time_token, size_token = self._predict_tokens(
                masked_ipd, masked_size, position
            )

            self.state["flow_ipd"] = self._insert_into_state_tensor(
                self.state["flow_ipd"], time_token, position
            )
            self.state["flow_size"] = self._insert_into_state_tensor(
                self.state["flow_size"], size_token, position
            )

            new_ipd_float = self._decode_single_token(self.time_vocab, time_token)
            new_size_float = self._decode_single_token(self.size_vocab, size_token)

            if "raw_ipd" in self.state and "raw_size" in self.state:
                self.state["raw_ipd"] = self._insert_into_state_tensor(
                    self.state["raw_ipd"], new_ipd_float, position
                )
                self.state["raw_size"] = self._insert_into_state_tensor(
                    self.state["raw_size"], new_size_float, position
                )

            if "real_length" in self.state:
                self.state["real_length"] = torch.clamp(
                    self.state["real_length"] + 1, max=self.seq_len
                )

            # Shift existing chaff indices and forbidden mask right by +1
            self.chaff_indices = [
                idx + 1 if idx >= position else idx for idx in self.chaff_indices
            ]
            bisect.insort(self.chaff_indices, position)
            self.chaff_indices = [
                idx for idx in self.chaff_indices if idx < self.seq_len
            ]

            self.forbidden_indices = self._insert_tensor_1d(
                self.forbidden_indices, True, position
            )
        else:
            # 2. Odd action: delay existing packet at `position` (IPD updated, size untouched)
            masked_ipd = self.state["flow_ipd"].clone().reshape(1, -1).to(self.device)
            masked_size = self.state["flow_size"].clone().reshape(1, -1).to(self.device)
            masked_ipd[0, position] = MASK_TOKEN_ID
            masked_size[0, position] = MASK_TOKEN_ID

            time_token, _size_token = self._predict_tokens(
                masked_ipd, masked_size, position
            )
            new_ipd_float = self._decode_single_token(self.time_vocab, time_token)

            if self.state["flow_ipd"].dim() == 2:
                self.state["flow_ipd"][0, position] = time_token
            else:
                self.state["flow_ipd"][position] = time_token

            if "raw_ipd" in self.state:
                if self.state["raw_ipd"].dim() == 2:
                    self.state["raw_ipd"][0, position] = new_ipd_float
                else:
                    self.state["raw_ipd"][position] = new_ipd_float

            self.forbidden_indices[position] = True

        self.state["forbidden_indices"] = self.forbidden_indices.clone()
        self.state["legal_action_mask"] = self.get_legal_action_mask()
        self.step_num += 1

    def step(self, action: torch.Tensor, target_multiplier: float = 0.5):
        self.apply_action(action)

        raw_ipds, raw_sizes = self._get_raw_traffic_dimensions()

        # Evaluate current state against IDS (expects up to 128 packets -> 256 features)
        eval_res = self.evaluator.predict(raw_ipds, raw_sizes, max_seq_len=128)
        if len(eval_res) == 3:
            _, current_score, pred_class = eval_res
        else:
            _, current_score = eval_res
            pred_class = 1

        is_benign = pred_class == 0

        if not hasattr(self, "prev_score") or self.prev_score is None:
            self.prev_score = current_score

        eval_mode = getattr(self.evaluator, "mode", "score_based")
        step_penalty = getattr(self, "step_penalty", 0.1)
        success_reward = getattr(self, "success_reward", 15.0)
        reward = -step_penalty
        done = False
        timeout = False

        if eval_mode == "hard_labeled":
            if is_benign:
                reward += success_reward
                done = True
                print(
                    f"[+] TARGETED EVASION SUCCESS -> Benign (Class 0) at step {self.step_num}!"
                )
            elif self.step_num >= self.rl_args.trainer.max_stop_step:
                reward -= 2.0
                done = True
                timeout = True
        else:
            total_reduction = self.baseline_score - current_score
            if pred_class == 2:
                reward -= 1.0
            else:
                reward += max(0.0, total_reduction * 5.0)

            if is_benign:
                reward += success_reward
                done = True
                print(
                    f"[+] TARGETED EVASION SUCCESS -> Benign (Class 0) at step {self.step_num}!"
                )
            elif self.step_num >= self.rl_args.trainer.max_stop_step:
                reward -= 2.0
                done = True
                timeout = True

        info = {
            "is_success": done and is_benign,
            "score": current_score,
            "timeout": timeout,
            "pred_class": pred_class,
            "chaff_indices": list(self.chaff_indices),
        }
        return self.state, float(reward), done, info

    def reset(self):
        self.step_num = 0
        self.chaff_indices = []
        if hasattr(self, "baseline_score"):
            self.prev_score = self.baseline_score

        self.forbidden_indices = torch.zeros(
            self.seq_len, dtype=torch.bool, device=self.device
        )
        self.forbidden_indices[0] = True

        if not hasattr(self, "state") or self.state is None:
            self.state = {}
        self.state["forbidden_indices"] = self.forbidden_indices.clone()
        self.state["legal_action_mask"] = self.get_legal_action_mask()

        return self.state

    def _get_raw_traffic_dimensions(self) -> tuple[list[float], list[float]]:
        real_len = (
            int(self.state["real_length"].item())
            if "real_length" in self.state
            else self.seq_len
        )
        if "raw_ipd" in self.state and "raw_size" in self.state:
            raw_ipds = self.state["raw_ipd"].squeeze().tolist()[:real_len]
            raw_sizes = self.state["raw_size"].squeeze().tolist()[:real_len]
        else:
            ipd_tokens = self.state["flow_ipd"].squeeze().tolist()[:real_len]
            size_tokens = self.state["flow_size"].squeeze().tolist()[:real_len]

            raw_ipds = [
                self._decode_single_token(self.time_vocab, int(t)) for t in ipd_tokens
            ]
            raw_sizes = [
                self._decode_single_token(self.size_vocab, int(t)) for t in size_tokens
            ]

        if len(raw_sizes) > 0 and raw_sizes[0] == 0.0:
            raw_ipds = raw_ipds[1:]
            raw_sizes = raw_sizes[1:]

        return raw_ipds, raw_sizes

    def _decode_single_token(self, vocab, tok: int) -> float:
        try:
            if hasattr(vocab, "special_size") and tok < vocab.special_size:
                return 0.0
            if hasattr(vocab, "itos"):
                if callable(vocab.itos):
                    return float(vocab.itos(tok))
                else:
                    return float(vocab.itos[tok])
            else:
                return float(vocab.data[tok - vocab.special_size])
        except Exception:
            return 0.0
