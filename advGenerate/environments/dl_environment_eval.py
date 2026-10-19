# src/advGenerate/environments/dl_environment_eval.py

import os
import numpy as np
import torch
import torch.nn as nn
import sklearn.preprocessing

from typing import Literal, Optional, Union

class DeepLearningIDS(nn.Module):
    def __init__(self, input_dim: int, num_classes: int = 2):
        super(DeepLearningIDS, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes),
        )

    def forward(self, x: torch.Tensor):
        return self.network(x)


class DL_IDS_Evaluator:
    _last_model_path = None
    _last_mode = "score_based"

    def __init__(
        self,
        models_dir: Optional[str] = None,
        mode: Union[Literal["score_based"], Literal["hard_labeled"], None] = None,
        threshold: Optional[float] = None,
    ):
        if models_dir is not None:
            DL_IDS_Evaluator._last_model_path = models_dir
        else:
            models_dir = DL_IDS_Evaluator._last_model_path

        if mode is not None:
            DL_IDS_Evaluator._last_mode = mode.lower()
        else:
            mode = DL_IDS_Evaluator._last_mode

        if models_dir is None:
            models_dir = "models/cic2023/dl"

        self.mode = mode.lower() if mode else "score_based"
        self.threshold = threshold
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if not os.path.exists(models_dir):
            raise FileNotFoundError(
                f"[!] Target classifier weights not found at: {models_dir}"
            )

        checkpoint = torch.load(
            models_dir, map_location=self.device, weights_only=False
        )
        self.scaler = checkpoint["scaler"]
        self.num_classes = checkpoint.get("num_classes", 2)
        self.world_type = checkpoint.get(
            "world_type", "closed_world" if self.num_classes == 2 else "open_world"
        )

        input_dim = getattr(self.scaler, "n_features_in_", self.scaler.scale_.shape[0])
        self.model = DeepLearningIDS(input_dim, num_classes=self.num_classes).to(
            self.device
        )
        self.model.load_state_dict(checkpoint["model_state"])
        self.model.eval()

        print(
            f"[+] Loaded DL-IDS Model from {models_dir} [{self.world_type.upper()} | {self.num_classes}-Class | {self.mode.upper()} Mode]"
        )

    def predict(
        self, raw_ipds: list[float], raw_sizes: list[int], max_seq_len: int = 128
    ) -> tuple[bool, float, int]:
        features = self._extract_features(raw_ipds, raw_sizes, max_seq_len)
        features_scaled = self.scaler.transform(features)
        tensor_features = torch.tensor(features_scaled, dtype=torch.float32).to(
            self.device
        )

        with torch.no_grad():
            logits = self.model(tensor_features)
            probs = torch.softmax(logits, dim=1)
            prob_benign = probs[0, 0].item()

            if self.threshold is not None:
                # Strict Threshold Mode
                if prob_benign >= self.threshold:
                    pred_class = 0
                else:
                    # If it doesn't meet the threshold, it cannot be benign.
                    # Mask out Class 0 and take the argmax of the remaining classes (Malicious vs UNK)
                    probs_masked = probs.clone()
                    probs_masked[0, 0] = -1.0
                    pred_class = int(torch.argmax(probs_masked, dim=1).item())
            else:
                # Default Argmax Mode
                pred_class = int(torch.argmax(probs, dim=1).item())

        # SUCCESS REQUIRING CLASS 0 (BENIGN) ONLY
        # Class 1 (Attack) and Class 2 (UNK / Anomaly) are both counted as evasion failures
        success = pred_class == 0

        if self.mode == "hard_labeled":
            distance = 0.0 if success else 1.0
        else:
            # Distance measures distance to pure Benign (1.0 - prob_benign)
            distance = 1.0 - prob_benign

        return success, distance, pred_class

    def evaluate_flow(
        self, raw_ipds: list[float], raw_sizes: list[int], max_seq_len: int = 128
    ):
        return self.predict(raw_ipds, raw_sizes, max_seq_len)

    def _extract_features(
        self, raw_ipds: list[float], raw_sizes: list[int], max_seq_len: int = 128
    ) -> np.ndarray:
        feature_row = []
        for i in range(max_seq_len):
            ipd = float(raw_ipds[i]) if i < len(raw_ipds) else 0.0
            size = float(raw_sizes[i]) if i < len(raw_sizes) else 0.0
            feature_row.extend([ipd, size])

        return np.array(feature_row, dtype=np.float32).reshape(1, -1)
