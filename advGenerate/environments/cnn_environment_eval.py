# src/advGenerate/environments/cnn_environment_eval.py

import os
import numpy as np
import torch
import torch.nn as nn
from typing import Literal, Optional, Union
from advGenerate.train_cnn_ids import CNN_ANN_IDS


class CNN_IDS_Evaluator:
    _last_model_path = None
    _last_mode = "score_based"

    def __init__(
        self,
        models_dir: Optional[str] = None,
        mode: Union[Literal["score_based"], Literal["hard_labeled"], None] = None,
        threshold: Optional[float] = None,
    ):
        if models_dir is not None:
            CNN_IDS_Evaluator._last_model_path = models_dir
        else:
            models_dir = CNN_IDS_Evaluator._last_model_path

        if mode is not None:
            CNN_IDS_Evaluator._last_mode = mode.lower()
        else:
            mode = CNN_IDS_Evaluator._last_mode

        if models_dir is None:
            models_dir = "models/cic2023/cnn"

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

        # Initialize model dynamically based on loaded num_classes
        self.model = CNN_ANN_IDS(in_channels=2, num_classes=self.num_classes).to(
            self.device
        )
        self.model.load_state_dict(checkpoint["model_state"])
        self.model.eval()

        print(
            f"[+] Loaded CNN-IDS Model from {models_dir} [{self.world_type.upper()} | {self.num_classes}-Class | {self.mode.upper()} Mode]"
        )

    def predict(
        self, raw_ipds: list[float], raw_sizes: list[float], max_seq_len: int = 128
    ) -> tuple[bool, float, int]:
        features = self._extract_features(raw_ipds, raw_sizes, max_seq_len)
        features_scaled = self.scaler.transform(features).astype(np.float32)
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
                    # If it doesn't meet the threshold, mask out Benign (Class 0)
                    # and select highest probability between Malicious (1) and UNK (2)
                    probs_masked = probs.clone()
                    probs_masked[0, 0] = -1.0
                    pred_class = int(torch.argmax(probs_masked, dim=1).item())
            else:
                # Argmax Fallback
                pred_class = int(torch.argmax(probs, dim=1).item())

        # Success requires Class 0 (Benign)
        success = pred_class == 0

        if self.mode == "hard_labeled":
            distance = 0.0 if success else 1.0
        else:
            # Distance measures distance to pure Benign (1.0 - prob_benign)
            distance = 1.0 - prob_benign

        return success, distance, pred_class

    def evaluate_flow(
        self, raw_ipds: list[float], raw_sizes: list[float], max_seq_len: int = 128
    ):
        return self.predict(raw_ipds, raw_sizes, max_seq_len)

    def _extract_features(
        self, raw_ipds: list[float], raw_sizes: list[float], max_seq_len: int = 128
    ) -> np.ndarray:
        feature_row = []
        for i in range(max_seq_len):
            ipd = float(raw_ipds[i]) if i < len(raw_ipds) else 0.0
            size = float(raw_sizes[i]) if i < len(raw_sizes) else 0.0
            feature_row.extend([ipd, size])

        return np.array(feature_row, dtype=np.float32).reshape(1, -1)
