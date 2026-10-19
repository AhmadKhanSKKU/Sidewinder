# src/advGenerate/environments/transformer_environment_eval.py

import torch
import torch.nn.functional as F
import numpy as np
import os
import joblib
from advGenerate.train_transformer_ids import Transformer_IDS


class Transformer_IDS_Evaluator:

    def __init__(
        self,
        models_dir: str,
        threshold: float = 0.5,
        mode: str = "score_based",
        num_classes: int = None,
    ):
        self.models_dir = models_dir
        self.threshold = threshold
        self.mode = mode
        self.device = (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )

        # 1. Determine num_classes (Auto-detect open_world from path if not provided)
        if num_classes is None:
            num_classes = 3 if "open_world" in models_dir else 2
        self.num_classes = num_classes

        # 2. Initialize model with correct num_classes
        self.model = Transformer_IDS(num_classes=self.num_classes)

        # 3. Load weights
        checkpoint = torch.load(
            models_dir, map_location=self.device, weights_only=False
        )
        if isinstance(checkpoint, dict) and "model_state" in checkpoint:
            self.model.load_state_dict(checkpoint["model_state"])
        else:
            self.model.load_state_dict(checkpoint)

        self.model.to(self.device)
        self.model.eval()

        # 4. Load scaler from checkpoint (with fallback for legacy external scalers)
        if isinstance(checkpoint, dict) and "scaler" in checkpoint:
            self.scaler = checkpoint["scaler"]
        else:
            scaler_path = models_dir.replace("_ids.pth", "_scaler.pkl")
            self.scaler = joblib.load(scaler_path)

    def predict(
        self, raw_ipd: list[float], raw_size: list[float], max_seq_len: int = 128
    ):
        # Format features
        feature_row = []
        for i in range(max_seq_len):
            if i < len(raw_ipd) and i < len(raw_size):
                feature_row.extend([float(raw_ipd[i]), float(raw_size[i])])
            else:
                feature_row.extend([0.0, 0.0])

        # Scale and Tensorize
        features_scaled = self.scaler.transform([feature_row]).astype(np.float32)
        x_tensor = torch.FloatTensor(features_scaled).to(self.device)

        # Predict using Softmax for probabilities
        with torch.no_grad():
            logits = self.model(x_tensor)
            probs = F.softmax(logits, dim=1)

            # Extract benign probability to match CNN logic
            prob_benign = probs[0, 0].item()

            if self.threshold is not None:
                # Strict Threshold Mode
                if prob_benign >= self.threshold:
                    pred_class = 0
                else:
                    # If it doesn't meet the threshold, mask out Benign (Class 0)
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
