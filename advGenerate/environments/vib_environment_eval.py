# src/advGenerate/environments/vib_environment_eval.py

import torch
import torch.nn.functional as F
import numpy as np
import os
import joblib
from advGenerate.environments.vib_ensemble import VIBEnsemble


class VIB_IDS_Evaluator:

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
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 1. Determine num_classes (Auto-detect open_world from path if not provided)
        if num_classes is None:
            num_classes = 3 if "open_world" in models_dir else 2
        self.num_classes = num_classes

        # 2. Resolve model path and load checkpoint
        model_path = (
            models_dir
            if models_dir.endswith(".pth")
            else os.path.join(models_dir, "vib_ensemble.pth")
        )
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"[!] VIB Ensemble weights not found at: {model_path}"
            )

        checkpoint = torch.load(
            model_path, map_location=self.device, weights_only=False
        )

        # 3. Load Scaler (Extract from checkpoint dictionary or fall back to .pkl)
        if isinstance(checkpoint, dict) and "scaler" in checkpoint:
            self.scaler = checkpoint["scaler"]
        else:
            scaler_path = models_dir.replace("_vib.pth", "_scaler.pkl")
            if not os.path.exists(scaler_path) and os.path.isdir(models_dir):
                scaler_path = os.path.join(models_dir, "vib_scaler.pkl")
            if not os.path.exists(scaler_path):
                raise FileNotFoundError(f"[!] VIB Scaler not found at: {scaler_path}")
            self.scaler = joblib.load(scaler_path)

        # 4. Initialize ensemble & load model weights
        self.ensemble = VIBEnsemble(
            num_models=5,
            input_dim=256,
            hidden_dim=256,
            latent_dim=8,
            num_classes=self.num_classes,
        )

        if isinstance(checkpoint, dict) and "model_state" in checkpoint:
            self.ensemble.load_state_dict(checkpoint["model_state"])
        else:
            self.ensemble.load_state_dict(checkpoint)

        self.ensemble.to(self.device)
        self.ensemble.eval()

    def predict(
        self, raw_ipd: list[float], raw_size: list[float], max_seq_len: int = 128
    ):
        features = self._extract_features(raw_ipd, raw_size, max_seq_len)
        features_scaled = self.scaler.transform(features).astype(np.float32)
        x_tensor = torch.FloatTensor(features_scaled).to(self.device)

        with torch.no_grad():
            mean_logits = self.ensemble(x_tensor)
            probs = F.softmax(mean_logits, dim=1)

            prob_benign = probs[0, 0].item()

            if self.threshold is not None:
                if prob_benign >= self.threshold:
                    pred_class = 0
                else:
                    probs_masked = probs.clone()
                    probs_masked[0, 0] = -1.0
                    pred_class = int(torch.argmax(probs_masked, dim=1).item())
            else:
                pred_class = int(torch.argmax(probs, dim=1).item())

        success = pred_class == 0

        if self.mode == "hard_labeled":
            distance = 0.0 if success else 1.0
        else:
            distance = 1.0 - prob_benign

        return success, distance, pred_class

    def evaluate_flow(
        self, raw_ipd: list[float], raw_size: list[float], max_seq_len: int = 128
    ):
        return self.predict(raw_ipd, raw_size, max_seq_len)

    def _extract_features(
        self, raw_ipd: list[float], raw_size: list[float], max_seq_len: int = 128
    ) -> np.ndarray:
        feature_row = []
        for i in range(max_seq_len):
            if i < len(raw_ipd) and i < len(raw_size):
                ipd = round(float(raw_ipd[i]), 4)
                size = round(float(raw_size[i]), 4)
            else:
                ipd, size = 0.0, 0.0
            feature_row.extend([ipd, size])

        return np.array(feature_row, dtype=np.float32).reshape(1, -1)
