# src/advGenerate/environments/ml_environment_eval.py

import joblib
import numpy as np
import os

class ClassicalML_IDS_Evaluator:

    def __init__(
        self,
        model_type: str = "rf",
        models_dir: str = "models",
        threshold: float = 0.5,
        mode: str = "score_based",
    ):
        self.threshold = threshold

        # Load the feature scaler used during model compilation
        self.scaler = joblib.load(f"{models_dir}/ml_scaler.pkl")

        # Load selected detection engine
        if model_type == "rf":
            print("[+] Initializing Classical ML IDS Evaluator with Random Forest backend.")
            self.model = joblib.load(f"{models_dir}/random_forest_ids.pkl")
        elif model_type == "svm":
            print("[+] Initializing Classical ML IDS Evaluator with Support Vector Machine backend.")
            self.model = joblib.load(f"{models_dir}/svm_ids.pkl")
        elif model_type == "xgb":
            print("[+] Initializing Classical ML IDS Evaluator with XGBoost backend.")
            self.model = joblib.load(f"{models_dir}/xgboost_ids.pkl")
        else:
            raise ValueError(f"Unknown classic ML model type: {model_type}")

        self.model = joblib.load(model_path)
        print(
            f"[+] Fast ML-IDS Evaluator initialized utilizing [{self.model_type.upper()}] backend. Mode: {self.mode}"
        )

    def predict(
        self, raw_ipd: list[float], raw_size: list[float], max_seq_len: int = 128
    ):
        # 1. Extract feature matrix (SINGULAR NAMING ENFORCED)
        features = self._extract_features(raw_ipd, raw_size, max_seq_len)

        # 2. APPLY SCALER to Float32 locked array
        features_scaled = self.scaler.transform(features).astype(np.float32)

        # 3. Compute probability
        prob_array = self.model.predict_proba(features_scaled)
        if prob_array.ndim > 1 and prob_array.shape[1] > 1:
            prob = float(prob_array[0][1])
        else:
            prob = float(prob_array[0])

        success = prob < self.threshold

        if self.mode == "hard_labeled":
            distance = 0.0 if success else 1.0
        else:
            distance = prob

        return success, distance

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
                # ALIGNMENT FIX: STRICT 4-DECIMAL FLOAT STANDARD
                ipd = round(float(raw_ipd[i]), 4)
                size = round(float(raw_size[i]), 4)
            else:
                ipd, size = 0.0, 0.0

            feature_row.extend([ipd, size])

        return np.array(feature_row, dtype=np.float32).reshape(1, -1)
