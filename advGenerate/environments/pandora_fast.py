# NetMasquerade-main/src/advGenerate/environments/pandora_fast.py

import sys
import torch
import numpy as np
import pandas as pd
from sklearn.preprocessing import QuantileTransformer

# IMPORTANT: Add the path to your PANDORA code directory
sys.path.append("/media/ahmad/Work/Experiments/Pandora/")
from pandora import SotaIDS_Probabilistic


class Pandora_Fast_Evaluate:
    def __init__(
        self,
        model_checkpoint_path: str,
        original_csv_path: str,
        train_csv_path: str,
        label_col: str = "benign_final",
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("[*] Initializing Fast In-Memory PANDORA Evaluator...")

        # 1. Load PANDORA Model
        checkpoint = torch.load(model_checkpoint_path, map_location=self.device)
        self.feature_groups = checkpoint["feature_groups"]
        self.all_features = checkpoint["all_features"]
        self.label_map = checkpoint["label_map"]
        hp = checkpoint["hyperparameters"]

        self.model = SotaIDS_Probabilistic(
            num_temporal_features=len(self.feature_groups["temporal"]),
            num_volumetric_features=len(self.feature_groups["volumetric"]),
            d_model=hp["d_model"],
            num_blocks=hp["num_blocks"],
            num_experts=hp["num_experts"],
            dropout_rate=hp["dropout_rate"],
            n_heads=hp["n_heads"],
        ).to(self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()

        self.scaler = QuantileTransformer(
            n_quantiles=checkpoint.get("quantile_n", 100),
            output_distribution="normal",
            random_state=42,
        )

        # 2. Pre-load Support Set
        self.sup_temp, self.sup_vol, self.sup_labels = self._prepare_support_set(
            train_csv_path, label_col
        )


        # Check if label_map keys are strings or ints and find the benign one
        for k, v in self.label_map.items():
            # Check keys
            if isinstance(k, str) and ("benign" in k.lower() or "normal" in k.lower()):
                self.benign_string = k
                self.benign_idx = v
                break
            # Check values (if keys are integers)
            if isinstance(v, str) and ("benign" in v.lower() or "normal" in v.lower()):
                self.benign_string = v
                self.benign_idx = k
                break

        print(
            f"[*] Detected Benign Label: '{self.benign_string}' at Index: {self.benign_idx}"
        )

        self.benign_idx = 10
        self.benign_string = "benign_final"
        print(
            f"[*] FINAL FORCED Benign Label: '{self.benign_string}' at Index: {self.benign_idx}"
        )

        # 3. Pre-load the Base Flow Stats (The starting point of the attack)
        df = pd.read_csv(original_csv_path)
        df.columns = df.columns.str.lower()

        search_col = label_col.lower()

        # Check if the label column actually exists in THIS specific file
        if search_col in df.columns:
            # Filter for ONLY malicious rows using the string we detected
            malware_df = df[
                df[search_col].astype(str).str.lower() != self.benign_string.lower()
            ]
            if not malware_df.empty:
                self.base_df = malware_df.head(1).copy()
                print(
                    f"[*] Found label column; successfully loaded malicious template."
                )
            else:
                self.base_df = df.head(1).copy()
                print(
                    "[!] Label column found but no malicious rows detected. Using first row."
                )
        else:
            # If no label column exists, just assume the file is what we want
            print(
                f"[*] Note: '{search_col}' not found in template CSV. Using first row as baseline."
            )
            self.base_df = df.head(1).copy()

    def _prepare_support_set(self, train_csv_path: str, label_col: str):
        df = pd.read_csv(train_csv_path)
        # support_df = df.groupby(label_col).head(10).copy()
        support_df = (
            df.groupby(label_col).sample(n=10, replace=True, random_state=42).copy()
        )
        support_df.columns = support_df.columns.str.lower()

        label_col_lower = label_col.lower()
        if support_df[label_col_lower].dtype == "O":
            support_labels = (
                support_df[label_col_lower].map(self.label_map).fillna(0).values
            )
        else:
            support_labels = support_df[label_col_lower].values

        # support_labels = torch.tensor(support_labels, dtype=torch.long).to(self.device)

        # Ensure support_labels are integers, not strings!
        if len(support_labels) > 0 and isinstance(support_labels[0], str):
            # Find unique labels and ensure 'Benign'/'Normal' gets mapped to 0
            unique_labels = sorted(list(set(support_labels)))
            if "Benign" in unique_labels:
                unique_labels.remove("Benign")
                unique_labels.insert(0, "Benign")
            elif "Normal" in unique_labels:
                unique_labels.remove("Normal")
                unique_labels.insert(0, "Normal")

            label_map = {label: idx for idx, label in enumerate(unique_labels)}
            print(f"[*] Auto-mapped string labels to integers: {label_map}")
            support_labels = [label_map[label] for label in support_labels]

        # Now safely convert to PyTorch tensor
        support_labels = torch.tensor(support_labels, dtype=torch.long).to(self.device)

        available_features = [f for f in self.all_features if f in support_df.columns]
        support_df[available_features] = (
            support_df[available_features]
            .apply(pd.to_numeric, errors="coerce")
            .fillna(0)
        )

        try:
            support_df[available_features] = self.scaler.transform(
                support_df[available_features]
            )
        except:
            support_df[available_features] = self.scaler.fit_transform(
                support_df[available_features]
            )

        sup_temp = torch.tensor(
            support_df[self.feature_groups["temporal"]].values, dtype=torch.float32
        ).to(self.device)
        sup_vol = torch.tensor(
            support_df[self.feature_groups["volumetric"]].values, dtype=torch.float32
        ).to(self.device)
        return sup_temp, sup_vol, support_labels

    def evaluate(self, current_ipd, current_size):
        """
        Calculates feature updates mathematically in-memory to bypass CICFlowMeter CLI.
        """
        # Create a fresh copy of the base attack stats
        df_query = self.base_df.copy()

        # Mathematically update Temporal stats based on Agent's IPD changes
        if len(current_ipd) > 0:
            ipds_micro = np.array(current_ipd) * 1e6  # CICFlowMeter uses microseconds
            ipds_micro = np.maximum(1.0, ipds_micro)
            df_query["flow duration"] = np.sum(ipds_micro)
            df_query["flow iat mean"] = np.mean(ipds_micro)
            df_query["flow iat std"] = np.std(ipds_micro)
            df_query["flow iat max"] = np.max(ipds_micro)
            df_query["flow iat min"] = np.min(ipds_micro)

        # Mathematically update Volumetric stats based on Agent's Padding changes
        if len(current_size) > 0:
            sizes = np.array(current_size)
            df_query["totlen fwd pkts"] = np.sum(sizes)
            df_query["fwd pkt len max"] = np.max(sizes)
            df_query["fwd pkt len min"] = np.min(sizes)
            df_query["fwd pkt len mean"] = np.mean(sizes)
            df_query["fwd pkt len std"] = np.std(sizes)
            df_query["pkt len var"] = np.var(sizes)
            df_query["pkt size avg"] = np.mean(sizes)

        # Preprocess features
        available_features = [f for f in self.all_features if f in df_query.columns]
        df_query = (
            df_query[available_features].apply(pd.to_numeric, errors="coerce").fillna(0)
        )

        try:
            scaled = self.scaler.transform(df_query[available_features])
        except:
            scaled = self.scaler.fit_transform(df_query[available_features])
        df_query[available_features] = scaled

        # Force a "batch" dimension of 1 using unsqueeze(0)
        # Removed .unsqueeze(0) because df_query.values is already a 2D batch (1 row)
        q_temp = torch.tensor(
            df_query[self.feature_groups["temporal"]].values, dtype=torch.float32
        ).to(self.device)

        q_vol = torch.tensor(
            df_query[self.feature_groups["volumetric"]].values, dtype=torch.float32
        ).to(self.device)

        # Batch size is exactly 1
        dummy_query_labels = torch.zeros(1, dtype=torch.long).to(self.device)

        # Query PANDORA
        with torch.no_grad():
            logits, _ = self.model(
                support_temporal=self.sup_temp,
                support_volumetric=self.sup_vol,
                support_labels=self.sup_labels,
                query_temporal=q_temp,
                query_volumetric=q_vol,
                query_labels_for_loss=dummy_query_labels,
            )
            # Use .item() to safely extract the integer regardless of dimensions
            predicted_class_idx = torch.argmax(logits).item()

            _, logvar = self.model.forward_encoder(q_temp, q_vol)
            # Use .mean().item() to safely get the scalar variance
            variance = torch.exp(logvar).mean().item()

        # Return Success/Fail based on PANDORA's Hard Label
        if predicted_class_idx == self.benign_idx:
            # Check if variance spikes! This is your proof of concept.
            print(f"[!] AGENT FOOLED CLASSIFICATION | PANDORA Variance: {variance:.4f}")
            return 0  # 0 means Agent Successfully Evaded
        else:
            return 1  # 1 means PANDORA Caught it
