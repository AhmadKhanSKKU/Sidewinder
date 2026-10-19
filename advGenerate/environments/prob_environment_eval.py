import torch
import torch.nn as nn
import numpy as np
import pickle
import os


class ProbabilisticClassifier(nn.Module):
    def __init__(self, input_dim: int = 256, latent_dim: int = 64):
        super(ProbabilisticClassifier, self).__init__()

        # Deeper, wider encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
        )
        self.fc_mu = nn.Linear(128, latent_dim)
        self.fc_var = nn.Linear(128, latent_dim)

        # Deeper Classifier Head
        self.classifier = nn.Sequential(
            nn.Linear(latent_dim, 64), nn.ReLU(), nn.Dropout(0.2), nn.Linear(64, 1)
        )

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor):
        # We enforce sampling during inference for robust evasion testing!
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        h = self.encoder(x)
        mu, logvar = self.fc_mu(h), self.fc_var(h)
        z = self.reparameterize(mu, logvar)
        return self.classifier(z), mu, logvar


class Probabilistic_IDS_Evaluator:
    def __init__(
        self, model_path: str = "prob_classifier_ids.pkl", threshold: float = 0.50
    ):
        self.threshold = threshold
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        with open(model_path, "rb") as f:
            save_dict = pickle.load(f)

        self.scaler = save_dict["scaler"]
        self.model = ProbabilisticClassifier(256, 64).to(self.device)
        self.model.load_state_dict(save_dict["model_state"])
        self.model.eval()
        print(f"[+] Probabilistic IDS loaded successfully.")

    def evaluate_flow(
        self, raw_ipds: list[float], raw_sizes: list[float], max_seq_len: int = 128
    ):
        feature_row = list(raw_ipds) + list(raw_sizes)
        input_dim = max_seq_len * 2
        features = np.array(feature_row, dtype=np.float32)
        if len(features) < input_dim:
            features = np.pad(features, (0, input_dim - len(features)), "constant")
        else:
            features = features[:input_dim]

        X_scaled = self.scaler.transform(features.reshape(1, -1))
        X_tensor = torch.FloatTensor(X_scaled).to(self.device)

        with torch.no_grad():
            logits, mu, logvar = self.model(X_tensor)
            prob = torch.sigmoid(logits).item()

            # Calculate how far the input's distribution is from "Normal"
            kld_score = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp()).item()

        is_evaded = (prob < self.threshold) and (kld_score < 5.0)

        # We can return the KLD as the anomaly score for logging
        kld_penalty = max(0.0, kld_score - 2.0) * 0.5 
        combined_score = prob + kld_penalty
        return is_evaded, combined_score

    def predict(
        self, raw_ipds: list[float], raw_sizes: list[float], max_seq_len: int = 128
    ):
        return self.evaluate_flow(raw_ipds, raw_sizes, max_seq_len)
