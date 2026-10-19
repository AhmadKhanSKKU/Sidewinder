# src/advGenerate/environments/vib_ensemble.py

import torch
import torch.nn as nn
import torch.nn.functional as F


class VIBModel(nn.Module):

    def __init__(
        self,
        input_dim: int = 256,
        hidden_dim: int = 256,
        latent_dim: int = 8,
        num_classes: int = 2,
    ):
        super(VIBModel, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

        self.classifier = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, num_classes),
        )

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        else:
            return mu

    def forward(self, x: torch.Tensor):
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        z = self.reparameterize(mu, logvar)
        logits = self.classifier(z)
        return logits, mu, logvar


class VIBEnsemble(nn.Module):

    def __init__(
        self,
        num_models: int = 5,
        input_dim: int = 256,
        hidden_dim: int = 256,
        latent_dim: int = 8,
        num_classes: int = 2,
    ):
        super(VIBEnsemble, self).__init__()
        self.models = nn.ModuleList(
            [
                VIBModel(
                    input_dim=input_dim,
                    hidden_dim=hidden_dim,
                    latent_dim=latent_dim,
                    num_classes=num_classes,
                )
                for _ in range(num_models)
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Average the output logits across all models in the ensemble
        all_logits = [model(x)[0] for model in self.models]
        return torch.stack(all_logits, dim=0).mean(dim=0)
