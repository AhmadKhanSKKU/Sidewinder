# my_attacker/src/advGenerate/train_prob_classifier.py

import torch
import torch.nn as nn
import pickle
import numpy as np
import os
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# 1. CONFIGURATION
MALICIOUS_PATH = "malicious.pkl"
BENIGN_PATH = "benign.pkl"
MODEL_SAVE_PATH = "prob_classifier_ids.pkl"

BATCH_SIZE = 512
INPUT_DIM = 256
LATENT_DIM = 64
EPOCHS = 15
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# 2. PROBABILISTIC CLASSIFIER ARCHITECTURE (VIB)
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

    def reparameterize(
        self, mu: torch.Tensor, logvar: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """This is where the magic robustness happens!"""
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        else:
            # During inference, we can just use the mean, or keep sampling to
            # create an ensemble effect. We'll use the mean for stable testing.
            return mu

    def forward(self, x: torch.Tensor):
        h = self.encoder(x)
        mu, logvar = self.fc_mu(h), self.fc_var(h)
        z = self.reparameterize(mu, logvar)
        logits = self.classifier(z)

        # We return mu and logvar to calculate the KL divergence penalty
        return logits, mu, logvar


# 3. DATA PIPELINE
def extract_features(filepath, label_val):
    print(f"[*] Loading data from {filepath}...")
    with open(filepath, "rb") as f:
        data = pickle.load(f)

    features_list, labels_list = [], []
    for item in data:
        if isinstance(item, tuple):
            ipds, sizes = item
        else:
            ipds = item.get("raw_ipd", item.get("ipds", []))
            sizes = item.get("raw_size", item.get("sizes", []))

        feats = np.array(list(ipds) + list(sizes), dtype=np.float32)
        if len(feats) < INPUT_DIM:
            feats = np.pad(feats, (0, INPUT_DIM - len(feats)), "constant")
        else:
            feats = feats[:INPUT_DIM]

        features_list.append(feats)
        labels_list.append(label_val)
    return features_list, labels_list


malicious_x, malicious_y = extract_features(MALICIOUS_PATH, label_val=1)
benign_x, benign_y = extract_features(BENIGN_PATH, label_val=0)

X_all = np.array(malicious_x + benign_x)
y_all = np.array(malicious_y + benign_y)

X_train, X_test, y_train, y_test = train_test_split(
    X_all, y_all, test_size=0.20, random_state=42, stratify=y_all
)

print("[*] Scaling features...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

train_loader = DataLoader(
    TensorDataset(torch.FloatTensor(X_train_scaled), torch.FloatTensor(y_train)),
    batch_size=BATCH_SIZE,
    shuffle=True,
)
test_loader = DataLoader(
    TensorDataset(torch.FloatTensor(X_test_scaled), torch.FloatTensor(y_test)),
    batch_size=BATCH_SIZE,
    shuffle=False,
)

# 4. TRAINING THE PROBABILISTIC CLASSIFIER
model = ProbabilisticClassifier(INPUT_DIM, LATENT_DIM).to(DEVICE)
optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)

# BCE loss for classification
bce_criterion = nn.BCEWithLogitsLoss()

print(f"\n[*] Starting Supervised Probabilistic Training on {DEVICE}...")
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0

    for batch_x, batch_y in train_loader:
        batch_x, batch_y = batch_x.to(DEVICE), batch_y.to(DEVICE).unsqueeze(1)

        optimizer.zero_grad()
        logits, mu, logvar = model(batch_x)

        # 1. Classification Loss (Is it an attack or not?)
        cls_loss = bce_criterion(logits, batch_y)

        # 2. Information Bottleneck Penalty (Forces it to create a smooth distribution)
        kld_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

        # Combine them (0.01 is a common weight for the KL term in VIB)
        loss = cls_loss + (0.01 * kld_loss)

        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    print(
        f"    Epoch {epoch + 1:02d}/{EPOCHS} | Train Loss: {total_loss/len(train_loader):.4f}"
    )

# 5. VALIDATION PHASE
print("\n[*] Starting Validation on Unseen Test Data...")
model.eval()

all_preds = []
all_true = []

with torch.no_grad():
    for batch_x, batch_y in test_loader:
        batch_x = batch_x.to(DEVICE)

        logits, _, _ = model(batch_x)
        probs = torch.sigmoid(logits)
        preds = (probs > 0.50).int().cpu().numpy().flatten()

        all_preds.extend(preds)
        all_true.extend(batch_y.numpy())

print("\n" + "=" * 50)
print("     PROBABILISTIC CLASSIFICATION REPORT")
print("=" * 50)
print(f"[+] Accuracy: {accuracy_score(all_true, all_preds):.4f}")
print("-" * 50)
print(
    classification_report(
        all_true, all_preds, digits=4, target_names=["Benign (0)", "Malicious (1)"]
    )
)

# SAVE
save_dict = {"model_state": model.state_dict(), "scaler": scaler}
with open(MODEL_SAVE_PATH, "wb") as f:
    pickle.dump(save_dict, f)
print(f"\n[+] Probabilistic Model saved to '{MODEL_SAVE_PATH}'")
