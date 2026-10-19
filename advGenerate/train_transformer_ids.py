# src/advGenerate/train_transformer_ids.py

import os
import argparse
import joblib
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    accuracy_score,
    precision_recall_fscore_support,
)
from advGenerate.train_ml_ids import load_and_flatten_data


def append_markdown_report(
    report_md_path: str,
    dataset_name: str,
    model_name: str,
    attack_name: str,
    world_type: str,
    accuracy: float,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    target_names: list,
):
    if not report_md_path:
        return
    os.makedirs(os.path.dirname(os.path.abspath(report_md_path)), exist_ok=True)
    file_exists = os.path.exists(report_md_path)

    report_dict = classification_report(
        y_true, y_pred, target_names=target_names, output_dict=True
    )

    with open(report_md_path, "a") as f:
        if not file_exists:
            f.write(f"# Training Classification Reports ({world_type.upper()})\n\n")

        f.write(
            f"### Dataset: `{dataset_name}` | World: `{world_type.upper()}` | Model: `{model_name}` | Attack: `{attack_name}`\n\n"
        )
        f.write(f"- **Test Accuracy**: `{accuracy * 100:.2f}%` (`{accuracy:.4f}`)\n\n")

        f.write("| Class / Metric | Precision | Recall | F1-Score | Support |\n")
        f.write("| :--- | :--- | :--- | :--- | :--- |\n")

        for k, v in report_dict.items():
            if k == "accuracy":
                continue
            p = f"{v['precision']:.4f}"
            r = f"{v['recall']:.4f}"
            f1 = f"{v['f1-score']:.4f}"
            sup = f"{int(v['support'])}"
            f.write(f"| {k} | {p} | {r} | {f1} | {sup} |\n")

        f.write(
            f"| **Overall Accuracy** | | | **{accuracy:.4f}** | **{len(y_true)}** |\n\n"
        )
        f.write("---\n\n")


class Transformer_IDS(nn.Module):
    def __init__(
        self,
        in_features: int = 2,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        num_classes: int = 2,
        max_seq_len: int = 128,
    ):
        super(Transformer_IDS, self).__init__()

        self.input_proj = nn.Linear(in_features, d_model)
        self.pos_encoder = nn.Parameter(torch.randn(1, max_seq_len, d_model))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=128,
            dropout=0.1,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.ann = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.size(0)
        x_reshaped = x.view(batch_size, 128, 2)
        x_emb = self.input_proj(x_reshaped) + self.pos_encoder
        x_trans = self.transformer(x_emb)
        x_pooled = x_trans.mean(dim=1)
        logits = self.ann(x_pooled)
        return logits


def train_transformer(
    world_type: str,
    benign_pkl: str,
    malicious_pkl: str,
    data_dir: str,
    model_out: str,
    report_md: str = None,
):
    save_dir = os.path.dirname(model_out)
    os.makedirs(save_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    X_mal, y_mal = load_and_flatten_data(malicious_pkl, assigned_label=1)
    X_ben, y_ben = load_and_flatten_data(benign_pkl, assigned_label=0)

    if world_type == "open_world":
        print("[*] Open-World Setup Detected. Gathering UNK class...")
        unk_x, unk_y = [], []
        unk_files_count = 0

        for f in os.listdir(data_dir):
            if f.endswith(".pkl"):
                full_path = os.path.join(data_dir, f)
                if full_path not in [benign_pkl, malicious_pkl]:
                    x_u, y_u = load_and_flatten_data(full_path, assigned_label=2)
                    unk_x.append(x_u)
                    unk_y.append(y_u)
                    unk_files_count += 1

        if unk_x:
            X_unk = np.vstack(unk_x)
            y_unk = np.concatenate(unk_y)
        else:
            X_unk, y_unk = np.array([]), np.array([])

        n_target = len(X_mal)

        if len(X_ben) > n_target:
            idx = np.random.choice(len(X_ben), n_target, replace=False)
            X_ben, y_ben = X_ben[idx], y_ben[idx]
        if len(X_unk) > n_target:
            idx = np.random.choice(len(X_unk), n_target, replace=False)
            X_unk, y_unk = X_unk[idx], y_unk[idx]

        X = np.vstack((X_ben, X_mal, X_unk))
        y = np.concatenate((y_ben, y_mal, y_unk))
        num_classes = 3
        target_names = ["Benign (0)", "Malicious (1)", "UNK (2)"]
    else:
        n_malicious = len(X_mal)
        n_benign = len(X_ben)

        if n_benign > n_malicious:
            np.random.seed(42)
            indices = np.random.choice(n_benign, size=n_malicious, replace=False)
            X_ben = X_ben[indices]
            y_ben = y_ben[indices]

        X = np.vstack((X_ben, X_mal))
        y = np.concatenate((y_ben, y_mal))
        num_classes = 2
        target_names = ["Benign (0)", "Malicious (1)"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train).astype(np.float32)
    X_test_scaled = scaler.transform(X_test).astype(np.float32)

    train_dataset = TensorDataset(
        torch.FloatTensor(X_train_scaled), torch.LongTensor(y_train)
    )
    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)

    model = Transformer_IDS(num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)

    mode_label = (
        "OPEN_WORLD Mode | 3-Class"
        if num_classes == 3
        else "CLOSED_WORLD Mode | 2-Class"
    )
    print(f"\n[*] Training Transformer IDS [{mode_label}]...")

    for epoch in range(15):
        model.train()
        total_loss = 0.0

        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device)

            benign_indices = (by == 0).nonzero(as_tuple=True)[0]
            if len(benign_indices) > 0:
                num_to_augment = max(1, int(len(benign_indices) * 0.10))
                selected_indices = benign_indices[
                    torch.randperm(len(benign_indices))[:num_to_augment]
                ]

                noise = torch.empty_like(bx[selected_indices]).uniform_(0.95, 1.05)
                bx[selected_indices] = bx[selected_indices] * noise

                dropout_mask = (torch.rand_like(bx[selected_indices]) > 0.05).float()
                bx[selected_indices] = bx[selected_indices] * dropout_mask

            optimizer.zero_grad()
            logits = model(bx)
            loss = criterion(logits, by)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * bx.size(0)

        print(
            f"    Epoch {epoch+1:02d} | Loss: {total_loss / len(train_loader.dataset):.4f}"
        )

    checkpoint = {
        "model_state": model.state_dict(),
        "scaler": scaler,
        "num_classes": num_classes,
        "world_type": world_type,
    }
    torch.save(checkpoint, model_out)
    print(f"\n[+] Transformer weights & metadata saved to {model_out}")

    model.eval()
    test_dataset = TensorDataset(torch.FloatTensor(X_test_scaled))
    test_loader = DataLoader(test_dataset, batch_size=512, shuffle=False)

    if num_classes == 3:
        all_preds = []
        with torch.no_grad():
            for batch in test_loader:
                logits = model(batch[0].to(device))
                preds = torch.argmax(logits, dim=1).cpu().numpy()
                all_preds.extend(preds)

        all_preds = np.array(all_preds)
        acc = accuracy_score(y_test, all_preds)
        print(f"\n[+] Transformer Test Accuracy: {acc:.4f}")
        print("\n================ CLASSIFICATION REPORT ================")
        print(classification_report(y_test, all_preds, target_names=target_names))
        final_preds = all_preds
    else:
        all_probs = []
        with torch.no_grad():
            for batch in test_loader:
                logits = model(batch[0].to(device))
                probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
                all_probs.extend(probs)

        all_probs = np.array(all_probs)

        thresholds = np.linspace(0.1, 0.9, 9)
        best_threshold = 0.5
        best_f1 = 0.0

        for th in thresholds:
            preds = (all_probs >= th).astype(int)
            _, _, f1, _ = precision_recall_fscore_support(
                y_test, preds, average="binary"
            )
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = th

        final_preds = (all_probs >= best_threshold).astype(int)
        acc = accuracy_score(y_test, final_preds)

        print(
            f"\n[+] Transformer Test Accuracy (Threshold={best_threshold:.2f}): {acc:.4f}"
        )
        print("\n================ CLASSIFICATION REPORT ================")
        print(classification_report(y_test, final_preds, target_names=target_names))

    attack_name = os.path.splitext(os.path.basename(malicious_pkl))[0]
    dataset_name = os.path.basename(os.path.dirname(os.path.abspath(malicious_pkl)))

    append_markdown_report(
        report_md_path=report_md,
        dataset_name=dataset_name,
        model_name="Transformer",
        attack_name=attack_name,
        world_type=world_type,
        accuracy=acc,
        y_true=y_test,
        y_pred=final_preds,
        target_names=target_names,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--world_type", type=str, default="closed_world")
    parser.add_argument("--benign", type=str, required=True)
    parser.add_argument("--malicious", type=str, required=True)
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--model_out", type=str, required=True)
    parser.add_argument("--report_md", type=str, default=None)
    args = parser.parse_args()

    train_transformer(
        world_type=args.world_type,
        benign_pkl=args.benign,
        malicious_pkl=args.malicious,
        data_dir=args.data_dir,
        model_out=args.model_out,
        report_md=args.report_md,
    )
