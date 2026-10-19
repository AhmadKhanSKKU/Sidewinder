# src/advGenerate/train_cnn_ids.py

import os
import argparse
import pickle
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


# ==========================================
# REPORT HELPER
# ==========================================
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


# ==========================================
# PURE CNN + ANN ARCHITECTURE
# ==========================================
class CNN_ANN_IDS(nn.Module):
    def __init__(self, in_channels: int = 2, num_classes: int = 2):
        super(CNN_ANN_IDS, self).__init__()

        self.cnn = nn.Sequential(
            nn.Conv1d(in_channels, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveMaxPool1d(1),
        )

        self.ann = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.size(0)
        x_reshaped = x.view(batch_size, 128, 2).permute(0, 2, 1)
        cnn_features = self.cnn(x_reshaped).view(batch_size, -1)
        logits = self.ann(cnn_features)
        return logits


# ==========================================
# DATA LOADING HELPER
# ==========================================
def load_and_flatten_data(pkl_path: str, assigned_label: int, max_seq_len: int = 128):
    if not os.path.exists(pkl_path):
        return np.empty((0, max_seq_len * 2), dtype=np.float32), np.empty(
            (0,), dtype=np.int64
        )

    with open(pkl_path, "rb") as f:
        data = pickle.load(f)

    X, y = [], []
    for item in data:
        if isinstance(item, tuple) and len(item) == 2:
            ipds, sizes = item
        else:
            ipds = item.get("ipds", [])
            sizes = item.get("sizes", [])

        padded_features = []
        for i in range(max_seq_len):
            if i < len(ipds) and i < len(sizes):
                padded_features.extend([float(ipds[i]), float(sizes[i])])
            else:
                padded_features.extend([0.0, 0.0])
        X.append(padded_features)
        y.append(assigned_label)
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int64)


# ==========================================
# TRAINING ENGINE
# ==========================================
def train_cnn(
    malicious_pkl: str,
    benign_pkl: str,
    save_path: str,
    world_type: str = "closed_world",
    data_dir: str = None,
    report_md: str = None,
):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    X_ben, y_ben = load_and_flatten_data(benign_pkl, assigned_label=0)
    X_mal, y_mal = load_and_flatten_data(malicious_pkl, assigned_label=1)

    np.random.seed(42)
    n_malicious = len(X_mal)

    if world_type == "open_world":
        num_classes = 3
        if data_dir is None:
            data_dir = os.path.dirname(malicious_pkl)

        all_files = [
            os.path.join(data_dir, f)
            for f in os.listdir(data_dir)
            if f.endswith(".pkl")
        ]
        ben_abs = os.path.abspath(benign_pkl)
        mal_abs = os.path.abspath(malicious_pkl)
        unk_files = [
            f for f in all_files if os.path.abspath(f) not in (ben_abs, mal_abs)
        ]

        print(
            f"[*] Open-World Setup Detected. Found {len(unk_files)} dataset files for UNK class."
        )

        X_unk_list, y_unk_list = [], []
        for unk_file in unk_files:
            x_u, y_u = load_and_flatten_data(unk_file, assigned_label=2)
            if len(x_u) > 0:
                X_unk_list.append(x_u)
                y_unk_list.append(y_u)

        if not X_unk_list:
            raise RuntimeError(
                f"[!] No valid UNK dataset files found in directory: {data_dir}"
            )

        X_unk = np.vstack(X_unk_list)
        y_unk = np.concatenate(y_unk_list)

        if len(X_ben) > n_malicious:
            idx = np.random.choice(len(X_ben), size=n_malicious, replace=False)
            X_ben, y_ben = X_ben[idx], y_ben[idx]

        if len(X_unk) > n_malicious:
            idx = np.random.choice(len(X_unk), size=n_malicious, replace=False)
            X_unk, y_unk = X_unk[idx], y_unk[idx]

        X = np.vstack((X_ben, X_mal, X_unk))
        y = np.concatenate((y_ben, y_mal, y_unk))
        target_names = ["Benign (0)", "Malicious (1)", "UNK (2)"]
    else:
        num_classes = 2
        if len(X_ben) > n_malicious:
            indices = np.random.choice(len(X_ben), size=n_malicious, replace=False)
            X_ben = X_ben[indices]
            y_ben = y_ben[indices]

        X = np.vstack((X_ben, X_mal))
        y = np.concatenate((y_ben, y_mal))
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

    model = CNN_ANN_IDS(in_channels=2, num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)

    print(
        f"\n[*] Training Pure CNN+ANN IDS [{world_type.upper()} Mode | {num_classes}-Class]..."
    )
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
    torch.save(checkpoint, save_path)
    print(f"\n[+] CNN+ANN weights & metadata saved to {save_path}")

    model.eval()
    test_dataset = TensorDataset(torch.FloatTensor(X_test_scaled))
    test_loader = DataLoader(test_dataset, batch_size=512, shuffle=False)

    all_preds = []
    with torch.no_grad():
        for batch in test_loader:
            logits = model(batch[0].to(device))
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            all_preds.extend(preds)

    all_preds = np.array(all_preds)
    acc = accuracy_score(y_test, all_preds)

    print(f"\n[+] CNN+ANN Test Accuracy: {acc:.4f}")
    print("\n================ CLASSIFICATION REPORT ================")
    print(classification_report(y_test, all_preds, target_names=target_names))
    print("=======================================================")

    attack_name = os.path.splitext(os.path.basename(malicious_pkl))[0]
    dataset_name = os.path.basename(os.path.dirname(os.path.abspath(malicious_pkl)))

    append_markdown_report(
        report_md_path=report_md,
        dataset_name=dataset_name,
        model_name="CNN",
        attack_name=attack_name,
        world_type=world_type,
        accuracy=acc,
        y_true=y_test,
        y_pred=all_preds,
        target_names=target_names,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--world_type",
        type=str,
        default="closed_world",
        choices=["closed_world", "open_world"],
    )
    parser.add_argument("--benign", type=str, required=True)
    parser.add_argument("--malicious", type=str, required=True)
    parser.add_argument(
        "--data_dir",
        type=str,
        default=None,
        help="Directory containing all dataset pkls",
    )
    parser.add_argument("--model_out", type=str, required=True)
    parser.add_argument("--report_md", type=str, default=None)
    args = parser.parse_args()

    train_cnn(
        malicious_pkl=args.malicious,
        benign_pkl=args.benign,
        save_path=args.model_out,
        world_type=args.world_type,
        data_dir=args.data_dir,
        report_md=args.report_md,
    )
