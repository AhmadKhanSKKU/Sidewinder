import os
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import argparse
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score


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


class DeepLearningIDS(nn.Module):
    def __init__(self, input_dim: int = 256, num_classes: int = 2):
        super(DeepLearningIDS, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


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


def train_dl(
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
    MIN_FLOW_THRESHOLD = 1500

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

        if n_malicious < MIN_FLOW_THRESHOLD:
            target_count = 3000
            if len(X_ben) > target_count:
                idx = np.random.choice(len(X_ben), size=target_count, replace=False)
                X_ben, y_ben = X_ben[idx], y_ben[idx]

            if len(X_unk) > target_count:
                idx = np.random.choice(len(X_unk), size=target_count, replace=False)
                X_unk, y_unk = X_unk[idx], y_unk[idx]

            mal_idx = np.random.choice(len(X_mal), size=target_count, replace=True)
            X_mal, y_mal = X_mal[mal_idx], y_mal[mal_idx]
        else:
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
        if n_malicious < MIN_FLOW_THRESHOLD:
            target_count = 3000
            if len(X_ben) > target_count:
                indices = np.random.choice(len(X_ben), size=target_count, replace=False)
                X_ben = X_ben[indices]
                y_ben = y_ben[indices]

            mal_indices = np.random.choice(len(X_mal), size=target_count, replace=True)
            X_mal = X_mal[mal_indices]
            y_mal = y_mal[mal_indices]
        else:
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

    input_dim = X_train_scaled.shape[1]
    model = DeepLearningIDS(input_dim=input_dim, num_classes=num_classes).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)

    print(f"\n[*] Training Pure DL (MLP) IDS [{world_type.upper()} Mode]...")
    for epoch in range(15):
        model.train()
        total_loss = 0.0

        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device)

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
    print(f"\n[+] DL weights & scaler saved to '{save_path}'")

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
    print(f"\n[+] DL Test Accuracy: {acc:.4f}")
    print("\n================ CLASSIFICATION REPORT ================")
    print(classification_report(y_test, all_preds, target_names=target_names))
    print("=======================================================")

    attack_name = os.path.splitext(os.path.basename(malicious_pkl))[0]
    dataset_name = os.path.basename(os.path.dirname(os.path.abspath(malicious_pkl)))

    append_markdown_report(
        report_md_path=report_md,
        dataset_name=dataset_name,
        model_name="DL",
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

    train_dl(
        malicious_pkl=args.malicious,
        benign_pkl=args.benign,
        save_path=args.model_out,
        world_type=args.world_type,
        data_dir=args.data_dir,
        report_md=args.report_md,
    )
