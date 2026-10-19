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
from sklearn.metrics import classification_report, accuracy_score
from typing import Tuple, Optional

try:
    from mamba_ssm import Mamba

    HAS_MAMBA_SSM = True
except ImportError:
    HAS_MAMBA_SSM = False


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


def load_pandora_file(
    file_path: str, label: int, max_seq_len: int = 128
) -> Tuple[np.ndarray, np.ndarray]:
    if not os.path.exists(file_path):
        return np.empty((0, max_seq_len * 2), dtype=np.float32), np.empty(
            (0,), dtype=np.int64
        )

    X, y = [], []
    with open(file_path, "rb") as f:
        flows = pickle.load(f)
        for flow in flows:
            if isinstance(flow, tuple) and len(flow) == 2:
                seq_a = list(flow[0])
                seq_b = list(flow[1])
            elif isinstance(flow, dict):
                seq_a = list(flow.get("ipds", []))
                seq_b = list(flow.get("sizes", []))
            else:
                continue

            if len(seq_a) > 0 and len(seq_b) > 0:
                if np.mean(seq_a) > np.mean(seq_b):
                    sizes, ipds = seq_a, seq_b
                else:
                    ipds, sizes = seq_a, seq_b
            else:
                ipds, sizes = seq_a, seq_b

            feature_row = []
            for i in range(max_seq_len):
                ipd_val = float(ipds[i]) if i < len(ipds) else 0.0
                size_val = float(sizes[i]) if i < len(sizes) else 0.0
                feature_row.extend([ipd_val, size_val])

            X.append(feature_row)
            y.append(label)

    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int64)


class Expert(nn.Module):
    def __init__(self, d_model: int, dropout_rate: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, 2 * d_model),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(2 * d_model, d_model),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MoE(nn.Module):
    def __init__(self, d_model: int, num_experts: int = 4, dropout_rate: float = 0.1):
        super().__init__()
        self.experts = nn.ModuleList(
            [Expert(d_model, dropout_rate) for _ in range(num_experts)]
        )
        self.gate = nn.Linear(d_model, num_experts)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scores = torch.softmax(self.gate(x), dim=-1)
        outputs = torch.stack([e(x) for e in self.experts], dim=-1)
        return torch.einsum("bse,bsde->bsd", scores, outputs)


class MambaBlock(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        if HAS_MAMBA_SSM:
            self.mamba = Mamba(d_model=d_model, d_state=16, d_conv=4, expand=2)
        else:
            self.conv1d = nn.Conv1d(
                in_channels=d_model,
                out_channels=d_model,
                kernel_size=4,
                padding=3,
                groups=d_model,
            )
            self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if HAS_MAMBA_SSM:
            return self.mamba(x)
        else:
            x_t = x.transpose(1, 2)
            x_conv = self.conv1d(x_t)[:, :, : x.shape[1]]
            return self.out_proj(x_conv.transpose(1, 2))


class MambaMoEBlock(nn.Module):
    def __init__(self, d_model: int, num_experts: int = 4, dropout_rate: float = 0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.mamba = MambaBlock(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.moe = MoE(d_model, num_experts, dropout_rate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.mamba(self.ln1(x))
        x = x + self.moe(self.ln2(x))
        return x


class CrossAttentionFusion(nn.Module):
    def __init__(self, d_model: int, n_heads: int = 4):
        super().__init__()
        self.cross_attn_A = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.cross_attn_B = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(d_model * 2, d_model * 2),
            nn.GELU(),
            nn.Linear(d_model * 2, d_model * 2),
        )
        self.ln = nn.LayerNorm(d_model * 2)

    def forward(self, z_A: torch.Tensor, z_B: torch.Tensor) -> torch.Tensor:
        z_A_seq, z_B_seq = z_A.unsqueeze(1), z_B.unsqueeze(1)
        z_A_prime, _ = self.cross_attn_A(query=z_A_seq, key=z_B_seq, value=z_B_seq)
        z_B_prime, _ = self.cross_attn_B(query=z_B_seq, key=z_A_seq, value=z_A_seq)
        fused = torch.cat([z_A_prime.squeeze(1), z_B_prime.squeeze(1)], dim=1)
        return self.ln(fused + self.ffn(fused))


class ProbabilisticSequenceEncoder(nn.Module):
    def __init__(
        self,
        in_channels: int,
        d_model: int,
        num_blocks: int,
        num_experts: int,
        dropout_rate: float = 0.1,
    ):
        super().__init__()
        self.embedding = nn.Linear(in_channels, d_model)
        self.encoder_blocks = nn.Sequential(
            *[
                MambaMoEBlock(d_model, num_experts, dropout_rate)
                for _ in range(num_blocks)
            ]
        )
        self.dropout = nn.Dropout(dropout_rate)
        self.fc_mu = nn.Linear(d_model, d_model)
        self.fc_logvar = nn.Linear(d_model, d_model)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x_embedded = self.embedding(x)
        x_encoded = self.encoder_blocks(x_embedded)
        x_encoded = x_encoded.mean(dim=1)
        x_encoded = self.dropout(x_encoded)

        mu = self.fc_mu(x_encoded)
        logvar = self.fc_logvar(x_encoded)
        return mu, logvar


class PandoraSeq(nn.Module):
    def __init__(
        self,
        d_model: int = 64,
        num_blocks: int = 2,
        num_experts: int = 4,
        dropout_rate: float = 0.1,
        n_heads: int = 4,
        num_classes: int = 2,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_classes = num_classes

        self.temporal_encoder = ProbabilisticSequenceEncoder(
            1, d_model, num_blocks, num_experts, dropout_rate
        )
        self.volumetric_encoder = ProbabilisticSequenceEncoder(
            1, d_model, num_blocks, num_experts, dropout_rate
        )

        self.fusion = CrossAttentionFusion(d_model, n_heads)

        self.classifier = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.ReLU(),
            nn.Linear(d_model, num_classes),
        )

        nn.init.zeros_(self.temporal_encoder.fc_logvar.weight)
        nn.init.zeros_(self.temporal_encoder.fc_logvar.bias)
        nn.init.zeros_(self.volumetric_encoder.fc_logvar.weight)
        nn.init.zeros_(self.volumetric_encoder.fc_logvar.bias)

    def encode(self, x_temporal: torch.Tensor, x_volumetric: torch.Tensor):
        mu_temp, logvar_temp = self.temporal_encoder(x_temporal)
        mu_vol, logvar_vol = self.volumetric_encoder(x_volumetric)

        fused_mu_intermediate = self.fusion(mu_temp, mu_vol)
        fused_logvar = torch.cat([logvar_temp, logvar_vol], dim=1)

        return fused_mu_intermediate, fused_logvar

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor):
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        return mu

    def forward(self, x: torch.Tensor):
        batch_size = x.size(0)
        x_reshaped = x.view(batch_size, 128, 2)
        x_temporal = x_reshaped[:, :, 0].unsqueeze(-1)
        x_volumetric = x_reshaped[:, :, 1].unsqueeze(-1)

        mu, logvar = self.encode(x_temporal, x_volumetric)
        z = self.reparameterize(mu, logvar)
        logits = self.classifier(z)

        return logits, mu, logvar


def train_pandora_seq(
    malicious_pkl: str,
    benign_pkl: str,
    save_dir: str,
    open_world: bool = False,
    data_dir: Optional[str] = None,
    ood_pkl: Optional[str] = None,
    report_md: Optional[str] = None,
):
    os.makedirs(save_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("[*] Loading primary classes (Benign & Malicious)...")
    X_ben, y_ben = load_pandora_file(benign_pkl, label=0)
    X_mal, y_mal = load_pandora_file(malicious_pkl, label=1)

    if len(X_mal) == 0:
        print(f"[-] ERROR: No malicious samples loaded from {malicious_pkl}")
        return

    np.random.seed(42)
    n_malicious = len(X_mal)

    if open_world:
        num_classes = 3
        search_dir = data_dir if data_dir else os.path.dirname(malicious_pkl)

        X_unk_list, y_unk_list = [], []

        if ood_pkl and os.path.exists(ood_pkl):
            x_u, y_u = load_pandora_file(ood_pkl, label=2)
            if len(x_u) > 0:
                X_unk_list.append(x_u)
                y_unk_list.append(y_u)

        if not X_unk_list and search_dir and os.path.exists(search_dir):
            all_files = [
                os.path.join(search_dir, f)
                for f in os.listdir(search_dir)
                if f.endswith(".pkl")
            ]
            ben_abs = os.path.abspath(benign_pkl)
            mal_abs = os.path.abspath(malicious_pkl)
            unk_files = [
                f for f in all_files if os.path.abspath(f) not in (ben_abs, mal_abs)
            ]

            for unk_file in unk_files:
                x_u, y_u = load_pandora_file(unk_file, label=2)
                if len(x_u) > 0:
                    X_unk_list.append(x_u)
                    y_unk_list.append(y_u)

        if not X_unk_list:
            raise RuntimeError(
                f"[!] No valid UNK dataset files found for Open-World mode in: {search_dir}"
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
            idx = np.random.choice(len(X_ben), size=n_malicious, replace=False)
            X_ben, y_ben = X_ben[idx], y_ben[idx]

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

    model = PandoraSeq(num_classes=num_classes).to(device)
    ce_criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    world_type_str = "open_world" if open_world else "closed_world"
    print(
        f"\n[*] Training PANDORA-Seq IDS [{world_type_str.upper()} | {num_classes}-Class]..."
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
            logits, mu, logvar = model(bx)
            loss_ce = ce_criterion(logits, by)

            loss_kl = (
                -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1).mean()
            )
            beta = 1e-4
            loss = loss_ce + (beta * loss_kl)

            loss.backward()
            optimizer.step()
            total_loss += loss.item() * bx.size(0)

        print(
            f"    Epoch {epoch+1:02d} | Loss: {total_loss / len(train_loader.dataset):.4f}"
        )

    attack_name = os.path.splitext(os.path.basename(malicious_pkl))[0]
    model_filename = (
        f"{attack_name}_ids_open_world.pth" if open_world else f"{attack_name}_ids.pth"
    )
    save_path = os.path.join(save_dir, model_filename)

    checkpoint = {
        "model_state": model.state_dict(),
        "scaler": scaler,
        "num_classes": num_classes,
        "world_type": world_type_str,
        "open_world": open_world,
    }
    torch.save(checkpoint, save_path)
    print(f"\n[+] PANDORA-Seq model & metadata saved to {save_path}")

    model.eval()
    test_dataset = TensorDataset(torch.FloatTensor(X_test_scaled))
    test_loader = DataLoader(test_dataset, batch_size=512, shuffle=False)

    all_preds = []
    with torch.no_grad():
        for batch in test_loader:
            logits, _, _ = model(batch[0].to(device))
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            all_preds.extend(preds)

    all_preds = np.array(all_preds)
    acc = accuracy_score(y_test, all_preds)
    print(f"\n[+] PANDORA-Seq Test Accuracy: {acc:.4f}")
    print("\n================ CLASSIFICATION REPORT ================")
    print(
        classification_report(
            y_test,
            all_preds,
            labels=list(range(num_classes)),
            target_names=target_names,
        )
    )
    print("=======================================================")

    dataset_name = os.path.basename(os.path.dirname(os.path.abspath(malicious_pkl)))

    append_markdown_report(
        report_md_path=report_md,
        dataset_name=dataset_name,
        model_name="PandoraSeq",
        attack_name=attack_name,
        world_type=world_type_str,
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
        default=None,
        choices=["closed_world", "open_world"],
    )
    parser.add_argument("--benign", type=str, default=None)
    parser.add_argument("--malicious", type=str, default=None)
    parser.add_argument("--data_dir", type=str, default=None)
    parser.add_argument("--save_dir", type=str, default=None)
    parser.add_argument("--report_md", type=str, default=None)

    parser.add_argument("--dataset", type=str, default="cic2023")
    parser.add_argument("--attack", type=str, default=None)
    parser.add_argument("--open_world", action="store_true")
    parser.add_argument("--ood_path", type=str, default=None)

    args = parser.parse_args()

    if args.world_type is not None:
        open_world_mode = args.world_type == "open_world"
    else:
        open_world_mode = args.open_world

    benign_path = (
        args.benign if args.benign else f"data/{args.dataset}/BenignTraffic.pkl"
    )

    if args.malicious:
        malicious_path = args.malicious
    elif args.attack:
        malicious_path = f"data/{args.dataset}/{args.attack}.pkl"
    else:
        raise ValueError(
            "Must specify target attack via --malicious or --attack argument."
        )

    dataset_dir = (
        args.data_dir
        if args.data_dir
        else (
            os.path.dirname(malicious_path)
            if os.path.exists(malicious_path)
            else f"data/{args.dataset}"
        )
    )

    if args.save_dir:
        output_dir = args.save_dir
    elif args.attack:
        output_dir = f"models/{args.dataset}/pandora_seq/{args.attack}"
    else:
        attack_name = os.path.splitext(os.path.basename(malicious_path))[0]
        world_str = "open_world" if open_world_mode else "closed_world"
        output_dir = f"models/{world_str}/{args.dataset}/pandora_seq/{attack_name}"

    train_pandora_seq(
        malicious_pkl=malicious_path,
        benign_pkl=benign_path,
        save_dir=output_dir,
        open_world=open_world_mode,
        data_dir=dataset_dir,
        ood_pkl=args.ood_path,
        report_md=args.report_md,
    )
