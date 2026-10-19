import os
import argparse
import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.manifold import TSNE
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
import torch
import warnings

warnings.filterwarnings("ignore")

from advGenerate.train import AdversarialTrainer
from advGenerate.environments.dl_environment_eval import DL_IDS_Evaluator
from advGenerate.environments.vib_environment_eval import VIB_IDS_Evaluator
from advGenerate.environments.transformer_environment_eval import Transformer_IDS_Evaluator
from advGenerate.environments.cnn_environment_eval import CNN_IDS_Evaluator
from advGenerate.environments.pandora_seq_environment_eval import PandoraSeq_IDS_Evaluator

def extract_features(ipds, sizes, max_seq_len=128):
    """Extracts 256D features (128 IPDs + 128 Sizes)."""
    feature_row = []
    for i in range(max_seq_len):
        ipd = float(ipds[i]) if i < len(ipds) else 0.0
        size = float(sizes[i]) if i < len(sizes) else 0.0
        feature_row.extend([ipd, size])
    return np.array(feature_row, dtype=np.float32)


def plot_tsne_decision_boundary(evaluator, benign_raw, malicious_raw, unknown_raw, morphed_records, is_open_world=False, save_path="tsne_boundary.png"):
    """
    Renders t-SNE projection and decision boundary.
    Supports 2-class (Closed-World) and 3-class (Open-World).
    """
    X_benign = np.array([extract_features(ipds, sizes) for ipds, sizes in benign_raw]) if len(benign_raw) > 0 else np.empty((0, 256))
    X_malic = np.array([extract_features(ipds, sizes) for ipds, sizes in malicious_raw]) if len(malicious_raw) > 0 else np.empty((0, 256))
    X_unk = np.array([extract_features(ipds, sizes) for ipds, sizes in unknown_raw]) if len(unknown_raw) > 0 else np.empty((0, 256))

    # Parse morphed outcomes: (orig_raw, morph_raw, pred_before, pred_after)
    morphed_benign = [m for _, m, _, c in morphed_records if c == 0]
    morphed_unknown = [m for _, m, _, c in morphed_records if c == 2]
    morphed_failed = [m for _, m, _, c in morphed_records if c == 1]

    X_m_benign = np.array([extract_features(ipds, sizes) for ipds, sizes in morphed_benign]) if len(morphed_benign) > 0 else np.empty((0, 256))
    X_m_unknown = np.array([extract_features(ipds, sizes) for ipds, sizes in morphed_unknown]) if len(morphed_unknown) > 0 else np.empty((0, 256))
    X_m_failed = np.array([extract_features(ipds, sizes) for ipds, sizes in morphed_failed]) if len(morphed_failed) > 0 else np.empty((0, 256))

    y_benign = np.zeros(len(X_benign), dtype=int)
    y_malic = np.ones(len(X_malic), dtype=int)
    y_unk = np.full(len(X_unk), 2, dtype=int)

    y_mb = np.zeros(len(X_m_benign), dtype=int)
    y_mu = np.full(len(X_m_unknown), 2, dtype=int)
    y_mf = np.ones(len(X_m_failed), dtype=int)

    all_blocks = [x for x in [X_benign, X_malic, X_unk, X_m_benign, X_m_unknown, X_m_failed] if len(x) > 0]
    all_labels = [y for y in [y_benign, y_malic, y_unk, y_mb, y_mu, y_mf] if len(y) > 0]
    
    if len(all_blocks) == 0:
        print("[!] Warning: No flow data available to plot.")
        return

    X_all = np.vstack(all_blocks)
    y_all = np.concatenate(all_labels)

    print(f"[*] Fitting t-SNE manifold on {len(X_all)} high-dimensional flows (Open World: {is_open_world})...")
    scaler = StandardScaler()
    X_all_scaled = scaler.fit_transform(X_all)

    perplexity = min(30, max(5, len(X_all) // 4))
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42, init="pca", learning_rate="auto")
    X_all_2d = tsne.fit_transform(X_all_scaled)

    # Slice 2D coordinates back
    idx = 0
    X_benign_2d = X_all_2d[idx : idx + len(X_benign)]; idx += len(X_benign)
    X_malic_2d = X_all_2d[idx : idx + len(X_malic)]; idx += len(X_malic)
    X_unk_2d = X_all_2d[idx : idx + len(X_unk)]; idx += len(X_unk)
    X_mb_2d = X_all_2d[idx : idx + len(X_m_benign)]; idx += len(X_m_benign)
    X_mu_2d = X_all_2d[idx : idx + len(X_m_unknown)]; idx += len(X_m_unknown)
    X_mf_2d = X_all_2d[idx : idx + len(X_m_failed)]; idx += len(X_m_failed)

    # Train 2D K-NN classifier on projected manifold using distance weighting to reduce mapping artifacts
    knn_boundary = KNeighborsClassifier(n_neighbors=5, weights='distance')
    knn_boundary.fit(X_all_2d, y_all)

    x_min, x_max = X_all_2d[:, 0].min() - 3, X_all_2d[:, 0].max() + 3
    y_min, y_max = X_all_2d[:, 1].min() - 3, X_all_2d[:, 1].max() + 3
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 250), np.linspace(y_min, y_max, 250))
    Z = knn_boundary.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

    print("[*] Rendering decision boundary and multi-class evasion pathways...")
    plt.figure(figsize=(11.5, 8.5), dpi=300)

    # High-contrast background colormaps (Replaced faint purple with darker #C084FC)
    if is_open_world:
        cmap_bg = ListedColormap(['#93C5FD', '#FCA5A5', '#C084FC'])  
        levels = [-0.5, 0.5, 1.5, 2.5]
    else:
        cmap_bg = ListedColormap(['#93C5FD', '#FCA5A5'])  
        levels = [-0.5, 0.5, 1.5]

    plt.contourf(xx, yy, Z, alpha=0.35, cmap=cmap_bg, levels=levels)

    # Plot baseline reference samples
    if len(X_benign_2d) > 0:
        plt.scatter(X_benign_2d[:, 0], X_benign_2d[:, 1], c='#1D4ED8', edgecolors='k', marker='o', s=70, label='True Benign', alpha=0.9)
    if len(X_malic_2d) > 0:
        plt.scatter(X_malic_2d[:, 0], X_malic_2d[:, 1], c='#B91C1C', edgecolors='k', marker='X', s=90, label='True Target Attack (DoS)', alpha=0.9)
    if is_open_world and len(X_unk_2d) > 0:
        plt.scatter(X_unk_2d[:, 0], X_unk_2d[:, 1], c='#6B21A8', edgecolors='k', marker='^', s=90, label='True Unknown Attack', alpha=0.9)

    # Plot trajectories for morphed flows
    orig_malic_map = {idx_m: X_malic_2d[idx_m] for idx_m in range(len(X_malic_2d))}

    mb_counter, mu_counter, mf_counter = 0, 0, 0
    for idx_rec, (orig_raw, morph_raw, pred_b, pred_a) in enumerate(morphed_records):
        if idx_rec not in orig_malic_map: continue
        orig_2d = orig_malic_map[idx_rec]

        if pred_a == 0 and mb_counter < len(X_mb_2d):
            morph_2d = X_mb_2d[mb_counter]
            mb_counter += 1
            plt.annotate('', xy=morph_2d, xytext=orig_2d,
                         arrowprops=dict(arrowstyle="->", color="#059669", lw=1.4, alpha=0.85, mutation_scale=12))
        elif pred_a == 2 and mu_counter < len(X_mu_2d):
            morph_2d = X_mu_2d[mu_counter]
            mu_counter += 1
            plt.annotate('', xy=morph_2d, xytext=orig_2d,
                         arrowprops=dict(arrowstyle="->", color="#D97706", lw=1.4, linestyle='--', alpha=0.85, mutation_scale=12))
        elif pred_a == 1 and mf_counter < len(X_mf_2d):
            morph_2d = X_mf_2d[mf_counter]
            mf_counter += 1
            plt.annotate('', xy=morph_2d, xytext=orig_2d,
                         arrowprops=dict(arrowstyle="->", color="#DC2626", lw=1.1, linestyle=':', alpha=0.6, mutation_scale=10))

    # Scatter Morphed Outcome Markers
    if len(X_mb_2d) > 0:
        plt.scatter(X_mb_2d[:, 0], X_mb_2d[:, 1], c='#10B981', edgecolors='k', marker='*', s=220, label='Morphed -> Benign (Evaded)', alpha=0.95)
    if len(X_mu_2d) > 0:
        plt.scatter(X_mu_2d[:, 0], X_mu_2d[:, 1], c='#F59E0B', edgecolors='k', marker='*', s=220, label='Morphed -> Unknown (Shifted)', alpha=0.95)
    if len(X_mf_2d) > 0:
        plt.scatter(X_mf_2d[:, 0], X_mf_2d[:, 1], c='#EF4444', edgecolors='k', marker='*', s=180, label='Morphed -> Remained DoS (Failed)', alpha=0.95)

    title_mode = "Open-World 3-Class" if is_open_world else "Closed-World 2-Class"
    plt.title(f"IDS Decision Boundary & Evasion Pathways ({title_mode} t-SNE Manifold)", fontsize=13, fontweight='bold', pad=15)
    plt.xlabel("t-SNE Dimension 1", fontweight='bold')
    plt.ylabel("t-SNE Dimension 2", fontweight='bold')
    plt.legend(loc="best", framealpha=0.9, edgecolor='gray')
    plt.grid(True, linestyle=":", alpha=0.5)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, bbox_inches="tight")
    print(f"[+] Saved t-SNE boundary plot to: {save_path}")
    plt.close()


def plot_confidence_and_perturbation(evaluator, morphed_records, save_path="confidence_perturbation.png"):
    """
    Plots confidence score shift and perturbation norms ($L_1$ and $L_2$).
    """
    if len(morphed_records) == 0:
        print("[!] Warning: No morphed records to plot in confidence analysis.")
        return

    scores_before = []
    scores_after = []
    l2_distances = []
    l1_distances = []

    print("[*] Extracting confidence scores and perturbation metrics...")
    for orig_raw, morph_raw, _, _ in morphed_records:
        orig_feat = extract_features(orig_raw[0], orig_raw[1])
        morph_feat = extract_features(morph_raw[0], morph_raw[1])

        res_orig = evaluator.predict(orig_raw[0], orig_raw[1], max_seq_len=128)
        res_morph = evaluator.predict(morph_raw[0], morph_raw[1], max_seq_len=128)

        # Extract index 0 for the Benign class probability
        score_orig = float(res_orig[0]) if len(res_orig) > 1 else 0.0
        score_morph = float(res_morph[0]) if len(res_morph) > 1 else 1.0

        scores_before.append(score_orig)
        scores_after.append(score_morph)

        l2_distances.append(np.linalg.norm(morph_feat - orig_feat))
        l1_distances.append(np.linalg.norm(morph_feat - orig_feat, ord=1))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5), dpi=300)

    # Panel 1: Confidence Score Jump
    sample_indices = np.arange(1, len(morphed_records) + 1)
    ax1.plot(sample_indices, scores_before, color='#DC2626', marker='o', linestyle='--', linewidth=1.5, label='Before Morphing', alpha=0.8)
    ax1.plot(sample_indices, scores_after, color='#059669', marker='s', linestyle='-', linewidth=2.0, label='After Morphing', alpha=0.9)
    ax1.fill_between(sample_indices, scores_before, scores_after, color='#10B981', alpha=0.15, label='Confidence Delta')
    
    ax1.set_title("A: IDS Benign Confidence Score Shift", fontsize=12, fontweight='bold')
    ax1.set_xlabel("Sample Flow Index", fontweight='bold')
    ax1.set_ylabel("Model Benign Confidence Score", fontweight='bold')
    ax1.set_ylim(-0.05, 1.05)
    ax1.legend(loc="best", framealpha=0.9)
    ax1.grid(True, linestyle=":", alpha=0.6)

    # Panel 2: Perturbation Magnitude
    width = 0.35
    ax2.bar(sample_indices - width/2, l2_distances, width, label=r'$L_2$ Distance', color='#2563EB', alpha=0.85)
    ax2.bar(sample_indices + width/2, l1_distances, width, label=r'$L_1$ Distance', color='#7C3AED', alpha=0.65)
    
    ax2.set_title(r"B: Feature Perturbation Magnitude ($L_2$ vs $L_1$)", fontsize=12, fontweight='bold')
    ax2.set_xlabel("Sample Flow Index", fontweight='bold')
    ax2.set_ylabel("Perturbation Distance Norm", fontweight='bold')
    ax2.legend(loc="best", framealpha=0.9)
    ax2.grid(True, linestyle=":", alpha=0.6)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, bbox_inches="tight")
    print(f"[+] Saved confidence and perturbation analysis to: {save_path}")
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize Decision Boundaries and Evasion Metrics")
    parser.add_argument("--world_type", type=str, default="open_world", choices=["open_world", "closed_world"])
    parser.add_argument("--dataset", type=str, default="cic2023")
    parser.add_argument("--attack", type=str, required=True)
    parser.add_argument("--defense", type=str, default="cnn")
    parser.add_argument("--feedback_mode", type=str, default="hard_labeled")
    parser.add_argument("--arch", type=str, default="mamba")
    parser.add_argument("--num_samples", type=int, default=40)
    args = parser.parse_args()

    is_open_world = (args.world_type == "open_world")

    # 1. Initialize Evaluator
    if args.defense == "dl":
        model_pth = f"models/{args.world_type}/{args.dataset}/dl/{args.attack}_ids.pth"
        evaluator = DL_IDS_Evaluator(models_dir=model_pth, mode=args.feedback_mode, threshold=0.3)
    elif args.defense == "vib":
        model_pth = f"models/{args.world_type}/{args.dataset}/vib/{args.attack}_vib.pth"
        evaluator = VIB_IDS_Evaluator(models_dir=model_pth, mode=args.feedback_mode, threshold=0.5, num_classes=3 if is_open_world else 2)
    elif args.defense == "transformer":
        model_pth = f"models/{args.world_type}/{args.dataset}/transformer/{args.attack}_ids.pth"
        evaluator = Transformer_IDS_Evaluator(models_dir=model_pth, mode=args.feedback_mode, threshold=0.5, num_classes=3 if is_open_world else 2)
    elif args.defense == "cnn":
        model_pth = f"models/{args.world_type}/{args.dataset}/cnn/{args.attack}_ids.pth"
        evaluator = CNN_IDS_Evaluator(models_dir=model_pth, mode=args.feedback_mode, threshold=0.5)
    elif args.defense == "pandora":
        model_pth = f"models/{args.world_type}/{args.dataset}/pandora_seq/{args.attack}"
        evaluator = PandoraSeq_IDS_Evaluator(models_dir=model_pth, mode=args.feedback_mode, threshold=0.5)
    else:
        raise ValueError("Unsupported defense type")

    # 2. Setup Trainer and Agent
    trainer = AdversarialTrainer(
        rl_yaml_path="advGenerate/config/sac.yaml",
        bert_yaml_path="trafficMimic/config/bert.yaml",
        evaluator=evaluator,
        arch=args.arch,
        disable_per=True,
        max_steps=100
    )

    attack_pkl = f"data/{args.dataset}/{args.attack}.pkl"
    benign_pkl = f"data/{args.dataset}/BenignTraffic.pkl"

    trainer.rl_args.trainer.train_data_pth = attack_pkl
    trainer.rl_args.trainer.test_data_pth = attack_pkl

    ablation_tag = f"PER-False_{args.arch}"
    agent_dir = f"results/{args.world_type}/{args.feedback_mode}/{args.dataset}/{args.defense}/{args.attack}/{ablation_tag}/checkpoints"
    best_model = os.path.join(agent_dir, "best_model")
    if os.path.exists(os.path.join(best_model, "actor.pth")):
        print(f"[*] Loading trained RL Agent from: {best_model}")
        trainer.agent.load_model(best_model)

    # 3. Collect True Benign Data (Class 0)
    benign_flows = []
    if os.path.exists(benign_pkl):
        print(f"[*] Loading Benign dataset from: {benign_pkl}")
        with open(benign_pkl, "rb") as f:
            raw_benign_data = pickle.load(f)
        for flow in raw_benign_data:
            ipds, sizes = flow[0], flow[1]
            eval_res = evaluator.predict(ipds, sizes, max_seq_len=128)
            pred_class = eval_res[2] if len(eval_res) == 3 else (0 if eval_res[0] else 1)
            if pred_class == 0:
                benign_flows.append((ipds, sizes))
            if len(benign_flows) >= args.num_samples:
                break

    # 4. Collect Malicious Data & Perform Morphing
    malic_flows = []
    unknown_flows = []
    morphed_records = []  # Stores (orig_raw, morph_raw, pred_before, pred_after)

    print(f"[*] Generating mutations (Open World = {is_open_world})...")
    for idx in range(len(trainer.test_dataset)):
        state_data = trainer.test_dataset[idx]
        real_len = int(state_data["real_length"].item())
        if real_len <= 1: continue

        trainer.env.state = {k: v.clone().detach().to(trainer.env.device) for k, v in state_data.items()}
        orig_ipds, orig_sizes = trainer.env._get_raw_traffic_dimensions()
        
        eval_res = evaluator.predict(orig_ipds, orig_sizes, max_seq_len=128)
        pred_class = eval_res[2] if len(eval_res) == 3 else (0 if eval_res[0] else 1)

        if is_open_world:
            # Open-World: morph non-benign flows (Class 1 or Class 2)
            if pred_class == 2 and len(unknown_flows) < args.num_samples:
                unknown_flows.append((orig_ipds, orig_sizes))

            if pred_class != 0 and len(malic_flows) < args.num_samples:
                malic_flows.append((orig_ipds, orig_sizes))

                trainer.env.step_num = 0
                trainer.env.forbidden_indices = torch.zeros(trainer.seq_len, dtype=torch.bool, device=trainer.env.device)
                trainer.env.forbidden_indices[0] = True
                trainer.env.state["forbidden_indices"] = trainer.env.forbidden_indices.clone()
                
                state = trainer.env.state
                done = False
                while not done:
                    action = trainer.agent.take_action(state, eval_mode=True)
                    next_state, _, done, info = trainer.env.step(action)
                    state = next_state
                    if done: break
                
                m_ipds, m_sizes = trainer.env._get_raw_traffic_dimensions()
                m_eval_res = evaluator.predict(m_ipds, m_sizes, max_seq_len=128)
                m_pred_class = m_eval_res[2] if len(m_eval_res) == 3 else (0 if m_eval_res[0] else 1)
                
                morphed_records.append(((orig_ipds, orig_sizes), (m_ipds, m_sizes), pred_class, m_pred_class))
        else:
            # Closed-World: filter strictly for successful evasion (Class 0)
            if pred_class != 0 and len(malic_flows) < args.num_samples:
                trainer.env.step_num = 0
                trainer.env.forbidden_indices = torch.zeros(trainer.seq_len, dtype=torch.bool, device=trainer.env.device)
                trainer.env.forbidden_indices[0] = True
                trainer.env.state["forbidden_indices"] = trainer.env.forbidden_indices.clone()
                
                state = trainer.env.state
                done = False
                while not done:
                    action = trainer.agent.take_action(state, eval_mode=True)
                    next_state, _, done, info = trainer.env.step(action)
                    state = next_state
                    if done: break
                
                m_ipds, m_sizes = trainer.env._get_raw_traffic_dimensions()
                m_eval_res = evaluator.predict(m_ipds, m_sizes, max_seq_len=128)
                m_pred_class = m_eval_res[2] if len(m_eval_res) == 3 else (0 if m_eval_res[0] else 1)

                malic_flows.append((orig_ipds, orig_sizes))
                morphed_records.append(((orig_ipds, orig_sizes), (m_ipds, m_sizes), pred_class, m_pred_class))
                
                # if m_pred_class == 0:
                #     malic_flows.append((orig_ipds, orig_sizes))
                #     morphed_records.append(((orig_ipds, orig_sizes), (m_ipds, m_sizes), pred_class, m_pred_class))

                
                

        if len(malic_flows) >= args.num_samples:
            break

    # Output directory
    out_dir = f"visualizations/{args.world_type}/{args.dataset}/{args.defense}"
    tsne_out = os.path.join(out_dir, f"{args.attack}_tsne_boundary.png")
    conf_out = os.path.join(out_dir, f"{args.attack}_confidence_perturbation.png")

    plot_tsne_decision_boundary(evaluator, benign_flows, malic_flows, unknown_flows, morphed_records, is_open_world=is_open_world, save_path=tsne_out)
    plot_confidence_and_perturbation(evaluator, morphed_records, save_path=conf_out)