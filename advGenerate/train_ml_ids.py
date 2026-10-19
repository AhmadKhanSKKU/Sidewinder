# src/advGenerate/train_ml_ids.py

import os
import argparse
import pickle
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.calibration import CalibratedClassifierCV


def load_and_flatten_data(pkl_path: str, assigned_label: int, max_seq_len: int = 128):
    """Loads a pkl file, flattens sequences, and applies the specified label (0 or 1)."""
    print(
        f"[*] Extracting features from {os.path.basename(pkl_path)} (Label: {assigned_label})..."
    )
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)

    X = []
    y = []

    for item in data:
        if isinstance(item, tuple) and len(item) == 2:
            raw_ipd, raw_size = item
        else:
            # ENFORCING SINGULAR NAMING STANDARD
            raw_ipd = item.get("raw_ipd", item.get("ipds", item.get("flow_ipd", [])))
            raw_size = item.get(
                "raw_size", item.get("sizes", item.get("flow_size", []))
            )

        # Convert PyTorch tensors to lists if necessary
        if hasattr(raw_ipd, "squeeze"):
            raw_ipd = raw_ipd.squeeze().tolist()
        if hasattr(raw_size, "squeeze"):
            raw_size = raw_size.squeeze().tolist()

        padded_features = []
        for i in range(max_seq_len):
            if i < len(raw_ipd) and i < len(raw_size):
                # STRICT 4-DECIMAL FLOAT STANDARD
                val_ipd = round(float(raw_ipd[i]), 4)
                val_size = round(float(raw_size[i]), 4)
                padded_features.extend([val_ipd, val_size])
            else:
                padded_features.extend([0.0, 0.0])

        X.append(padded_features)
        y.append(assigned_label)

    # Lock into float32
    return np.array(X, dtype=np.float32), np.array(y)


def train_baselines(malicious_pkl: str, benign_pkl: str, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)

    X_mal, y_mal = load_and_flatten_data(malicious_pkl, assigned_label=1)
    X_ben, y_ben = load_and_flatten_data(benign_pkl, assigned_label=0)

    X = np.vstack((X_ben, X_mal))
    y = np.concatenate((y_ben, y_mal))

    print(
        f"[*] Total dataset size: {len(y)} samples (Benign: {len(y_ben)}, Malicious: {len(y_mal)})"
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 1. TRAIN AND SAVE THE SCALER (Using Float32)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train).astype(np.float32)
    X_test_scaled = scaler.transform(X_test).astype(np.float32)

    joblib.dump(scaler, os.path.join(output_dir, "ml_scaler.pkl"))
    print("[+] Feature Scaler saved successfully.")

    # ==========================================
    # MODEL 1: RANDOM FOREST
    # ==========================================
    print("\n[*] Training RandomForest Classifier...")
    rf = RandomForestClassifier(
        n_estimators=100, max_depth=15, random_state=42, n_jobs=-1
    )
    rf.fit(X_train_scaled, y_train)
    rf_preds = rf.predict(X_test_scaled)
    print(f"[+] RF Accuracy: {accuracy_score(y_test, rf_preds):.4f}")
    joblib.dump(rf, os.path.join(output_dir, "random_forest_ids.pkl"))

    # ==========================================
    # MODEL 2: SUPPORT VECTOR MACHINE (SVM)
    # ==========================================
    print("\n[*] Training Support Vector Machine (SVM)... (Calibrating Probabilities)")
    SVM_MAX_SAMPLES = 15000
    if len(X_train_scaled) > SVM_MAX_SAMPLES:
        X_svm_train, _, y_svm_train, _ = train_test_split(
            X_train_scaled,
            y_train,
            train_size=SVM_MAX_SAMPLES,
            stratify=y_train,
            random_state=42,
        )
    else:
        X_svm_train, y_svm_train = X_train_scaled, y_train

    base_svm = SVC(kernel="rbf", C=1.0, random_state=42)
    svm = CalibratedClassifierCV(base_svm, ensemble=False, cv=3)
    svm.fit(X_svm_train, y_svm_train)
    svm_preds = svm.predict(X_test_scaled)
    print(f"[+] SVM Accuracy: {accuracy_score(y_test, svm_preds):.4f}")
    joblib.dump(svm, os.path.join(output_dir, "svm_ids.pkl"))

    # ==========================================
    # MODEL 3: XGBOOST
    # ==========================================
    print("\n[*] Training Extreme Gradient Boosting (XGBoost)...")
    xgb = XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        random_state=42,
        eval_metric="logloss",
    )
    xgb.fit(X_train_scaled, y_train)
    xgb_preds = xgb.predict(X_test_scaled)
    print(f"[+] XGBoost Accuracy: {accuracy_score(y_test, xgb_preds):.4f}")
    joblib.dump(xgb, os.path.join(output_dir, "xgboost_ids.pkl"))

    print(f"\n[SUCCESS] All classic ML engines exported to '{output_dir}/'")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train ML IDS Baselines")
    parser.add_argument("--dataset", type=str, default="cic2023")
    parser.add_argument("--attack", type=str, required=True)
    args = parser.parse_args()

    malicious_pkl = f"data/{args.dataset}/{args.attack}.pkl"
    benign_pkl = f"data/{args.dataset}/BenignTraffic.pkl"
    save_dir = f"models/{args.dataset}/ml/{args.attack}"

    if os.path.exists(malicious_pkl) and os.path.exists(benign_pkl):
        train_baselines(malicious_pkl, benign_pkl, output_dir=save_dir)
    else:
        print("[!] Error: Data files not found.")
