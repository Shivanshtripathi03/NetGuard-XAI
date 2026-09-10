"""
Phase 2 — Autoencoder Anomaly Detector Training
Trains on Normal-only traffic. Sets threshold at 95th percentile of normal reconstruction error.
Evaluates on full test set (normal + all attack categories).
"""

import os
import sys
import json
import pickle
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import roc_auc_score, average_precision_score, confusion_matrix

from src.models import Autoencoder, train_autoencoder

PROCESSED_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "processed")
MODELS_DIR    = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")
RESULTS_DIR   = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

THRESHOLD_PERCENTILE = 95  # documented choice: balance sensitivity vs FPR


def load_data():
    train = pd.read_pickle(os.path.join(PROCESSED_DIR, "train_split.pkl"))
    val   = pd.read_pickle(os.path.join(PROCESSED_DIR, "val_split.pkl"))
    test  = pd.read_pickle(os.path.join(PROCESSED_DIR, "test_split.pkl"))
    with open(os.path.join(PROCESSED_DIR, "preprocessors.pkl"), "rb") as f:
        artifacts = pickle.load(f)
    neural = np.load(os.path.join(PROCESSED_DIR, "neural_data.npz"), allow_pickle=True)
    return train, val, test, artifacts, neural


def main():
    print("=== Phase 2: Autoencoder Anomaly Detector ===")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"[AE] Using device: {device}")

    train_df, val_df, test_df, artifacts, neural = load_data()

    # --- Filter to Normal-only rows (STRICTLY from training split) ---
    train_normal_mask = train_df['attack_cat'].str.lower().str.strip() == 'normal'
    val_normal_mask   = val_df['attack_cat'].str.lower().str.strip() == 'normal'

    print(f"[AE] Normal train rows: {train_normal_mask.sum():,} | Normal val rows: {val_normal_mask.sum():,}")

    # Verify NO held-out attack categories leaked into AE training
    assert train_df.loc[train_normal_mask, 'label'].sum() == 0, "Non-zero label in Normal train rows!"

    num_cols = artifacts['num_cols']

    # Fit a fresh StandardScaler ONLY on normal training rows
    from sklearn.preprocessing import StandardScaler
    ae_scaler = StandardScaler()
    X_train_normal = ae_scaler.fit_transform(train_df.loc[train_normal_mask, num_cols].values)
    X_val_normal   = ae_scaler.transform(val_df.loc[val_normal_mask, num_cols].values)

    # Save AE-specific scaler (different from the general scaler, fitted on Normal-only subset)
    ae_scaler_path = os.path.join(MODELS_DIR, "ae_scaler.pkl")
    with open(ae_scaler_path, "wb") as f:
        pickle.dump(ae_scaler, f)
    print(f"[AE] Saved AE scaler (fitted on Normal-only training rows) to {ae_scaler_path}")

    # Build DataLoaders
    X_train_t = torch.tensor(X_train_normal, dtype=torch.float32)
    X_val_t   = torch.tensor(X_val_normal,   dtype=torch.float32)
    train_loader = DataLoader(TensorDataset(X_train_t), batch_size=512, shuffle=True)
    val_loader   = DataLoader(TensorDataset(X_val_t),   batch_size=512, shuffle=False)

    # Train Autoencoder
    input_dim  = X_train_normal.shape[1]
    latent_dim = max(8, input_dim // 2)
    model = Autoencoder(input_dim=input_dim, latent_dim=latent_dim)
    print(f"[AE] Input dim: {input_dim} | Latent dim: {latent_dim}")

    model, best_val_loss = train_autoencoder(
        model, train_loader, val_loader,
        epochs=80, lr=1e-3, patience=8, device=device
    )

    # Save model
    model_path = os.path.join(MODELS_DIR, "autoencoder.pt")
    torch.save(model.state_dict(), model_path)
    print(f"[AE] Saved model to {model_path}")

    # --- Compute anomaly threshold on val Normal rows ---
    model.eval()
    model.to(device)
    with torch.no_grad():
        recon_val = model(X_val_t.to(device)).cpu().numpy()
    val_errors = np.mean((X_val_normal - recon_val) ** 2, axis=1)
    threshold = float(np.percentile(val_errors, THRESHOLD_PERCENTILE))
    print(f"[AE] Anomaly threshold set at {THRESHOLD_PERCENTILE}th percentile of val Normal MSE: {threshold:.6f}")

    # --- Evaluate on full official test set ---
    test_num_cols = test_df[num_cols].values
    X_test_scaled = ae_scaler.transform(test_num_cols)
    X_test_t = torch.tensor(X_test_scaled, dtype=torch.float32)
    with torch.no_grad():
        recon_test = model(X_test_t.to(device)).cpu().numpy()
    test_errors = np.mean((X_test_scaled - recon_test) ** 2, axis=1)

    y_test = test_df['label'].astype(int).values
    y_pred = (test_errors > threshold).astype(int)

    # Compute metrics
    roc_auc = roc_auc_score(y_test, test_errors)
    pr_auc  = average_precision_score(y_test, test_errors)
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    detection_rate = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0

    metrics = {
        "model": "autoencoder",
        "threshold_percentile": THRESHOLD_PERCENTILE,
        "anomaly_threshold": threshold,
        "best_val_loss": float(best_val_loss),
        "test_roc_auc": float(roc_auc),
        "test_pr_auc": float(pr_auc),
        "test_detection_rate": float(detection_rate),
        "test_fpr": float(fpr),
        "confusion_matrix": cm.tolist()
    }

    metrics_path = os.path.join(RESULTS_DIR, "autoencoder_test_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"[AE] Detection Rate: {detection_rate:.4f} | FPR: {fpr:.4f} | ROC-AUC: {roc_auc:.4f} | PR-AUC: {pr_auc:.4f}")
    print(f"[AE] Metrics saved to {metrics_path}")
    print("\n[✓] Phase 2 Autoencoder checkpoint complete.")
    return model, ae_scaler, threshold, metrics


if __name__ == "__main__":
    main()
