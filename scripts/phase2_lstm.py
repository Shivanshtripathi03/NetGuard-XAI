"""
Phase 2 — LSTM / GRU Behavioral Sequence Model Training
Builds rolling-window sequences from ordered events, trains binary classifier,
evaluates on test-derived sequences.
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
from sklearn.metrics import (
    classification_report, roc_auc_score, average_precision_score,
    confusion_matrix, f1_score
)

from src.models import LSTMBehaviorModel, train_lstm, create_sequences

PROCESSED_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "processed")
MODELS_DIR    = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")
RESULTS_DIR   = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

SEQ_LENGTH = 10  # rolling window size — documented choice


def load_data():
    train = pd.read_pickle(os.path.join(PROCESSED_DIR, "train_split.pkl"))
    val   = pd.read_pickle(os.path.join(PROCESSED_DIR, "val_split.pkl"))
    test  = pd.read_pickle(os.path.join(PROCESSED_DIR, "test_split.pkl"))
    with open(os.path.join(PROCESSED_DIR, "preprocessors.pkl"), "rb") as f:
        artifacts = pickle.load(f)
    return train, val, test, artifacts


def build_sequences_from_df(df, num_cols, scaler):
    """
    Sequence construction logic:
    - Sort by 'stime' if available, otherwise preserve original order (proxy for temporal order)
    - Group by 'srcip' if available; else treat entire split as one ordered stream
    - Use rolling window of SEQ_LENGTH; label = label of last event in window
    - This logic is saved and reused identically for all phases (including Phase 4 RL env)
    """
    if 'stime' in df.columns:
        df = df.sort_values('stime').reset_index(drop=True)

    X_raw = scaler.transform(df[num_cols].values)
    y_raw = df['label'].astype(int).values

    X_seq, y_seq = create_sequences(X_raw, y_raw, seq_length=SEQ_LENGTH)
    print(f"  Constructed {len(X_seq):,} sequences from {len(df):,} events")
    return X_seq, y_seq


def main():
    print("=== Phase 2: LSTM Behavioral Model ===")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"[LSTM] Using device: {device}")

    train_df, val_df, test_df, artifacts = load_data()
    num_cols = artifacts['num_cols']
    scaler   = artifacts['scaler']

    print("[LSTM] Building training sequences...")
    X_train_seq, y_train_seq = build_sequences_from_df(train_df, num_cols, scaler)
    print("[LSTM] Building validation sequences...")
    X_val_seq, y_val_seq = build_sequences_from_df(val_df, num_cols, scaler)
    print("[LSTM] Building test sequences...")
    X_test_seq, y_test_seq = build_sequences_from_df(test_df, num_cols, scaler)

    # Save sequence-building metadata
    seq_meta = {
        "seq_length": SEQ_LENGTH,
        "num_cols": num_cols,
        "train_sequences": len(X_train_seq),
        "val_sequences": len(X_val_seq),
        "test_sequences": len(X_test_seq),
        "label_scheme": "last-event label in window"
    }
    with open(os.path.join(RESULTS_DIR, "lstm_sequence_metadata.json"), "w") as f:
        json.dump(seq_meta, f, indent=2)

    # DataLoaders
    train_loader = DataLoader(
        TensorDataset(
            torch.tensor(X_train_seq, dtype=torch.float32),
            torch.tensor(y_train_seq, dtype=torch.float32)
        ), batch_size=256, shuffle=True
    )
    val_loader = DataLoader(
        TensorDataset(
            torch.tensor(X_val_seq, dtype=torch.float32),
            torch.tensor(y_val_seq, dtype=torch.float32)
        ), batch_size=256, shuffle=False
    )

    input_dim = X_train_seq.shape[2]
    model = LSTMBehaviorModel(input_dim=input_dim, hidden_dim=64, num_layers=2, dropout=0.3)
    print(f"[LSTM] Input dim: {input_dim} | Sequence length: {SEQ_LENGTH}")

    model, best_val_loss = train_lstm(
        model, train_loader, val_loader,
        epochs=50, lr=5e-4, patience=7, device=device
    )

    # Save model
    model_path = os.path.join(MODELS_DIR, "lstm.pt")
    torch.save(model.state_dict(), model_path)
    print(f"[LSTM] Model saved to {model_path}")

    # Evaluate on test sequences
    model.eval()
    model.to(device)
    X_test_t = torch.tensor(X_test_seq, dtype=torch.float32)
    with torch.no_grad():
        y_prob = model(X_test_t.to(device)).cpu().numpy()
    y_pred = (y_prob >= 0.5).astype(int)
    y_true = y_test_seq

    macro_f1  = f1_score(y_true, y_pred, average='macro')
    roc_auc   = roc_auc_score(y_true, y_prob)
    pr_auc    = average_precision_score(y_true, y_prob)
    cm        = confusion_matrix(y_true, y_pred)
    report    = classification_report(y_true, y_pred, output_dict=True)

    metrics = {
        "model": "lstm",
        "seq_length": SEQ_LENGTH,
        "test_macro_f1": float(macro_f1),
        "test_roc_auc": float(roc_auc),
        "test_pr_auc": float(pr_auc),
        "classification_report": report,
        "confusion_matrix": cm.tolist()
    }

    metrics_path = os.path.join(RESULTS_DIR, "lstm_test_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"[LSTM] Macro-F1: {macro_f1:.4f} | ROC-AUC: {roc_auc:.4f} | PR-AUC: {pr_auc:.4f}")
    print(f"[LSTM] Metrics saved to {metrics_path}")
    print("\n[✓] Phase 2 LSTM checkpoint complete.")
    return model, metrics


if __name__ == "__main__":
    main()
