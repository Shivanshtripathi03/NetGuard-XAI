"""
Save Restricted Model Set for Live Dashboard
============================================
Trains the Phase 5 "restricted" pipeline (Reconnaissance + Fuzzers held out)
and saves all artifacts to models/restricted/ so the API can load them.

Output files:
  models/restricted/xgboost_restricted.json
  models/restricted/autoencoder_restricted.pt
  models/restricted/lstm_restricted.pt
  models/restricted/ae_scaler_restricted.pkl
  models/restricted/lstm_scaler_restricted.pkl
  models/restricted/ae_threshold_restricted.json
  models/restricted/training_manifest.json   <- canonical record of held-out cats
"""

import os
import sys
import json
import pickle
import argparse
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'

import torch
torch.set_num_threads(1)
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
import xgboost as xgb

from src.models import Autoencoder, LSTMBehaviorModel, create_sequences, train_autoencoder, train_lstm

PROCESSED_DIR  = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "processed")
MODELS_DIR     = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")
RESTRICTED_DIR = os.path.join(MODELS_DIR, "restricted")
RESULTS_DIR    = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")

HELD_OUT_CATEGORIES = ["Reconnaissance", "Fuzzers"]
SEQ_LENGTH = 10


def restrict_df(df, held_out):
    return df[~df['attack_cat'].str.strip().isin(held_out)].reset_index(drop=True)


def leakage_check(df, held_out, split_name):
    count = df['attack_cat'].str.strip().isin(held_out).sum()
    assert count == 0, f"LEAKAGE in {split_name}: {count} rows!"
    print(f"  [ok] {split_name}: 0 held-out rows")


def train_xgb(train_r, val_r, feature_cols):
    X_tr, y_tr = train_r[feature_cols].values, train_r['label'].astype(int).values
    X_va, y_va = val_r[feature_cols].values,   val_r['label'].astype(int).values
    spw = (y_tr == 0).sum() / max((y_tr == 1).sum(), 1)
    m = xgb.XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.1,
                           subsample=0.8, scale_pos_weight=spw,
                           use_label_encoder=False, eval_metric='logloss',
                           random_state=42, n_jobs=-1, verbosity=0)
    m.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
    return m


def train_ae(train_r, val_r, num_cols, device):
    nm_tr = train_r['attack_cat'].str.lower().str.strip() == 'normal'
    nm_va = val_r['attack_cat'].str.lower().str.strip() == 'normal'
    sc = StandardScaler()
    X_tr = sc.fit_transform(train_r.loc[nm_tr, num_cols].values)
    X_va = sc.transform(val_r.loc[nm_va, num_cols].values)
    idim = X_tr.shape[1]
    ae = Autoencoder(idim, max(8, idim // 2))
    trl = DataLoader(TensorDataset(torch.tensor(X_tr, dtype=torch.float32)), batch_size=512, shuffle=True)
    val = DataLoader(TensorDataset(torch.tensor(X_va, dtype=torch.float32)), batch_size=512)
    ae, _ = train_autoencoder(ae, trl, val, epochs=50, patience=6, device=device)
    ae.eval()
    with torch.no_grad():
        recon = ae(torch.tensor(X_va, dtype=torch.float32).to(device)).cpu().numpy()
    thr = float(np.percentile(np.mean((X_va - recon) ** 2, axis=1), 95))
    return ae, sc, thr


def train_lstm_model(train_r, val_r, num_cols, device):
    sc = StandardScaler()
    X_tr = sc.fit_transform(train_r[num_cols].values)
    X_va = sc.transform(val_r[num_cols].values)
    y_tr, y_va = train_r['label'].astype(int).values, val_r['label'].astype(int).values
    Xs_tr, ys_tr = create_sequences(X_tr, y_tr, SEQ_LENGTH)
    Xs_va, ys_va = create_sequences(X_va, y_va, SEQ_LENGTH)
    lstm = LSTMBehaviorModel(Xs_tr.shape[2], hidden_dim=64, num_layers=2, dropout=0.3)
    trl = DataLoader(TensorDataset(torch.tensor(Xs_tr, dtype=torch.float32),
                                   torch.tensor(ys_tr, dtype=torch.float32)), batch_size=256, shuffle=True)
    val = DataLoader(TensorDataset(torch.tensor(Xs_va, dtype=torch.float32),
                                   torch.tensor(ys_va, dtype=torch.float32)), batch_size=256)
    lstm, _ = train_lstm(lstm, trl, val, epochs=30, patience=5, device=device)
    return lstm, sc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    os.makedirs(RESTRICTED_DIR, exist_ok=True)
    manifest_path = os.path.join(RESTRICTED_DIR, "training_manifest.json")

    if not args.force and os.path.exists(manifest_path):
        with open(manifest_path) as f:
            mf = json.load(f)
        print(f"[RESTRICTED] Already trained. Held-out: {mf['held_out_categories']}")
        print("  Use --force to retrain.")
        return

    print("=" * 60)
    print("Training Restricted Model Set (Reconnaissance + Fuzzers held out)")
    print("=" * 60)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"  Device: {device}")

    train_df = pd.read_pickle(os.path.join(PROCESSED_DIR, "train_split.pkl"))
    val_df   = pd.read_pickle(os.path.join(PROCESSED_DIR, "val_split.pkl"))
    with open(os.path.join(PROCESSED_DIR, "preprocessors.pkl"), "rb") as f:
        artifacts = pickle.load(f)

    feature_cols = artifacts['tree_feature_cols']
    num_cols     = artifacts['num_cols']
    all_train_cats = sorted(train_df['attack_cat'].str.strip().unique().tolist())

    train_r = restrict_df(train_df, HELD_OUT_CATEGORIES)
    val_r   = restrict_df(val_df,   HELD_OUT_CATEGORIES)

    leakage_check(train_r, HELD_OUT_CATEGORIES, "restricted_train")
    leakage_check(val_r,   HELD_OUT_CATEGORIES, "restricted_val")

    cats_in_train = sorted(train_r['attack_cat'].str.strip().unique().tolist())
    print(f"  Cats in restricted train: {cats_in_train}")
    print(f"  Train: {len(train_r):,}  Val: {len(val_r):,}")

    print("\n[3] XGBoost...")
    xgb_r = train_xgb(train_r, val_r, feature_cols)
    xgb_r.save_model(os.path.join(RESTRICTED_DIR, "xgboost_restricted.json"))
    print("  saved xgboost_restricted.json")

    print("\n[4] Autoencoder...")
    ae_r, ae_sc_r, ae_thr_r = train_ae(train_r, val_r, num_cols, device)
    torch.save(ae_r.state_dict(), os.path.join(RESTRICTED_DIR, "autoencoder_restricted.pt"))
    with open(os.path.join(RESTRICTED_DIR, "ae_scaler_restricted.pkl"), "wb") as f:
        pickle.dump(ae_sc_r, f)
    with open(os.path.join(RESTRICTED_DIR, "ae_threshold_restricted.json"), "w") as f:
        json.dump({"anomaly_threshold": ae_thr_r}, f, indent=2)
    print(f"  saved autoencoder_restricted.pt  threshold={ae_thr_r:.6f}")

    print("\n[5] LSTM...")
    lstm_r, lstm_sc_r = train_lstm_model(train_r, val_r, num_cols, device)
    torch.save(lstm_r.state_dict(), os.path.join(RESTRICTED_DIR, "lstm_restricted.pt"))
    with open(os.path.join(RESTRICTED_DIR, "lstm_scaler_restricted.pkl"), "wb") as f:
        pickle.dump(lstm_sc_r, f)
    print("  saved lstm_restricted.pt")

    manifest = {
        "held_out_categories": HELD_OUT_CATEGORIES,
        "trained_categories": cats_in_train,
        "all_full_train_categories": all_train_cats,
        "restricted_train_rows": int(len(train_r)),
        "restricted_val_rows": int(len(val_r)),
        "leakage_status": "PASSED",
        "seq_length": SEQ_LENGTH,
        "note": (
            "DEPLOYED restricted model set. "
            "Reconnaissance and Fuzzers were NEVER seen during training. "
            "These categories trigger genuine zero-day anomaly-based detection."
        )
    }
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\n[OK] Restricted models saved to {RESTRICTED_DIR}")
    print(f"     Trained: {cats_in_train}")
    print(f"     Held-out: {HELD_OUT_CATEGORIES}")

if __name__ == "__main__":
    main()
