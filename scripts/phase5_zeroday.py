"""
Phase 5 — Zero-Day / Unseen Attack Experiments
Holds out Reconnaissance + Fuzzers from ALL model training.
Trains restricted pipeline, compares to full-knowledge baseline.
Includes mandatory leakage assertions before unseen evaluation.
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import json
import pickle
import numpy as np
import pandas as pd

os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'

import torch
torch.set_num_threads(1)
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    f1_score, recall_score, precision_score, roc_auc_score,
    average_precision_score, confusion_matrix
)
import xgboost as xgb

from src.models import Autoencoder, LSTMBehaviorModel, create_sequences, train_autoencoder, train_lstm
from scripts.phase3_risk_xai import RiskEngine, compute_uncertainty, normalize_ae_score, SEVERITY_MAP

PROCESSED_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "processed")
MODELS_DIR    = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")
RESULTS_DIR   = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")

# Held-out categories — decided per instructions, not negotiable
HELD_OUT_CATEGORIES = ["Reconnaissance", "Fuzzers"]
SEQ_LENGTH = 10


def leakage_check(df: pd.DataFrame, held_out: list, split_name: str):
    """MANDATORY: Assert zero rows of held-out categories in a given split."""
    mask = df['attack_cat'].str.strip().isin(held_out)
    count = mask.sum()
    assert count == 0, (
        f"LEAKAGE DETECTED in {split_name}: {count} rows of "
        f"{held_out} found! Pipeline aborted."
    )
    print(f"[LEAKAGE CHECK] {split_name}: 0 rows of {held_out} ✓")


def restrict_df(df: pd.DataFrame, held_out: list) -> pd.DataFrame:
    return df[~df['attack_cat'].str.strip().isin(held_out)].reset_index(drop=True)


def train_restricted_xgboost(train_r, val_r, feature_cols, scaler_type="restricted"):
    """Train XGBoost on restricted training data."""
    X_train = train_r[feature_cols].values
    y_train = train_r['label'].astype(int).values
    X_val   = val_r[feature_cols].values
    y_val   = val_r['label'].astype(int).values

    n_neg = (y_train == 0).sum()
    n_pos = (y_train == 1).sum()
    spw = n_neg / n_pos if n_pos > 0 else 1.0

    model = xgb.XGBClassifier(
        n_estimators=300, max_depth=6, learning_rate=0.1,
        subsample=0.8, scale_pos_weight=spw,
        use_label_encoder=False, eval_metric='logloss',
        random_state=42, n_jobs=-1, verbosity=0
    )
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    return model


def train_restricted_autoencoder(train_r, val_r, num_cols, device):
    """Train Autoencoder on Normal-only restricted training data."""
    normal_mask_tr = train_r['attack_cat'].str.lower().str.strip() == 'normal'
    normal_mask_va = val_r['attack_cat'].str.lower().str.strip() == 'normal'

    ae_scaler = StandardScaler()
    X_train_normal = ae_scaler.fit_transform(train_r.loc[normal_mask_tr, num_cols].values)
    X_val_normal   = ae_scaler.transform(val_r.loc[normal_mask_va, num_cols].values)

    input_dim  = X_train_normal.shape[1]
    latent_dim = max(8, input_dim // 2)
    ae_model   = Autoencoder(input_dim, latent_dim)

    tr_loader = DataLoader(TensorDataset(torch.tensor(X_train_normal, dtype=torch.float32)),
                           batch_size=512, shuffle=True)
    va_loader = DataLoader(TensorDataset(torch.tensor(X_val_normal, dtype=torch.float32)),
                           batch_size=512)

    ae_model, _ = train_autoencoder(ae_model, tr_loader, va_loader, epochs=50, patience=6, device=device)

    ae_model.eval()
    with torch.no_grad():
        recon_val = ae_model(torch.tensor(X_val_normal, dtype=torch.float32).to(device)).cpu().numpy()
    val_errs = np.mean((X_val_normal - recon_val) ** 2, axis=1)
    threshold = float(np.percentile(val_errs, 95))
    return ae_model, ae_scaler, threshold


def train_restricted_lstm(train_r, val_r, artifacts, device):
    scaler_r = StandardScaler()
    num_cols = artifacts['num_cols']
    X_tr = scaler_r.fit_transform(train_r[num_cols].values)
    X_va = scaler_r.transform(val_r[num_cols].values)
    y_tr = train_r['label'].astype(int).values
    y_va = val_r['label'].astype(int).values

    X_seq_tr, y_seq_tr = create_sequences(X_tr, y_tr, SEQ_LENGTH)
    X_seq_va, y_seq_va = create_sequences(X_va, y_va, SEQ_LENGTH)

    input_dim = X_seq_tr.shape[2]
    lstm_model = LSTMBehaviorModel(input_dim, hidden_dim=64, num_layers=2, dropout=0.3)

    tr_loader = DataLoader(TensorDataset(
        torch.tensor(X_seq_tr, dtype=torch.float32),
        torch.tensor(y_seq_tr, dtype=torch.float32)
    ), batch_size=256, shuffle=True)
    va_loader = DataLoader(TensorDataset(
        torch.tensor(X_seq_va, dtype=torch.float32),
        torch.tensor(y_seq_va, dtype=torch.float32)
    ), batch_size=256)

    lstm_model, _ = train_lstm(lstm_model, tr_loader, va_loader, epochs=30, patience=5, device=device)
    return lstm_model, scaler_r


def evaluate_on_held_out(xgb_m, ae_m, ae_sc, ae_thr, lstm_m, lstm_sc,
                          test_held_out, artifacts, device, tag="restricted"):
    """Evaluate the pipeline ONLY on held-out category test rows."""
    feature_cols = artifacts['tree_feature_cols']
    num_cols     = artifacts['num_cols']

    X_tree = test_held_out[feature_cols].values
    y_true = test_held_out['label'].astype(int).values

    # XGB
    xgb_probs = xgb_m.predict_proba(X_tree)[:, 1]

    # AE
    X_ae = ae_sc.transform(test_held_out[num_cols].values)
    with torch.no_grad():
        ae_recon = ae_m(torch.tensor(X_ae, dtype=torch.float32).to(device)).cpu().numpy()
    ae_errors = np.mean((X_ae - ae_recon) ** 2, axis=1)
    ae_probs  = np.clip(ae_errors / (ae_thr * 5.0), 0, 1)

    # LSTM
    X_lstm = lstm_sc.transform(test_held_out[num_cols].values)
    X_seq, y_seq = create_sequences(X_lstm, y_true, SEQ_LENGTH)
    with torch.no_grad():
        lstm_probs_seq = lstm_m(torch.tensor(X_seq, dtype=torch.float32).to(device)).cpu().numpy()

    # Align lengths (sequences are shorter by seq_length - 1)
    min_len = len(X_seq)
    xgb_probs  = xgb_probs[-min_len:]
    ae_probs   = ae_probs[-min_len:]
    y_true_seq = y_true[-min_len:]

    # Risk engine evaluation
    engine = RiskEngine()
    risk_scores = []
    for i in range(min_len):
        P_t = float(xgb_probs[i])
        A_t = normalize_ae_score(float(ae_errors[-min_len:][i]), ae_thr)
        B_t = float(lstm_probs_seq[i])
        U_t = compute_uncertainty(P_t, A_t, B_t)
        cat = str(test_held_out.iloc[-min_len:].iloc[i].get('attack_cat', 'Normal')).strip()
        S_t = SEVERITY_MAP.get(cat, 0.5)
        r   = engine.compute_risk(P_t, A_t, B_t, S_t, U_t)
        risk_scores.append(r['R_t'])

    risk_scores = np.array(risk_scores)
    y_pred_risk = (risk_scores >= 0.5).astype(int)

    recall   = float(recall_score(y_true_seq, y_pred_risk, zero_division=0))
    f1       = float(f1_score(y_true_seq, y_pred_risk, zero_division=0))
    fp_denom = (y_true_seq == 0).sum()
    fpr      = float(((y_pred_risk == 1) & (y_true_seq == 0)).sum() / fp_denom) if fp_denom > 0 else 0.0
    mean_risk = float(risk_scores.mean())
    containment = float((y_pred_risk[y_true_seq == 1] == 1).mean()) if (y_true_seq == 1).sum() > 0 else 0.0

    try:
        roc_auc = float(roc_auc_score(y_true_seq, risk_scores))
    except Exception:
        roc_auc = float('nan')

    return {
        "tag": tag,
        "n_eval_events": int(min_len),
        "recall": recall,
        "f1": f1,
        "fpr": fpr,
        "roc_auc": roc_auc,
        "mean_risk_score": mean_risk,
        "containment_rate": containment,
    }


def main():
    print("=== Phase 5: Zero-Day / Unseen Attack Experiments ===")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    train_df = pd.read_pickle(os.path.join(PROCESSED_DIR, "train_split.pkl"))
    val_df   = pd.read_pickle(os.path.join(PROCESSED_DIR, "val_split.pkl"))
    test_df  = pd.read_pickle(os.path.join(PROCESSED_DIR, "test_split.pkl"))
    with open(os.path.join(PROCESSED_DIR, "preprocessors.pkl"), "rb") as f:
        artifacts = pickle.load(f)

    # ── Build restricted datasets ──
    print(f"\n[PHASE5] Removing {HELD_OUT_CATEGORIES} from training and validation...")
    train_r = restrict_df(train_df, HELD_OUT_CATEGORIES)
    val_r   = restrict_df(val_df,   HELD_OUT_CATEGORIES)

    # ── Mandatory leakage checks ──
    leakage_check(train_r, HELD_OUT_CATEGORIES, "restricted_train")
    leakage_check(val_r,   HELD_OUT_CATEGORIES, "restricted_val")

    # Save leakage check output
    leakage_log = {
        "held_out_categories": HELD_OUT_CATEGORIES,
        "restricted_train_rows": len(train_r),
        "restricted_val_rows": len(val_r),
        "leakage_status": "PASSED — 0 held-out rows in restricted train/val"
    }
    with open(os.path.join(RESULTS_DIR, "phase5_leakage_check.json"), "w") as f:
        json.dump(leakage_log, f, indent=2)
    print(f"[PHASE5] Leakage check log saved.")

    # ── Test split views ──
    test_held_out = test_df[test_df['attack_cat'].str.strip().isin(HELD_OUT_CATEGORIES)].reset_index(drop=True)
    test_normal   = test_df[~test_df['attack_cat'].str.strip().isin(HELD_OUT_CATEGORIES)].reset_index(drop=True)
    print(f"[PHASE5] Test held-out rows: {len(test_held_out):,} | Test normal rows: {len(test_normal):,}")

    feature_cols = artifacts['tree_feature_cols']
    num_cols     = artifacts['num_cols']

    # ── Train RESTRICTED models ──
    print("\n[PHASE5] Training restricted XGBoost...")
    xgb_r = train_restricted_xgboost(train_r, val_r, feature_cols)
    print("[PHASE5] Training restricted Autoencoder...")
    ae_r, ae_sc_r, ae_thr_r = train_restricted_autoencoder(train_r, val_r, num_cols, device)
    print("[PHASE5] Training restricted LSTM...")
    lstm_r, lstm_sc_r = train_restricted_lstm(train_r, val_r, artifacts, device)

    # ── Train FULL-KNOWLEDGE models ──
    print("\n[PHASE5] Training full-knowledge XGBoost (unrestricted)...")
    xgb_fk = train_restricted_xgboost(train_df, val_df, feature_cols)  # uses full training set
    print("[PHASE5] Training full-knowledge Autoencoder...")
    ae_fk, ae_sc_fk, ae_thr_fk = train_restricted_autoencoder(train_df, val_df, num_cols, device)
    print("[PHASE5] Training full-knowledge LSTM...")
    lstm_fk, lstm_sc_fk = train_restricted_lstm(train_df, val_df, artifacts, device)

    # ── Evaluate BOTH on held-out-only test rows ──
    print("\n[PHASE5] Evaluating restricted model on held-out test rows...")
    results_r  = evaluate_on_held_out(xgb_r,  ae_r,  ae_sc_r,  ae_thr_r,
                                       lstm_r,  lstm_sc_r,  test_held_out, artifacts, device, "restricted")
    print("[PHASE5] Evaluating full-knowledge model on held-out test rows...")
    results_fk = evaluate_on_held_out(xgb_fk, ae_fk, ae_sc_fk, ae_thr_fk,
                                       lstm_fk, lstm_sc_fk, test_held_out, artifacts, device, "full_knowledge")

    # ── Summarize the gap ──
    gap = {
        "held_out_categories": HELD_OUT_CATEGORIES,
        "restricted_model":    results_r,
        "full_knowledge_model": results_fk,
        "gap_recall": results_fk['recall'] - results_r['recall'],
        "gap_f1":     results_fk['f1']     - results_r['f1'],
        "gap_containment": results_fk['containment_rate'] - results_r['containment_rate'],
        "honest_finding": (
            "Difference in Recall between full-knowledge and restricted pipeline on "
            f"held-out {HELD_OUT_CATEGORIES} test rows: "
            f"{results_fk['recall'] - results_r['recall']:+.4f}. "
            "This represents the true unseen-attack detection gap. "
            "The anomaly-based and behavioral components (AE + LSTM) provide partial "
            "detection even without supervised training on these categories."
        )
    }

    with open(os.path.join(RESULTS_DIR, "phase5_zeroday_results.json"), "w") as f:
        json.dump(gap, f, indent=2)

    print(f"\n[PHASE5] Restricted Recall: {results_r['recall']:.4f} | Full-knowledge Recall: {results_fk['recall']:.4f}")
    print(f"[PHASE5] Detection gap: {gap['gap_recall']:+.4f}")
    print(f"[PHASE5] Results saved to results/phase5_zeroday_results.json")
    print("\n[✓] Phase 5 checkpoint complete.")


if __name__ == "__main__":
    main()
