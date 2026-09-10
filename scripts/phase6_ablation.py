"""
Phase 6 — Comparative and Ablation Study
System A: Static Threshold, System B: Risk-based Rule, System C: Full NetGuard-XAI+RL
Ablation: XGB → +AE → +LSTM → +Risk → +SHAP → +RL
"""

import os
import sys
import json
import pickle
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from sklearn.metrics import f1_score, recall_score, precision_score, roc_auc_score
import xgboost as xgb

from src.models import Autoencoder, LSTMBehaviorModel, create_sequences
from scripts.phase3_risk_xai import RiskEngine, compute_uncertainty, normalize_ae_score, SEVERITY_MAP
from scripts.phase4_rl import NetGuardSecurityEnv, ACTION_NAMES

PROCESSED_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "processed")
MODELS_DIR    = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")
RESULTS_DIR   = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")


def load_all_models():
    from stable_baselines3 import DQN
    with open(os.path.join(PROCESSED_DIR, "preprocessors.pkl"), "rb") as f:
        artifacts = pickle.load(f)
    test_df  = pd.read_pickle(os.path.join(PROCESSED_DIR, "test_split.pkl"))
    num_cols = artifacts['num_cols']

    xgb_model = xgb.XGBClassifier()
    xgb_model.load_model(os.path.join(MODELS_DIR, "xgboost_model.json"))

    with open(os.path.join(RESULTS_DIR, "autoencoder_test_metrics.json")) as f:
        ae_metrics = json.load(f)
    ae_threshold = ae_metrics['anomaly_threshold']
    with open(os.path.join(MODELS_DIR, "ae_scaler.pkl"), "rb") as f:
        ae_scaler = pickle.load(f)

    ae_model = Autoencoder(input_dim=len(num_cols), latent_dim=max(8, len(num_cols)//2))
    ae_model.load_state_dict(torch.load(os.path.join(MODELS_DIR, "autoencoder.pt"), weights_only=True))
    ae_model.eval()

    lstm_model = LSTMBehaviorModel(input_dim=len(num_cols), hidden_dim=64, num_layers=2, dropout=0.3)
    lstm_model.load_state_dict(torch.load(os.path.join(MODELS_DIR, "lstm.pt"), weights_only=True))
    lstm_model.eval()

    dqn_model = DQN.load(os.path.join(MODELS_DIR, "dqn_policy"))
    return test_df, artifacts, xgb_model, ae_model, ae_scaler, ae_threshold, lstm_model, dqn_model


def metrics_from_arrays(y_true, y_pred, y_prob=None):
    result = {
        "recall":    float(recall_score(y_true, y_pred, zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "f1":        float(f1_score(y_true, y_pred, zero_division=0)),
        "macro_f1":  float(f1_score(y_true, y_pred, average='macro', zero_division=0)),
    }
    if y_prob is not None:
        try:
            result["roc_auc"] = float(roc_auc_score(y_true, y_prob))
        except Exception:
            result["roc_auc"] = float('nan')
    return result


def system_a_static_threshold(test_df, artifacts, xgb_model, threshold=0.5):
    """System A: plain XGBoost probability with a fixed threshold."""
    feature_cols = artifacts['tree_feature_cols']
    X = test_df[feature_cols].values
    y_true = test_df['label'].astype(int).values
    y_prob = xgb_model.predict_proba(X)[:, 1]
    y_pred = (y_prob >= threshold).astype(int)
    return metrics_from_arrays(y_true, y_pred, y_prob)


def system_b_risk_rule(test_df, artifacts, xgb_model, ae_model, ae_scaler, ae_threshold, lstm_model, device):
    """System B: Risk engine + hand-crafted rule (block if R_t > 0.5)."""
    from src.models import create_sequences
    num_cols = artifacts['num_cols']
    feature_cols = artifacts['tree_feature_cols']

    X_tree = test_df[feature_cols].values
    xgb_probs = xgb_model.predict_proba(X_tree)[:, 1]

    X_ae = ae_scaler.transform(test_df[num_cols].values)
    with torch.no_grad():
        ae_recon = ae_model(torch.tensor(X_ae, dtype=torch.float32).to(device)).cpu().numpy()
    ae_errors = np.mean((X_ae - ae_recon) ** 2, axis=1)

    scaler = artifacts['scaler']
    X_lstm_raw = scaler.transform(test_df[num_cols].values)
    y_raw = test_df['label'].astype(int).values
    X_seq, y_seq = create_sequences(X_lstm_raw, y_raw, 10)
    with torch.no_grad():
        lstm_probs = lstm_model(torch.tensor(X_seq, dtype=torch.float32).to(device)).cpu().numpy()

    n = len(X_seq)
    engine = RiskEngine()
    risk_scores = []
    for i in range(n):
        P_t = float(xgb_probs[-n:][i])
        A_t = normalize_ae_score(float(ae_errors[-n:][i]), ae_threshold)
        B_t = float(lstm_probs[i])
        S_t = SEVERITY_MAP.get(str(test_df.iloc[-n:].iloc[i]['attack_cat']).strip(), 0.5)
        U_t = compute_uncertainty(P_t, A_t, B_t)
        r = engine.compute_risk(P_t, A_t, B_t, S_t, U_t)
        risk_scores.append(r['R_t'])

    y_true = y_raw[-n:]
    y_pred = (np.array(risk_scores) >= 0.5).astype(int)
    return metrics_from_arrays(y_true, y_pred, np.array(risk_scores))


def main():
    print("=== Phase 6: Comparative and Ablation Study ===")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    test_df, artifacts, xgb_model, ae_model, ae_scaler, ae_threshold, lstm_model, dqn_model = load_all_models()

    print("\n[ABL] System A — Static Threshold (XGBoost @0.5)...")
    sys_a = system_a_static_threshold(test_df, artifacts, xgb_model)

    print("[ABL] System B — Risk-based Rule (R_t ≥ 0.5)...")
    sys_b = system_b_risk_rule(test_df, artifacts, xgb_model, ae_model, ae_scaler, ae_threshold, lstm_model, device)

    print("[ABL] System C — Full NetGuard-XAI with RL Policy...")
    rl_eval_path = os.path.join(RESULTS_DIR, "dqn_test_eval.json")
    with open(rl_eval_path) as f:
        rl_eval = json.load(f)
    sys_c = {
        "containment_rate": rl_eval['attack_containment_rate'],
        "fp_rate": rl_eval['false_positive_intervention_rate'],
        "avg_reward": rl_eval['average_reward'],
        "note": "RL policy result from Phase 4 evaluation"
    }

    comparison_table = {
        "System A (Static Threshold)": sys_a,
        "System B (Risk-Rule)":        sys_b,
        "System C (NetGuard-XAI+RL)":  sys_c,
    }

    # --- Ablation: XGB → +AE → +LSTM → +Risk → +RL ---
    feature_cols = artifacts['tree_feature_cols']
    num_cols     = artifacts['num_cols']
    y_test_full  = test_df['label'].astype(int).values
    X_tree       = test_df[feature_cols].values
    xgb_probs    = xgb_model.predict_proba(X_tree)[:, 1]

    ablation = {}

    # Step 1: XGBoost only
    ablation["1_XGBoost_only"] = metrics_from_arrays(
        y_test_full, (xgb_probs >= 0.5).astype(int), xgb_probs
    )

    # Step 2: +Autoencoder (OR fusion)
    X_ae = ae_scaler.transform(test_df[num_cols].values)
    with torch.no_grad():
        ae_recon = ae_model(torch.tensor(X_ae, dtype=torch.float32).to(device)).cpu().numpy()
    ae_mse = np.mean((X_ae - ae_recon) ** 2, axis=1)
    ae_preds = (ae_mse > ae_threshold).astype(int)
    fused_ae = ((xgb_probs >= 0.5) | (ae_preds == 1)).astype(int)
    ablation["2_XGB_plus_AE"] = metrics_from_arrays(y_test_full, fused_ae)

    # Step 3: +LSTM
    scaler = artifacts['scaler']
    X_lstm_raw = scaler.transform(test_df[num_cols].values)
    X_seq, y_seq = create_sequences(X_lstm_raw, y_test_full, 10)
    with torch.no_grad():
        lstm_probs = lstm_model(torch.tensor(X_seq, dtype=torch.float32).to(device)).cpu().numpy()
    n = len(X_seq)
    lstm_preds = (lstm_probs >= 0.5).astype(int)
    fused_all3 = (
        ((xgb_probs[-n:] >= 0.5) | (ae_preds[-n:] == 1) | (lstm_preds == 1))
    ).astype(int)
    ablation["3_XGB_AE_LSTM"] = metrics_from_arrays(y_seq, fused_all3)

    # Step 4: +Risk Engine (already computed in sys_b)
    ablation["4_plus_Risk_Engine"] = {
        "recall": sys_b["recall"], "f1": sys_b["f1"],
        "macro_f1": sys_b["macro_f1"], "roc_auc": sys_b.get("roc_auc", float('nan'))
    }

    # Step 5: +SHAP (no metric change, SHAP is explanatory only — document this)
    ablation["5_plus_SHAP"] = {
        **ablation["4_plus_Risk_Engine"],
        "note": "SHAP adds explainability per incident; detection metrics unchanged from step 4"
    }

    # Step 6: +RL
    ablation["6_full_NetGuard_XAI_RL"] = {
        "containment_rate": rl_eval['attack_containment_rate'],
        "fp_rate": rl_eval['false_positive_intervention_rate'],
        "note": "RL optimizes action policy, not raw detection — see Phase 4 for eval details"
    }

    final = {
        "system_comparison": comparison_table,
        "ablation_study": ablation
    }
    path = os.path.join(RESULTS_DIR, "phase6_ablation.json")
    with open(path, "w") as f:
        json.dump(final, f, indent=2)

    print("\n── System Comparison ──────────────────────────────────")
    for name, m in comparison_table.items():
        print(f"  {name}: {m}")
    print("\n── Ablation Study ─────────────────────────────────────")
    for step, m in ablation.items():
        print(f"  {step}: {m}")
    print(f"\n[✓] Phase 6 results saved to {path}")


if __name__ == "__main__":
    main()
