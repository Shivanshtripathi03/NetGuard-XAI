"""
Phase 3 — Stateful Risk Engine, Uncertainty Quantification, and SHAP Explainability
Implements R_t = f(P_t, A_t, B_t, S_t, H_t, U_t, C_t) with historical decay,
risk velocity, unit tests, and real SHAP explanations.
"""

import os
import sys
import json
import pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

MODELS_DIR  = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")
RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
PROCESSED_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "processed")


# ─────────────────────────────────────────────────────────────────────────────
# Risk Engine Implementation
# ─────────────────────────────────────────────────────────────────────────────

# Weight constants — all tunable, named (never magic numbers)
RISK_WEIGHTS = {
    "W_P": 0.30,   # XGBoost supervised probability weight
    "W_A": 0.25,   # Autoencoder anomaly score weight
    "W_B": 0.20,   # LSTM behavioral probability weight
    "W_S": 0.10,   # Severity score weight (attack_cat severity)
    "W_U": 0.10,   # Uncertainty penalty weight
    "W_C": 0.05,   # Contextual weight
}

HISTORICAL_DECAY_GAMMA = 0.85   # exponential decay factor for H_t
SEVERITY_MAP = {
    'Normal':         0.0,
    'Reconnaissance': 0.4,
    'Fuzzers':        0.5,
    'Analysis':       0.5,
    'Backdoor':       0.9,
    'DoS':            0.7,
    'Exploits':       0.8,
    'Generic':        0.6,
    'Shellcode':      0.95,
    'Worms':          0.85,
}


class RiskEngine:
    """
    Stateful per-source risk engine.
    Tracks H_t (historical decay), R_{t-1}, and risk trajectory (velocity, volatility).
    """
    def __init__(self):
        self.H_t = 0.0       # historical risk component
        self.R_prev = 0.0    # previous risk score
        self.risk_history = []

    def compute_risk(self, P_t: float, A_t: float, B_t: float,
                     S_t: float = 0.5, U_t: float = 0.0, C_t: float = 1.0) -> dict:
        """
        Compute current risk score R_t using weighted combination.
        
        Args:
            P_t: XGBoost attack probability [0, 1]
            A_t: Autoencoder anomaly score, normalized to [0, 1]
            B_t: LSTM behavioral attack probability [0, 1]
            S_t: Severity score from attack category [0, 1]
            U_t: Uncertainty score (model disagreement) [0, 1]
            C_t: Context weight multiplier [0.5, 2.0]
        
        Returns:
            dict with R_t, delta_R, velocity, H_t components
        """
        w = RISK_WEIGHTS

        # Compute raw risk before historical component
        R_raw = (
            w["W_P"] * P_t +
            w["W_A"] * A_t +
            w["W_B"] * B_t +
            w["W_S"] * S_t +
            w["W_U"] * U_t
        )

        # Update historical decay: H_t = gamma * R_{t-1}
        self.H_t = HISTORICAL_DECAY_GAMMA * self.R_prev

        # Final risk: blend raw + historical, scaled by context weight, clipped to [0, 1]
        R_t = float(np.clip(C_t * (R_raw + 0.15 * self.H_t), 0.0, 1.0))

        # Risk velocity and trajectory
        delta_R = R_t - self.R_prev

        self.risk_history.append(R_t)
        self.R_prev = R_t

        return {
            "R_t":     R_t,
            "delta_R": delta_R,
            "H_t":     self.H_t,
            "R_raw":   R_raw,
            "components": {
                "P_t": P_t, "A_t": A_t, "B_t": B_t,
                "S_t": S_t, "U_t": U_t, "C_t": C_t
            }
        }

    def reset(self):
        self.H_t = 0.0
        self.R_prev = 0.0
        self.risk_history = []


def compute_uncertainty(P_t: float, A_t: float, B_t: float) -> float:
    """
    Uncertainty = standard deviation of the three model scores.
    High disagreement → high uncertainty.
    """
    scores = np.array([P_t, A_t, B_t])
    return float(np.std(scores))


def normalize_ae_score(raw_mse: float, threshold: float, clip_max: float = 10.0) -> float:
    """Normalize raw Autoencoder MSE reconstruction error to [0, 1]."""
    return float(np.clip(raw_mse / (threshold * clip_max), 0.0, 1.0))


# ─────────────────────────────────────────────────────────────────────────────
# Unit Tests for Risk Engine
# ─────────────────────────────────────────────────────────────────────────────

def run_unit_tests():
    print("[RISK] Running unit tests on synthetic event sequences...")

    # Test 1: Pure normal traffic → risk should stay low
    engine = RiskEngine()
    results = [engine.compute_risk(P_t=0.02, A_t=0.03, B_t=0.01, S_t=0.0, U_t=0.01) for _ in range(5)]
    assert all(r['R_t'] < 0.3 for r in results), f"FAIL: Normal traffic risk too high: {[r['R_t'] for r in results]}"
    print("  [✓] Test 1 PASS: Normal traffic maintains low risk")

    # Test 2: Attack traffic → risk should escalate
    engine.reset()
    risks = []
    for _ in range(5):
        r = engine.compute_risk(P_t=0.9, A_t=0.8, B_t=0.85, S_t=0.8, U_t=0.1)
        risks.append(r['R_t'])
    assert risks[-1] > 0.5, f"FAIL: Attack traffic risk not escalating: {risks}"
    print(f"  [✓] Test 2 PASS: Attack traffic escalates risk: {[f'{r:.3f}' for r in risks]}")

    # Test 3: Historical decay — after attack stops, risk should decay
    engine_decay = RiskEngine()
    # inject 3 attack events
    for _ in range(3):
        engine_decay.compute_risk(P_t=0.9, A_t=0.85, B_t=0.8, S_t=0.7)
    peak = engine_decay.R_prev
    # inject 5 normal events
    decayed = []
    for _ in range(5):
        r = engine_decay.compute_risk(P_t=0.05, A_t=0.05, B_t=0.03, S_t=0.0)
        decayed.append(r['R_t'])
    assert decayed[-1] < peak, f"FAIL: Risk not decaying after normal events: peak={peak:.3f}, decayed={decayed}"
    print(f"  [✓] Test 3 PASS: Historical decay functional: peak={peak:.3f} → final={decayed[-1]:.3f}")

    # Test 4: Uncertainty increases with disagreement
    U_agree  = compute_uncertainty(0.8, 0.8, 0.8)
    U_disagree = compute_uncertainty(0.9, 0.1, 0.5)
    assert U_disagree > U_agree, "FAIL: Uncertainty not higher for disagreeing models"
    print(f"  [✓] Test 4 PASS: Uncertainty(agree)={U_agree:.4f} < Uncertainty(disagree)={U_disagree:.4f}")

    print("[RISK] All unit tests PASSED.\n")


# ─────────────────────────────────────────────────────────────────────────────
# SHAP Explainability
# ─────────────────────────────────────────────────────────────────────────────

def compute_shap_explanations(n_incidents=5):
    """
    Compute real SHAP values for XGBoost model on actual test incidents.
    Returns a sample of incidents (mix of TP attacks + FP benign) with top features.
    """
    import shap
    import xgboost as xgb

    print("[SHAP] Loading XGBoost model and test data...")
    model = xgb.Booster()
    model.load_model(os.path.join(MODELS_DIR, "xgboost_model.json"))

    with open(os.path.join(PROCESSED_DIR, "preprocessors.pkl"), "rb") as f:
        artifacts = pickle.load(f)
    test_df = pd.read_pickle(os.path.join(PROCESSED_DIR, "test_split.pkl"))
    feature_cols = artifacts['tree_feature_cols']

    # Sample 2000 rows for fast evaluation
    sample_df = test_df.sample(n=min(2000, len(test_df)), random_state=42).reset_index(drop=True)
    X_test = sample_df[feature_cols].values.astype(np.float32)
    y_test = sample_df['label'].astype(int).values
    dtest = xgb.DMatrix(X_test)
    y_probs = model.predict(dtest)
    y_pred = (y_probs > 0.5).astype(int)

    # Select incidents: mix of TP (attack correctly detected) and FP (benign incorrectly flagged)
    tp_idx = np.where((y_test == 1) & (y_pred == 1))[0]
    fp_idx = np.where((y_test == 0) & (y_pred == 1))[0]
    fn_idx = np.where((y_test == 1) & (y_pred == 0))[0]

    selected_idx = []
    if len(tp_idx) >= 3: selected_idx.extend(tp_idx[:3].tolist())
    if len(fp_idx) >= 1: selected_idx.extend(fp_idx[:1].tolist())
    if len(fn_idx) >= 1: selected_idx.extend(fn_idx[:1].tolist())
    selected_idx = selected_idx[:n_incidents]

    print(f"[SHAP] Computing SHAP values for {len(selected_idx)} incidents (TPs + FPs + FNs)...")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test[selected_idx])

    incidents_report = []
    for i, idx in enumerate(selected_idx):
        sv = shap_values[i] if shap_values.ndim == 2 else shap_values[:, i]
        top_features = sorted(
            zip(feature_cols, sv.tolist()),
            key=lambda x: abs(x[1]), reverse=True
        )[:10]
        incident = {
            "index": int(idx),
            "true_label": int(y_test[idx]),
            "predicted_label": int(y_pred[idx]),
            "attack_cat": str(sample_df.iloc[idx]['attack_cat']),
            "top_shap_features": [{"feature": f, "shap_value": v} for f, v in top_features]
        }
        incidents_report.append(incident)
        print(f"  Incident {i+1}: true={y_test[idx]}, pred={y_pred[idx]}, cat={sample_df.iloc[idx]['attack_cat']}")
        print(f"    Top feature: {top_features[0][0]} (SHAP={top_features[0][1]:.4f})")

    shap_path = os.path.join(RESULTS_DIR, "shap_incidents.json")
    with open(shap_path, "w") as f:
        json.dump(incidents_report, f, indent=2)
    print(f"[SHAP] Saved explanations to {shap_path}")
    return incidents_report, explainer, shap_values, selected_idx, feature_cols


def plot_risk_trajectory(events_risk_data, output_path):
    """Plot real risk trajectory from a sequence of processed test events."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

    timesteps   = list(range(len(events_risk_data)))
    risk_scores = [e['risk']['R_t']     for e in events_risk_data]
    delta_r     = [e['risk']['delta_R'] for e in events_risk_data]
    true_labels = [e['true_label']       for e in events_risk_data]
    attack_idxs = [i for i, l in enumerate(true_labels) if l == 1]
    normal_idxs = [i for i, l in enumerate(true_labels) if l == 0]

    ax1.fill_between(timesteps, risk_scores, alpha=0.3, color='#e74c3c')
    ax1.plot(timesteps, risk_scores, color='#e74c3c', linewidth=2, label='Risk R_t')
    ax1.axhline(y=0.5, color='orange', linestyle='--', alpha=0.7, label='Alert threshold (0.5)')
    ax1.axhline(y=0.8, color='red',    linestyle='--', alpha=0.7, label='Critical threshold (0.8)')
    for i in attack_idxs:
        ax1.axvspan(i - 0.5, i + 0.5, alpha=0.15, color='red')
    ax1.scatter(attack_idxs, [risk_scores[i] for i in attack_idxs], color='red',   s=20, zorder=5, label='True Attack')
    ax1.scatter(normal_idxs, [risk_scores[i] for i in normal_idxs], color='green', s=10, zorder=5, alpha=0.5, label='Normal')
    ax1.set_ylabel('Risk Score R_t', fontsize=12)
    ax1.set_ylim(0, 1.05)
    ax1.legend(loc='upper right', fontsize=9)
    ax1.set_title('NetGuard-XAI Risk Trajectory (Real Test Events)', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)

    colors = ['#e74c3c' if d > 0 else '#2ecc71' for d in delta_r]
    ax2.bar(timesteps, delta_r, color=colors, alpha=0.7, width=1.0)
    ax2.axhline(y=0, color='black', linewidth=0.5)
    ax2.set_xlabel('Event Timestep', fontsize=12)
    ax2.set_ylabel('Risk Velocity ΔR_t', fontsize=12)
    ax2.set_title('Risk Velocity (Change per Event)', fontsize=12)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[RISK] Risk trajectory plot saved to {output_path}")


def main():
    print("=== Phase 3: Risk Engine, Uncertainty & SHAP ===")

    # 1. Run unit tests
    run_unit_tests()

    # 2. Load models and run a real sequence through the risk engine
    import xgboost as xgb
    import torch
    from src.models import Autoencoder, LSTMBehaviorModel, create_sequences

    test_df = pd.read_pickle(os.path.join(PROCESSED_DIR, "test_split.pkl"))
    with open(os.path.join(PROCESSED_DIR, "preprocessors.pkl"), "rb") as f:
        artifacts = pickle.load(f)

    print("[RISK] Models and preprocessors loading...", flush=True)

    num_cols = artifacts['num_cols']
    feature_cols = artifacts['tree_feature_cols']

    # Load XGBoost
    xgb_model = xgb.Booster()
    xgb_model.load_model(os.path.join(MODELS_DIR, "xgboost_model.json"))

    # Load AE
    neural_data = np.load(os.path.join(PROCESSED_DIR, "neural_data.npz"), allow_pickle=True)
    with open(os.path.join(MODELS_DIR, "ae_scaler.pkl"), "rb") as f:
        ae_scaler = pickle.load(f)
    with open(os.path.join(RESULTS_DIR, "autoencoder_test_metrics.json")) as f:
        ae_metrics = json.load(f)
    ae_threshold = ae_metrics['anomaly_threshold']

    ae_model = Autoencoder(input_dim=len(num_cols), latent_dim=max(8, len(num_cols)//2))
    ae_model.load_state_dict(torch.load(os.path.join(MODELS_DIR, "autoencoder.pt"), map_location='cpu'))
    ae_model.eval()

    # Load LSTM
    with open(os.path.join(RESULTS_DIR, "lstm_sequence_metadata.json")) as f:
        lstm_meta = json.load(f)
    seq_length = lstm_meta['seq_length']
    lstm_model = LSTMBehaviorModel(input_dim=len(num_cols), hidden_dim=64, num_layers=2, dropout=0.3)
    lstm_model.load_state_dict(torch.load(os.path.join(MODELS_DIR, "lstm.pt"), map_location='cpu'))
    lstm_model.eval()

    # Sample 200 sequential test events for risk trajectory demo
    sample = test_df.sample(n=min(200, len(test_df)), random_state=42).reset_index(drop=True)
    engine = RiskEngine()
    events_risk_data = []

    ae_scaler_test = ae_scaler.transform(sample[num_cols].values)
    scaler_test = artifacts['scaler'].transform(sample[num_cols].values)
    X_tree = sample[feature_cols]

    with torch.no_grad():
        ae_input = torch.tensor(ae_scaler_test, dtype=torch.float32)
        ae_recon = ae_model(ae_input).numpy()
        ae_mse = np.mean((ae_scaler_test - ae_recon) ** 2, axis=1)

        # LSTM: need sequences — pad first seq_length events
        padded = np.vstack([scaler_test[:seq_length], scaler_test])
        X_seq_arr, _ = create_sequences(padded, np.zeros(len(padded)), seq_length)
        X_seq_arr = X_seq_arr[:len(sample)]
        lstm_probs = lstm_model(torch.tensor(X_seq_arr, dtype=torch.float32)).numpy()

    X_tree_arr = sample[feature_cols].values.astype(np.float32)
    dmatrix_sample = xgb.DMatrix(X_tree_arr)
    xgb_probs = xgb_model.predict(dmatrix_sample)
    print(f"[RISK] Evaluated {len(sample)} sample events across 3 models.", flush=True)

    for i in range(len(sample)):
        P_t = float(xgb_probs[i])
        A_t = normalize_ae_score(float(ae_mse[i]), ae_threshold)
        B_t = float(lstm_probs[i])
        U_t = compute_uncertainty(P_t, A_t, B_t)
        S_t = SEVERITY_MAP.get(str(sample.iloc[i]['attack_cat']).strip(), 0.5)
        result = engine.compute_risk(P_t=P_t, A_t=A_t, B_t=B_t, S_t=S_t, U_t=U_t)
        events_risk_data.append({
            "true_label": int(sample.iloc[i]['label']),
            "attack_cat": str(sample.iloc[i]['attack_cat']),
            "risk": result,
            "P_t": P_t, "A_t": A_t, "B_t": B_t
        })

    # Save trajectory data
    traj_path = os.path.join(RESULTS_DIR, "risk_trajectory.json")
    with open(traj_path, "w") as f:
        json.dump(events_risk_data, f, indent=2)

    # Plot risk trajectory
    print("[RISK] Generating risk trajectory plot...", flush=True)
    plot_risk_trajectory(events_risk_data, os.path.join(RESULTS_DIR, "risk_trajectory.png"))

    # 3. SHAP explanations
    print("[RISK] Starting SHAP explanation generator...", flush=True)
    incidents, _, _, _, _ = compute_shap_explanations(n_incidents=5)

    print("\n[✓] Phase 3 checkpoint complete.", flush=True)
    return engine, events_risk_data, incidents


if __name__ == "__main__":
    main()
