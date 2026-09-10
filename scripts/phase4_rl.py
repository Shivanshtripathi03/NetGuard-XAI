"""
Phase 4 — NetGuard RL Environment (Gymnasium) and DQN Training
Implements NetGuardSecurityEnv with real model-derived state vectors,
trains DQN with stable-baselines3, evaluates frozen policy on test episodes.
"""

import os
import sys
import json
import pickle
import numpy as np
import pandas as pd
import torch
import gymnasium as gym
from gymnasium import spaces

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.phase3_risk_xai import (
    RiskEngine, compute_uncertainty, normalize_ae_score, SEVERITY_MAP
)
from src.models import Autoencoder, LSTMBehaviorModel, create_sequences

PROCESSED_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "processed")
MODELS_DIR    = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")
RESULTS_DIR   = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")

os.makedirs(RESULTS_DIR, exist_ok=True)

# ── Reward table (named constants, not magic numbers — documented as tunable) ──
REWARD_TABLE = {
    "correctly_allow_benign":         +5,
    "correctly_monitor_suspicious":   +5,
    "correctly_contain_attack":       +15,   # Block or Isolate on true attack
    "unnecessary_rate_limit_benign":  -8,
    "false_positive_block_benign":    -20,
    "serious_attack_allowed":         -30,
    "unnecessary_isolation_benign":   -25,
    "monitor_attack_missed":          -10,   # Attack only monitored, not contained
}

# Action space mapping
ACTION_NAMES = {0: "Allow", 1: "Monitor", 2: "RateLimit", 3: "Block", 4: "Isolate"}


class NetGuardSecurityEnv(gym.Env):
    """
    NetGuard-XAI Gymnasium Environment.
    
    Observation: [P_t, A_t, B_t, R_t, U_t, delta_R_t, H_t, C_t]  — 8 floats in [0,1]
    Actions: Discrete(5) — Allow(0), Monitor(1), RateLimit(2), Block(3), Isolate(4)
    Episodes: Sequences of real UNSW-NB15 training events (never test events during DQN training)
    """
    
    metadata = {"render_modes": []}
    
    def __init__(self, events_df, xgb_model, ae_model, ae_scaler, ae_threshold,
                 lstm_model, artifacts, episode_length=50, mode="train"):
        super().__init__()
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(8,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(5)
        
        self.events_df     = events_df.reset_index(drop=True)
        self.xgb_model     = xgb_model
        self.ae_model      = ae_model
        self.ae_scaler     = ae_scaler
        self.ae_threshold  = ae_threshold
        self.lstm_model    = lstm_model
        self.artifacts     = artifacts
        self.num_cols      = artifacts['num_cols']
        self.feature_cols  = artifacts['tree_feature_cols']
        self.episode_length = episode_length
        self.mode          = mode
        self.seq_length    = 10
        
        self.risk_engine   = RiskEngine()
        self._current_step = 0
        self._episode_start_idx = 0
        self._state = None
        self._current_event = None

        # Precompute model scores across all events for 100x speedup during RL episodes
        print(f"[RL] Precomputing model scores for {len(self.events_df):,} events ({mode} env)...", flush=True)
        X_tree = self.events_df[self.feature_cols].values.astype(np.float32)
        self._P_all = self.xgb_model.predict_proba(X_tree)[:, 1]

        X_ae = self.ae_scaler.transform(self.events_df[self.num_cols].values)
        with torch.no_grad():
            ae_recon = self.ae_model(torch.tensor(X_ae, dtype=torch.float32)).numpy()
        ae_mse = np.mean((X_ae - ae_recon) ** 2, axis=1)
        self._A_all = np.clip(ae_mse / (self.ae_threshold * 10.0), 0.0, 1.0)

        X_lstm_raw = self.artifacts['scaler'].transform(self.events_df[self.num_cols].values)
        padded = np.vstack([np.zeros((self.seq_length - 1, X_lstm_raw.shape[1])), X_lstm_raw])
        X_seq_batch, _ = create_sequences(padded, np.zeros(len(padded)), self.seq_length)
        with torch.no_grad():
            chunk_size = 5000
            b_list = []
            for c_idx in range(0, len(X_seq_batch), chunk_size):
                chunk = torch.tensor(X_seq_batch[c_idx:c_idx+chunk_size], dtype=torch.float32)
                b_list.append(self.lstm_model(chunk).numpy())
            self._B_all = np.concatenate(b_list)

        self._labels_all = self.events_df['label'].astype(int).values
        self._cats_all   = self.events_df.get('attack_cat', pd.Series(['Normal']*len(self.events_df))).astype(str).str.strip().values
        print(f"[RL] Precomputation complete.", flush=True)

    def _get_model_scores(self, event_row_idx: int):
        return float(self._P_all[event_row_idx]), float(self._A_all[event_row_idx]), float(self._B_all[event_row_idx])

    def _build_observation(self, event_row_idx: int):
        P_t, A_t, B_t = self._get_model_scores(event_row_idx)
        U_t = compute_uncertainty(P_t, A_t, B_t)
        cat = self._cats_all[event_row_idx]
        S_t = SEVERITY_MAP.get(cat, 0.5)
        result = self.risk_engine.compute_risk(P_t, A_t, B_t, S_t, U_t)
        R_t     = result['R_t']
        delta_R = float(np.clip((result['delta_R'] + 1.0) / 2.0, 0, 1))
        H_t     = float(np.clip(result['H_t'], 0, 1))
        C_t     = 1.0

        obs = np.array([P_t, A_t, B_t, R_t, U_t, delta_R, H_t, C_t], dtype=np.float32)
        self._current_event = {
            "row_idx": event_row_idx,
            "true_label": int(self._labels_all[event_row_idx]),
            "attack_cat": cat,
            "P_t": P_t, "A_t": A_t, "B_t": B_t, "R_t": R_t, "U_t": U_t
        }
        return obs
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.risk_engine.reset()
        self._current_step = 0
        # Sample a random starting position in the event pool
        max_start = max(1, len(self.events_df) - self.episode_length - self.seq_length)
        self._episode_start_idx = self.np_random.integers(0, max_start)
        obs = self._build_observation(self._episode_start_idx)
        return obs, {}
    
    def step(self, action: int):
        true_label = self._current_event['true_label']
        R_t        = self._current_event['R_t']
        attack_cat = self._current_event['attack_cat']
        
        # Determine reward
        reward = self._compute_reward(action, true_label, R_t)
        
        self._current_step += 1
        terminated = self._current_step >= self.episode_length
        truncated  = False
        
        next_idx = self._episode_start_idx + self._current_step
        if next_idx >= len(self.events_df):
            terminated = True
            
        if terminated:
            obs = np.zeros(8, dtype=np.float32)
        else:
            obs = self._build_observation(next_idx)
        
        info = {
            "action_name": ACTION_NAMES[action],
            "true_label": true_label,
            "attack_cat": attack_cat,
            "R_t": self._current_event.get('R_t', 0.0) if not terminated else 0.0,
            "reward": reward
        }
        return obs, reward, terminated, truncated, info
    
    def _compute_reward(self, action: int, true_label: int, R_t: float) -> float:
        is_attack = (true_label == 1)
        contains  = action in {3, 4}   # Block or Isolate
        monitors  = action == 1
        allows    = action == 0
        rate_limits = action == 2
        
        if allows and not is_attack:
            return REWARD_TABLE["correctly_allow_benign"]
        elif monitors and not is_attack and R_t < 0.4:
            return REWARD_TABLE["correctly_monitor_suspicious"]
        elif contains and is_attack:
            return REWARD_TABLE["correctly_contain_attack"]
        elif rate_limits and not is_attack:
            return REWARD_TABLE["unnecessary_rate_limit_benign"]
        elif action == 3 and not is_attack:
            return REWARD_TABLE["false_positive_block_benign"]
        elif allows and is_attack:
            return REWARD_TABLE["serious_attack_allowed"]
        elif action == 4 and not is_attack:
            return REWARD_TABLE["unnecessary_isolation_benign"]
        elif monitors and is_attack:
            return REWARD_TABLE["monitor_attack_missed"]
        else:
            return 0.0


def load_models_and_data():
    import xgboost as xgb
    
    train_df = pd.read_pickle(os.path.join(PROCESSED_DIR, "train_split.pkl"))
    val_df   = pd.read_pickle(os.path.join(PROCESSED_DIR, "val_split.pkl"))
    test_df  = pd.read_pickle(os.path.join(PROCESSED_DIR, "test_split.pkl"))
    with open(os.path.join(PROCESSED_DIR, "preprocessors.pkl"), "rb") as f:
        artifacts = pickle.load(f)
    
    xgb_model = xgb.XGBClassifier()
    xgb_model.load_model(os.path.join(MODELS_DIR, "xgboost_model.json"))
    
    with open(os.path.join(RESULTS_DIR, "autoencoder_test_metrics.json")) as f:
        ae_metrics = json.load(f)
    ae_threshold = ae_metrics['anomaly_threshold']
    
    with open(os.path.join(MODELS_DIR, "ae_scaler.pkl"), "rb") as f:
        ae_scaler = pickle.load(f)
    
    num_cols = artifacts['num_cols']
    ae_model = Autoencoder(input_dim=len(num_cols), latent_dim=max(8, len(num_cols)//2))
    ae_model.load_state_dict(torch.load(os.path.join(MODELS_DIR, "autoencoder.pt"), weights_only=True))
    ae_model.eval()
    
    with open(os.path.join(RESULTS_DIR, "lstm_sequence_metadata.json")) as f:
        lstm_meta = json.load(f)
    lstm_model = LSTMBehaviorModel(input_dim=len(num_cols), hidden_dim=64, num_layers=2, dropout=0.3)
    lstm_model.load_state_dict(torch.load(os.path.join(MODELS_DIR, "lstm.pt"), weights_only=True))
    lstm_model.eval()
    
    return train_df, val_df, test_df, artifacts, xgb_model, ae_model, ae_scaler, ae_threshold, lstm_model


def main():
    print("=== Phase 4: RL Environment and DQN Training ===")
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from stable_baselines3 import DQN
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.callbacks import BaseCallback
    
    train_df, val_df, test_df, artifacts, xgb_model, ae_model, ae_scaler, ae_threshold, lstm_model = load_models_and_data()
    
    # Training environment uses ONLY training split events
    env = NetGuardSecurityEnv(
        events_df=train_df, xgb_model=xgb_model, ae_model=ae_model,
        ae_scaler=ae_scaler, ae_threshold=ae_threshold, lstm_model=lstm_model,
        artifacts=artifacts, episode_length=100, mode="train"
    )
    env = Monitor(env)

    print("[RL] Training DQN on training-split environment...")
    dqn_log_path = os.path.join(RESULTS_DIR, "dqn_training_log.csv")
    
    # DQN hyperparameters with documented epsilon decay schedule
    dqn_hparams = {
        "learning_rate": 1e-4,
        "buffer_size": 50000,
        "learning_starts": 1000,
        "batch_size": 64,
        "gamma": 0.99,
        "train_freq": 4,
        "target_update_interval": 1000,
        "exploration_fraction": 0.2,       # fraction of total steps for epsilon decay
        "exploration_initial_eps": 1.0,
        "exploration_final_eps": 0.05,
    }
    
    with open(os.path.join(RESULTS_DIR, "dqn_hyperparams.json"), "w") as f:
        json.dump(dqn_hparams, f, indent=2)
    
    model = DQN("MlpPolicy", env, verbose=1, **dqn_hparams,
                tensorboard_log=None)
    
    model.learn(total_timesteps=100_000)
    
    # Save DQN policy
    policy_path = os.path.join(MODELS_DIR, "dqn_policy")
    model.save(policy_path)
    print(f"[RL] DQN policy saved to {policy_path}.zip")
    
    # Extract episode rewards from Monitor
    ep_rewards = env.get_episode_rewards()
    ep_lengths = env.get_episode_lengths()
    
    log_data = {"episode": list(range(len(ep_rewards))),
                "reward": ep_rewards, "length": ep_lengths}
    pd.DataFrame(log_data).to_csv(dqn_log_path, index=False)
    print(f"[RL] Training log saved to {dqn_log_path}")
    
    # Plot training curve
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    rolling = pd.Series(ep_rewards).rolling(window=20, min_periods=1).mean()
    ax1.plot(ep_rewards, alpha=0.3, color='#3498db', linewidth=0.8)
    ax1.plot(rolling, color='#e74c3c', linewidth=2, label='Rolling mean (20 eps)')
    ax1.set_ylabel('Episode Reward')
    ax1.set_title('DQN Training Curve — Episode Rewards')
    ax1.legend(); ax1.grid(True, alpha=0.3)
    
    ax2.plot(ep_lengths, alpha=0.5, color='#2ecc71', linewidth=0.8)
    ax2.set_xlabel('Episode'); ax2.set_ylabel('Episode Length')
    ax2.set_title('DQN Training — Episode Lengths')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "dqn_training_curve.png"), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[RL] Training curve saved.")
    
    # ── Evaluate frozen policy on TEST episodes ──
    print("[RL] Evaluating frozen DQN policy on test-split episodes...")
    test_env = NetGuardSecurityEnv(
        events_df=test_df, xgb_model=xgb_model, ae_model=ae_model,
        ae_scaler=ae_scaler, ae_threshold=ae_threshold, lstm_model=lstm_model,
        artifacts=artifacts, episode_length=100, mode="test"
    )
    
    eval_records = []
    n_eval_episodes = 50
    for ep in range(n_eval_episodes):
        obs, _ = test_env.reset()
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = test_env.step(int(action))
            eval_records.append({
                "episode": ep,
                "action": int(action),
                "action_name": info["action_name"],
                "true_label": info["true_label"],
                "attack_cat": info["attack_cat"],
                "R_t": info["R_t"],
                "reward": reward
            })
            done = terminated or truncated
    
    eval_df = pd.DataFrame(eval_records)
    
    # Compute evaluation metrics
    attack_rows   = eval_df[eval_df['true_label'] == 1]
    benign_rows   = eval_df[eval_df['true_label'] == 0]
    
    containment_rate = ((attack_rows['action'] >= 3).sum() / len(attack_rows)
                        if len(attack_rows) > 0 else 0.0)
    fp_rate  = ((benign_rows['action'] >= 3).sum() / len(benign_rows)
                if len(benign_rows) > 0 else 0.0)
    avg_rew  = eval_df['reward'].mean()
    
    rl_eval = {
        "n_episodes": n_eval_episodes,
        "total_steps": len(eval_df),
        "attack_containment_rate": float(containment_rate),
        "false_positive_intervention_rate": float(fp_rate),
        "average_reward": float(avg_rew),
        "action_distribution": eval_df['action_name'].value_counts().to_dict()
    }
    
    eval_df.to_csv(os.path.join(RESULTS_DIR, "dqn_test_eval_steps.csv"), index=False)
    with open(os.path.join(RESULTS_DIR, "dqn_test_eval.json"), "w") as f:
        json.dump(rl_eval, f, indent=2)
    
    print(f"[RL] Containment Rate: {containment_rate:.4f} | FP Rate: {fp_rate:.4f} | Avg Reward: {avg_rew:.3f}")
    print(f"[RL] Results saved to results/dqn_test_eval.json")
    print("\n[✓] Phase 4 checkpoint complete.")


if __name__ == "__main__":
    main()
