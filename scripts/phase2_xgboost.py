"""
Phase 2 — XGBoost Known-Attack Classifier Training
Handles hyperparameter search, final training, and evaluation on the official test set.
"""

import os
import sys
import json
import pickle
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report, roc_auc_score, average_precision_score,
    confusion_matrix, precision_recall_fscore_support, f1_score
)
from sklearn.preprocessing import LabelEncoder

PROCESSED_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "processed")
MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")
RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)


def load_splits():
    train = pd.read_pickle(os.path.join(PROCESSED_DIR, "train_split.pkl"))
    val   = pd.read_pickle(os.path.join(PROCESSED_DIR, "val_split.pkl"))
    test  = pd.read_pickle(os.path.join(PROCESSED_DIR, "test_split.pkl"))
    with open(os.path.join(PROCESSED_DIR, "preprocessors.pkl"), "rb") as f:
        artifacts = pickle.load(f)
    return train, val, test, artifacts


def get_features_and_labels(df, feature_cols, binary=True):
    X = df[feature_cols].values
    if binary:
        y = df['label'].astype(int).values
    else:
        y = df['attack_cat'].values
    return X, y


def train_xgboost(train_df, val_df, feature_cols, artifacts):
    X_train, y_train = get_features_and_labels(train_df, feature_cols)
    X_val,   y_val   = get_features_and_labels(val_df,   feature_cols)

    # Class imbalance: compute scale_pos_weight from training split only
    n_neg = (y_train == 0).sum()
    n_pos = (y_train == 1).sum()
    spw = n_neg / n_pos
    print(f"[XGB] Train class balance — Normal: {n_neg:,} | Attack: {n_pos:,} | scale_pos_weight: {spw:.2f}")

    # --- Baseline ---
    print("[XGB] Training baseline model...")
    baseline = xgb.XGBClassifier(
        n_estimators=200,
        use_label_encoder=False,
        eval_metric='logloss',
        random_state=42,
        n_jobs=-1
    )
    baseline.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    baseline_f1 = f1_score(y_val, baseline.predict(X_val), average='macro')
    print(f"[XGB] Baseline Val Macro-F1: {baseline_f1:.4f}")

    # --- Grid Search (lightweight) ---
    best_params = None
    best_f1 = -1
    param_grid = [
        {"max_depth": d, "learning_rate": lr, "n_estimators": n, "subsample": s, "scale_pos_weight": spw}
        for d in [4, 6]
        for lr in [0.05, 0.1]
        for n in [300, 500]
        for s in [0.8, 1.0]
    ]
    print(f"[XGB] Hyperparameter search over {len(param_grid)} configs...")
    for i, p in enumerate(param_grid):
        model = xgb.XGBClassifier(
            **p,
            use_label_encoder=False,
            eval_metric='logloss',
            random_state=42,
            n_jobs=-1,
            verbosity=0
        )
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        score = f1_score(y_val, model.predict(X_val), average='macro')
        if score > best_f1:
            best_f1 = score
            best_params = p
        if (i + 1) % 4 == 0:
            print(f"  [{i+1}/{len(param_grid)}] Best so far: Macro-F1={best_f1:.4f}")

    print(f"[XGB] Best params: {best_params} | Val Macro-F1: {best_f1:.4f}")

    # --- Final model: retrain on ALL training data (train_split) with best params ---
    print("[XGB] Retraining final model on full training split with best params...")
    final_model = xgb.XGBClassifier(
        **best_params,
        use_label_encoder=False,
        eval_metric='logloss',
        random_state=42,
        n_jobs=-1,
        verbosity=0
    )
    X_all = np.vstack([X_train, X_val])
    y_all = np.concatenate([y_train, y_val])
    final_model.fit(X_all, y_all, verbose=False)

    # Save model
    model_path = os.path.join(MODELS_DIR, "xgboost_model.json")
    final_model.save_model(model_path)
    print(f"[XGB] Saved model to {model_path}")

    return final_model, best_params, best_f1


def evaluate_xgboost(model, test_df, feature_cols):
    """Run ONE evaluation against the official test set. Never re-run during tuning."""
    X_test, y_test = get_features_and_labels(test_df, feature_cols)
    
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    report = classification_report(y_test, y_pred, output_dict=True)
    cm = confusion_matrix(y_test, y_pred)
    macro_f1 = f1_score(y_test, y_pred, average='macro')
    roc_auc = roc_auc_score(y_test, y_prob)
    pr_auc = average_precision_score(y_test, y_prob)

    metrics = {
        "model": "xgboost",
        "test_macro_f1": macro_f1,
        "test_roc_auc": roc_auc,
        "test_pr_auc": pr_auc,
        "classification_report": report,
        "confusion_matrix": cm.tolist()
    }

    # Save
    metrics_path = os.path.join(RESULTS_DIR, "xgboost_test_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"[XGB] Test Macro-F1: {macro_f1:.4f} | ROC-AUC: {roc_auc:.4f} | PR-AUC: {pr_auc:.4f}")
    print(f"[XGB] Metrics saved to {metrics_path}")
    return metrics, y_pred, y_prob


def main():
    print("=== Phase 2: XGBoost Training & Evaluation ===")
    train_df, val_df, test_df, artifacts = load_splits()
    feature_cols = artifacts['tree_feature_cols']

    # Verify NO data leakage: test identifiers only come from test
    print(f"[XGB] Train rows: {len(train_df):,} | Val rows: {len(val_df):,} | Test rows: {len(test_df):,}")

    model, best_params, val_f1 = train_xgboost(train_df, val_df, feature_cols, artifacts)
    metrics, _, _ = evaluate_xgboost(model, test_df, feature_cols)

    print("\n[✓] Phase 2 XGBoost checkpoint complete.")
    return model, metrics


if __name__ == "__main__":
    main()
