# Phase 2 — Detection Models Checkpoint Report

## Summary

Three complementary detection models were trained on the official UNSW-NB15 train/val splits and evaluated **once** on the frozen official test set (82,332 rows). All models are frozen and saved as artifacts.

---

## XGBoost (Known-Attack Classifier)

- **Train split**: 140,272 rows (80%) | **Val split**: 35,069 rows (20%) | **Test**: 82,332 rows
- **Hyperparameter search**: Grid over `max_depth ∈ {4,6}`, `learning_rate ∈ {0.05,0.1}`, `n_estimators ∈ {300,500}`, `subsample ∈ {0.8,1.0}` — 16 configs evaluated on val split
- **Class imbalance**: `scale_pos_weight` computed from training split class ratio
- **Final model**: Retrained on full training data (train + val) with best params

| Metric | Value |
|:---|:---:|
| **Test Macro-F1** | 0.8961 |
| **Test ROC-AUC** | 0.9833 |
| **Test PR-AUC** | 0.9877 |
| **Accuracy** | 0.8990 |

**Confusion Matrix (Test Set)**:

| | Predicted Normal | Predicted Attack |
|:---|:---:|:---:|
| **Actual Normal** (37,000) | 30,125 (TN) | 6,875 (FP) |
| **Actual Attack** (45,332) | 1,441 (FN) | 43,891 (TP) |

- Precision (Normal): 0.954 | Recall (Normal): 0.814
- Precision (Attack): 0.865 | Recall (Attack): 0.968

**Artifact**: `models/xgboost_model.json`

---

## Autoencoder (Anomaly Detector)

- **Training data**: Normal-only rows from training split (44,800 rows)
- **Architecture**: Symmetric encoder-decoder (input → 64 → 32 → latent → 32 → 64 → input), BatchNorm + ReLU
- **Latent dim**: max(8, input_dim // 2)
- **Scaler**: Fresh StandardScaler fitted strictly on Normal-only training rows
- **Anomaly threshold**: 95th percentile of reconstruction MSE on Normal validation data = **0.04651**

| Metric | Value |
|:---|:---:|
| **Test ROC-AUC** | 0.8526 |
| **Test PR-AUC** | 0.8687 |
| **Test Detection Rate (TPR)** | 0.5709 |
| **Test FPR** | 0.0920 |

**Confusion Matrix (Test Set)**:

| | Predicted Normal | Predicted Anomaly |
|:---|:---:|:---:|
| **Actual Normal** (37,000) | 33,595 (TN) | 3,405 (FP) |
| **Actual Attack** (45,332) | 19,453 (FN) | 25,879 (TP) |

**Artifacts**: `models/autoencoder.pt`, `models/ae_scaler.pkl`

---

## LSTM/GRU (Behavioral Sequence Model)

- **Sequence construction**: Rolling window of 10 events, ordered by original dataset order (proxy for temporal order); label = last event label
- **Architecture**: 2-layer LSTM (hidden_dim=64, dropout=0.3) → FC(64→32→1) → Sigmoid
- **Training sequences**: 140,263 | **Val sequences**: 35,060 | **Test sequences**: 82,323

| Metric | Value |
|:---|:---:|
| **Test Macro-F1** | 0.8157 |
| **Test ROC-AUC** | 0.9463 |
| **Test PR-AUC** | 0.9577 |
| **Accuracy** | 0.8259 |

**Confusion Matrix (Test Set)**:

| | Predicted Normal | Predicted Attack |
|:---|:---:|:---:|
| **Actual Normal** (36,991) | 24,327 (TN) | 12,664 (FP) |
| **Actual Attack** (45,332) | 1,671 (FN) | 43,661 (TP) |

**Artifact**: `models/lstm.pt`

---

## Combined Model Comparison

| Model | Macro-F1 | ROC-AUC | PR-AUC | Detection Rate | FPR |
|:---|:---:|:---:|:---:|:---:|:---:|
| **XGBoost** | **0.8961** | **0.9833** | **0.9877** | 0.968 | 0.186 |
| **Autoencoder** | — | 0.8526 | 0.8687 | 0.571 | **0.092** |
| **LSTM** | 0.8157 | 0.9463 | 0.9577 | **0.963** | 0.342 |

---

## Honest Assessment (3–5 sentences)

XGBoost performs strongly as the primary supervised classifier with 0.896 macro-F1 and 0.983 ROC-AUC, achieving high attack recall (96.8%) at the cost of moderate false positive rate (18.6%). The Autoencoder trained on Normal-only data correctly identifies anomalies with the lowest FPR (9.2%) but lower detection rate (57.1%), which is expected since it has no supervised signal — its value is as a complementary unsupervised signal, not a standalone detector. The LSTM behavioral model achieves strong attack recall (96.3%) but higher FPR (34.2%), likely because the rolling-window sequence construction without true source-IP grouping (identifiers were dropped per instructions) treats the entire stream as one sequence, limiting its ability to capture per-host behavioral patterns. All three models are complementary and designed to be fused via the Phase 3 risk engine, where the Autoencoder's low FPR and LSTM's high recall should jointly improve on XGBoost's standalone performance.
