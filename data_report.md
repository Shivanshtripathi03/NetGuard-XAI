# Phase 1 — Data Pipeline Checkpoint Report

## 1. Dataset Summary & Diagnostics
- **Official Raw Train Set**: 175,341 rows, 45 columns
- **Official Raw Test Set**: 82,332 rows, 45 columns
- **Null Values Count (Train)**: 0
- **Null Values Count (Test)**: 0
- **Infinite Values Count (Train)**: 0
- **Infinite Values Count (Test)**: 0

## 2. Frozen Split Breakdown
From the official training set (175,341 rows), a stratified 80/20 train/validation split was created:
- **Training Split**: 140,272 rows (80% of training CSV)
- **Validation Split**: 35,069 rows (20% of training CSV)
- **Official Test Set**: 82,332 rows (100% frozen testing CSV)

## 3. Class Distribution across Splits (attack_cat)

| Attack Category | Raw Train Count | Train Split (80%) | Val Split (20%) | Official Test Count |
| :--- | :---: | :---: | :---: | :---: |
| **Analysis** | 2,000 | 1,600 | 400 | 677 |
| **Backdoor** | 1,746 | 1,397 | 349 | 583 |
| **DoS** | 12,264 | 9,811 | 2,453 | 4,089 |
| **Exploits** | 33,393 | 26,714 | 6,679 | 11,132 |
| **Fuzzers** | 18,184 | 14,547 | 3,637 | 6,062 |
| **Generic** | 40,000 | 32,000 | 8,000 | 18,871 |
| **Normal** | 56,000 | 44,800 | 11,200 | 37,000 |
| **Reconnaissance** | 10,491 | 8,393 | 2,098 | 3,496 |
| **Shellcode** | 1,133 | 906 | 227 | 378 |
| **Worms** | 130 | 104 | 26 | 44 |

## 4. Preprocessing & Feature Engineering Summary
- **Identifiers Dropped**: `id` (and any `srcip`, `sport`, `dstip`, `dsport`, `stime`, `ltime` if present)
- **Categorical Columns**: `proto`, `service`, `state`
  - **Tree Models (XGBoost)**: Label encoded integer mappings (`proto_enc`, `service_enc`, `state_enc`)
  - **Neural Models (Autoencoder / LSTM)**: One-Hot Encoded sparse matrix
- **Numerical Scaling**: `StandardScaler` fitted **strictly** on the training split (140,272 rows). Zero data leakage to validation or test sets.
- **Model Feature Count**: 45 tabular features for tree models.

## 5. Phase 1 Summary
- **Status**: Completed & Verified.
- **Data Leakage Safeguard**: Strict separation enforced; scalers and encoders fit solely on the 80% train split.
- **Artifacts Saved**: `data/processed/train_split.pkl`, `val_split.pkl`, `test_split.pkl`, `preprocessors.pkl`, `neural_data.npz`.
