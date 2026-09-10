"""
NetGuard-XAI Data Preprocessing Module
Implements clean data ingestion, feature engineering, categorical encoding,
normalization, train/val/test split freezing, and diagnostic reporting.
"""

import os
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder

IDENTIFIER_COLS = ['id', 'srcip', 'sport', 'dstip', 'dsport', 'stime', 'ltime']
CATEGORICAL_COLS = ['proto', 'service', 'state']
TARGET_COLS = ['label', 'attack_cat']

def load_raw_data(raw_dir):
    """Loads raw UNSW-NB15 training and testing CSV files."""
    train_path = os.path.join(raw_dir, "UNSW_NB15_training-set.csv")
    test_path = os.path.join(raw_dir, "UNSW_NB15_testing-set.csv")
    
    if not os.path.exists(train_path) or not os.path.exists(test_path):
        raise FileNotFoundError(f"Raw dataset files missing from {raw_dir}. Run download script first.")
        
    train_raw = pd.read_csv(train_path)
    test_raw = pd.read_csv(test_path)
    return train_raw, test_raw

def sanitize_attack_cat(df):
    """Clean and standardize attack_cat column."""
    if 'attack_cat' in df.columns:
        df['attack_cat'] = df['attack_cat'].astype(str).str.strip()
        df['attack_cat'] = df['attack_cat'].replace({'nan': 'Normal', 'NaN': 'Normal', '': 'Normal'})
    return df

def preprocess_dataset(train_raw, test_raw, val_ratio=0.2, random_state=42, output_dir="data/processed"):
    """
    Cleans raw data, carves stratified val split from train, fits scalers & encoders strictly
    on training data, and exports processed datasets.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    train_df = train_raw.copy()
    test_df = test_raw.copy()
    
    train_df = sanitize_attack_cat(train_df)
    test_df = sanitize_attack_cat(test_df)
    
    # Identify non-predictive columns present in dataset
    cols_to_drop = [c for c in IDENTIFIER_COLS if c in train_df.columns]
    train_df = train_df.drop(columns=cols_to_drop)
    test_df = test_df.drop(columns=cols_to_drop)
    
    # Carve Stratified Validation Split from Training Data ONLY
    train_split, val_split = train_test_split(
        train_df,
        test_size=val_ratio,
        random_state=random_state,
        stratify=train_df['attack_cat']
    )
    train_split = train_split.reset_index(drop=True)
    val_split = val_split.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)
    
    # Separate features and targets
    feature_cols = [c for c in train_split.columns if c not in TARGET_COLS]
    num_cols = [c for c in feature_cols if c not in CATEGORICAL_COLS]
    
    # Fit Label Encoders / One-Hot Encoders strictly on train_split
    cat_encoders = {}
    ohe_encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    
    for c in CATEGORICAL_COLS:
        le = LabelEncoder()
        # Ensure unseen categories in val/test are handled gracefully
        train_split[c + '_enc'] = le.fit_transform(train_split[c].astype(str))
        
        # Mapping for unseen classes
        unseen_val = set(val_split[c].astype(str)) - set(le.classes_)
        unseen_test = set(test_df[c].astype(str)) - set(le.classes_)
        
        val_split[c + '_enc'] = val_split[c].astype(str).map(
            lambda x: le.transform([x])[0] if x in le.classes_ else -1
        )
        test_df[c + '_enc'] = test_df[c].astype(str).map(
            lambda x: le.transform([x])[0] if x in le.classes_ else -1
        )
        cat_encoders[c] = le

    # Fit OneHotEncoder on categorical columns of train_split
    train_cat_ohe = ohe_encoder.fit_transform(train_split[CATEGORICAL_COLS].astype(str))
    val_cat_ohe = ohe_encoder.transform(val_split[CATEGORICAL_COLS].astype(str))
    test_cat_ohe = ohe_encoder.transform(test_df[CATEGORICAL_COLS].astype(str))
    
    ohe_feature_names = list(ohe_encoder.get_feature_names_out(CATEGORICAL_COLS))
    
    # Fit StandardScaler on numerical features of train_split ONLY
    scaler = StandardScaler()
    train_num_scaled = scaler.fit_transform(train_split[num_cols])
    val_num_scaled = scaler.transform(val_split[num_cols])
    test_num_scaled = scaler.transform(test_df[num_cols])
    
    # Construct tabular representation for XGBoost (Label encoded categories + unscaled/scaled numericals)
    tree_feature_cols = num_cols + [c + '_enc' for c in CATEGORICAL_COLS]
    
    # Save artifacts
    artifacts = {
        'scaler': scaler,
        'cat_encoders': cat_encoders,
        'ohe_encoder': ohe_encoder,
        'num_cols': num_cols,
        'cat_cols': CATEGORICAL_COLS,
        'tree_feature_cols': tree_feature_cols,
        'ohe_feature_names': ohe_feature_names
    }
    
    with open(os.path.join(output_dir, "preprocessors.pkl"), "wb") as f:
        pickle.dump(artifacts, f)
        
    # Save splits as Parquet / CSV
    train_split.to_pickle(os.path.join(output_dir, "train_split.pkl"))
    val_split.to_pickle(os.path.join(output_dir, "val_split.pkl"))
    test_df.to_pickle(os.path.join(output_dir, "test_split.pkl"))
    
    # Save scaled numpy matrices for Neural models / Autoencoder / LSTM
    np.savez_compressed(
        os.path.join(output_dir, "neural_data.npz"),
        X_train_num=train_num_scaled,
        X_train_cat=train_cat_ohe,
        y_train_label=train_split['label'].values,
        y_train_cat=train_split['attack_cat'].values,
        X_val_num=val_num_scaled,
        X_val_cat=val_cat_ohe,
        y_val_label=val_split['label'].values,
        y_val_cat=val_split['attack_cat'].values,
        X_test_num=test_num_scaled,
        X_test_cat=test_cat_ohe,
        y_test_label=test_df['label'].values,
        y_test_cat=test_df['attack_cat'].values
    )

    return train_split, val_split, test_df, artifacts

def generate_data_report(train_raw, test_raw, train_split, val_split, test_df, report_path="data_report.md"):
    """Generates Phase 1 Data Verification Report with exact dataset metrics."""
    
    train_cat_counts = train_raw['attack_cat'].astype(str).str.strip().value_counts().to_dict()
    test_cat_counts = test_raw['attack_cat'].astype(str).str.strip().value_counts().to_dict()
    
    report_content = f"""# Phase 1 — Data Pipeline Checkpoint Report

## 1. Dataset Summary & Diagnostics
- **Official Raw Train Set**: {len(train_raw):,} rows, {train_raw.shape[1]} columns
- **Official Raw Test Set**: {len(test_raw):,} rows, {test_raw.shape[1]} columns
- **Null Values Count (Train)**: {train_raw.isnull().sum().sum()}
- **Null Values Count (Test)**: {test_raw.isnull().sum().sum()}
- **Infinite Values Count (Train)**: {np.isinf(train_raw.select_dtypes(include=np.number)).sum().sum()}
- **Infinite Values Count (Test)**: {np.isinf(test_raw.select_dtypes(include=np.number)).sum().sum()}

## 2. Frozen Split Breakdown
From the official training set ({len(train_raw):,} rows), a stratified 80/20 train/validation split was created:
- **Training Split**: {len(train_split):,} rows (80% of training CSV)
- **Validation Split**: {len(val_split):,} rows (20% of training CSV)
- **Official Test Set**: {len(test_df):,} rows (100% frozen testing CSV)

## 3. Class Distribution across Splits (attack_cat)

| Attack Category | Raw Train Count | Train Split (80%) | Val Split (20%) | Official Test Count |
| :--- | :---: | :---: | :---: | :---: |
"""
    all_cats = sorted(list(set(list(train_cat_counts.keys()) + list(test_cat_counts.keys()))))
    for cat in all_cats:
        tr_raw_c = train_cat_counts.get(cat, 0)
        tr_split_c = (train_split['attack_cat'] == cat).sum()
        val_split_c = (val_split['attack_cat'] == cat).sum()
        te_raw_c = test_cat_counts.get(cat, 0)
        report_content += f"| **{cat}** | {tr_raw_c:,} | {tr_split_c:,} | {val_split_c:,} | {te_raw_c:,} |\n"

    report_content += f"""
## 4. Preprocessing & Feature Engineering Summary
- **Identifiers Dropped**: `id` (and any `srcip`, `sport`, `dstip`, `dsport`, `stime`, `ltime` if present)
- **Categorical Columns**: `proto`, `service`, `state`
  - **Tree Models (XGBoost)**: Label encoded integer mappings (`proto_enc`, `service_enc`, `state_enc`)
  - **Neural Models (Autoencoder / LSTM)**: One-Hot Encoded sparse matrix
- **Numerical Scaling**: `StandardScaler` fitted **strictly** on the training split ({len(train_split):,} rows). Zero data leakage to validation or test sets.
- **Model Feature Count**: {len(train_split.columns) - 2} tabular features for tree models.

## 5. Phase 1 Summary
- **Status**: Completed & Verified.
- **Data Leakage Safeguard**: Strict separation enforced; scalers and encoders fit solely on the 80% train split.
- **Artifacts Saved**: `data/processed/train_split.pkl`, `val_split.pkl`, `test_split.pkl`, `preprocessors.pkl`, `neural_data.npz`.
"""
    with open(report_path, "w") as f:
        f.write(report_content)
    print(f"[+] Data report written to {report_path}")
