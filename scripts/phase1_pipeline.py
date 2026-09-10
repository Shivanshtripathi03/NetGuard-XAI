#!/usr/bin/env python3
"""
Phase 1 Execution Script
Downloads datasets, runs preprocessing, verifies splits, and produces data_report.md checkpoint.
"""

import sys
import os

# Add root directory to sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.download_dataset import main as download_main
from src.data_preprocessing import load_raw_data, preprocess_dataset, generate_data_report

def main():
    print("==================================================")
    print("        NetGuard-XAI Phase 1 Data Pipeline        ")
    print("==================================================")
    
    # 1. Download official CSVs
    download_main()
    
    # 2. Load Raw Datasets
    raw_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "raw")
    print("\n[*] Loading raw CSV files...")
    train_raw, test_raw = load_raw_data(raw_dir)
    print(f"[+] Loaded Train Shape: {train_raw.shape} | Test Shape: {test_raw.shape}")
    
    # 3. Preprocess & Freeze Splits
    print("\n[*] Running Data Preprocessing & Validation Split...")
    train_split, val_split, test_df, artifacts = preprocess_dataset(train_raw, test_raw)
    print(f"[+] Processed Train Split: {train_split.shape}")
    print(f"[+] Processed Val Split:   {val_split.shape}")
    print(f"[+] Processed Test Set:    {test_df.shape}")
    
    # 4. Generate Diagnostic Checkpoint Report
    report_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data_report.md")
    print(f"\n[*] Generating Data Diagnostic Report: {report_path}...")
    generate_data_report(train_raw, test_raw, train_split, val_split, test_df, report_path)
    
    print("\n[✓] Phase 1 Completed Successfully!")

if __name__ == "__main__":
    main()
