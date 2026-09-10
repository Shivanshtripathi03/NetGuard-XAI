#!/usr/bin/env python3
"""
Dataset Downloader for UNSW-NB15
Downloads official pre-split training and testing CSV files and feature descriptions.
"""

import os
import sys
import ssl
import urllib.request
import pandas as pd

RAW_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "raw")
os.makedirs(RAW_DIR, exist_ok=True)

SOURCES = {
    "UNSW_NB15_training-set.csv": [
        "https://git.tu-berlin.de/b.falkner/network-intrusion-detection/-/raw/master/UNSW_NB15_training-set.csv",
        "https://raw.githubusercontent.com/646510425/UNSW-NB15/master/UNSW_NB15_training-set.csv",
        "https://media.githubusercontent.com/media/NourMoustafa/UNSW-NB15/master/UNSW_NB15_training-set.csv",
    ],
    "UNSW_NB15_testing-set.csv": [
        "https://git.tu-berlin.de/b.falkner/network-intrusion-detection/-/raw/master/UNSW_NB15_testing-set.csv",
        "https://raw.githubusercontent.com/646510425/UNSW-NB15/master/UNSW_NB15_testing-set.csv",
        "https://media.githubusercontent.com/media/NourMoustafa/UNSW-NB15/master/UNSW_NB15_testing-set.csv",
    ],
    "UNSW-NB15_features.csv": [
        "https://git.tu-berlin.de/b.falkner/network-intrusion-detection/-/raw/master/UNSW-NB15_features.csv",
        "https://raw.githubusercontent.com/646510425/UNSW-NB15/master/UNSW-NB15_features.csv",
    ]
}

EXPECTED_ROWS = {
    "UNSW_NB15_training-set.csv": 175341,
    "UNSW_NB15_testing-set.csv": 82332,
}

def download_file(filename, urls):
    dest_path = os.path.join(RAW_DIR, filename)
    if os.path.exists(dest_path) and os.path.getsize(dest_path) > 1000:
        print(f"[+] {filename} already exists at {dest_path} ({os.path.getsize(dest_path)} bytes).")
        return dest_path

    ssl_context = ssl._create_unverified_context()
    
    for url in urls:
        print(f"[*] Attempting download for {filename} from {url}...")
        try:
            req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
            with urllib.request.urlopen(req, context=ssl_context, timeout=60) as response, open(dest_path, 'wb') as out_file:
                out_file.write(response.read())
            print(f"[+] Successfully downloaded {filename} ({os.path.getsize(dest_path)} bytes).")
            return dest_path
        except Exception as e:
            print(f"[-] Failed from {url}: {e}")
            if os.path.exists(dest_path):
                os.remove(dest_path)
                
    raise RuntimeError(f"Could not download {filename} from any configured source URL.")

def verify_dataset():
    print("\n--- Verifying Dataset Files ---")
    for filename, expected_count in EXPECTED_ROWS.items():
        file_path = os.path.join(RAW_DIR, filename)
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Missing file: {file_path}")
        df = pd.read_csv(file_path)
        actual_count = len(df)
        print(f"File: {filename} | Rows: {actual_count} | Expected: {expected_count}")
        if actual_count != expected_count:
            print(f"[!] WARNING: Row count mismatch for {filename}! Expected {expected_count}, got {actual_count}")
        else:
            print(f"[✓] {filename} row count verified successfully.")

def main():
    print("=== Phase 1: UNSW-NB15 Dataset Download ===")
    for filename, urls in SOURCES.items():
        download_file(filename, urls)
    verify_dataset()
    print("=== Dataset Download and Verification Complete ===")

if __name__ == "__main__":
    main()
