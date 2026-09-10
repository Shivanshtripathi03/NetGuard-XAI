#!/usr/bin/env python3
"""
NetGuard-XAI Full Pipeline Runner
Executes all phases in order. Run: python3 run_pipeline.py
"""

import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def header(title):
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60)

def main():
    header("NetGuard-XAI Pipeline — Start")
    t0 = time.time()

    # Phase 1
    header("Phase 1: Data Pipeline")
    from scripts.phase1_pipeline import main as phase1
    phase1()

    # Phase 2: XGBoost
    header("Phase 2a: XGBoost Detection Model")
    from scripts.phase2_xgboost import main as phase2_xgb
    phase2_xgb()

    # Phase 2: Autoencoder
    header("Phase 2b: Autoencoder Anomaly Detector")
    from scripts.phase2_autoencoder import main as phase2_ae
    phase2_ae()

    # Phase 2: LSTM
    header("Phase 2c: LSTM Behavioral Model")
    from scripts.phase2_lstm import main as phase2_lstm
    phase2_lstm()

    # Phase 3
    header("Phase 3: Risk Engine, Uncertainty & SHAP")
    from scripts.phase3_risk_xai import main as phase3
    phase3()

    # Phase 4
    header("Phase 4: RL Environment & DQN Training")
    from scripts.phase4_rl import main as phase4
    phase4()

    # Phase 5
    header("Phase 5: Zero-Day Experiments")
    from scripts.phase5_zeroday import main as phase5
    phase5()

    # Phase 6
    header("Phase 6: Ablation & Comparative Study")
    from scripts.phase6_ablation import main as phase6
    phase6()

    elapsed = time.time() - t0
    header(f"Pipeline Complete! Total time: {elapsed/60:.1f} min")
    print("\nTo launch the dashboard:")
    print("  1. Start API:  python3 -m uvicorn src.api:app --host 0.0.0.0 --port 8000")
    print("  2. Dashboard:  python3 -m streamlit run src/dashboard.py")

if __name__ == "__main__":
    main()
