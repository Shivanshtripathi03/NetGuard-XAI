"""
NetGuard-XAI Unit & Integration Tests
Verifies core data processing, risk engine calculations, model loading, and result files.
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.phase3_risk_xai import RiskEngine, compute_uncertainty

class TestNetGuardPipeline(unittest.TestCase):

    def test_risk_engine(self):
        """Test risk engine formula output range and boundary behavior."""
        engine = RiskEngine()
        res = engine.compute_risk(
            P_t=0.9,
            A_t=0.5,
            B_t=0.8,
            S_t=0.8,
            U_t=0.1
        )
        risk = res['R_t']
        self.assertGreaterEqual(risk, 0.0)
        self.assertLessEqual(risk, 1.0)
        self.assertGreater(risk, 0.5)

    def test_uncertainty_calculation(self):
        """Test uncertainty metric based on model agreement/disagreement."""
        unc_high = compute_uncertainty(P_t=0.9, A_t=0.1, B_t=0.1)
        unc_low = compute_uncertainty(P_t=0.9, A_t=0.9, B_t=0.9)
        self.assertGreater(unc_high, unc_low)

    def test_saved_model_artifacts_exist(self):
        """Verify all trained model artifacts exist on disk."""
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.assertTrue(os.path.exists(os.path.join(base_dir, "models", "xgboost_model.json")))
        self.assertTrue(os.path.exists(os.path.join(base_dir, "models", "autoencoder.pt")))
        self.assertTrue(os.path.exists(os.path.join(base_dir, "models", "ae_scaler.pkl")))
        self.assertTrue(os.path.exists(os.path.join(base_dir, "models", "lstm.pt")))
        self.assertTrue(os.path.exists(os.path.join(base_dir, "models", "dqn_policy.zip")))

    def test_saved_results_exist(self):
        """Verify Phase 1-6 evaluation metrics files exist."""
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.assertTrue(os.path.exists(os.path.join(base_dir, "results", "xgboost_test_metrics.json")))
        self.assertTrue(os.path.exists(os.path.join(base_dir, "results", "autoencoder_test_metrics.json")))
        self.assertTrue(os.path.exists(os.path.join(base_dir, "results", "lstm_test_metrics.json")))
        self.assertTrue(os.path.exists(os.path.join(base_dir, "results", "phase5_zeroday_results.json")))
        self.assertTrue(os.path.exists(os.path.join(base_dir, "results", "phase6_ablation.json")))

if __name__ == "__main__":
    unittest.main()
