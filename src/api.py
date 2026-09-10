"""
NetGuard-XAI FastAPI Inference Service
Exposes /analyze endpoint for real-time network flow analysis.
Runs XGBoost + Autoencoder + LSTM → Risk Engine → SHAP → RL decision.
Persists all incidents to SQLite/PostgreSQL.
"""

import os
import sys
import json
import pickle
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
from datetime import datetime
import torch
from sqlalchemy.orm import Session

from src.models import Autoencoder, LSTMBehaviorModel
from src.database import get_engine, save_incident, get_recent_incidents, get_incidents_by_risk, Incident
from scripts.phase3_risk_xai import RiskEngine, compute_uncertainty, normalize_ae_score, SEVERITY_MAP

MODELS_DIR    = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")
RESULTS_DIR   = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
PROCESSED_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "processed")

app = FastAPI(
    title="NetGuard-XAI API",
    description="Real-time Network Intrusion Detection with SHAP Explanations and RL-based Response",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Global model cache (loaded once at startup) ──
_models = {}
_engine = None
_risk_engines: Dict[str, RiskEngine] = {}  # per-source stateful engines


class FlowFeatures(BaseModel):
    """Network flow features for inference."""
    features: Dict[str, float]  # keyed by feature name
    srcip: Optional[str] = "unknown"
    dstip: Optional[str] = "unknown"
    proto: Optional[str] = "tcp"
    true_label: Optional[int] = None   # only available in batch/test mode
    attack_cat: Optional[str] = None

class AnalysisResponse(BaseModel):
    incident_id: int
    risk_score: float
    risk_delta: float
    xgb_prob: float
    ae_score: float
    lstm_prob: float
    uncertainty: float
    rl_action: int
    action_name: str
    is_alert: bool
    shap_top_features: List[Dict]
    timestamp: str


@app.on_event("startup")
async def load_models():
    global _models, _engine
    import xgboost as xgb
    from stable_baselines3 import DQN

    with open(os.path.join(PROCESSED_DIR, "preprocessors.pkl"), "rb") as f:
        artifacts = pickle.load(f)

    with open(os.path.join(RESULTS_DIR, "autoencoder_test_metrics.json")) as f:
        ae_metrics = json.load(f)
    with open(os.path.join(MODELS_DIR, "ae_scaler.pkl"), "rb") as f:
        ae_scaler = pickle.load(f)

    num_cols = artifacts['num_cols']

    xgb_model = xgb.XGBClassifier()
    xgb_model.load_model(os.path.join(MODELS_DIR, "xgboost_model.json"))

    ae_model = Autoencoder(input_dim=len(num_cols), latent_dim=max(8, len(num_cols)//2))
    ae_model.load_state_dict(torch.load(os.path.join(MODELS_DIR, "autoencoder.pt"), weights_only=True))
    ae_model.eval()

    lstm_model = LSTMBehaviorModel(input_dim=len(num_cols), hidden_dim=64, num_layers=2, dropout=0.3)
    lstm_model.load_state_dict(torch.load(os.path.join(MODELS_DIR, "lstm.pt"), weights_only=True))
    lstm_model.eval()

    dqn_model = DQN.load(os.path.join(MODELS_DIR, "dqn_policy"))

    import shap
    explainer = shap.TreeExplainer(xgb_model)

    _models.update({
        "xgb": xgb_model,
        "ae": ae_model,
        "ae_scaler": ae_scaler,
        "ae_threshold": ae_metrics["anomaly_threshold"],
        "lstm": lstm_model,
        "dqn": dqn_model,
        "explainer": explainer,
        "artifacts": artifacts,
        "num_cols": num_cols,
        "feature_cols": artifacts["tree_feature_cols"],
        "seq_buffer": {},  # per-source sequence buffer for LSTM
    })

    _engine = get_engine()
    print("[API] All models loaded. NetGuard-XAI API ready.")


@app.post("/analyze", response_model=AnalysisResponse)
async def analyze_flow(flow: FlowFeatures):
    if not _models:
        raise HTTPException(status_code=503, detail="Models not loaded yet")

    artifacts  = _models["artifacts"]
    num_cols   = _models["num_cols"]
    feat_cols  = _models["feature_cols"]

    # Build feature vectors
    try:
        X_num  = np.array([[flow.features.get(c, 0.0) for c in num_cols]], dtype=np.float32)
        X_tree = np.array([[flow.features.get(c, 0.0) for c in feat_cols]], dtype=np.float32)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Feature extraction error: {e}")

    # XGBoost
    P_t = float(_models["xgb"].predict_proba(X_tree)[0, 1])

    # Autoencoder
    X_ae = _models["ae_scaler"].transform(X_num)
    with torch.no_grad():
        ae_recon = _models["ae"](torch.tensor(X_ae, dtype=torch.float32)).numpy()
    mse = float(np.mean((X_ae - ae_recon) ** 2))
    A_t = normalize_ae_score(mse, _models["ae_threshold"])

    # LSTM (maintain per-source rolling buffer)
    src = flow.srcip or "unknown"
    if src not in _models["seq_buffer"]:
        _models["seq_buffer"][src] = []
    buf = _models["seq_buffer"][src]
    scaler = artifacts['scaler']
    buf.append(scaler.transform(X_num)[0])
    if len(buf) > 10:
        buf.pop(0)
    if len(buf) < 10:
        padded = np.zeros((10, X_num.shape[1]))
        padded[-len(buf):] = buf
    else:
        padded = np.array(buf)
    X_seq = torch.tensor(padded.reshape(1, 10, -1), dtype=torch.float32)
    with torch.no_grad():
        B_t = float(_models["lstm"](X_seq).item())

    # Risk Engine (stateful per source)
    if src not in _risk_engines:
        _risk_engines[src] = RiskEngine()
    engine = _risk_engines[src]

    U_t = compute_uncertainty(P_t, A_t, B_t)
    S_t = SEVERITY_MAP.get(flow.attack_cat or "Normal", 0.5) if flow.attack_cat else 0.5
    risk_result = engine.compute_risk(P_t, A_t, B_t, S_t, U_t)
    R_t     = risk_result["R_t"]
    delta_R = risk_result["delta_R"]
    H_t     = risk_result["H_t"]

    # RL Action
    obs = np.array([P_t, A_t, B_t, R_t, U_t,
                    float(np.clip((delta_R + 1) / 2, 0, 1)),
                    float(np.clip(H_t, 0, 1)), 1.0], dtype=np.float32)
    action, _ = _models["dqn"].predict(obs, deterministic=True)
    action = int(action)
    action_names = {0: "Allow", 1: "Monitor", 2: "RateLimit", 3: "Block", 4: "Isolate"}
    action_name = action_names[action]

    # SHAP
    sv = _models["explainer"].shap_values(X_tree)[0]
    top_shap = sorted(
        [{"feature": f, "shap_value": float(v)} for f, v in zip(feat_cols, sv)],
        key=lambda x: abs(x["shap_value"]), reverse=True
    )[:8]

    is_alert = R_t >= 0.5

    # Persist to database
    incident_data = {
        "timestamp": datetime.utcnow(),
        "srcip": flow.srcip,
        "dstip": flow.dstip,
        "proto": flow.proto,
        "true_label": flow.true_label,
        "attack_cat": flow.attack_cat,
        "xgb_prob": P_t,
        "ae_score": A_t,
        "lstm_prob": B_t,
        "risk_score": R_t,
        "risk_delta": delta_R,
        "risk_hist": H_t,
        "uncertainty": U_t,
        "rl_action": action,
        "action_name": action_name,
        "shap_top": top_shap,
        "is_alert": is_alert,
    }
    with Session(_engine) as session:
        inc = save_incident(session, incident_data)
        incident_id = inc.id

    return AnalysisResponse(
        incident_id=incident_id,
        risk_score=R_t,
        risk_delta=delta_R,
        xgb_prob=P_t,
        ae_score=A_t,
        lstm_prob=B_t,
        uncertainty=U_t,
        rl_action=action,
        action_name=action_name,
        is_alert=is_alert,
        shap_top_features=top_shap,
        timestamp=datetime.utcnow().isoformat()
    )


@app.get("/incidents/recent")
async def recent_incidents(limit: int = 50):
    with Session(_engine) as session:
        incidents = get_recent_incidents(session, limit)
        return [
            {
                "id": i.id,
                "timestamp": i.timestamp.isoformat() if i.timestamp else None,
                "srcip": i.srcip,
                "attack_cat": i.attack_cat,
                "risk_score": i.risk_score,
                "action_name": i.action_name,
                "is_alert": i.is_alert,
                "xgb_prob": i.xgb_prob,
                "ae_score": i.ae_score,
                "lstm_prob": i.lstm_prob,
                "uncertainty": i.uncertainty,
                "shap_top": i.shap_top,
            }
            for i in incidents
        ]


@app.get("/incidents/alerts")
async def alert_incidents(min_risk: float = 0.5, limit: int = 50):
    with Session(_engine) as session:
        incidents = get_incidents_by_risk(session, min_risk, limit)
        return [
            {
                "id": i.id,
                "timestamp": i.timestamp.isoformat() if i.timestamp else None,
                "risk_score": i.risk_score,
                "attack_cat": i.attack_cat,
                "action_name": i.action_name,
                "shap_top": i.shap_top,
            }
            for i in incidents
        ]


@app.get("/health")
async def health():
    return {"status": "ok", "models_loaded": bool(_models)}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.api:app", host="0.0.0.0", port=8000, reload=False)
