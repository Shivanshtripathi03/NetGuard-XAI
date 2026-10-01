"""
NetGuard-XAI FastAPI Inference Service
=======================================
Exposes /analyze and /simulate endpoints for real-time and demonstration use.
The /simulate endpoint uses the RESTRICTED model set (Phase 5 — Reconnaissance
and Fuzzers held out) so that selecting those categories demonstrates genuine
zero-day handling, not a scripted simulation.

Design principle: /simulate routes through exactly the same code path as
/analyze — no shortcut. All results are persisted to the incident database.
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'

import json
import pickle
import random
import numpy as np
import pandas as pd
import torch
torch.set_num_threads(1)

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
from datetime import datetime
from sqlalchemy.orm import Session

from src.models import Autoencoder, LSTMBehaviorModel
from src.database import get_engine, save_incident, get_recent_incidents, get_incidents_by_risk, Incident
from scripts.phase3_risk_xai import RiskEngine, compute_uncertainty, normalize_ae_score, SEVERITY_MAP

ROOT_DIR      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR    = os.path.join(ROOT_DIR, "models")
RESTRICTED_DIR= os.path.join(MODELS_DIR, "restricted")
RESULTS_DIR   = os.path.join(ROOT_DIR, "results")
PROCESSED_DIR = os.path.join(ROOT_DIR, "data", "processed")
RAW_DIR       = os.path.join(ROOT_DIR, "data", "raw")

app = FastAPI(
    title="NetGuard-XAI API",
    description="Real-time Network Intrusion Detection — SHAP · RL · Zero-Day",
    version="2.0.0"
)

from fastapi.staticfiles import StaticFiles

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

if os.path.isdir(RESULTS_DIR):
    app.mount("/results", StaticFiles(directory=RESULTS_DIR), name="results")

# ── Global caches ──
_models: Dict[str, Any] = {}          # production (full) models
_restricted: Dict[str, Any] = {}      # restricted (zero-day dashboard) models
_engine = None
_risk_engines: Dict[str, RiskEngine] = {}     # per-source stateful engines
_sim_risk_engine = RiskEngine()               # shared engine for simulation trajectory


# ─────────────────────────────────────────────────────────────────────────────
# Pydantic Schemas
# ─────────────────────────────────────────────────────────────────────────────

class FlowFeatures(BaseModel):
    features: Dict[str, float]
    srcip: Optional[str] = "unknown"
    dstip: Optional[str] = "unknown"
    proto: Optional[str] = "tcp"
    true_label: Optional[int] = None
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
    row_index: Optional[int] = None


class SimulateRequest(BaseModel):
    attack_cat: str


class SimulateResponse(BaseModel):
    # Sampled event info
    incident_id: int
    row_index: int
    true_label: int
    true_attack_cat: str
    is_known: bool          # determined programmatically from training manifest
    known_label: str        # "Known — seen in training" or "Unseen — held out (zero-day test)"
    test_set_size: int      # number of rows for this category in test set

    # Per-model scores
    xgb_prob: float
    ae_score: float
    lstm_prob: float
    uncertainty: float

    # Risk engine
    risk_score: float
    risk_delta: float
    risk_hist: float

    # RL decision
    rl_action: int
    action_name: str
    is_alert: bool

    # SHAP (XGBoost only — labeled explicitly in response)
    shap_top_features: List[Dict]

    # Correctness verdict derived from ground truth (not model self-report)
    correctness_verdict: str   # "Correct containment" | "False positive" | "Missed attack" | "Correct allow"
    correctness_ok: bool

    # Human-readable narrative (template filled with real numbers)
    narrative: str

    # Aggregate performance for this category (from saved eval files)
    category_stats: Dict[str, Any]

    timestamp: str


class CategoryInfoResponse(BaseModel):
    categories: List[Dict[str, Any]]


# ─────────────────────────────────────────────────────────────────────────────
# Startup: load both model sets
# ─────────────────────────────────────────────────────────────────────────────

@app.on_event("startup")
async def load_models():
    global _models, _restricted, _engine
    import xgboost as xgb
    from stable_baselines3 import DQN
    import shap

    # ── Load shared preprocessors ──
    with open(os.path.join(PROCESSED_DIR, "preprocessors.pkl"), "rb") as f:
        artifacts = pickle.load(f)

    num_cols     = artifacts['num_cols']
    feature_cols = artifacts['tree_feature_cols']

    # ── Load AE threshold from production metrics (used for /analyze) ──
    with open(os.path.join(RESULTS_DIR, "autoencoder_test_metrics.json")) as f:
        ae_metrics_prod = json.load(f)

    # ── Production (full) models ── used by /analyze
    xgb_prod = xgb.XGBClassifier()
    xgb_prod.load_model(os.path.join(MODELS_DIR, "xgboost_model.json"))

    ae_prod = Autoencoder(input_dim=len(num_cols), latent_dim=max(8, len(num_cols)//2))
    ae_prod.load_state_dict(torch.load(os.path.join(MODELS_DIR, "autoencoder.pt"), weights_only=True))
    ae_prod.eval()

    lstm_prod = LSTMBehaviorModel(input_dim=len(num_cols), hidden_dim=64, num_layers=2, dropout=0.3)
    lstm_prod.load_state_dict(torch.load(os.path.join(MODELS_DIR, "lstm.pt"), weights_only=True))
    lstm_prod.eval()

    with open(os.path.join(MODELS_DIR, "ae_scaler.pkl"), "rb") as f:
        ae_scaler_prod = pickle.load(f)

    dqn_model = DQN.load(os.path.join(MODELS_DIR, "dqn_policy"))
    explainer_prod = shap.TreeExplainer(xgb_prod)

    _models.update({
        "xgb": xgb_prod, "ae": ae_prod, "ae_scaler": ae_scaler_prod,
        "ae_threshold": ae_metrics_prod["anomaly_threshold"],
        "lstm": lstm_prod, "dqn": dqn_model,
        "explainer": explainer_prod,
        "artifacts": artifacts, "num_cols": num_cols, "feature_cols": feature_cols,
        "seq_buffer": {},
        "test_df": pd.read_pickle(os.path.join(PROCESSED_DIR, "test_split.pkl")),
        "train_df": pd.read_pickle(os.path.join(PROCESSED_DIR, "train_split.pkl")),
    })

    # ── Restricted (zero-day) models ── used by /simulate
    manifest_path = os.path.join(RESTRICTED_DIR, "training_manifest.json")
    if os.path.exists(manifest_path):
        with open(manifest_path) as f:
            manifest = json.load(f)

        xgb_r = xgb.XGBClassifier()
        xgb_r.load_model(os.path.join(RESTRICTED_DIR, "xgboost_restricted.json"))

        ae_r = Autoencoder(input_dim=len(num_cols), latent_dim=max(8, len(num_cols)//2))
        ae_r.load_state_dict(torch.load(
            os.path.join(RESTRICTED_DIR, "autoencoder_restricted.pt"), weights_only=True))
        ae_r.eval()

        lstm_r = LSTMBehaviorModel(input_dim=len(num_cols), hidden_dim=64, num_layers=2, dropout=0.3)
        lstm_r.load_state_dict(torch.load(
            os.path.join(RESTRICTED_DIR, "lstm_restricted.pt"), weights_only=True))
        lstm_r.eval()

        with open(os.path.join(RESTRICTED_DIR, "ae_scaler_restricted.pkl"), "rb") as f:
            ae_sc_r = pickle.load(f)
        with open(os.path.join(RESTRICTED_DIR, "lstm_scaler_restricted.pkl"), "rb") as f:
            lstm_sc_r = pickle.load(f)
        with open(os.path.join(RESTRICTED_DIR, "ae_threshold_restricted.json")) as f:
            ae_thr_r = json.load(f)["anomaly_threshold"]

        explainer_r = shap.TreeExplainer(xgb_r)

        _restricted.update({
            "xgb": xgb_r, "ae": ae_r, "ae_scaler": ae_sc_r,
            "ae_threshold": ae_thr_r, "lstm": lstm_r, "lstm_scaler": lstm_sc_r,
            "dqn": dqn_model,           # DQN is shared (action policy is category-agnostic)
            "explainer": explainer_r,
            "manifest": manifest,
            "trained_categories": set(manifest["trained_categories"]),
            "seq_buffer": {},
        })
        print(f"[API] Restricted models loaded. Trained on: {manifest['trained_categories']}")
        print(f"[API] Held out: {manifest['held_out_categories']}")
    else:
        print("[API] WARNING: Restricted models not found. Run: python3 scripts/save_restricted_models.py")
        print(f"[API] Expected path: {manifest_path}")

    _engine = get_engine()
    print("[API] NetGuard-XAI API v2.0 ready.")


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

ACTION_NAMES = {0: "Allow", 1: "Monitor", 2: "RateLimit", 3: "Block", 4: "Isolate"}

def _run_pipeline(X_num, X_tree, models_dict, src_key="api", attack_cat=None):
    """
    Shared inference pipeline used by both /analyze and /simulate.
    Returns a dict of all intermediate scores.
    """
    artifacts  = _models["artifacts"]
    num_cols   = _models["num_cols"]
    feat_cols  = _models["feature_cols"]

    xgb_m      = models_dict["xgb"]
    ae_m       = models_dict["ae"]
    ae_sc      = models_dict["ae_scaler"]
    ae_thr     = models_dict["ae_threshold"]
    lstm_m     = models_dict["lstm"]
    dqn_m      = models_dict["dqn"]
    explainer  = models_dict["explainer"]
    seq_buf    = models_dict["seq_buffer"]

    # XGBoost probability
    P_t = float(xgb_m.predict_proba(X_tree)[0, 1])

    # Autoencoder anomaly score
    X_ae = ae_sc.transform(X_num)
    with torch.no_grad():
        ae_recon = ae_m(torch.tensor(X_ae, dtype=torch.float32)).numpy()
    mse = float(np.mean((X_ae - ae_recon) ** 2))
    A_t = normalize_ae_score(mse, ae_thr)

    # LSTM — per-source rolling buffer
    # For restricted models, use lstm_scaler if present; else fall back to artifacts scaler
    if "lstm_scaler" in models_dict:
        lstm_sc = models_dict["lstm_scaler"]
        scaled_row = lstm_sc.transform(X_num)[0]
    else:
        scaled_row = artifacts['scaler'].transform(X_num)[0]

    if src_key not in seq_buf:
        seq_buf[src_key] = []
    buf = seq_buf[src_key]
    buf.append(scaled_row)
    if len(buf) > 10:
        buf.pop(0)
    padded = np.zeros((10, X_num.shape[1]))
    padded[-len(buf):] = buf
    X_seq = torch.tensor(padded.reshape(1, 10, -1), dtype=torch.float32)
    with torch.no_grad():
        B_t = float(lstm_m(X_seq).item())

    # Risk Engine — use shared simulation engine for /simulate, per-source for /analyze
    U_t = compute_uncertainty(P_t, A_t, B_t)
    S_t = SEVERITY_MAP.get(attack_cat or "Normal", 0.5)

    if src_key.startswith("_sim_"):
        engine = _sim_risk_engine
    else:
        if src_key not in _risk_engines:
            _risk_engines[src_key] = RiskEngine()
        engine = _risk_engines[src_key]

    risk_result = engine.compute_risk(P_t, A_t, B_t, S_t, U_t)
    R_t     = risk_result["R_t"]
    delta_R = risk_result["delta_R"]
    H_t     = risk_result["H_t"]

    # RL action
    obs = np.array([P_t, A_t, B_t, R_t, U_t,
                    float(np.clip((delta_R + 1) / 2, 0, 1)),
                    float(np.clip(H_t, 0, 1)), 1.0], dtype=np.float32)
    action, _ = dqn_m.predict(obs, deterministic=True)
    action = int(action)

    # SHAP (XGBoost only)
    sv = explainer.shap_values(X_tree)[0]
    top_shap = sorted(
        [{"feature": f, "shap_value": float(v)} for f, v in zip(feat_cols, sv)],
        key=lambda x: abs(x["shap_value"]), reverse=True
    )[:8]

    return {
        "P_t": P_t, "A_t": A_t, "B_t": B_t,
        "mse": mse, "U_t": U_t, "S_t": S_t,
        "R_t": R_t, "delta_R": delta_R, "H_t": H_t,
        "action": action, "action_name": ACTION_NAMES[action],
        "shap": top_shap,
        "is_alert": R_t >= 0.5,
    }


def _compute_correctness(true_label: int, action_name: str):
    """
    Derive correctness verdict from ground-truth label and RL action taken.
    Never asks the model to self-report correctness.
    """
    is_attack  = (true_label == 1)
    is_contain = action_name in ("Block", "Isolate", "RateLimit")
    is_monitor = action_name == "Monitor"

    if is_attack and is_contain:
        return "Correct containment", True
    elif is_attack and is_monitor:
        return "Missed attack (monitoring only)", False
    elif is_attack and not is_contain:
        return "Missed attack", False
    elif not is_attack and is_contain:
        return "False positive", False
    else:
        return "Correct allow", True


def _build_narrative(is_known: bool, cat: str, A_t: float, B_t: float,
                     P_t: float, action_name: str, U_t: float) -> str:
    """Fill a template with real numbers — never write a static per-category string."""
    if is_known:
        detection_basis = (
            f"Signature detection active (XGBoost prob={P_t:.3f}). "
            f"Anomaly={A_t:.3f}, Behavioral={B_t:.3f}."
        )
    else:
        detection_basis = (
            f"This category was excluded from training. "
            f"Detection relied on anomaly ({A_t:.3f}) and behavioral deviation ({B_t:.3f}) "
            f"rather than a known-attack signature (XGBoost prob={P_t:.3f})."
        )
    return (
        f"{detection_basis} "
        f"Model disagreement (uncertainty)={U_t:.4f}. "
        f"The system chose {action_name.upper()}."
    )


def _get_category_stats(attack_cat: str, is_known: bool) -> Dict[str, Any]:
    """
    Return real aggregate performance numbers for the category from saved eval files.
    """
    stats = {"source": "phase results files"}

    # Phase 5 zero-day results — relevant for held-out categories
    p5_path = os.path.join(RESULTS_DIR, "phase5_zeroday_results.json")
    if os.path.exists(p5_path):
        with open(p5_path) as f:
            p5 = json.load(f)
        if not is_known:
            rm = p5.get("restricted_model", {})
            stats.update({
                "zero_day_recall": rm.get("recall"),
                "zero_day_f1": rm.get("f1"),
                "zero_day_containment_rate": rm.get("containment_rate"),
                "zero_day_n_eval": rm.get("n_eval_events"),
                "zero_day_note": (
                    f"Restricted model evaluated on all held-out "
                    f"({p5.get('held_out_categories', [])}) test rows together."
                ),
                "honest_finding": p5.get("honest_finding", ""),
            })

    # Overall XGBoost metrics (for known categories)
    xgb_path = os.path.join(RESULTS_DIR, "xgboost_test_metrics.json")
    if is_known and os.path.exists(xgb_path):
        with open(xgb_path) as f:
            xm = json.load(f)
        stats.update({
            "overall_xgb_f1": xm.get("test_macro_f1"),
            "overall_xgb_roc_auc": xm.get("test_roc_auc"),
        })

    # Phase 6 ablation
    abl_path = os.path.join(RESULTS_DIR, "phase6_ablation.json")
    if os.path.exists(abl_path):
        with open(abl_path) as f:
            abl = json.load(f)
        sys_c = abl.get("system_comparison", {}).get("System C (NetGuard-XAI+RL)", {})
        stats["full_system_containment_rate"] = sys_c.get("containment_rate")
        stats["full_system_fp_rate"] = sys_c.get("fp_rate")

    return stats


# ─────────────────────────────────────────────────────────────────────────────
# /analyze — production real-time endpoint
# ─────────────────────────────────────────────────────────────────────────────

@app.post("/analyze", response_model=AnalysisResponse)
async def analyze_flow(flow: FlowFeatures):
    if not _models:
        raise HTTPException(status_code=503, detail="Models not loaded yet")

    num_cols  = _models["num_cols"]
    feat_cols = _models["feature_cols"]

    try:
        X_num  = np.array([[flow.features.get(c, 0.0) for c in num_cols]], dtype=np.float32)
        X_tree = np.array([[flow.features.get(c, 0.0) for c in feat_cols]], dtype=np.float32)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Feature extraction error: {e}")

    src = flow.srcip or "unknown"
    res = _run_pipeline(X_num, X_tree, _models, src_key=src, attack_cat=flow.attack_cat)

    incident_data = {
        "timestamp": datetime.utcnow(), "srcip": flow.srcip, "dstip": flow.dstip,
        "proto": flow.proto, "true_label": flow.true_label, "attack_cat": flow.attack_cat,
        "xgb_prob": res["P_t"], "ae_score": res["A_t"], "lstm_prob": res["B_t"],
        "risk_score": res["R_t"], "risk_delta": res["delta_R"], "risk_hist": res["H_t"],
        "uncertainty": res["U_t"], "rl_action": res["action"], "action_name": res["action_name"],
        "shap_top": res["shap"], "is_alert": res["is_alert"],
    }
    with Session(_engine) as session:
        inc = save_incident(session, incident_data)
        incident_id = inc.id

    return AnalysisResponse(
        incident_id=incident_id, risk_score=res["R_t"], risk_delta=res["delta_R"],
        xgb_prob=res["P_t"], ae_score=res["A_t"], lstm_prob=res["B_t"],
        uncertainty=res["U_t"], rl_action=res["action"], action_name=res["action_name"],
        is_alert=res["is_alert"], shap_top_features=res["shap"],
        timestamp=datetime.utcnow().isoformat()
    )


# ─────────────────────────────────────────────────────────────────────────────
# /simulate — zero-day demo endpoint (restricted model set)
# ─────────────────────────────────────────────────────────────────────────────

@app.post("/simulate", response_model=SimulateResponse)
async def simulate_attack(req: SimulateRequest):
    if not _models:
        raise HTTPException(status_code=503, detail="Models not loaded yet")
    if not _restricted:
        raise HTTPException(
            status_code=503,
            detail=(
                "Restricted models not loaded. "
                "Run: python3 scripts/save_restricted_models.py  then restart the API."
            )
        )

    # ── Determine known vs unseen from training manifest (programmatic, not hardcoded) ──
    trained_cats = _restricted["trained_categories"]
    is_known = req.attack_cat in trained_cats

    known_label = (
        "Known — seen in training" if is_known
        else "Unseen — held out (zero-day test)"
    )

    # ── Sample a REAL row from the test CSV for this category ──
    test_df = _models["test_df"]
    cat_df  = test_df[test_df["attack_cat"].str.strip() == req.attack_cat]
    if cat_df.empty:
        raise HTTPException(
            status_code=404,
            detail=f"No test samples found for category: {req.attack_cat}"
        )

    test_set_size = int(len(cat_df))

    # Sample without replacement from test set — different row each call
    row = cat_df.sample(n=1).iloc[0]
    row_index = int(row.name)
    true_label = int(row.get("label", 0))

    num_cols  = _models["num_cols"]
    feat_cols = _models["feature_cols"]

    X_num  = np.array([[row.get(c, 0.0) for c in num_cols]], dtype=np.float32)
    X_tree = np.array([[row.get(c, 0.0) for c in feat_cols]], dtype=np.float32)

    # ── Run through restricted pipeline ── (same code path as production)
    src_key = f"_sim_{req.attack_cat}"
    res = _run_pipeline(X_num, X_tree, _restricted, src_key=src_key, attack_cat=req.attack_cat)

    # ── Persist to incident DB (same path as /analyze) ──
    srcip = f"10.{random.randint(0,255)}.{random.randint(0,255)}.{random.randint(1,254)}"
    dstip = f"192.168.{random.randint(0,255)}.{random.randint(1,254)}"

    incident_data = {
        "timestamp": datetime.utcnow(),
        "srcip": srcip, "dstip": dstip, "proto": str(row.get("proto", "tcp")),
        "true_label": true_label, "attack_cat": req.attack_cat,
        "xgb_prob": res["P_t"], "ae_score": res["A_t"], "lstm_prob": res["B_t"],
        "risk_score": res["R_t"], "risk_delta": res["delta_R"], "risk_hist": res["H_t"],
        "uncertainty": res["U_t"], "rl_action": res["action"], "action_name": res["action_name"],
        "shap_top": res["shap"], "is_alert": res["is_alert"],
    }
    with Session(_engine) as session:
        inc = save_incident(session, incident_data)
        incident_id = inc.id

    # ── Correctness verdict from ground truth ──
    verdict, verdict_ok = _compute_correctness(true_label, res["action_name"])

    # ── Narrative from real numbers ──
    narrative = _build_narrative(
        is_known, req.attack_cat,
        res["A_t"], res["B_t"], res["P_t"],
        res["action_name"], res["U_t"]
    )

    # ── Aggregate category stats from saved files ──
    cat_stats = _get_category_stats(req.attack_cat, is_known)
    cat_stats["test_set_row_count"] = test_set_size
    cat_stats["row_index_sampled"] = row_index

    return SimulateResponse(
        incident_id=incident_id,
        row_index=row_index,
        true_label=true_label,
        true_attack_cat=req.attack_cat,
        is_known=is_known,
        known_label=known_label,
        test_set_size=test_set_size,
        xgb_prob=res["P_t"],
        ae_score=res["A_t"],
        lstm_prob=res["B_t"],
        uncertainty=res["U_t"],
        risk_score=res["R_t"],
        risk_delta=res["delta_R"],
        risk_hist=res["H_t"],
        rl_action=res["action"],
        action_name=res["action_name"],
        is_alert=res["is_alert"],
        shap_top_features=res["shap"],
        correctness_verdict=verdict,
        correctness_ok=verdict_ok,
        narrative=narrative,
        category_stats=cat_stats,
        timestamp=datetime.utcnow().isoformat(),
    )


# ─────────────────────────────────────────────────────────────────────────────
# /categories — returns all 10 labels with known/unseen status from manifest
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/categories")
async def get_categories():
    """
    Returns the 10 UNSW-NB15 attack categories with their known/unseen status
    (determined programmatically from the training manifest, not a hardcoded list)
    plus test-set row counts for transparency.
    """
    if not _models:
        raise HTTPException(status_code=503, detail="Models not loaded yet")

    test_df     = _models["test_df"]
    trained_cats = _restricted.get("trained_categories", set()) if _restricted else set()

    counts = test_df['attack_cat'].str.strip().value_counts().to_dict()

    all_cats = sorted(counts.keys())
    result = []
    for cat in all_cats:
        is_known = cat in trained_cats if trained_cats else True
        result.append({
            "name": cat,
            "is_known": is_known,
            "known_label": (
                "Known — seen in training" if is_known
                else "Unseen — held out (zero-day test)"
            ),
            "test_set_count": counts.get(cat, 0),
            "small_sample_warning": counts.get(cat, 0) < 100,
        })

    return {"categories": result, "restricted_models_loaded": bool(_restricted)}


# ─────────────────────────────────────────────────────────────────────────────
# /incidents/recent  /incidents/alerts  /health
# ─────────────────────────────────────────────────────────────────────────────

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
    return {
        "status": "ok",
        "models_loaded": bool(_models),
        "restricted_models_loaded": bool(_restricted),
        "held_out_categories": (
            _restricted["manifest"]["held_out_categories"] if _restricted else []
        ),
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.api:app", host="0.0.0.0", port=8000, reload=False)
