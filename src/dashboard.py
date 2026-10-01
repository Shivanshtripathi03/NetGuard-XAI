"""
NetGuard-XAI Streamlit Dashboard (Phase 7 + Simulation)
==========================================================
Live interactive dashboard. All data comes from real model runs — no mocks.

Tab 5: Attack Simulation
  - Uses the restricted (Phase 5) model set — Reconnaissance and Fuzzers
    are genuinely unseen, not staged.
  - Samples a real test-set row every run (different each time).
  - Displays: model scores, risk panel, SHAP chart, RL decision,
    correctness verdict, narrative, and aggregate performance for the category.
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'

import torch
torch.set_num_threads(1)

import json
import pickle
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import requests
import time

# ─────────────────────────────────────────────────────────────────────────────
# Page Config
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="NetGuard-XAI | Real-time Intrusion Detection",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────────────────────────────────────
# Premium CSS — dark cyberpunk aesthetic
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700;900&family=JetBrains+Mono:wght@400;700&display=swap');

:root {
    --bg-primary:   #0a0e1a;
    --bg-secondary: #0d1120;
    --bg-card:      #111827;
    --accent-blue:  #3b82f6;
    --accent-cyan:  #06b6d4;
    --accent-green: #10b981;
    --accent-red:   #ef4444;
    --accent-amber: #f59e0b;
    --accent-purple:#8b5cf6;
    --text-primary: #f1f5f9;
    --text-muted:   #64748b;
    --border:       #1e293b;
    --glow-blue:    0 0 24px rgba(59,130,246,0.35);
    --glow-red:     0 0 24px rgba(239,68,68,0.4);
    --glow-green:   0 0 24px rgba(16,185,129,0.35);
    --glow-cyan:    0 0 24px rgba(6,182,212,0.35);
}

html, body, [data-testid="stAppViewContainer"] {
    background-color: var(--bg-primary) !important;
    font-family: 'Inter', sans-serif !important;
    color: var(--text-primary) !important;
}

::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: var(--bg-primary); }
::-webkit-scrollbar-thumb { background: #1e293b; border-radius: 4px; }
::-webkit-scrollbar-thumb:hover { background: var(--accent-blue); }

[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0d1120 0%, #0a0e1a 100%) !important;
    border-right: 1px solid var(--border) !important;
}

.ng-header {
    background: linear-gradient(135deg, rgba(13,17,32,0.95) 0%, rgba(17,24,39,0.95) 50%, rgba(10,14,26,0.95) 100%);
    backdrop-filter: blur(16px);
    border: 1px solid rgba(59,130,246,0.25);
    border-radius: 16px;
    padding: 24px 32px;
    margin-bottom: 24px;
    box-shadow: var(--glow-blue);
    position: relative;
    overflow: hidden;
}
.ng-header::before {
    content: '';
    position: absolute;
    top: 0; left: -100%; width: 300%;
    height: 2px;
    background: linear-gradient(90deg, transparent, var(--accent-cyan), var(--accent-blue), var(--accent-purple), transparent);
    animation: shimmerLine 6s linear infinite;
}
@keyframes shimmerLine { 0% { transform: translateX(0); } 100% { transform: translateX(50%); } }

.ng-title {
    font-size: 2.2rem; font-weight: 900;
    background: linear-gradient(135deg, #3b82f6 0%, #06b6d4 50%, #8b5cf6 100%);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    margin: 0; letter-spacing: -0.5px;
}
.ng-subtitle {
    font-size: 0.9rem; color: var(--text-muted); margin-top: 6px;
    font-family: 'JetBrains Mono', monospace;
}

.metric-card {
    background: rgba(17,24,39,0.75);
    backdrop-filter: blur(12px);
    border: 1px solid var(--border);
    border-radius: 14px; padding: 20px; text-align: center;
    transition: transform 0.25s cubic-bezier(0.16,1,0.3,1), box-shadow 0.25s, border-color 0.25s;
    position: relative; overflow: hidden;
}
.metric-card::before {
    content: ''; position: absolute; top: 0; left: 0; right: 0;
    height: 2px; transition: height 0.25s ease;
}
.metric-card:hover { transform: translateY(-4px); border-color: rgba(59,130,246,0.5); }
.metric-card.green::before  { background: var(--accent-green); }
.metric-card.red::before    { background: var(--accent-red); }
.metric-card.blue::before   { background: var(--accent-blue); }
.metric-card.amber::before  { background: var(--accent-amber); }
.metric-card.purple::before { background: var(--accent-purple); }
.metric-card.cyan::before   { background: var(--accent-cyan); }

.metric-value { font-size: 2.3rem; font-weight: 700; font-family: 'JetBrains Mono', monospace; letter-spacing: -0.5px; }
.metric-label { font-size: 0.72rem; color: var(--text-muted); text-transform: uppercase; letter-spacing: 1.5px; margin-top: 6px; font-weight: 600; }

.status-badge {
    display: inline-block; padding: 3px 12px; border-radius: 999px;
    font-size: 0.75rem; font-weight: 600; font-family: 'JetBrains Mono', monospace; letter-spacing: 0.5px;
}
.badge-alert  { background: rgba(239,68,68,0.18); color: #ef4444; border: 1px solid rgba(239,68,68,0.5); animation: alertPulse 2s ease-in-out infinite; }
.badge-normal { background: rgba(16,185,129,0.15); color: #10b981; border: 1px solid rgba(16,185,129,0.4); }
.badge-warn   { background: rgba(245,158,11,0.18); color: #f59e0b; border: 1px solid rgba(245,158,11,0.5); }
.badge-known  { background: rgba(59,130,246,0.15); color: #3b82f6; border: 1px solid rgba(59,130,246,0.4); }
.badge-unseen { background: rgba(139,92,246,0.15); color: #8b5cf6; border: 1px solid rgba(139,92,246,0.5); animation: unseenPulse 3s ease-in-out infinite; }
.badge-correct { background: rgba(16,185,129,0.15); color: #10b981; border: 1px solid rgba(16,185,129,0.5); }
.badge-wrong   { background: rgba(239,68,68,0.15); color: #ef4444; border: 1px solid rgba(239,68,68,0.5); }

@keyframes alertPulse { 0%,100%{box-shadow:0 0 4px rgba(239,68,68,0.3);} 50%{box-shadow:0 0 16px rgba(239,68,68,0.7);} }
@keyframes unseenPulse { 0%,100%{box-shadow:0 0 4px rgba(139,92,246,0.3);} 50%{box-shadow:0 0 16px rgba(139,92,246,0.7);} }

.section-header {
    font-size: 0.95rem; font-weight: 700; color: var(--accent-cyan);
    text-transform: uppercase; letter-spacing: 2px;
    border-bottom: 1px solid var(--border); padding-bottom: 8px; margin-bottom: 16px;
    font-family: 'JetBrains Mono', monospace;
}
.live-dot {
    display: inline-block; width: 8px; height: 8px; border-radius: 50%;
    background: var(--accent-green); animation: pulse 1.5s ease-in-out infinite; margin-right: 6px;
}
@keyframes pulse { 0%,100%{opacity:1;box-shadow:0 0 0 0 rgba(16,185,129,0.7);} 50%{opacity:0.8;box-shadow:0 0 0 8px rgba(16,185,129,0);} }

.sim-event-box {
    background: rgba(13,17,32,0.85);
    border: 1px solid rgba(59,130,246,0.3);
    border-radius: 12px; padding: 16px 20px; margin: 8px 0;
    font-family: 'JetBrains Mono', monospace; font-size: 0.82rem;
}
.narrative-box {
    background: rgba(139,92,246,0.08);
    border: 1px solid rgba(139,92,246,0.3);
    border-radius: 12px; padding: 16px 20px; margin: 8px 0;
    font-size: 0.9rem; line-height: 1.6; color: var(--text-primary);
}
.verdict-box {
    border-radius: 12px; padding: 16px; text-align: center; margin: 8px 0;
}

button[data-baseweb="tab"] {
    background: transparent !important;
    border-radius: 8px 8px 0 0 !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.85rem !important; font-weight: 600 !important;
    color: var(--text-muted) !important; transition: color 0.2s ease !important;
}
button[data-baseweb="tab"][aria-selected="true"] { color: var(--accent-cyan) !important; }

[data-testid="stMetricValue"] { color: var(--text-primary) !important; }
div[data-testid="stDataFrame"] { border-radius: 10px; overflow: hidden; border: 1px solid var(--border); }

.cat-button {
    background: rgba(17,24,39,0.9);
    border: 1px solid rgba(59,130,246,0.3);
    border-radius: 10px; padding: 10px 14px; margin: 4px;
    cursor: pointer; transition: all 0.2s ease;
    font-family: 'JetBrains Mono', monospace; font-size: 0.82rem;
    display: inline-block; text-align: center;
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────
API_BASE     = "http://localhost:8000"
RESULTS_DIR  = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")

ACTION_COLORS = {
    "Allow":     "#10b981",
    "Monitor":   "#3b82f6",
    "RateLimit": "#f59e0b",
    "Block":     "#ef4444",
    "Isolate":   "#8b5cf6",
}
RISK_THRESHOLDS = {"normal": 0.3, "warning": 0.5, "critical": 0.8}

# Category ordering (deterministic)
CATEGORIES_ORDERED = ["Normal", "Generic", "Exploits", "DoS", "Analysis",
                      "Backdoor", "Shellcode", "Worms", "Fuzzers", "Reconnaissance"]

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data(ttl=5)
def fetch_recent_incidents(limit=200):
    try:
        r = requests.get(f"{API_BASE}/incidents/recent", params={"limit": limit}, timeout=3)
        if r.status_code == 200:
            return pd.DataFrame(r.json())
    except Exception:
        pass
    return _load_from_results_fallback()


def _load_from_results_fallback():
    traj_path = os.path.join(RESULTS_DIR, "risk_trajectory.json")
    if os.path.exists(traj_path):
        with open(traj_path) as f:
            data = json.load(f)
        rows = [
            {
                "id": i, "timestamp": datetime.utcnow().isoformat(),
                "risk_score": e["risk"]["R_t"], "risk_delta": e["risk"]["delta_R"],
                "xgb_prob": e.get("P_t", 0), "ae_score": e.get("A_t", 0),
                "lstm_prob": e.get("B_t", 0), "true_label": e.get("true_label", 0),
                "attack_cat": e.get("attack_cat", "Normal"),
                "is_alert": e["risk"]["R_t"] >= 0.5, "action_name": "N/A", "uncertainty": 0.0, "shap_top": []
            }
            for i, e in enumerate(data)
        ]
        return pd.DataFrame(rows)
    return pd.DataFrame()


@st.cache_data(ttl=30)
def fetch_categories():
    """Fetch category info from API (programmatic known/unseen from manifest)."""
    try:
        r = requests.get(f"{API_BASE}/categories", timeout=5)
        if r.status_code == 200:
            return r.json()
    except Exception:
        pass
    return None


def risk_color(score: float):
    if score >= RISK_THRESHOLDS["critical"]:  return "#ef4444"
    if score >= RISK_THRESHOLDS["warning"]:   return "#f59e0b"
    return "#10b981"


def risk_badge(score: float):
    if score >= RISK_THRESHOLDS["critical"]:
        return '<span class="status-badge badge-alert">⚠ CRITICAL</span>'
    if score >= RISK_THRESHOLDS["warning"]:
        return '<span class="status-badge badge-warn">⚡ WARNING</span>'
    return '<span class="status-badge badge-normal">✓ NORMAL</span>'


# ─────────────────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<p class="section-header">⚙ Control Panel</p>', unsafe_allow_html=True)
    auto_refresh  = st.toggle("Auto-Refresh (Live Mode)", value=True)
    refresh_rate  = st.slider("Refresh interval (s)", 3, 30, 5)
    st.divider()
    st.markdown('<p class="section-header">🎯 Alert Thresholds</p>', unsafe_allow_html=True)
    warn_thresh = st.slider("Warning threshold",  0.1, 0.9, 0.5, 0.05)
    crit_thresh = st.slider("Critical threshold", 0.1, 1.0, 0.8, 0.05)
    st.divider()
    st.markdown('<p class="section-header">📊 View Settings</p>', unsafe_allow_html=True)
    max_incidents = st.selectbox("Incidents to show", [50, 100, 200], index=1)
    show_normal   = st.checkbox("Show Normal traffic", value=True)
    st.divider()
    st.caption("NetGuard-XAI v2.0 | UNSW-NB15")
    st.caption("Restricted model deployed — Reconnaissance & Fuzzers are genuine zero-day categories")

# ─────────────────────────────────────────────────────────────────────────────
# Header
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="ng-header">
    <h1 class="ng-title">🛡️ NetGuard-XAI</h1>
    <p class="ng-subtitle">
        <span class="live-dot"></span>
        Real-time Intrusion Detection · Adaptive RL Containment · SHAP Explainability · Zero-Day Demo
    </p>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# Tabs
# ─────────────────────────────────────────────────────────────────────────────
tab_main, tab_shap, tab_rl, tab_phase_results, tab_simulate = st.tabs([
    "📡 Live Monitor", "🔍 SHAP Explanations", "🤖 RL Policy",
    "📊 Phase Results", "⚡ Simulate Attack"
])

# ═══════════════════════════════════════════════════════════════════════════
# TAB 1: Live Monitor
# ═══════════════════════════════════════════════════════════════════════════
with tab_main:
    df = fetch_recent_incidents(max_incidents)

    if df.empty:
        st.warning("No incidents found. Start the API: `python3 -m uvicorn src.api:app --port 8000`")
    else:
        if not show_normal and 'is_alert' in df.columns:
            df = df[df['is_alert'] == True]

        total      = len(df)
        alerts     = int(df['is_alert'].sum()) if 'is_alert' in df.columns else 0
        avg_risk   = float(df['risk_score'].mean()) if 'risk_score' in df.columns else 0.0
        avg_unc    = float(df['uncertainty'].mean()) if 'uncertainty' in df.columns else 0.0
        attack_rate= alerts / total * 100 if total > 0 else 0

        cols = st.columns(5)
        kpis = [
            ("Total Events",   f"{total:,}",           "blue",   "📡"),
            ("Active Alerts",  f"{alerts:,}",           "red" if alerts > 0 else "green", "🚨"),
            ("Attack Rate",    f"{attack_rate:.1f}%",   "amber",  "📈"),
            ("Avg Risk Score", f"{avg_risk:.3f}",       "red" if avg_risk > 0.5 else "green", "⚡"),
            ("Avg Uncertainty",f"{avg_unc:.3f}",        "purple", "🎲"),
        ]
        for col, (label, val, color, icon) in zip(cols, kpis):
            with col:
                hue = {"blue":"accent-blue","red":"accent-red","amber":"accent-amber",
                       "green":"accent-green","purple":"accent-purple"}.get(color,"accent-blue")
                st.markdown(f"""
                <div class="metric-card {color}">
                    <div style="font-size:1.6rem">{icon}</div>
                    <div class="metric-value" style="color:var(--{hue})">{val}</div>
                    <div class="metric-label">{label}</div>
                </div>
                """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        if 'risk_score' in df.columns:
            st.markdown('<p class="section-header">📈 Risk Score Timeline</p>', unsafe_allow_html=True)
            x_vals    = list(range(len(df)))
            risk_vals = df['risk_score'].tolist()
            fig_tl = go.Figure()
            fig_tl.add_trace(go.Scatter(
                x=x_vals, y=risk_vals,
                fill='tozeroy', fillcolor='rgba(59,130,246,0.08)',
                line=dict(color='#3b82f6', width=2), name='Risk R_t', mode='lines'
            ))
            alert_idx  = [i for i, v in enumerate(df.get('is_alert', pd.Series()).tolist()) if v]
            alert_vals = [risk_vals[i] for i in alert_idx]
            if alert_idx:
                fig_tl.add_trace(go.Scatter(
                    x=alert_idx, y=alert_vals, mode='markers',
                    marker=dict(color='#ef4444', size=8, symbol='x'), name='Alert'
                ))
            fig_tl.add_hline(y=warn_thresh, line_dash="dash", line_color="#f59e0b",
                             annotation_text=f"Warning ({warn_thresh})")
            fig_tl.add_hline(y=crit_thresh, line_dash="dash", line_color="#ef4444",
                             annotation_text=f"Critical ({crit_thresh})")
            fig_tl.update_layout(
                paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                font=dict(color='#94a3b8'), height=280,
                legend=dict(bgcolor='rgba(0,0,0,0)'), margin=dict(t=20,b=20,l=0,r=0),
                yaxis=dict(range=[0,1.05], gridcolor='rgba(30,41,59,0.8)'),
                xaxis=dict(gridcolor='rgba(30,41,59,0.8)')
            )
            st.plotly_chart(fig_tl, use_container_width=True)

        c1, c2 = st.columns([2, 1])
        with c1:
            if all(c in df.columns for c in ['xgb_prob', 'ae_score', 'lstm_prob']):
                st.markdown('<p class="section-header">🎯 Model Score Distribution</p>', unsafe_allow_html=True)
                fig_s = go.Figure()
                for col_name, color, name in [
                    ('xgb_prob', '#3b82f6', 'XGBoost'),
                    ('ae_score',  '#06b6d4', 'Autoencoder'),
                    ('lstm_prob', '#8b5cf6', 'LSTM')
                ]:
                    if col_name in df.columns:
                        fig_s.add_trace(go.Box(y=df[col_name], name=name, marker_color=color, boxmean=True))
                fig_s.update_layout(
                    paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                    font=dict(color='#94a3b8'), height=260, margin=dict(t=20,b=20),
                    showlegend=False, yaxis=dict(range=[0,1], gridcolor='rgba(30,41,59,0.8)')
                )
                st.plotly_chart(fig_s, use_container_width=True)

        with c2:
            if 'attack_cat' in df.columns:
                st.markdown('<p class="section-header">🗂️ Attack Categories</p>', unsafe_allow_html=True)
                cat_counts = (df[df['is_alert'] == True]['attack_cat'].value_counts()
                              if 'is_alert' in df.columns else df['attack_cat'].value_counts())
                if not cat_counts.empty:
                    fig_pie = go.Figure(go.Pie(
                        labels=cat_counts.index, values=cat_counts.values, hole=0.55,
                        marker=dict(colors=['#ef4444','#f59e0b','#3b82f6','#8b5cf6',
                                            '#06b6d4','#10b981','#ec4899','#14b8a6','#f97316'])
                    ))
                    fig_pie.update_layout(
                        paper_bgcolor='rgba(0,0,0,0)', font=dict(color='#94a3b8'),
                        height=260, showlegend=True,
                        legend=dict(bgcolor='rgba(0,0,0,0)', font=dict(size=10)),
                        margin=dict(t=10,b=10,l=10,r=10)
                    )
                    st.plotly_chart(fig_pie, use_container_width=True)

        st.markdown('<p class="section-header">📋 Recent Incident Log</p>', unsafe_allow_html=True)
        display_cols = [c for c in ['id','timestamp','attack_cat','risk_score','action_name',
                                     'xgb_prob','ae_score','lstm_prob','uncertainty','is_alert']
                        if c in df.columns]
        st.dataframe(
            df[display_cols].head(30).style
                .format({c: '{:.3f}' for c in ['risk_score','xgb_prob','ae_score','lstm_prob','uncertainty']
                         if c in df.columns})
                .background_gradient(subset=['risk_score'] if 'risk_score' in display_cols else [], cmap='RdYlGn_r'),
            use_container_width=True, height=300
        )

# ═══════════════════════════════════════════════════════════════════════════
# TAB 2: SHAP Explanations
# ═══════════════════════════════════════════════════════════════════════════
with tab_shap:
    st.markdown('<p class="section-header">🔍 SHAP Incident Explanations</p>', unsafe_allow_html=True)
    shap_path = os.path.join(RESULTS_DIR, "shap_incidents.json")
    if os.path.exists(shap_path):
        with open(shap_path) as f:
            shap_data = json.load(f)

        incident_labels = [
            f"Incident {i+1}: true={d['true_label']} pred={d['predicted_label']} ({d['attack_cat']})"
            for i, d in enumerate(shap_data)
        ]
        selected = st.selectbox("Select incident to explain:", incident_labels)
        idx = incident_labels.index(selected)
        inc = shap_data[idx]

        col1, col2 = st.columns([1, 2])
        with col1:
            st.markdown(f"""
            <div class="metric-card {'red' if inc['true_label'] == 1 else 'green'}">
                <div style="font-size:2rem">{'🔴' if inc['true_label'] == 1 else '🟢'}</div>
                <div class="metric-value">{inc['attack_cat']}</div>
                <div class="metric-label">Ground Truth</div>
            </div>
            """, unsafe_allow_html=True)
            outcome = {(1,1):"✅ True Positive",(0,1):"❌ False Positive",
                       (1,0):"⚠️ False Negative",(0,0):"✅ True Negative"}.get(
                       (inc['true_label'], inc['predicted_label']), "?")
            st.markdown(f"**Outcome:** {outcome}")

        with col2:
            features = [d['feature'] for d in inc['top_shap_features']]
            values   = [d['shap_value'] for d in inc['top_shap_features']]
            fig_shap = go.Figure(go.Bar(
                x=values, y=features, orientation='h',
                marker_color=['#ef4444' if v > 0 else '#10b981' for v in values]
            ))
            fig_shap.update_layout(
                title="SHAP Feature Contributions (XGBoost)",
                paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                font=dict(color='#94a3b8'), height=320,
                xaxis=dict(title="SHAP Value", gridcolor='rgba(30,41,59,0.8)'),
                yaxis=dict(autorange='reversed'), margin=dict(t=40,b=20,l=20,r=20)
            )
            st.plotly_chart(fig_shap, use_container_width=True)
    else:
        st.info("SHAP results not found. Run Phase 3 pipeline first.")

# ═══════════════════════════════════════════════════════════════════════════
# TAB 3: RL Policy
# ═══════════════════════════════════════════════════════════════════════════
with tab_rl:
    st.markdown('<p class="section-header">🤖 DQN Policy Performance</p>', unsafe_allow_html=True)
    dqn_path   = os.path.join(RESULTS_DIR, "dqn_test_eval.json")
    curve_path = os.path.join(RESULTS_DIR, "dqn_training_curve.png")

    if os.path.exists(dqn_path):
        with open(dqn_path) as f:
            rl_eval = json.load(f)
        c1, c2, c3 = st.columns(3)
        with c1: st.metric("Attack Containment Rate", f"{rl_eval['attack_containment_rate']:.1%}")
        with c2: st.metric("FP Intervention Rate",    f"{rl_eval['false_positive_intervention_rate']:.1%}")
        with c3: st.metric("Avg Episode Reward",       f"{rl_eval['average_reward']:.2f}")

        if 'action_distribution' in rl_eval:
            st.markdown('<p class="section-header">⚡ Action Distribution</p>', unsafe_allow_html=True)
            actions = rl_eval['action_distribution']
            fig_bar = go.Figure(go.Bar(
                x=list(actions.keys()), y=list(actions.values()),
                marker_color=[ACTION_COLORS.get(a, '#3b82f6') for a in actions.keys()]
            ))
            fig_bar.update_layout(
                paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                font=dict(color='#94a3b8'), height=280,
                yaxis=dict(gridcolor='rgba(30,41,59,0.8)'), margin=dict(t=20,b=20)
            )
            st.plotly_chart(fig_bar, use_container_width=True)
    else:
        st.info("RL evaluation results not found. Run Phase 4 pipeline first.")

    if os.path.exists(curve_path):
        st.markdown('<p class="section-header">📈 DQN Training Curve</p>', unsafe_allow_html=True)
        st.image(curve_path, use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════
# TAB 4: Phase Results
# ═══════════════════════════════════════════════════════════════════════════
with tab_phase_results:
    st.markdown('<p class="section-header">📊 All Phase Metrics</p>', unsafe_allow_html=True)
    results_files = {
        "XGBoost (Phase 2)":     "xgboost_test_metrics.json",
        "Autoencoder (Phase 2)": "autoencoder_test_metrics.json",
        "LSTM (Phase 2)":        "lstm_test_metrics.json",
        "Phase 5 Zero-Day":      "phase5_zeroday_results.json",
        "Phase 6 Ablation":      "phase6_ablation.json",
    }
    for label, fname in results_files.items():
        path = os.path.join(RESULTS_DIR, fname)
        if os.path.exists(path):
            with open(path) as f:
                data = json.load(f)
            with st.expander(f"📄 {label}", expanded=False):
                st.json(data)
        else:
            st.caption(f"⏳ {label} — not yet generated")

    traj_img = os.path.join(RESULTS_DIR, "risk_trajectory.png")
    if os.path.exists(traj_img):
        st.markdown('<p class="section-header">📈 Risk Trajectory (Phase 3)</p>', unsafe_allow_html=True)
        st.image(traj_img, use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════
# TAB 5: Simulate Attack — the main new feature
# ═══════════════════════════════════════════════════════════════════════════
with tab_simulate:
    st.markdown('<p class="section-header">⚡ Live Attack Simulation — Restricted Model</p>',
                unsafe_allow_html=True)

    # ── Introductory explainer ──
    st.markdown("""
    <div style="background:rgba(13,17,32,0.9);border:1px solid rgba(59,130,246,0.25);
                border-radius:12px;padding:16px 20px;margin-bottom:16px;">
        <p style="margin:0;font-size:0.88rem;color:#94a3b8;line-height:1.7;">
            <strong style="color:#06b6d4;">Deployed model set:</strong>
            The <em>restricted</em> pipeline trained with Reconnaissance and Fuzzers excluded from
            all training data (Phase 5). This means:<br>
            &nbsp;&nbsp;• <span style="color:#3b82f6;">Known categories</span> (Normal, Generic, Exploits, DoS, Analysis, Backdoor, Shellcode, Worms)
            trigger <strong>trained-signature detection</strong> via XGBoost + AE + LSTM.<br>
            &nbsp;&nbsp;• <span style="color:#8b5cf6;">Unseen categories</span> (Reconnaissance, Fuzzers)
            trigger <strong>genuine zero-day handling</strong> — the models truly never saw these during training.
            Detection relies on anomaly (AE) and behavioral (LSTM) signals only.
        </p>
        <p style="margin:8px 0 0 0;font-size:0.8rem;color:#64748b;">
            Every run samples a different real row from the UNSW-NB15 test CSV. The row index is shown
            for reproducibility — you can verify against the original dataset.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # ── Fetch categories from API (dynamic known/unseen from manifest) ──
    cat_info = fetch_categories()

    if cat_info is None:
        st.warning("API offline. Start: `python3 -m uvicorn src.api:app --host 0.0.0.0 --port 8000`")
        cat_info = {
            "categories": [
                {"name": c, "is_known": c not in ["Reconnaissance", "Fuzzers"],
                 "known_label": ("Known — seen in training" if c not in ["Reconnaissance","Fuzzers"]
                                 else "Unseen — held out (zero-day test)"),
                 "test_set_count": 0, "small_sample_warning": c in ["Worms","Shellcode"]}
                for c in CATEGORIES_ORDERED
            ],
            "restricted_models_loaded": False
        }

    if not cat_info.get("restricted_models_loaded", False):
        st.error("""
        **Restricted models not loaded.**
        Run this once to train and save them:
        ```
        python3 scripts/save_restricted_models.py
        ```
        Then restart the API.
        """)

    cats_by_name = {c["name"]: c for c in cat_info["categories"]}

    # Sort to display Known first, then Unseen
    known_cats  = [c for c in CATEGORIES_ORDERED if cats_by_name.get(c, {}).get("is_known", True)]
    unseen_cats = [c for c in CATEGORIES_ORDERED if not cats_by_name.get(c, {}).get("is_known", True)]

    # ── Category selector with visible Known/Unseen tags ──
    st.markdown('<p class="section-header">🎯 Select Attack Category</p>', unsafe_allow_html=True)

    col_known, col_unseen = st.columns([3, 2])

    with col_known:
        st.markdown("**Known — Seen in Training** <span class='status-badge badge-known'>Trained signature detection</span>",
                    unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        for cat in known_cats:
            info = cats_by_name.get(cat, {})
            count = info.get("test_set_count", 0)
            warn  = info.get("small_sample_warning", False)
            label_suffix = f" ⚠ ({count} samples)" if warn else f" ({count:,} test rows)"
            st.markdown(f"<small style='color:#64748b'>{cat}{label_suffix}</small>",
                        unsafe_allow_html=True)

    with col_unseen:
        st.markdown("**Unseen — Held Out** <span class='status-badge badge-unseen'>Zero-day test</span>",
                    unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        for cat in unseen_cats:
            info = cats_by_name.get(cat, {})
            count = info.get("test_set_count", 0)
            st.markdown(
                f"<small style='color:#8b5cf6'>{cat} ({count:,} test rows) — "
                f"no training examples of this category exist in the deployed model</small>",
                unsafe_allow_html=True)

    # Build dropdown options with tags
    all_cats_ordered = known_cats + unseen_cats
    dropdown_options = []
    for cat in all_cats_ordered:
        info = cats_by_name.get(cat, {})
        is_known = info.get("is_known", True)
        count = info.get("test_set_count", 0)
        warn  = info.get("small_sample_warning", False)
        tag = "✅ Known" if is_known else "🔮 Unseen"
        size_note = " ⚠ small sample" if warn else ""
        dropdown_options.append(f"{cat} [{tag}{size_note}] — {count:,} test rows")

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<p class="section-header">🚀 Run Simulation</p>', unsafe_allow_html=True)

    selected_display = st.selectbox(
        "Attack category to simulate:",
        dropdown_options,
        index=0,
        key="sim_cat_select"
    )
    # Extract category name from display string
    selected_cat = selected_display.split(" [")[0]
    selected_info = cats_by_name.get(selected_cat, {})
    is_selected_known = selected_info.get("is_known", True)
    selected_count = selected_info.get("test_set_count", 0)
    selected_warn = selected_info.get("small_sample_warning", False)

    # Warn on small sample categories
    if selected_warn:
        st.warning(
            f"⚠️ **Small sample warning:** {selected_cat} has only {selected_count} rows in the "
            f"test set. A single result provides limited statistical evidence — interpret alongside "
            f"the aggregate metrics shown below."
        )

    # Known/unseen tag display
    if is_selected_known:
        st.markdown(
            f'<span class="status-badge badge-known">✅ Known — seen in training</span> '
            f'<small style="color:#64748b;margin-left:8px;">{selected_cat} was present in restricted training data</small>',
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            f'<span class="status-badge badge-unseen">🔮 Unseen — held out (zero-day test)</span> '
            f'<small style="color:#8b5cf6;margin-left:8px;">{selected_cat} was NEVER in restricted training — this is genuine zero-day handling</small>',
            unsafe_allow_html=True
        )

    st.markdown("<br>", unsafe_allow_html=True)
    btn_col1, btn_col2 = st.columns([2, 1])
    with btn_col1:
        run_btn = st.button(
            f"🚀 Run Simulation — {selected_cat}",
            type="primary", use_container_width=True, key="run_sim"
        )
    with btn_col2:
        run_again_btn = st.button(
            "🔄 Run Another (different row)",
            use_container_width=True, key="run_again"
        )

    # ── State for trajectory tracking ──
    if "sim_trajectory" not in st.session_state:
        st.session_state["sim_trajectory"] = []
    if "last_sim_cat" not in st.session_state:
        st.session_state["last_sim_cat"] = None

    should_run = run_btn or run_again_btn

    if should_run:
        with st.spinner(f"Pulling real {selected_cat} event from test set → running through restricted pipeline..."):
            try:
                resp = requests.post(f"{API_BASE}/simulate",
                                     json={"attack_cat": selected_cat}, timeout=30)
                if resp.status_code == 200:
                    result = resp.json()
                    error  = None
                else:
                    result = None
                    error  = f"API {resp.status_code}: {resp.text}"
            except Exception as e:
                result = None
                error  = f"API unreachable: {e}"

        if error:
            st.error(f"Simulation failed: {error}")
        else:
            # Store in trajectory
            st.session_state["sim_trajectory"].append({
                "cat": selected_cat,
                "risk": result["risk_score"],
                "action": result["action_name"],
                "row_index": result["row_index"],
                "incident_id": result["incident_id"],
            })
            st.session_state["last_sim_cat"] = selected_cat

            # ─────────────────────── RESULTS PANEL ───────────────────────

            # Header
            tag_class = "badge-known" if result["is_known"] else "badge-unseen"
            tag_text  = result["known_label"]
            st.markdown(f"""
            <div style="background:rgba(13,17,32,0.9);border:1px solid rgba(59,130,246,0.3);
                        border-radius:12px;padding:16px 20px;margin:12px 0;">
                <div style="display:flex;align-items:center;gap:12px;flex-wrap:wrap;">
                    <span style="font-size:1.1rem;font-weight:700;">{selected_cat}</span>
                    <span class="status-badge {tag_class}">{tag_text}</span>
                    <span style="color:#64748b;font-family:'JetBrains Mono';font-size:0.8rem;">
                        Incident #{result['incident_id']} · Test row {result['row_index']}
                        · True label: {'Attack' if result['true_label']==1 else 'Normal'}
                    </span>
                </div>
            </div>
            """, unsafe_allow_html=True)

            # ─── Section 1: Event Summary ───
            st.markdown('<p class="section-header" style="margin-top:16px;">📋 Sampled Event</p>',
                        unsafe_allow_html=True)
            e1, e2, e3, e4 = st.columns(4)
            with e1:
                st.markdown(f"""<div class="metric-card blue">
                    <div class="metric-value" style="color:var(--accent-blue);font-size:1.2rem">{selected_cat}</div>
                    <div class="metric-label">True Category</div></div>""", unsafe_allow_html=True)
            with e2:
                st.markdown(f"""<div class="metric-card {'red' if result['true_label']==1 else 'green'}">
                    <div class="metric-value" style="color:var(--accent-{'red' if result['true_label']==1 else 'green'})">
                        {'Attack' if result['true_label']==1 else 'Normal'}</div>
                    <div class="metric-label">True Label</div></div>""", unsafe_allow_html=True)
            with e3:
                st.markdown(f"""<div class="metric-card purple">
                    <div class="metric-value" style="color:var(--accent-purple);font-size:1.3rem">
                        #{result['row_index']}</div>
                    <div class="metric-label">Test Row Index</div></div>""", unsafe_allow_html=True)
            with e4:
                st.markdown(f"""<div class="metric-card amber">
                    <div class="metric-value" style="color:var(--accent-amber);font-size:1.3rem">
                        {result['test_set_size']:,}</div>
                    <div class="metric-label">Category Test Size</div></div>""", unsafe_allow_html=True)

            # ─── Section 2: Per-Model Scores ───
            st.markdown('<p class="section-header" style="margin-top:16px;">🎯 Model Scores</p>',
                        unsafe_allow_html=True)
            m1, m2, m3, m4 = st.columns(4)
            xgb_col  = "red" if result["xgb_prob"] > 0.5 else "green"
            ae_col   = "red" if result["ae_score"]  > 0.5 else "green"
            lstm_col = "red" if result["lstm_prob"] > 0.5 else "green"
            unc_col  = "amber"
            for col, label, val, color, note in [
                (m1, "XGBoost Prob",   result["xgb_prob"],  xgb_col,  "Supervised classifier"),
                (m2, "AE Anomaly",     result["ae_score"],  ae_col,   "Reconstruction error"),
                (m3, "LSTM Behavioral",result["lstm_prob"], lstm_col, "Sequential pattern"),
                (m4, "Uncertainty",    result["uncertainty"],unc_col, "Model disagreement (σ)"),
            ]:
                hue = {"red":"accent-red","green":"accent-green","amber":"accent-amber"}.get(color,"accent-blue")
                with col:
                    st.markdown(f"""
                    <div class="metric-card {color}">
                        <div class="metric-value" style="color:var(--{hue})">{val:.3f}</div>
                        <div class="metric-label">{label}</div>
                        <div style="font-size:0.65rem;color:#64748b;margin-top:4px">{note}</div>
                    </div>
                    """, unsafe_allow_html=True)

            # ─── Section 3: Risk Panel ───
            st.markdown('<p class="section-header" style="margin-top:16px;">⚡ Risk Engine</p>',
                        unsafe_allow_html=True)
            r1, r2, r3 = st.columns(3)
            risk_c = "red" if result["risk_score"] >= 0.8 else "amber" if result["risk_score"] >= 0.5 else "green"
            risk_hue = {"red":"accent-red","amber":"accent-amber","green":"accent-green"}[risk_c]
            with r1:
                st.markdown(f"""<div class="metric-card {risk_c}">
                    {risk_badge(result['risk_score'])}
                    <div class="metric-value" style="color:var(--{risk_hue})">{result['risk_score']:.3f}</div>
                    <div class="metric-label">Current Risk R_t</div></div>""", unsafe_allow_html=True)
            with r2:
                delta_c = "red" if result["risk_delta"] > 0.05 else "green" if result["risk_delta"] < -0.05 else "blue"
                delta_hue = {"red":"accent-red","green":"accent-green","blue":"accent-blue"}[delta_c]
                arrow = "↑" if result["risk_delta"] > 0 else "↓" if result["risk_delta"] < 0 else "→"
                st.markdown(f"""<div class="metric-card {delta_c}">
                    <div class="metric-value" style="color:var(--{delta_hue})">{arrow} {result['risk_delta']:+.3f}</div>
                    <div class="metric-label">Risk Velocity ΔR</div></div>""", unsafe_allow_html=True)
            with r3:
                st.markdown(f"""<div class="metric-card blue">
                    <div class="metric-value" style="color:var(--accent-blue)">{result['risk_hist']:.3f}</div>
                    <div class="metric-label">Historical State H_t</div></div>""", unsafe_allow_html=True)

            # Trajectory mini-chart (multi-run)
            traj = st.session_state["sim_trajectory"]
            if len(traj) > 1:
                st.markdown("**Risk trajectory across runs:**", unsafe_allow_html=False)
                fig_traj = go.Figure()
                traj_risks = [t["risk"] for t in traj]
                traj_labels= [f"{t['cat']} #{t['row_index']}" for t in traj]
                fig_traj.add_trace(go.Scatter(
                    x=list(range(len(traj_risks))), y=traj_risks,
                    mode='lines+markers', line=dict(color='#06b6d4', width=2),
                    marker=dict(size=8, color=['#ef4444' if r >= 0.5 else '#10b981' for r in traj_risks]),
                    text=traj_labels, hovertemplate='%{text}<br>Risk: %{y:.3f}',
                ))
                fig_traj.add_hline(y=0.5, line_dash="dash", line_color="#f59e0b",
                                   annotation_text="Alert threshold")
                fig_traj.update_layout(
                    paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                    font=dict(color='#94a3b8'), height=180,
                    margin=dict(t=10,b=10,l=0,r=0),
                    xaxis=dict(gridcolor='rgba(30,41,59,0.4)'),
                    yaxis=dict(range=[0,1.05], gridcolor='rgba(30,41,59,0.4)')
                )
                st.plotly_chart(fig_traj, use_container_width=True)

            # ─── Section 4: SHAP (XGBoost only — labeled explicitly) ───
            if result.get("shap_top_features"):
                st.markdown('<p class="section-header" style="margin-top:16px;">🔍 SHAP Feature Attribution</p>',
                            unsafe_allow_html=True)
                st.caption(
                    "⚠️ SHAP values below are computed from the XGBoost supervised component only. "
                    "They do not explain the Autoencoder or LSTM signals, which are model-specific and "
                    "not amenable to standard TreeSHAP attribution."
                )
                shap_df = pd.DataFrame(result["shap_top_features"])
                fig_shap = go.Figure(go.Bar(
                    x=shap_df["shap_value"], y=shap_df["feature"], orientation='h',
                    marker=dict(
                        color=['#ef4444' if v > 0 else '#10b981' for v in shap_df["shap_value"]],
                        line=dict(width=0)
                    )
                ))
                fig_shap.update_layout(
                    title=f"XGBoost SHAP — Top 8 features (row {result['row_index']})",
                    paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                    font=dict(color='#94a3b8'), height=300,
                    xaxis=dict(title="SHAP Value", gridcolor='rgba(30,41,59,0.6)'),
                    yaxis=dict(autorange='reversed'), margin=dict(t=40,b=10,l=20,r=20)
                )
                st.plotly_chart(fig_shap, use_container_width=True)

            # ─── Section 5: RL Decision ───
            st.markdown('<p class="section-header" style="margin-top:16px;">🤖 RL Policy Decision</p>',
                        unsafe_allow_html=True)
            action_col = ACTION_COLORS.get(result["action_name"], "#3b82f6")

            st.markdown(f"""
            <div style="background:rgba(0,0,0,0.3);border:1px solid {action_col};
                        border-radius:14px;padding:20px;text-align:center;">
                <div style="font-size:0.72rem;color:#64748b;letter-spacing:2px;text-transform:uppercase">
                    DQN Policy Decision
                </div>
                <div style="font-size:2.8rem;font-weight:900;color:{action_col};
                            font-family:'JetBrains Mono';margin:8px 0;
                            text-shadow:0 0 24px {action_col}80">
                    {result['action_name'].upper()}
                </div>
                <div style="font-size:0.82rem;color:#94a3b8">
                    Uncertainty: {result['uncertainty']:.4f} &nbsp;·&nbsp;
                    Risk: {result['risk_score']:.3f}
                </div>
            </div>
            """, unsafe_allow_html=True)

            # ─── Section 6: Narrative ───
            st.markdown('<p class="section-header" style="margin-top:16px;">📝 System Narrative</p>',
                        unsafe_allow_html=True)
            st.markdown(f"""
            <div class="narrative-box">
                <span style="font-size:1rem">💬</span>
                {result['narrative']}
            </div>
            """, unsafe_allow_html=True)

            # ─── Section 7: Correctness Verdict ───
            st.markdown('<p class="section-header" style="margin-top:16px;">🏆 Correctness Verdict</p>',
                        unsafe_allow_html=True)
            v_ok     = result["correctness_ok"]
            v_text   = result["correctness_verdict"]
            v_color  = "#10b981" if v_ok else "#ef4444"
            v_bg     = "rgba(16,185,129,0.08)" if v_ok else "rgba(239,68,68,0.08)"
            v_border = "rgba(16,185,129,0.4)"  if v_ok else "rgba(239,68,68,0.4)"
            v_icon   = "✅" if v_ok else "❌"

            st.markdown(f"""
            <div style="background:{v_bg};border:1px solid {v_border};border-radius:12px;
                        padding:16px;text-align:center;margin:8px 0;">
                <div style="font-size:2rem">{v_icon}</div>
                <div style="font-size:1.15rem;font-weight:700;color:{v_color};
                            font-family:'JetBrains Mono';margin-top:4px">
                    {v_text.upper()}
                </div>
                <div style="font-size:0.8rem;color:#64748b;margin-top:8px">
                    Ground-truth label: {'Attack (label=1)' if result['true_label']==1 else 'Normal (label=0)'}
                    &nbsp;→&nbsp; Action taken: {result['action_name']}
                </div>
            </div>
            """, unsafe_allow_html=True)

            # ─── Section 8: Aggregate Category Performance ───
            cat_stats = result.get("category_stats", {})
            if cat_stats:
                st.markdown('<p class="section-header" style="margin-top:20px;">📊 Category Aggregate Performance</p>',
                            unsafe_allow_html=True)
                st.caption(
                    "Real measured performance from saved evaluation result files — "
                    "not extrapolated from this single run."
                )

                if not result["is_known"]:
                    # Zero-day stats from Phase 5
                    s1, s2, s3, s4 = st.columns(4)
                    zdr   = cat_stats.get("zero_day_recall")
                    zdf1  = cat_stats.get("zero_day_f1")
                    zdcr  = cat_stats.get("zero_day_containment_rate")
                    zdn   = cat_stats.get("zero_day_n_eval")
                    for col, label, val in [
                        (s1, "Recall (restricted model)",      f"{zdr:.3f}" if zdr is not None else "N/A"),
                        (s2, "F1 (restricted model)",          f"{zdf1:.3f}" if zdf1 is not None else "N/A"),
                        (s3, "Containment Rate",               f"{zdcr:.3f}" if zdcr is not None else "N/A"),
                        (s4, "Eval Events",                    f"{zdn:,}" if zdn is not None else "N/A"),
                    ]:
                        with col:
                            st.metric(label, val)
                    if cat_stats.get("honest_finding"):
                        st.info(f"📝 {cat_stats['honest_finding']}")
                else:
                    # Full pipeline stats
                    s1, s2, s3, s4 = st.columns(4)
                    xf1   = cat_stats.get("overall_xgb_f1")
                    xroc  = cat_stats.get("overall_xgb_roc_auc")
                    fscr  = cat_stats.get("full_system_containment_rate")
                    fsfpr = cat_stats.get("full_system_fp_rate")
                    for col, label, val in [
                        (s1, "XGBoost Macro F1",         f"{xf1:.3f}"   if xf1   is not None else "N/A"),
                        (s2, "XGBoost ROC-AUC",          f"{xroc:.3f}"  if xroc  is not None else "N/A"),
                        (s3, "System Containment Rate",  f"{fscr:.3f}"  if fscr  is not None else "N/A"),
                        (s4, "System FP Rate",           f"{fsfpr:.3f}" if fsfpr is not None else "N/A"),
                    ]:
                        with col:
                            st.metric(label, val)
                    st.caption(
                        "XGBoost metrics: binary (binary classification). "
                        "System containment rate: from Phase 4 DQN RL evaluation."
                    )

                st.markdown(
                    f"<small style='color:#64748b'>Test set rows for this category: "
                    f"{cat_stats.get('test_set_row_count','?'):,} · "
                    f"Row sampled this run: #{cat_stats.get('row_index_sampled','?')}</small>",
                    unsafe_allow_html=True
                )

    # ── Simulation run history ──
    if st.session_state["sim_trajectory"]:
        with st.expander(f"🕑 Run History ({len(st.session_state['sim_trajectory'])} runs)", expanded=False):
            hist_df = pd.DataFrame(st.session_state["sim_trajectory"])
            st.dataframe(hist_df.style.format({"risk": "{:.3f}"}), use_container_width=True)
            if st.button("Clear history", key="clear_hist"):
                st.session_state["sim_trajectory"] = []
                st.rerun()

# ─────────────────────────────────────────────────────────────────────────────
# Auto-refresh
# ─────────────────────────────────────────────────────────────────────────────
if auto_refresh:
    time.sleep(refresh_rate)
    st.rerun()
