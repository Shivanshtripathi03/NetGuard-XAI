"""
NetGuard-XAI Streamlit Dashboard (Phase 7)
Live interactive dashboard reading from the incident database.
No mocked fixtures — all data comes from real model runs.
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
import threading

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
# Custom CSS — premium dark cyberpunk aesthetic
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

/* Custom modern dark scrollbar */
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
    -webkit-backdrop-filter: blur(16px);
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

@keyframes shimmerLine {
    0% { transform: translateX(0); }
    100% { transform: translateX(50%); }
}

.ng-title {
    font-size: 2.2rem;
    font-weight: 900;
    background: linear-gradient(135deg, #3b82f6 0%, #06b6d4 50%, #8b5cf6 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin: 0;
    letter-spacing: -0.5px;
}

.ng-subtitle {
    font-size: 0.9rem;
    color: var(--text-muted);
    margin-top: 6px;
    font-family: 'JetBrains Mono', monospace;
    display: flex;
    align-items: center;
    gap: 8px;
}

/* Spotlight glassmorphism metric cards */
.metric-card {
    background: rgba(17, 24, 39, 0.75);
    backdrop-filter: blur(12px);
    -webkit-backdrop-filter: blur(12px);
    border: 1px solid var(--border);
    border-radius: 14px;
    padding: 20px;
    text-align: center;
    transition: transform 0.25s cubic-bezier(0.16, 1, 0.3, 1), box-shadow 0.25s cubic-bezier(0.16, 1, 0.3, 1), border-color 0.25s ease;
    position: relative;
    overflow: hidden;
}

.metric-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    transition: height 0.25s ease;
}

.metric-card:hover {
    transform: translateY(-4px);
    box-shadow: 0 16px 32px -8px rgba(0,0,0,0.6), var(--glow-blue);
    border-color: rgba(59,130,246,0.5);
}

.metric-card:hover::before {
    height: 3px;
}

.metric-card.green::before  { background: var(--accent-green); }
.metric-card.red::before    { background: var(--accent-red); }
.metric-card.blue::before   { background: var(--accent-blue); }
.metric-card.amber::before  { background: var(--accent-amber); }
.metric-card.purple::before { background: var(--accent-purple); }

.metric-card.green:hover  { box-shadow: 0 16px 32px -8px rgba(0,0,0,0.6), var(--glow-green); border-color: rgba(16,185,129,0.5); }
.metric-card.red:hover    { box-shadow: 0 16px 32px -8px rgba(0,0,0,0.6), var(--glow-red); border-color: rgba(239,68,68,0.5); }
.metric-card.amber:hover  { box-shadow: 0 16px 32px -8px rgba(0,0,0,0.6), 0 0 24px rgba(245,158,11,0.35); border-color: rgba(245,158,11,0.5); }
.metric-card.purple:hover { box-shadow: 0 16px 32px -8px rgba(0,0,0,0.6), 0 0 24px rgba(139,92,246,0.35); border-color: rgba(139,92,246,0.5); }

.metric-value {
    font-size: 2.3rem;
    font-weight: 700;
    font-family: 'JetBrains Mono', monospace;
    letter-spacing: -0.5px;
}
.metric-label {
    font-size: 0.72rem;
    color: var(--text-muted);
    text-transform: uppercase;
    letter-spacing: 1.5px;
    margin-top: 6px;
    font-weight: 600;
}

.status-badge {
    display: inline-block;
    padding: 3px 12px;
    border-radius: 999px;
    font-size: 0.75rem;
    font-weight: 600;
    font-family: 'JetBrains Mono', monospace;
    letter-spacing: 0.5px;
}
.badge-alert  {
    background: rgba(239,68,68,0.18);
    color: #ef4444;
    border: 1px solid rgba(239,68,68,0.5);
    animation: alertPulse 2s ease-in-out infinite;
}
.badge-normal { background: rgba(16,185,129,0.15); color: #10b981; border: 1px solid rgba(16,185,129,0.4); }
.badge-warn   { background: rgba(245,158,11,0.18); color: #f59e0b; border: 1px solid rgba(245,158,11,0.5); }

@keyframes alertPulse {
    0%, 100% { box-shadow: 0 0 4px rgba(239,68,68,0.3); }
    50%       { box-shadow: 0 0 16px rgba(239,68,68,0.7); }
}

.section-header {
    font-size: 0.95rem;
    font-weight: 700;
    color: var(--accent-cyan);
    text-transform: uppercase;
    letter-spacing: 2px;
    border-bottom: 1px solid var(--border);
    padding-bottom: 8px;
    margin-bottom: 16px;
    font-family: 'JetBrains Mono', monospace;
}

.live-dot {
    display: inline-block;
    width: 8px; height: 8px;
    border-radius: 50%;
    background: var(--accent-green);
    animation: pulse 1.5s ease-in-out infinite;
    margin-right: 6px;
}
@keyframes pulse {
    0%, 100% { opacity: 1; box-shadow: 0 0 0 0 rgba(16,185,129,0.7); }
    50%       { opacity: 0.8; box-shadow: 0 0 0 8px rgba(16,185,129,0); }
}

/* Tab styling enhancements */
button[data-baseweb="tab"] {
    background: transparent !important;
    border-radius: 8px 8px 0 0 !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.85rem !important;
    font-weight: 600 !important;
    color: var(--text-muted) !important;
    transition: color 0.2s ease !important;
}
button[data-baseweb="tab"][aria-selected="true"] {
    color: var(--accent-cyan) !important;
    border-bottom-color: var(--accent-cyan) !important;
}

[data-testid="stMetricValue"] { color: var(--text-primary) !important; }
div[data-testid="stDataFrame"] {
    border-radius: 10px;
    overflow: hidden;
    border: 1px solid var(--border);
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────
API_BASE    = "http://localhost:8000"
RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
PROCESSED_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "processed")
MODELS_DIR  = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")

ACTION_COLORS = {
    "Allow":     "#10b981",
    "Monitor":   "#3b82f6",
    "RateLimit": "#f59e0b",
    "Block":     "#ef4444",
    "Isolate":   "#8b5cf6",
}
RISK_THRESHOLDS = {"normal": 0.3, "warning": 0.5, "critical": 0.8}

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
    """Load cached risk trajectory from Phase 3 results if API is offline."""
    traj_path = os.path.join(RESULTS_DIR, "risk_trajectory.json")
    if os.path.exists(traj_path):
        with open(traj_path) as f:
            data = json.load(f)
        rows = []
        for i, e in enumerate(data):
            rows.append({
                "id": i,
                "timestamp": datetime.utcnow().isoformat(),
                "risk_score": e["risk"]["R_t"],
                "risk_delta": e["risk"]["delta_R"],
                "xgb_prob":   e.get("P_t", 0),
                "ae_score":   e.get("A_t", 0),
                "lstm_prob":  e.get("B_t", 0),
                "true_label": e.get("true_label", 0),
                "attack_cat": e.get("attack_cat", "Normal"),
                "is_alert":   e["risk"]["R_t"] >= 0.5,
                "action_name":"N/A",
                "uncertainty": 0.0,
                "shap_top": []
            })
        return pd.DataFrame(rows)
    return pd.DataFrame()


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

    auto_refresh = st.toggle("Auto-Refresh (Live Mode)", value=True)
    refresh_rate = st.slider("Refresh interval (s)", 3, 30, 5)

    st.divider()
    st.markdown('<p class="section-header">🎯 Alert Thresholds</p>', unsafe_allow_html=True)
    warn_thresh = st.slider("Warning threshold",  0.1, 0.9, 0.5, 0.05)
    crit_thresh = st.slider("Critical threshold", 0.1, 1.0, 0.8, 0.05)

    st.divider()
    st.markdown('<p class="section-header">📊 View Settings</p>', unsafe_allow_html=True)
    max_incidents = st.selectbox("Incidents to show", [50, 100, 200], index=1)
    show_normal = st.checkbox("Show Normal traffic", value=True)

    st.divider()
    st.caption("NetGuard-XAI v1.0.0 | UNSW-NB15")
    st.caption("Built for real-time intrusion detection and adaptive containment")

# ─────────────────────────────────────────────────────────────────────────────
# Simulate Live Attack (Feed test data through API)
# ─────────────────────────────────────────────────────────────────────────────
def simulate_live_attack():
    """Sends a sample test event through the live API and captures the response."""
    test_pkl = os.path.join(PROCESSED_DIR, "test_split.pkl")
    prox_pkl = os.path.join(PROCESSED_DIR, "preprocessors.pkl")
    if not os.path.exists(test_pkl):
        return None, "Test data not found. Run Phase 1 first."
    test_df  = pd.read_pickle(test_pkl)
    with open(prox_pkl, "rb") as f:
        artifacts = pickle.load(f)

    # Sample one attack row
    attack_rows = test_df[test_df['label'] == 1]
    if len(attack_rows) == 0:
        return None, "No attack rows in test set."
    row = attack_rows.sample(1).iloc[0]
    feat_cols = artifacts['tree_feature_cols']
    features = {c: float(row[c]) if pd.notna(row.get(c, 0)) else 0.0 for c in feat_cols}
    payload = {
        "features": features,
        "srcip":    str(row.get('srcip', '10.0.0.1')),
        "dstip":    str(row.get('dstip', '192.168.1.1')),
        "proto":    str(row.get('proto', 'tcp')),
        "true_label": int(row['label']),
        "attack_cat": str(row.get('attack_cat', 'Unknown'))
    }
    try:
        resp = requests.post(f"{API_BASE}/analyze", json=payload, timeout=10)
        if resp.status_code == 200:
            return resp.json(), None
        return None, f"API error: {resp.status_code} {resp.text}"
    except Exception as e:
        return None, f"API unreachable: {e}"


# ─────────────────────────────────────────────────────────────────────────────
# Main Dashboard
# ─────────────────────────────────────────────────────────────────────────────

# Header
st.markdown("""
<div class="ng-header">
    <h1 class="ng-title">🛡️ NetGuard-XAI</h1>
    <p class="ng-subtitle">
        <span class="live-dot"></span>
        Real-time Network Intrusion Detection · Adaptive RL Containment · SHAP Explainability
    </p>
</div>
""", unsafe_allow_html=True)

# Tab layout
tab_main, tab_shap, tab_rl, tab_phase_results, tab_simulate = st.tabs([
    "📡 Live Monitor", "🔍 SHAP Explanations", "🤖 RL Policy", "📊 Phase Results", "⚡ Simulate Attack"
])

# ── TAB 1: Live Monitor ──
with tab_main:
    df = fetch_recent_incidents(max_incidents)

    if df.empty:
        st.warning("No incidents found. Run Phase 1-3 pipeline first or check that the API is running.")
    else:
        if not show_normal and 'is_alert' in df.columns:
            df = df[df['is_alert'] == True]

        # KPI Row
        total = len(df)
        alerts = int(df['is_alert'].sum()) if 'is_alert' in df.columns else 0
        avg_risk = float(df['risk_score'].mean()) if 'risk_score' in df.columns else 0.0
        avg_unc  = float(df['uncertainty'].mean()) if 'uncertainty' in df.columns else 0.0
        attack_rate = alerts / total * 100 if total > 0 else 0

        cols = st.columns(5)
        kpis = [
            ("Total Events", f"{total:,}", "blue", "📡"),
            ("Active Alerts", f"{alerts:,}", "red" if alerts > 0 else "green", "🚨"),
            ("Attack Rate", f"{attack_rate:.1f}%", "amber", "📈"),
            ("Avg Risk Score", f"{avg_risk:.3f}", "red" if avg_risk > 0.5 else "green", "⚡"),
            ("Avg Uncertainty", f"{avg_unc:.3f}", "purple", "🎲"),
        ]
        for col, (label, val, color, icon) in zip(cols, kpis):
            with col:
                st.markdown(f"""
                <div class="metric-card {color}">
                    <div style="font-size:1.6rem">{icon}</div>
                    <div class="metric-value" style="color: var(--accent-{'blue' if color=='blue' else 'red' if color=='red' else 'amber' if color=='amber' else 'green' if color=='green' else 'purple'})">{val}</div>
                    <div class="metric-label">{label}</div>
                </div>
                """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Risk Timeline
        if 'risk_score' in df.columns:
            st.markdown('<p class="section-header">📈 Risk Score Timeline</p>', unsafe_allow_html=True)
            fig_timeline = go.Figure()
            x_vals = list(range(len(df)))
            risk_vals = df['risk_score'].tolist()
            attack_cats = df.get('attack_cat', ['Normal'] * len(df)).tolist()

            fig_timeline.add_trace(go.Scatter(
                x=x_vals, y=risk_vals,
                fill='tozeroy', fillcolor='rgba(59,130,246,0.08)',
                line=dict(color='#3b82f6', width=2),
                name='Risk R_t', mode='lines',
                hovertemplate='Event %{x}<br>Risk: %{y:.3f}<br>Cat: ' +
                              str(attack_cats[0]) if attack_cats else 'Risk: %{y:.3f}'
            ))
            # Mark alerts
            alert_idx  = [i for i, v in enumerate(df['is_alert'].tolist() if 'is_alert' in df.columns else []) if v]
            alert_vals = [risk_vals[i] for i in alert_idx]
            if alert_idx:
                fig_timeline.add_trace(go.Scatter(
                    x=alert_idx, y=alert_vals,
                    mode='markers', marker=dict(color='#ef4444', size=8, symbol='x'),
                    name='Alert'
                ))
            fig_timeline.add_hline(y=warn_thresh, line_dash="dash", line_color="#f59e0b",
                                   annotation_text=f"Warning ({warn_thresh})")
            fig_timeline.add_hline(y=crit_thresh, line_dash="dash", line_color="#ef4444",
                                   annotation_text=f"Critical ({crit_thresh})")
            fig_timeline.update_layout(
                paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                font=dict(color='#94a3b8'), height=280,
                legend=dict(bgcolor='rgba(0,0,0,0)'),
                margin=dict(t=20, b=20, l=0, r=0),
                yaxis=dict(range=[0, 1.05], gridcolor='rgba(30,41,59,0.8)'),
                xaxis=dict(gridcolor='rgba(30,41,59,0.8)')
            )
            st.plotly_chart(fig_timeline, use_container_width=True)

        c1, c2 = st.columns([2, 1])

        with c1:
            # Model Score Comparison
            if all(c in df.columns for c in ['xgb_prob', 'ae_score', 'lstm_prob']):
                st.markdown('<p class="section-header">🎯 Model Score Distribution</p>', unsafe_allow_html=True)
                fig_scores = go.Figure()
                for col_name, color, name in [
                    ('xgb_prob', '#3b82f6', 'XGBoost'),
                    ('ae_score',  '#06b6d4', 'Autoencoder'),
                    ('lstm_prob', '#8b5cf6', 'LSTM')
                ]:
                    if col_name in df.columns:
                        fig_scores.add_trace(go.Box(
                            y=df[col_name], name=name,
                            marker_color=color, boxmean=True
                        ))
                fig_scores.update_layout(
                    paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                    font=dict(color='#94a3b8'), height=260,
                    margin=dict(t=20, b=20), showlegend=False,
                    yaxis=dict(range=[0,1], gridcolor='rgba(30,41,59,0.8)')
                )
                st.plotly_chart(fig_scores, use_container_width=True)

        with c2:
            # Attack Category Breakdown
            if 'attack_cat' in df.columns:
                st.markdown('<p class="section-header">🗂️ Attack Categories</p>', unsafe_allow_html=True)
                cat_counts = df[df['is_alert'] == True]['attack_cat'].value_counts() if 'is_alert' in df.columns else df['attack_cat'].value_counts()
                if not cat_counts.empty:
                    fig_pie = go.Figure(go.Pie(
                        labels=cat_counts.index,
                        values=cat_counts.values,
                        hole=0.55,
                        marker=dict(colors=['#ef4444','#f59e0b','#3b82f6','#8b5cf6','#06b6d4','#10b981','#ec4899','#14b8a6','#f97316']),
                    ))
                    fig_pie.update_layout(
                        paper_bgcolor='rgba(0,0,0,0)',
                        font=dict(color='#94a3b8'), height=260,
                        showlegend=True,
                        legend=dict(bgcolor='rgba(0,0,0,0)', font=dict(size=10)),
                        margin=dict(t=10, b=10, l=10, r=10)
                    )
                    st.plotly_chart(fig_pie, use_container_width=True)

        # Recent Incidents Table
        st.markdown('<p class="section-header">📋 Recent Incident Log</p>', unsafe_allow_html=True)
        display_cols = [c for c in ['id','timestamp','attack_cat','risk_score','action_name','xgb_prob','ae_score','lstm_prob','uncertainty','is_alert'] if c in df.columns]
        st.dataframe(
            df[display_cols].head(30).style
                .format({c: '{:.3f}' for c in ['risk_score','xgb_prob','ae_score','lstm_prob','uncertainty'] if c in df.columns})
                .background_gradient(subset=['risk_score'] if 'risk_score' in display_cols else [], cmap='RdYlGn_r'),
            use_container_width=True, height=300
        )

# ── TAB 2: SHAP Explanations ──
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

            outcome = ""
            if inc['true_label'] == 1 and inc['predicted_label'] == 1:
                outcome = "✅ True Positive"
            elif inc['true_label'] == 0 and inc['predicted_label'] == 1:
                outcome = "❌ False Positive"
            elif inc['true_label'] == 1 and inc['predicted_label'] == 0:
                outcome = "⚠️ False Negative"
            else:
                outcome = "✅ True Negative"
            st.markdown(f"**Outcome:** {outcome}")

        with col2:
            features = [d['feature'] for d in inc['top_shap_features']]
            values   = [d['shap_value'] for d in inc['top_shap_features']]
            colors   = ['#ef4444' if v > 0 else '#10b981' for v in values]
            fig_shap = go.Figure(go.Bar(
                x=values, y=features,
                orientation='h',
                marker_color=colors,
            ))
            fig_shap.update_layout(
                title="SHAP Feature Contributions",
                paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                font=dict(color='#94a3b8'), height=320,
                xaxis=dict(title="SHAP Value", gridcolor='rgba(30,41,59,0.8)'),
                yaxis=dict(autorange='reversed'),
                margin=dict(t=40, b=20, l=20, r=20)
            )
            st.plotly_chart(fig_shap, use_container_width=True)
    else:
        st.info("SHAP results not found. Run Phase 3 pipeline first.")

# ── TAB 3: RL Policy ──
with tab_rl:
    st.markdown('<p class="section-header">🤖 DQN Policy Performance</p>', unsafe_allow_html=True)

    dqn_path = os.path.join(RESULTS_DIR, "dqn_test_eval.json")
    curve_path = os.path.join(RESULTS_DIR, "dqn_training_curve.png")

    if os.path.exists(dqn_path):
        with open(dqn_path) as f:
            rl_eval = json.load(f)

        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("Attack Containment Rate", f"{rl_eval['attack_containment_rate']:.1%}")
        with c2:
            st.metric("FP Intervention Rate", f"{rl_eval['false_positive_intervention_rate']:.1%}")
        with c3:
            st.metric("Avg Episode Reward", f"{rl_eval['average_reward']:.2f}")

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
                yaxis=dict(gridcolor='rgba(30,41,59,0.8)'),
                margin=dict(t=20, b=20)
            )
            st.plotly_chart(fig_bar, use_container_width=True)
    else:
        st.info("RL evaluation results not found. Run Phase 4 pipeline first.")

    if os.path.exists(curve_path):
        st.markdown('<p class="section-header">📈 DQN Training Curve</p>', unsafe_allow_html=True)
        st.image(curve_path, use_container_width=True)

# ── TAB 4: Phase Results ──
with tab_phase_results:
    st.markdown('<p class="section-header">📊 All Phase Metrics</p>', unsafe_allow_html=True)

    results_files = {
        "XGBoost (Phase 2)":    "xgboost_test_metrics.json",
        "Autoencoder (Phase 2)":"autoencoder_test_metrics.json",
        "LSTM (Phase 2)":       "lstm_test_metrics.json",
        "Phase 5 Zero-Day":     "phase5_zeroday_results.json",
        "Phase 6 Ablation":     "phase6_ablation.json",
    }

    for label, fname in results_files.items():
        path = os.path.join(RESULTS_DIR, fname)
        if os.path.exists(path):
            with open(path) as f:
                data = json.load(f)
            with st.expander(f"📄 {label}", expanded=False):
                st.json(data)
        else:
            st.caption(f"⏳ {label} — not yet generated (run pipeline phase)")

    # Phase trajectory image
    traj_img = os.path.join(RESULTS_DIR, "risk_trajectory.png")
    if os.path.exists(traj_img):
        st.markdown('<p class="section-header">📈 Risk Trajectory (Phase 3)</p>', unsafe_allow_html=True)
        st.image(traj_img, use_container_width=True)

# ── TAB 5: Simulate Live Attack ──
with tab_simulate:
    st.markdown('<p class="section-header">⚡ Live Attack Simulation</p>', unsafe_allow_html=True)
    st.info("This sends a real UNSW-NB15 test-set attack event through the live FastAPI pipeline and shows the full response, which is then persisted to the incident database.")

    if st.button("🚀 Trigger Simulated Attack Event", type="primary", use_container_width=True):
        with st.spinner("Sending event through pipeline..."):
            result, error = simulate_live_attack()

        if error:
            st.error(f"Simulation failed: {error}")
            st.info("Make sure to start the API first: `python3 -m uvicorn src.api:app --host 0.0.0.0 --port 8000`")
        else:
            st.success(f"✅ Event processed! Incident ID: {result['incident_id']}")

            mc1, mc2, mc3, mc4 = st.columns(4)
            with mc1: st.metric("Risk Score",   f"{result['risk_score']:.3f}")
            with mc2: st.metric("XGBoost Prob", f"{result['xgb_prob']:.3f}")
            with mc3: st.metric("AE Score",     f"{result['ae_score']:.3f}")
            with mc4: st.metric("LSTM Prob",    f"{result['lstm_prob']:.3f}")

            action_col = ACTION_COLORS.get(result['action_name'], '#3b82f6')
            st.markdown(f"""
            <div style="background: rgba(0,0,0,0.3); border: 1px solid {action_col}; border-radius: 12px;
                        padding: 16px; margin: 12px 0; text-align: center;">
                <div style="font-size: 0.75rem; color: #64748b; letter-spacing: 2px; text-transform: uppercase;">RL Policy Decision</div>
                <div style="font-size: 2rem; font-weight: 900; color: {action_col}; font-family: 'JetBrains Mono';">
                    {result['action_name']}
                </div>
                <div style="font-size: 0.85rem; color: #64748b;">Uncertainty: {result['uncertainty']:.4f}</div>
            </div>
            """, unsafe_allow_html=True)

            if result['shap_top_features']:
                st.markdown('<p class="section-header" style="margin-top:16px">🔍 SHAP Explanation</p>', unsafe_allow_html=True)
                shap_df = pd.DataFrame(result['shap_top_features'])
                fig_shap = go.Figure(go.Bar(
                    x=shap_df['shap_value'],
                    y=shap_df['feature'],
                    orientation='h',
                    marker_color=['#ef4444' if v > 0 else '#10b981' for v in shap_df['shap_value']]
                ))
                fig_shap.update_layout(
                    paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                    font=dict(color='#94a3b8'), height=280,
                    yaxis=dict(autorange='reversed'),
                    margin=dict(t=10, b=10)
                )
                st.plotly_chart(fig_shap, use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────────
# Auto-refresh
# ─────────────────────────────────────────────────────────────────────────────
if auto_refresh:
    time.sleep(refresh_rate)
    st.rerun()
