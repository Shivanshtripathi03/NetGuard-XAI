# 🛡️ NetGuard-XAI

**Real-time, closed-loop network intrusion detection and adaptive response** — combining supervised detection, unsupervised anomaly detection, temporal behavioral analysis, a dynamic multi-signal risk engine, SHAP-based explainability, and a DQN reinforcement-learning response policy.

> **Evaluated on the official UNSW-NB15 dataset.** All metrics are derived from actual executed runs against real data — no fabricated or placeholder numbers anywhere in this repository.

---

## 📋 Table of Contents

- [Research Claim](#-research-claim)
- [Architecture](#-architecture)
- [Tech Stack](#-tech-stack)
- [Dataset](#-dataset)
- [Project Structure](#-project-structure)
- [Quick Start](#-quick-start)
- [Phased Build Plan](#-phased-build-plan)
- [Running the Pipeline](#-running-the-pipeline)
- [API & Dashboard](#-api--dashboard)
- [Scientific Honesty Notice](#-scientific-honesty-notice)
- [Citation](#-citation)
- [License](#-license)

---

## 🔬 Research Claim

> *A closed-loop system that fuses complementary detection signals (supervised, anomaly-based, behavioral) into an evolving risk estimate, and uses that risk plus uncertainty and temporal trend to drive an RL-based containment policy, **can detect and adaptively contain network attack behavior that was never seen during supervised training** — better than a static-threshold or rule-based baseline.*

This is a testable hypothesis, not an assertion. Phase 5 runs the actual unseen-attack experiment (Reconnaissance + Fuzzers held out) and reports honest comparison numbers including cases where the system underperforms.

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         NetGuard-XAI Pipeline                           │
├────────────────┬────────────────┬───────────────────────────────────────┤
│  Phase 1       │  Phase 2       │  Phase 3                              │
│  Data Pipeline │  Detection     │  Risk Engine + XAI                    │
│                │  Models        │                                        │
│  UNSW-NB15     │  ┌──────────┐  │  R_t = f(P_t, A_t, B_t, S_t, H_t,   │
│  175,341 train │  │ XGBoost  │  │          U_t, C_t)                    │
│  82,332 test   │  │ P_t ──►  │  │                                        │
│                │  ├──────────┤  │  Uncertainty: agreement-based          │
│  Cleaned,      │  │AutoEnc.  │  │  SHAP: real TreeExplainer values       │
│  encoded,      │  │ A_t ──►  ├──►  Historical decay: H_t×0.85           │
│  normalized    │  ├──────────┤  │  Risk trajectory (ΔR_t, velocity)     │
│  (no leakage)  │  │ LSTM/GRU │  │                                        │
│                │  │ B_t ──►  │  │                                        │
│                │  └──────────┘  │                                        │
├────────────────┴────────────────┴──────────────────────┬───────────────┤
│  Phase 4: DQN RL Policy                                │  Phase 5      │
│                                                        │  Zero-Day     │
│  State: [P_t, A_t, B_t, R_t, U_t, ΔR_t, H_t, C_t]   │  Experiments  │
│  Actions: Allow │ Monitor │ RateLimit │ Block │ Isolate│               │
│  Reward table (named constants, tunable):              │  Hold out:    │
│    +15 correct contain, −30 attack allowed through     │  Recon+Fuzz   │
│    −20 FP block, −25 FP isolate                        │  Leakage chk  │
│                                                        │  vs full-     │
│  stable-baselines3 DQN, ε-greedy decay, γ=0.99        │  knowledge    │
├────────────────────────────────────────────────────────┤  control      │
│  Phase 6: Ablation & Comparative Study                 │               │
│  System A (static threshold) vs B (risk rules) vs C   │               │
│  XGB→+AE→+LSTM→+Risk→+SHAP→+RL ablation table        │               │
├────────────────────────────────────────────────────────┴───────────────┤
│  Phase 7: FastAPI Microservice + Streamlit Dashboard                   │
│  POST /analyze → full inference pipeline → SQLite/PostgreSQL           │
│  Dashboard reads live from incident DB — no mocked fixtures            │
└─────────────────────────────────────────────────────────────────────────┘
```

### Risk Formula

```
R_t = clip(W_P·P_t + W_A·A_t + W_B·B_t + W_S·S_t + W_U·U_t + W_C·C_t + W_H·H_t, 0, 1)
```

| Weight | Signal | Description |
|--------|--------|-------------|
| W_P = 0.30 | P_t | XGBoost attack probability |
| W_A = 0.25 | A_t | Autoencoder anomaly score (normalized MSE) |
| W_B = 0.20 | B_t | LSTM behavioral deviation probability |
| W_S = 0.10 | S_t | Attack category severity (0.0–0.95) |
| W_U = 0.10 | U_t | Uncertainty penalty (model disagreement) |
| W_C = 0.05 | C_t | Contextual signal |
| — | H_t | Historical risk (exponential decay γ=0.85) |

---

## 🧰 Tech Stack

| Component | Technology |
|-----------|-----------|
| Language | Python 3.11 |
| Environment | `venv` |
| Supervised detection | XGBoost |
| Anomaly detection | PyTorch Autoencoder (MSE reconstruction) |
| Behavioral analysis | PyTorch LSTM/GRU |
| Explainability | SHAP (TreeExplainer) |
| RL policy | stable-baselines3 DQN + custom Gymnasium env |
| API serving | FastAPI + uvicorn |
| Dashboard | Streamlit + Plotly |
| Incident DB | SQLAlchemy (SQLite dev / PostgreSQL prod) |
| Orchestration | Docker Compose (microservice-style, local/single-VM) |

---

## 📊 Dataset

**UNSW-NB15** — official pre-split CSVs from the UNSW Canberra Cyber Range Lab.

| Split | Rows | Source |
|-------|------|--------|
| Training | 175,341 | `UNSW_NB15_training-set.csv` |
| Testing | 82,332 | `UNSW_NB15_testing-set.csv` |

**Attack categories (10 classes):**
`Normal`, `Reconnaissance`, `Fuzzers`, `Analysis`, `Backdoor`, `DoS`, `Exploits`, `Generic`, `Shellcode`, `Worms`

**Dataset access:**
The official CSVs must be downloaded manually from [https://research.unsw.edu.au/projects/unsw-nb15-dataset](https://research.unsw.edu.au/projects/unsw-nb15-dataset) (UNSW SharePoint, requires manual browser download of large files).

Place downloaded files in `data/raw/`:
```
data/raw/
├── UNSW_NB15_training-set.csv
├── UNSW_NB15_testing-set.csv
└── UNSW-NB15_features.csv      # optional feature descriptions
```

Then run Phase 1 to validate, clean, and produce all derived artifacts.

> ⚠️ **Do NOT substitute a Kaggle re-upload** without verifying it matches the official split (175,341/82,332 rows). Different re-uploads have been found to have different preprocessing applied.

**Citation requirement:** Using this dataset requires citing Moustafa & Slay's 2015 MilCIS paper. See [`CITATION.md`](./CITATION.md).

---

## 📁 Project Structure

```
NetGuard-XAI/
│
├── data/
│   ├── raw/                   # Official UNSW-NB15 CSVs (place here manually)
│   └── processed/             # Generated by Phase 1: train/val/test splits, scalers
│
├── models/                    # Saved model artifacts (generated by Phase 2+)
│   ├── xgboost_model.json
│   ├── autoencoder.pt
│   ├── ae_scaler.pkl
│   ├── lstm.pt
│   └── dqn_policy.zip
│
├── results/                   # All metrics & evaluation outputs (generated by pipeline)
│   ├── xgboost_test_metrics.json
│   ├── autoencoder_test_metrics.json
│   ├── lstm_test_metrics.json
│   ├── phase2_metrics.md
│   ├── dqn_training_curve.png
│   ├── dqn_training_log.csv
│   ├── dqn_test_eval.json
│   ├── dqn_hyperparams.json
│   └── phase6_ablation_results.json
│
├── scripts/                   # Executable phase scripts
│   ├── download_dataset.py    # Dataset validation/download helper
│   ├── phase1_pipeline.py     # Data pipeline entry point
│   ├── phase2_xgboost.py      # XGBoost training & evaluation
│   ├── phase2_autoencoder.py  # Autoencoder training & evaluation
│   ├── phase2_lstm.py         # LSTM training & evaluation
│   ├── phase3_risk_xai.py     # Risk engine + uncertainty + SHAP
│   ├── phase4_rl.py           # RL environment + DQN training
│   ├── phase5_zeroday.py      # Zero-day / unseen-attack experiments
│   └── phase6_ablation.py     # Ablation & comparative study
│
├── src/                       # Core library modules
│   ├── __init__.py
│   ├── data_preprocessing.py  # Feature engineering, encoding, scaling
│   ├── models.py              # PyTorch Autoencoder, LSTM + training utils
│   ├── database.py            # SQLAlchemy ORM (Incident table)
│   ├── api.py                 # FastAPI inference service
│   └── dashboard.py           # Streamlit live dashboard
│
├── tests/                     # Unit tests
├── run_pipeline.py            # Full pipeline runner (all phases in sequence)
├── data_report.md             # Phase 1 checkpoint: real data diagnostics
├── CITATION.md                # Required academic citations for UNSW-NB15
├── requirements.txt           # Python dependencies
└── .gitignore
```

---

## ⚡ Quick Start

### 1. Clone & set up environment

```bash
git clone https://github.com/Shivanshtripathi03/NetGuard-XAI.git
cd NetGuard-XAI

python3.11 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

### 2. Download the dataset

Download the official UNSW-NB15 pre-split CSVs from [the UNSW official page](https://research.unsw.edu.au/projects/unsw-nb15-dataset) and place them in `data/raw/`.

```
data/raw/UNSW_NB15_training-set.csv    # 175,341 rows
data/raw/UNSW_NB15_testing-set.csv     # 82,332 rows
```

### 3. Run Phase 1 (data pipeline + diagnostics)

```bash
python scripts/phase1_pipeline.py
```

Outputs: `data/processed/` artifacts + `data_report.md`

### 4. Train all models (Phases 2–4)

```bash
python scripts/phase2_xgboost.py
python scripts/phase2_autoencoder.py
python scripts/phase2_lstm.py
python scripts/phase3_risk_xai.py
python scripts/phase4_rl.py
```

### 5. Run zero-day & ablation experiments (Phases 5–6)

```bash
python scripts/phase5_zeroday.py
python scripts/phase6_ablation.py
```

### 6. Launch API + Dashboard

```bash
# Terminal 1 — FastAPI inference service
uvicorn src.api:app --host 0.0.0.0 --port 8000

# Terminal 2 — Streamlit dashboard
streamlit run src/dashboard.py
```

Dashboard: [http://localhost:8501](http://localhost:8501)  
API docs: [http://localhost:8000/docs](http://localhost:8000/docs)

### 7. Run the full pipeline at once

```bash
python run_pipeline.py
```

> ⚠️ Full pipeline takes significant time (model training). Run phases individually during development.

---

## 🔄 Phased Build Plan

| Phase | Description | Key Outputs |
|-------|-------------|-------------|
| **Phase 1** | Data pipeline — load, validate, clean, encode, split UNSW-NB15 | `data_report.md`, processed splits |
| **Phase 2** | Detection models — XGBoost, Autoencoder, LSTM/GRU (each with hyperparameter search, frozen artifacts) | `*_test_metrics.json`, `phase2_metrics.md` |
| **Phase 3** | Risk engine + uncertainty + SHAP — `R_t` formula, historical decay, real SHAP on test predictions | Risk trajectory plots, SHAP per incident |
| **Phase 4** | RL — `NetGuardSecurityEnv` (Gymnasium), DQN training on training-split episodes only, eval on test | `dqn_policy.zip`, training curves, eval table |
| **Phase 5** | Zero-day experiments — hold out Recon + Fuzzers, leakage check, restricted vs. full-knowledge control | Unseen-attack metrics table |
| **Phase 6** | Ablation & comparative study — System A/B/C comparison, stepwise ablation | `phase6_ablation_results.json` |
| **Phase 7** | FastAPI service + Streamlit dashboard reading live from incident DB | Live dashboard verified against DB |
| **Phase 8** | Final report — architecture diagram, full metrics tables, honest limitations | `final_report.md` |

---

## 🌐 API & Dashboard

### API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/analyze` | Analyze a network flow (full inference pipeline) |
| `GET` | `/incidents/recent` | Fetch recent incidents from DB |
| `GET` | `/incidents/alerts` | Fetch high-risk alerts (min_risk configurable) |
| `GET` | `/health` | Service health check |

### Example `/analyze` request

```json
{
  "srcip": "192.168.1.100",
  "dstip": "10.0.0.1",
  "proto": "tcp",
  "features": {
    "dur": 0.021,
    "spkts": 6,
    "dpkts": 5,
    "sbytes": 496,
    "dbytes": 2000,
    "rate": 476.19
  }
}
```

### Example response

```json
{
  "incident_id": 42,
  "risk_score": 0.847,
  "xgb_prob": 0.912,
  "ae_score": 0.743,
  "lstm_prob": 0.831,
  "uncertainty": 0.089,
  "rl_action": 3,
  "action_name": "Block",
  "is_alert": true,
  "shap_top_features": [
    {"feature": "ct_state_ttl", "shap_value": 0.412},
    {"feature": "sbytes", "shap_value": -0.301}
  ]
}
```

---

## ⚖️ Scientific Honesty Notice

This project explicitly avoids the claim *"prevents zero-day attacks."*

The correct, defensible claim is:

> **NetGuard-XAI demonstrates detection and adaptive containment of attack behavior (Reconnaissance, Fuzzers) that was withheld from supervised training** — measured against a full-knowledge control baseline on the same test rows.

All limitations are reported honestly in Phase 5's results, including scenarios where the restricted model underperforms. A valid experimental finding — positive or negative — is still a valid contribution.

---

## 📦 Requirements

See [`requirements.txt`](./requirements.txt) for the full pinned dependency list.

Core dependencies:

```
python>=3.11
xgboost
torch
scikit-learn
shap
stable-baselines3[extra]
gymnasium
fastapi
uvicorn
streamlit
plotly
sqlalchemy
pandas
numpy
matplotlib
```

---

## 📖 Citation

If you use this codebase or the UNSW-NB15 dataset in your work, you **must** cite the original dataset papers per their academic license. See [`CITATION.md`](./CITATION.md) for the full BibTeX entries.

Primary citation:

```bibtex
@inproceedings{moustafa2015unsw,
  title={UNSW-NB15: a comprehensive data set for network intrusion detection systems},
  author={Moustafa, Nour and Slay, Jill},
  booktitle={2015 Military Communications and Information Systems Conference (MilCIS)},
  pages={1--6},
  year={2015},
  organization={IEEE}
}
```

---

## 📄 License

This project is released for academic and research use. The UNSW-NB15 dataset is separately licensed by UNSW Canberra — see [CITATION.md](./CITATION.md) for usage requirements.

---

*Built by [@Shivanshtripathi03](https://github.com/Shivanshtripathi03)*
