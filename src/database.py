"""
NetGuard-XAI Incident Database
SQLAlchemy ORM models and helper functions for SQLite/PostgreSQL storage.
Stores events, risk scores, SHAP explanations, and RL actions.
"""

import os
from datetime import datetime
from sqlalchemy import (
    create_engine, Column, Integer, Float, String, DateTime,
    Text, Boolean, JSON
)
from sqlalchemy.orm import DeclarativeBase, Session
from sqlalchemy import select

DB_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data", "netguard.db"
)

class Base(DeclarativeBase):
    pass


class Incident(Base):
    __tablename__ = "incidents"

    id         = Column(Integer, primary_key=True, autoincrement=True)
    timestamp  = Column(DateTime, default=datetime.utcnow)
    # Raw flow identifiers
    srcip      = Column(String(64),  nullable=True)
    dstip      = Column(String(64),  nullable=True)
    proto      = Column(String(16),  nullable=True)
    # Ground truth (from dataset — present in batch mode only)
    true_label = Column(Integer,     nullable=True)
    attack_cat = Column(String(64),  nullable=True)
    # Model scores
    xgb_prob   = Column(Float,       nullable=True)
    ae_score   = Column(Float,       nullable=True)
    lstm_prob  = Column(Float,       nullable=True)
    # Risk engine output
    risk_score = Column(Float,       nullable=False)
    risk_delta = Column(Float,       nullable=True)
    risk_hist  = Column(Float,       nullable=True)
    uncertainty= Column(Float,       nullable=True)
    # RL response
    rl_action  = Column(Integer,     nullable=True)   # 0-4
    action_name= Column(String(16),  nullable=True)
    reward     = Column(Float,       nullable=True)
    # SHAP explanation (top features as JSON)
    shap_top   = Column(JSON,        nullable=True)
    # Alert flag
    is_alert   = Column(Boolean,     default=False)


def get_engine(db_url: str = None):
    url = db_url or f"sqlite:///{DB_PATH}"
    engine = create_engine(url, echo=False)
    Base.metadata.create_all(engine)
    return engine


def save_incident(session: Session, incident_data: dict) -> Incident:
    inc = Incident(**incident_data)
    session.add(inc)
    session.commit()
    session.refresh(inc)
    return inc


def get_recent_incidents(session: Session, limit: int = 100):
    stmt = select(Incident).order_by(Incident.timestamp.desc()).limit(limit)
    return session.scalars(stmt).all()


def get_incidents_by_risk(session: Session, min_risk: float = 0.5, limit: int = 100):
    stmt = (
        select(Incident)
        .where(Incident.risk_score >= min_risk)
        .order_by(Incident.risk_score.desc())
        .limit(limit)
    )
    return session.scalars(stmt).all()
