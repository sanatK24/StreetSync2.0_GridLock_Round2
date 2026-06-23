"""
feedback_store.py
==================
Addresses the third "Why It's Hard Today" pain point head-on: "No post-event
learning system." Every time an event is resolved, ops logs what actually
happened (real duration, real personnel used, whether it escalated beyond
the forecast). That gets appended to feedback/outcomes_log.csv in the same
shape as the original ASTraM export, so it can be folded straight back into
the training set and the models can be re-fit on `python -m
src.train_models` (also exposed as a button in the Streamlit app).

This is a CSV-backed store deliberately - no database server to stand up,
no extra paid service, works the same on a laptop or on Streamlit Community
Cloud's free tier. For a production deployment this would move to whatever
datastore BTP/ASTraM already runs (the schema doesn't need to change).
"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

FEEDBACK_COLUMNS = [
    "id", "event_type", "latitude", "longitude", "address", "event_cause",
    "requires_road_closure", "start_datetime", "end_datetime", "status",
    "priority", "corridor", "created_date", "police_station",
    "closed_datetime", "resolved_datetime", "zone", "junction",
    # feedback-specific extra columns (not in the original export, additive)
    "logged_manpower_used", "forecast_impact_score", "escalated_beyond_forecast",
    "feedback_notes", "feedback_logged_at",
]


def feedback_path(root: Path) -> Path:
    return Path(root) / "feedback" / "outcomes_log.csv"


def load_feedback_log(root: Path) -> pd.DataFrame:
    path = feedback_path(root)
    if path.exists():
        return pd.read_csv(path)
    return pd.DataFrame(columns=FEEDBACK_COLUMNS)


def log_outcome(root: Path, *, event_cause: str, event_type: str, lat: float, lon: float,
                 address: str, priority: str, corridor: str, police_station: str,
                 start_datetime: str, actual_end_datetime: str,
                 requires_road_closure: bool, manpower_used: int,
                 forecast_impact_score: float, escalated_beyond_forecast: bool,
                 notes: str = "") -> pd.DataFrame:
    """Appends one resolved event to the feedback log and returns the full
    updated log. Designed to be called from the Streamlit 'Log outcome'
    form once an event the team forecasted has actually concluded."""
    row = {
        "id": f"fb-{uuid.uuid4().hex[:10]}",
        "event_type": event_type, "latitude": lat, "longitude": lon,
        "address": address, "event_cause": event_cause,
        "requires_road_closure": requires_road_closure,
        "start_datetime": start_datetime, "end_datetime": actual_end_datetime,
        "status": "resolved", "priority": priority, "corridor": corridor,
        "created_date": start_datetime, "police_station": police_station,
        "closed_datetime": actual_end_datetime, "resolved_datetime": actual_end_datetime,
        "zone": None, "junction": None,
        "logged_manpower_used": manpower_used,
        "forecast_impact_score": forecast_impact_score,
        "escalated_beyond_forecast": escalated_beyond_forecast,
        "feedback_notes": notes,
        "feedback_logged_at": datetime.now(timezone.utc).isoformat(),
    }
    log = load_feedback_log(root)
    log = pd.concat([log, pd.DataFrame([row])], ignore_index=True)
    path = feedback_path(root)
    path.parent.mkdir(parents=True, exist_ok=True)
    log.to_csv(path, index=False)
    return log


def build_augmented_dataset(root: Path) -> Path:
    """Merges the original ASTraM export with everything logged through the
    feedback form into one CSV that train_models.py can run on directly -
    i.e. the actual retraining step of the learning loop. Returns the path
    to the augmented file."""
    root = Path(root)
    original = pd.read_csv(root / "data" / "astram_event_data.csv", low_memory=False)
    feedback = load_feedback_log(root)
    if feedback.empty:
        return root / "data" / "astram_event_data.csv"

    feedback_aligned = feedback.reindex(columns=original.columns)
    combined = pd.concat([original, feedback_aligned], ignore_index=True)
    out_path = root / "data" / "astram_event_data_augmented.csv"
    combined.to_csv(out_path, index=False)
    return out_path
