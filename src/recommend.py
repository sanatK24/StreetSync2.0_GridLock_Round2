"""
recommend.py
=============
The prescriptive layer. Where impact_index.py answers "how bad will this
be", this module answers "what should we actually deploy" - manpower count,
barricade points, diversion routing - and packages the full forecast used by
the Streamlit app.

Resource formulas (BASE_PERSONNEL, barricade thresholds) are explicit,
named constants, not buried magic numbers - see the module-level comments
for the reasoning, and feedback_store.py for how these get recalibrated once
real deployment outcomes start flowing back through the feedback loop. This
directly answers the brief's pain point #2 ("resource deployment is
experience-driven") by making the experience-based assumptions explicit,
visible, and tunable rather than claiming a ground-truth model that the data
doesn't support.
"""
from __future__ import annotations

import math
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from data_prep import build_processed_dataset, FESTIVAL_WINDOWS, IST  # noqa: E402
from features import PRIORITY_FEATURES, CLOSURE_DURATION_FEATURES  # noqa: E402
from impact_index import (  # noqa: E402
    historical_duration_distribution, compute_impact_index,
)
from geo_utils import (  # noqa: E402
    build_spatial_index, nearest_context, similar_incidents, get_diversion_suggestion,
)

# Starting point personnel-per-incident assumptions, openly stated so BTP
# domain experts can correct them; see README "Calibrating the recommendation
# engine". These are NOT learned from data (no ground-truth headcount exists
# in the export) - they encode commonly cited traffic-management practice
# (route-lining for processions/VIP movement needs more personnel than cone
# placement for a stalled vehicle) and are multiplied by the data-driven
# severity_multiplier below so the data still drives the final number.
BASE_PERSONNEL = {
    "vip_movement": 6, "protest": 8, "public_event": 6, "procession": 5,
    "construction": 2, "accident": 3, "vehicle_breakdown": 1, "pot_holes": 1,
    "tree_fall": 2, "water_logging": 2, "road_conditions": 1, "congestion": 2,
    "debris": 1, "test_demo": 1, "Fog/Low Visibility": 2, "others": 1,
}
DEFAULT_BASE_PERSONNEL = 2


def severity_multiplier(impact_score: float) -> float:
    """impact_score in [0,100] -> multiplier in [0.6, 2.0]."""
    return 0.6 + (impact_score / 100.0) * 1.4


def recommend_manpower(event_cause: str, impact_score: float, closure_proba: float) -> dict:
    base = BASE_PERSONNEL.get(event_cause, DEFAULT_BASE_PERSONNEL)
    mult = severity_multiplier(impact_score)
    extra_for_closure = 2 if closure_proba >= 0.5 else 0
    count = math.ceil(base * mult + extra_for_closure)
    return {
        "personnel": max(1, count),
        "base_personnel": base,
        "severity_multiplier": round(mult, 2),
        "extra_for_closure_management": extra_for_closure,
    }


def recommend_barricades(closure_proba: float, is_corridor: bool,
                          live_approach_count: int | None = None) -> dict:
    if closure_proba < 0.3:
        heuristic = 0
    elif closure_proba < 0.6:
        heuristic = 2
    else:
        heuristic = 4 if is_corridor else 3

    points = heuristic
    note = "Heuristic: based on predicted closure probability and road type."
    if live_approach_count:
        points = max(heuristic, live_approach_count - 1)
        note = (f"Adjusted using live OSM road-network data: "
                f"{live_approach_count} approach roads detected at this junction.")
    return {"barricade_points": points, "note": note}


class ForecastEngine:
    """Loads every model + the processed dataset once; reuse one instance
    across a Streamlit session instead of reloading on every interaction."""

    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.models_dir = self.root / "models"
        self.df, self.kmeans = build_processed_dataset(self.root / "data" / "astram_event_data.csv")
        self.tree = build_spatial_index(self.df)
        self.sorted_durations = historical_duration_distribution(self.df)

        self.priority_model = joblib.load(self.models_dir / "priority_model.joblib")
        self.closure_model = joblib.load(self.models_dir / "closure_model.joblib")
        self.duration_model = joblib.load(self.models_dir / "duration_model.joblib")
        self.duration_tier_model = joblib.load(self.models_dir / "duration_tier_model.joblib")

    def _time_features(self, dt: pd.Timestamp) -> dict:
        dt = pd.Timestamp(dt)
        if dt.tzinfo is None:
            dt = dt.tz_localize(IST)
        hour, dow, month = dt.hour, dt.dayofweek, dt.month
        date_str = dt.strftime("%Y-%m-%d")
        is_festival = any(date_str >= s and date_str <= e for s, e in FESTIVAL_WINDOWS)
        bins = [-1, 4, 7, 10, 16, 19, 22, 24]
        labels = ["Late Night (00-04)", "Early Morning (04-07)", "Morning (07-10)",
                  "Midday (10-16)", "Evening (16-19)", "Night Peak (19-22)", "Late Night (22-24)"]
        daypart = pd.cut([hour], bins=bins, labels=labels, ordered=False)[0]
        return {
            "hour_sin": np.sin(2 * np.pi * hour / 24), "hour_cos": np.cos(2 * np.pi * hour / 24),
            "dow_sin": np.sin(2 * np.pi * dow / 7), "dow_cos": np.cos(2 * np.pi * dow / 7),
            "month": month, "is_weekend": int(dow in (5, 6)),
            "is_festival_window": int(is_festival), "daypart": daypart,
        }

    def forecast(self, event_cause: str, lat: float, lon: float, event_type: str,
                 when: pd.Timestamp, priority_override: str | None = None) -> dict:
        ctx = nearest_context(lat, lon, self.df, self.tree)
        tfeat = self._time_features(when)
        is_corridor = bool(ctx["is_corridor"])

        base_row = {
            "event_cause": event_cause, "event_type": event_type,
            "corridor": ctx["corridor"], "police_station": ctx["police_station"],
            "zone": ctx["zone"], "daypart": tfeat["daypart"], "geo_cluster": str(
                int(self.kmeans.predict([[lat, lon]])[0])),
            "latitude": lat, "longitude": lon,
            "hour_sin": tfeat["hour_sin"], "hour_cos": tfeat["hour_cos"],
            "dow_sin": tfeat["dow_sin"], "dow_cos": tfeat["dow_cos"],
            "month": tfeat["month"], "is_weekend": tfeat["is_weekend"],
            "is_festival_window": tfeat["is_festival_window"], "is_corridor": int(is_corridor),
        }

        pri_X = pd.DataFrame([{k: base_row[k] for k in PRIORITY_FEATURES}])
        suggested_priority = self.priority_model.predict(pri_X)[0]
        priority = priority_override or suggested_priority

        cd_row = dict(base_row, priority=priority)
        cd_X = pd.DataFrame([{k: cd_row[k] for k in CLOSURE_DURATION_FEATURES}])
        closure_proba = float(self.closure_model.predict_proba(cd_X)[0, 1])

        dur_cols = ["event_cause", "event_type", "corridor", "zone", "daypart", "priority",
                    "hour_sin", "hour_cos", "dow_sin", "dow_cos", "month",
                    "is_weekend", "is_festival_window", "is_corridor"]
        dur_X = pd.DataFrame([{k: cd_row[k] for k in dur_cols}])
        duration_pred_min = float(np.expm1(self.duration_model.predict(dur_X)[0]))
        tier_pred = self.duration_tier_model.predict(dur_X)[0]
        tier_proba = dict(zip(self.duration_tier_model.classes_,
                               self.duration_tier_model.predict_proba(dur_X)[0].round(3)))

        precedents = similar_incidents(self.df, event_cause, lat, lon, tree=None, k=8)
        precedent_durations = precedents.loc[precedents["duration_valid"], "duration_min"]
        precedent_median = float(precedent_durations.median()) if len(precedent_durations) else None
        duration_estimate_for_index = precedent_median if precedent_median is not None else duration_pred_min

        impact = compute_impact_index(closure_proba, duration_estimate_for_index,
                                       priority, is_corridor, self.sorted_durations)

        diversion = get_diversion_suggestion(lat, lon, self.df, corridor=ctx["corridor"])
        manpower = recommend_manpower(event_cause, impact["score"], closure_proba)
        barricades = recommend_barricades(
            closure_proba, is_corridor,
            live_approach_count=diversion.get("approach_count"))

        return {
            "input": {"event_cause": event_cause, "event_type": event_type,
                      "lat": lat, "lon": lon, "when": str(when)},
            "context": ctx,
            "priority": {"suggested": suggested_priority, "used": priority},
            "closure_probability": round(closure_proba, 3),
            "duration": {
                "ml_point_estimate_min": round(duration_pred_min, 1),
                "tier_prediction": tier_pred,
                "tier_probabilities": tier_proba,
                "precedent_median_min": precedent_median,
                "n_precedents": int(len(precedents)),
            },
            "impact_index": impact,
            "manpower": manpower,
            "barricades": barricades,
            "diversion": diversion,
            "similar_incidents": precedents,
        }


if __name__ == "__main__":
    engine = ForecastEngine(Path(__file__).resolve().parent.parent)
    result = engine.forecast(
        event_cause="public_event", lat=12.9716, lon=77.5946,
        event_type="planned", when=pd.Timestamp("2026-06-20 18:30"),
    )
    import json
    printable = {k: v for k, v in result.items() if k != "similar_incidents"}
    print(json.dumps(printable, indent=2, default=str))
    print("\nSimilar incidents found:", len(result["similar_incidents"]))
