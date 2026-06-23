"""
impact_index.py
================
Combines the two validated model outputs (road-closure probability, expected
duration) plus two cheap-but-informative priors (priority tag, corridor
arterial status) into a single 0-100 "Impact Index" that the rest of the
system plans resources against.

This is DELIBERATELY a transparent weighted scorecard, not a fifth black-box
model trained on a made-up target - there is no ground-truth "impact" label
in the source data to train against, so manufacturing one and reporting a
fake accuracy on it would be dishonest. Every component below is either a
model output we've validated on held-out data (closure probability) or an
empirically grounded prior straight from the historical distribution
(duration percentile, corridor/priority weighting derived from the EDA).

Weights (out of 100) and the reasoning behind each:
  - Closure probability   : 40 pts - directly stops traffic; the single
                             biggest real-world driver of congestion.
  - Duration percentile    : 35 pts - how long the disruption is expected to
                             persist relative to the historical distribution.
  - Priority tag           : 15 pts - currently BTP's own triage signal.
  - Corridor (arterial)    : 10 pts - same incident on an arterial carries
                             more through-traffic than on a side street.
These weights are configurable constants below (WEIGHTS), explicitly so a
domain expert at BTP can recalibrate them once real outcome data (escalation
flags, actual congestion spillover) comes back through the feedback loop -
see feedback_store.py.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

WEIGHTS = {
    "closure": 40,
    "duration": 35,
    "priority": 15,
    "corridor": 10,
}

BANDS = [
    (0, 25, "Low", "#34D399"),
    (25, 50, "Medium", "#F5A623"),
    (50, 75, "High", "#F2784B"),
    (75, 100.0001, "Critical", "#E5484D"),
]


def historical_duration_distribution(df: pd.DataFrame) -> np.ndarray:
    return np.sort(df.loc[df["duration_valid"], "duration_min"].to_numpy())


def duration_percentile(minutes: float, sorted_durations: np.ndarray) -> float:
    if len(sorted_durations) == 0 or minutes is None or np.isnan(minutes):
        return 0.5
    return float(np.searchsorted(sorted_durations, minutes) / len(sorted_durations))


def band_for_score(score: float):
    for lo, hi, name, color in BANDS:
        if lo <= score < hi:
            return name, color
    return "Critical", "#E5484D"


def compute_impact_index(closure_proba: float, duration_minutes_estimate: float,
                          priority: str, is_corridor: bool,
                          sorted_durations: np.ndarray) -> dict:
    dur_pct = duration_percentile(duration_minutes_estimate, sorted_durations)
    priority_frac = 1.0 if str(priority).lower() == "high" else 0.3
    corridor_frac = 1.0 if is_corridor else 0.25

    components = {
        "closure": closure_proba * WEIGHTS["closure"],
        "duration": dur_pct * WEIGHTS["duration"],
        "priority": priority_frac * WEIGHTS["priority"],
        "corridor": corridor_frac * WEIGHTS["corridor"],
    }
    score = float(sum(components.values()))
    band, color = band_for_score(score)
    return {
        "score": round(score, 1),
        "band": band,
        "color": color,
        "components": {k: round(v, 1) for k, v in components.items()},
        "duration_percentile": round(dur_pct * 100, 1),
        "max_components": WEIGHTS,
    }
