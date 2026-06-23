"""
features.py
===========
Single source of truth for which columns feed the models and how they're
preprocessed. Every model (priority suggestion, road-closure classifier,
duration regressor) is trained on the SAME feature contract defined here, so
the recommendation engine only has to build one feature row per scenario and
can hand it to any of the three models.

Why these features and not others:
- event_cause, event_type: the strongest signals in the data (see EDA -
  event_cause alone separates 0% to 80% road-closure rate across causes).
- corridor / is_corridor / police_station / geo_cluster: spatial context.
  geo_cluster (KMeans over lat/lon) is the fallback that's always populated,
  even for a brand-new point on the map that doesn't match any historical
  corridor/police_station string exactly.
- hour/dow as sin-cos pairs + month + is_weekend + is_festival_window +
  daypart: temporal context, in IST.
- priority: included as a feature for the closure/duration models because in
  real BTP workflow it is assigned at intake (often by fixed policy, e.g.
  VIP movement is always High) - i.e. it IS known before closure/duration
  play out. It is never used as a feature for predicting itself.

NUMERIC_FEATURES / CATEGORICAL_FEATURES are deliberately kept as plain
Python lists (not hardcoded inline in train_models.py) so adding a feature
later means editing exactly one place.
"""

from __future__ import annotations
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

CATEGORICAL_FEATURES = [
    "event_cause", "event_type", "corridor", "police_station", "zone",
    "daypart", "geo_cluster",
]

NUMERIC_FEATURES = [
    "latitude", "longitude", "hour_sin", "hour_cos", "dow_sin", "dow_cos",
    "month", "is_weekend", "is_festival_window", "is_corridor",
]

PRIORITY_FEATURES = CATEGORICAL_FEATURES + NUMERIC_FEATURES
# closure & duration models additionally see the (known-at-intake) priority
CLOSURE_DURATION_FEATURES = PRIORITY_FEATURES + ["priority"]


def build_preprocessor(categorical_cols, numeric_cols) -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
            ("num", "passthrough", numeric_cols),
        ]
    )


def make_xy(df: pd.DataFrame, feature_cols: list[str], target_col: str):
    cols = [c for c in feature_cols if c in df.columns]
    X = df[cols].copy()
    # geo_cluster is an integer label from KMeans - treat as categorical
    if "geo_cluster" in X.columns:
        X["geo_cluster"] = X["geo_cluster"].astype(str)
    y = df[target_col]
    return X, y
