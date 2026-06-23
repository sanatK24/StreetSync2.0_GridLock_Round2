"""
data_prep.py
============
Loads the raw ASTraM (Bengaluru Traffic Police) event export and turns it into
a clean, feature-engineered table that the forecasting models and the
recommendation engine both consume.

Design notes (read this before touching feature logic):
- The raw export has 46 columns; most operational metadata (comment,
  meta_data, map_file, citizen_accident_id, assigned_to_police_id, ...) is
  >90% empty in this anonymized cut and is dropped rather than imputed, to
  avoid manufacturing signal that isn't really there.
- All timestamps in the source are UTC. Every derived time feature
  (hour, day-of-week, month) is converted to Asia/Kolkata first, because the
  consumers of this system (BTP ops) think in local time, not UTC.
- `corridor` (0.2% missing) and `police_station` (0% missing) are the most
  reliable categorical location fields. `zone` (58% missing) and `junction`
  (69% missing) are kept as optional context but never required, because a
  model that only works when junction is populated would be useless for most
  incoming reports. Spatial generalisation instead comes from lat/lon via a
  KMeans grid (see `add_geo_cluster`), which is always available.
- `requires_road_closure` and a derived `duration_min` are the two real,
  observed labels in this data. Everything the models predict is grounded in
  one of those two columns - we do not invent a ground-truth "impact score"
  and pretend it was observed; see impact_index.py for how the composite
  score is built transparently from these two predictions instead.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path

IST = "Asia/Kolkata"

# Multi-day Indian festival/holiday windows that fall inside the dataset's
# observed range (2023-11-09 -> 2024-04-08). Extend this list if you bring in
# more months of data. Dates are IST calendar dates.
FESTIVAL_WINDOWS = [
    ("2023-11-10", "2023-11-15"),  # Diwali / Deepavali
    ("2023-12-24", "2023-12-26"),  # Christmas
    ("2023-12-31", "2024-01-01"),  # New Year's Eve / Day
    ("2024-01-14", "2024-01-16"),  # Makar Sankranti
    ("2024-01-26", "2024-01-26"),  # Republic Day
    ("2024-03-08", "2024-03-08"),  # Maha Shivaratri
    ("2024-03-25", "2024-03-26"),  # Holi
]

RAW_DATE_COLS = [
    "start_datetime", "end_datetime", "modified_datetime", "created_date",
    "closed_datetime", "resolved_datetime",
]

# Columns we keep from the raw export. Everything else is either an internal
# ID, near-100%-empty, or a post-hoc field that wouldn't be known at
# forecast time (e.g. resolved_at_address is filled in AFTER the event ends).
KEEP_COLS = [
    "id", "event_type", "latitude", "longitude", "address", "event_cause",
    "requires_road_closure", "start_datetime", "end_datetime", "status",
    "priority", "corridor", "created_date", "police_station",
    "closed_datetime", "resolved_datetime", "zone", "junction",
]


def load_raw(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)
    df = df[[c for c in KEEP_COLS if c in df.columns]].copy()
    return df


def _to_ist(series: pd.Series) -> pd.Series:
    # format="mixed": the original bulk export uses "YYYY-MM-DD HH:MM:SS.fff+00"
    # while rows appended later through the feedback loop (feedback_store.py)
    # use standard ISO-8601 with a "Z" suffix. Without format="mixed", pandas'
    # vectorised parser locks onto whichever format it sees first and silently
    # coerces every row in a different format to NaT (verified during
    # development - this quietly dropped every feedback-logged row from
    # retraining). format="mixed" parses each value independently instead.
    return pd.to_datetime(series, errors="coerce", utc=True, format="mixed").dt.tz_convert(IST)


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    df["start_ts"] = _to_ist(df["start_datetime"])
    df["hour"] = df["start_ts"].dt.hour
    df["dow"] = df["start_ts"].dt.dayofweek          # 0=Mon
    df["dow_name"] = df["start_ts"].dt.day_name()
    df["month"] = df["start_ts"].dt.month
    df["is_weekend"] = df["dow"].isin([5, 6]).astype(int)

    # cyclical encodings so "23:00" and "00:00" are close to a model
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"].fillna(0) / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"].fillna(0) / 24)
    df["dow_sin"] = np.sin(2 * np.pi * df["dow"].fillna(0) / 7)
    df["dow_cos"] = np.cos(2 * np.pi * df["dow"].fillna(0) / 7)

    # daypart bucket - more interpretable than raw hour for the dashboard
    bins = [-1, 4, 7, 10, 16, 19, 22, 24]
    labels = ["Late Night (00-04)", "Early Morning (04-07)", "Morning (07-10)",
              "Midday (10-16)", "Evening (16-19)", "Night Peak (19-22)", "Late Night (22-24)"]
    df["daypart"] = pd.cut(df["hour"], bins=bins, labels=labels, ordered=False)

    date_str = df["start_ts"].dt.strftime("%Y-%m-%d")
    in_festival = pd.Series(False, index=df.index)
    for start, end in FESTIVAL_WINDOWS:
        in_festival |= (date_str >= start) & (date_str <= end)
    df["is_festival_window"] = in_festival.astype(int)
    return df


def add_duration(df: pd.DataFrame) -> pd.DataFrame:
    start = pd.to_datetime(df["start_datetime"], errors="coerce", utc=True, format="mixed")
    resolved = pd.to_datetime(df["resolved_datetime"], errors="coerce", utc=True, format="mixed")
    closed = pd.to_datetime(df["closed_datetime"], errors="coerce", utc=True, format="mixed")
    end_dt = pd.to_datetime(df["end_datetime"], errors="coerce", utc=True, format="mixed")
    effective_end = resolved.fillna(closed).fillna(end_dt)
    dur = (effective_end - start).dt.total_seconds() / 60.0
    # guard against bad data: negative or absurdly long (>48h) durations are
    # logging errors, not real multi-day closures - excluded from training,
    # not silently clipped, so the model never learns from corrupted labels.
    df["duration_min"] = dur
    df["duration_valid"] = (dur > 0) & (dur < 48 * 60)
    return df


def add_corridor_features(df: pd.DataFrame) -> pd.DataFrame:
    df["corridor"] = df["corridor"].fillna("Non-corridor")
    df["is_corridor"] = (df["corridor"] != "Non-corridor").astype(int)
    return df


def add_geo_cluster(df: pd.DataFrame, n_clusters: int = 50, random_state: int = 42):
    """K-Means grid over lat/lon so every record (and every brand-new point a
    user clicks on the map later) gets a spatial bucket, even when corridor/
    junction/zone are missing. Returns (df_with_cluster_col, fitted_kmeans)."""
    from sklearn.cluster import KMeans

    coords = df[["latitude", "longitude"]].to_numpy()
    km = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    df["geo_cluster"] = km.fit_predict(coords)
    return df, km


def clean_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    df["event_cause"] = df["event_cause"].fillna("others").str.strip()
    # normalise duplicate casing seen in the raw export ("Debris" vs "debris")
    df["event_cause"] = df["event_cause"].replace({"Debris": "debris"})
    df["event_type"] = df["event_type"].fillna("unplanned")
    df["priority"] = df["priority"].fillna("Low")
    df["police_station"] = df["police_station"].fillna("Unknown")
    df["zone"] = df["zone"].fillna("Unknown")
    df["requires_road_closure"] = df["requires_road_closure"].astype(bool)
    return df


def build_processed_dataset(raw_csv_path: str | Path, n_geo_clusters: int = 50):
    """Main entry point. Returns (processed_df, geo_kmeans_model)."""
    df = load_raw(raw_csv_path)
    df = clean_categoricals(df)
    df = add_time_features(df)
    df = add_duration(df)
    df = add_corridor_features(df)
    df, km = add_geo_cluster(df, n_clusters=n_geo_clusters)

    # drop rows with no usable coordinates or no parseable start time - a
    # forecasting system needs both "where" and "when" to be useful, and a
    # handful of source rows (~1.4%) have a malformed start_datetime that
    # fails to parse entirely. Imputing a time for those would be worse than
    # dropping them.
    df = df.dropna(subset=["latitude", "longitude", "month"]).reset_index(drop=True)
    return df, km


if __name__ == "__main__":
    here = Path(__file__).resolve().parent.parent
    df, km = build_processed_dataset(here / "data" / "astram_event_data.csv")
    print("Processed shape:", df.shape)
    print(df[["event_type", "event_cause", "hour", "daypart", "is_festival_window",
               "is_corridor", "geo_cluster", "duration_min", "duration_valid"]].head(8))
    print("\nValid-duration rows:", df["duration_valid"].sum(), "/", len(df))
    out_path = here / "data" / "processed_events.parquet"
    try:
        df.to_parquet(out_path)
        print("Saved:", out_path)
    except Exception as e:
        csv_path = here / "data" / "processed_events.csv"
        df.to_csv(csv_path, index=False)
        print("Parquet failed (", e, ") - saved CSV instead:", csv_path)
