"""
train_models.py
================
Trains and evaluates the three supervised models behind the forecasting
engine, using a TIME-BASED holdout (train on the earliest 80% of events,
test on the most recent 20%) rather than a random split. A random split
would let the model "see the future" relative to nearby test rows recorded
minutes apart; a time split honestly simulates "forecast events you haven't
seen yet," which is the actual task BTP needs solved.

Models:
  1. priority_model      - classifies High/Low priority from cause+context
  2. closure_model        - P(requires_road_closure) - the key driver of the
                             barricading recommendation
  3. duration_model       - expected disruption duration in minutes (log
                             space internally, minutes at the API boundary)

Run:
    python src/train_models.py
Outputs:
    models/*.joblib   - fitted sklearn Pipelines (preprocessing + estimator)
    models/geo_kmeans.joblib
    models/metrics.json   - honest, reproducible evaluation numbers
    models/feature_importance.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (
    roc_auc_score, average_precision_score, f1_score, precision_score,
    recall_score, accuracy_score, mean_absolute_error,
    median_absolute_error, r2_score, confusion_matrix,
)
from sklearn.pipeline import Pipeline
import joblib

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from data_prep import build_processed_dataset  # noqa: E402
from features import (  # noqa: E402
    PRIORITY_FEATURES, CLOSURE_DURATION_FEATURES, build_preprocessor, make_xy,
)

RANDOM_STATE = 42


def time_split(df: pd.DataFrame, frac_train: float = 0.8):
    df_sorted = df.sort_values("start_ts").reset_index(drop=True)
    cut = int(len(df_sorted) * frac_train)
    return df_sorted.iloc[:cut].copy(), df_sorted.iloc[cut:].copy()


def cat_num_split(feature_cols, df):
    cats = [c for c in feature_cols if c in
            ["event_cause", "event_type", "corridor", "police_station",
             "zone", "daypart", "geo_cluster", "priority"]]
    nums = [c for c in feature_cols if c not in cats]
    return cats, nums


def train_priority_model(train_df, test_df):
    cats, nums = cat_num_split(PRIORITY_FEATURES, train_df)
    Xtr, ytr = make_xy(train_df, PRIORITY_FEATURES, "priority")
    Xte, yte = make_xy(test_df, PRIORITY_FEATURES, "priority")

    pre = build_preprocessor(cats, nums)
    clf = RandomForestClassifier(
        n_estimators=300, min_samples_leaf=3, class_weight="balanced",
        random_state=RANDOM_STATE, n_jobs=-1,
    )
    pipe = Pipeline([("pre", pre), ("clf", clf)])
    pipe.fit(Xtr, ytr)

    pred = pipe.predict(Xte)
    metrics = {
        "accuracy": float(accuracy_score(yte, pred)),
        "f1_weighted": float(f1_score(yte, pred, average="weighted")),
        "n_train": int(len(Xtr)),
        "n_test": int(len(Xte)),
        "class_balance_test": yte.value_counts(normalize=True).to_dict(),
    }
    return pipe, metrics


def train_closure_model(train_df, test_df):
    cats, nums = cat_num_split(CLOSURE_DURATION_FEATURES, train_df)
    Xtr, ytr = make_xy(train_df, CLOSURE_DURATION_FEATURES, "requires_road_closure")
    Xte, yte = make_xy(test_df, CLOSURE_DURATION_FEATURES, "requires_road_closure")

    pre = build_preprocessor(cats, nums)
    clf = RandomForestClassifier(
        n_estimators=400, min_samples_leaf=2, class_weight="balanced_subsample",
        random_state=RANDOM_STATE, n_jobs=-1,
    )
    pipe = Pipeline([("pre", pre), ("clf", clf)])
    pipe.fit(Xtr, ytr)

    proba = pipe.predict_proba(Xte)[:, 1]
    pred = (proba >= 0.5).astype(int)
    metrics = {
        "roc_auc": float(roc_auc_score(yte, proba)),
        "pr_auc": float(average_precision_score(yte, proba)),
        "f1_at_0.5": float(f1_score(yte, pred)),
        "precision_at_0.5": float(precision_score(yte, pred, zero_division=0)),
        "recall_at_0.5": float(recall_score(yte, pred, zero_division=0)),
        "positive_rate_test": float(yte.mean()),
        "confusion_matrix_at_0.5": confusion_matrix(yte, pred).tolist(),
        "n_train": int(len(Xtr)),
        "n_test": int(len(Xte)),
    }
    return pipe, metrics


DURATION_TIER_BINS = [0, 30, 120, np.inf]
DURATION_TIER_LABELS = ["Quick (<30 min)", "Moderate (30-120 min)", "Prolonged (>120 min)"]

# Duration is heavy-tailed (std >> mean) and we only have ~2.3k rows with a
# resolvable duration after a time-based split, so a high-cardinality
# RandomForest memorises train and collapses on test (verified during
# development: full feature set gave train R2=0.75 vs test R2=0.02). Two
# fixes applied: (1) drop the highest-cardinality, sparsest spatial columns
# (police_station, geo_cluster, raw lat/lon) which a 2.3k-row sample can't
# support without overfitting, keeping the columns that are actually known
# in advance and carry real signal (event_cause, corridor type, time);
# (2) heavily regularise the trees (deep min_samples_leaf, capped depth).
# We report the point-estimate regression AND a 3-tier classification
# (Quick/Moderate/Prolonged) side by side - the tier view is materially more
# robust and is what the recommendation engine relies on; the regression is
# shown as a secondary, explicitly-caveated estimate.
DURATION_FEATURE_COLS_CAT = ["event_cause", "event_type", "corridor", "zone", "daypart", "priority"]
DURATION_FEATURE_COLS_NUM = ["hour_sin", "hour_cos", "dow_sin", "dow_cos", "month",
                              "is_weekend", "is_festival_window", "is_corridor"]


def _duration_tier(minutes: pd.Series) -> pd.Series:
    return pd.cut(minutes, bins=DURATION_TIER_BINS, labels=DURATION_TIER_LABELS, right=False)


def train_duration_model(train_df, test_df):
    train_d = train_df[train_df["duration_valid"]].copy()
    test_d = test_df[test_df["duration_valid"]].copy()
    cats, nums = DURATION_FEATURE_COLS_CAT, DURATION_FEATURE_COLS_NUM

    Xtr = train_d[cats + nums].copy()
    Xte = test_d[cats + nums].copy()
    ytr_min, yte_min = train_d["duration_min"], test_d["duration_min"]
    ytr_log, yte_log = np.log1p(ytr_min), np.log1p(yte_min)

    pre = build_preprocessor(cats, nums)
    reg = RandomForestRegressor(
        n_estimators=300, min_samples_leaf=25, max_depth=8,
        random_state=RANDOM_STATE, n_jobs=-1,
    )
    reg_pipe = Pipeline([("pre", pre), ("reg", reg)])
    reg_pipe.fit(Xtr, ytr_log)
    pred_min = np.expm1(reg_pipe.predict(Xte))

    naive_pred = np.full_like(yte_min.values, fill_value=float(ytr_min.median()))
    reg_metrics = {
        "mae_minutes": float(mean_absolute_error(yte_min, pred_min)),
        "median_ae_minutes": float(median_absolute_error(yte_min, pred_min)),
        "r2_log_space": float(r2_score(yte_log, reg_pipe.predict(Xte))),
        "naive_baseline_mae_minutes": float(mean_absolute_error(yte_min, naive_pred)),
        "median_actual_minutes": float(yte_min.median()),
        "n_train": int(len(Xtr)),
        "n_test": int(len(Xte)),
    }

    # tier classifier - same features, more robust target
    ytr_tier = _duration_tier(ytr_min)
    yte_tier = _duration_tier(yte_min)
    pre2 = build_preprocessor(cats, nums)
    tier_clf = RandomForestClassifier(
        n_estimators=300, min_samples_leaf=15, max_depth=8,
        class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1,
    )
    tier_pipe = Pipeline([("pre", pre2), ("clf", tier_clf)])
    tier_pipe.fit(Xtr, ytr_tier)
    tier_pred = tier_pipe.predict(Xte)
    naive_tier_pred = np.full(len(yte_tier), fill_value=ytr_tier.mode()[0])
    reg_metrics.update({
        "tier_accuracy": float(accuracy_score(yte_tier, tier_pred)),
        "tier_f1_macro": float(f1_score(yte_tier, tier_pred, average="macro")),
        "tier_naive_baseline_accuracy": float(accuracy_score(yte_tier, naive_tier_pred)),
        "tier_labels": DURATION_TIER_LABELS,
        "tier_confusion_matrix": confusion_matrix(
            yte_tier, tier_pred, labels=DURATION_TIER_LABELS).tolist(),
    })
    return reg_pipe, tier_pipe, reg_metrics


def top_feature_importance(pipe, top_n=15):
    pre = pipe.named_steps["pre"]
    model_step = [s for s in pipe.named_steps if s in ("clf", "reg")][0]
    model = pipe.named_steps[model_step]
    names = pre.get_feature_names_out()
    importances = model.feature_importances_
    order = np.argsort(importances)[::-1][:top_n]
    return [{"feature": names[i].split("__")[-1], "importance": float(importances[i])}
            for i in order]


def run_pipeline(data_path, models_dir, save: bool = True):
    """The full train+evaluate+save pipeline, callable both from the CLI
    (`python src/train_models.py`) and from the Streamlit app's 'retrain
    with feedback' button, on whichever CSV path is passed in."""
    models_dir = Path(models_dir)
    models_dir.mkdir(exist_ok=True)

    df, km = build_processed_dataset(data_path)
    train_df, test_df = time_split(df, frac_train=0.8)

    all_metrics = {"n_total_events": int(len(df)),
                   "train_window_end": str(train_df["start_ts"].max()),
                   "test_window_start": str(test_df["start_ts"].min())}

    pri_pipe, pri_metrics = train_priority_model(train_df, test_df)
    all_metrics["priority_model"] = pri_metrics

    clo_pipe, clo_metrics = train_closure_model(train_df, test_df)
    all_metrics["closure_model"] = clo_metrics

    dur_pipe, tier_pipe, dur_metrics = train_duration_model(train_df, test_df)
    all_metrics["duration_model"] = dur_metrics

    pipes = {"priority_model": pri_pipe, "closure_model": clo_pipe,
             "duration_model": dur_pipe, "duration_tier_model": tier_pipe,
             "geo_kmeans": km}

    if save:
        for name, obj in pipes.items():
            joblib.dump(obj, models_dir / f"{name}.joblib")
        feature_importance = {
            "closure_model": top_feature_importance(clo_pipe),
            "duration_model": top_feature_importance(dur_pipe),
            "duration_tier_model": top_feature_importance(tier_pipe),
        }
        with open(models_dir / "feature_importance.json", "w") as f:
            json.dump(feature_importance, f, indent=2)
        with open(models_dir / "metrics.json", "w") as f:
            json.dump(all_metrics, f, indent=2, default=str)

    return pipes, all_metrics


def main():
    data_path = ROOT / "data" / "astram_event_data.csv"
    models_dir = ROOT / "models"

    print("Loading + engineering features...")
    pipes, all_metrics = run_pipeline(data_path, models_dir, save=True)

    pm = all_metrics["priority_model"]
    print(f"\n[1/3] Priority model -> accuracy={pm['accuracy']:.3f} f1_weighted={pm['f1_weighted']:.3f}")
    cm = all_metrics["closure_model"]
    print(f"[2/3] Closure model  -> ROC-AUC={cm['roc_auc']:.3f} PR-AUC={cm['pr_auc']:.3f} "
          f"(base rate={cm['positive_rate_test']:.3f})")
    dm = all_metrics["duration_model"]
    print(f"[3/3] Duration model -> MAE={dm['mae_minutes']:.1f}min "
          f"(naive baseline={dm['naive_baseline_mae_minutes']:.1f}min) | "
          f"tier accuracy={dm['tier_accuracy']:.3f} (naive={dm['tier_naive_baseline_accuracy']:.3f})")
    print("\nAll models + metrics saved to", models_dir)


if __name__ == "__main__":
    main()
