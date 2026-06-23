"""
tests/test_pipeline.py
=======================
Covers the parts of this system that are easy to silently break: the data
pipeline (datetime parsing, feature engineering), the trained models loading
and producing sane outputs, the recommendation engine's output contract, and
the feedback round-trip.

Run with pytest (recommended):
    pip install pytest
    pytest tests/ -v

Or standalone, no pytest required:
    python tests/test_pipeline.py
"""
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from data_prep import build_processed_dataset  # noqa: E402
from impact_index import compute_impact_index, historical_duration_distribution  # noqa: E402
from recommend import ForecastEngine  # noqa: E402
from geo_utils import offset_point, bearing_between, compute_deployment_points, geocode_address  # noqa: E402
import feedback_store  # noqa: E402

DATA_CSV = ROOT / "data" / "astram_event_data.csv"


def test_processed_dataset_has_no_critical_nulls():
    df, _ = build_processed_dataset(DATA_CSV)
    assert len(df) > 8000, "expected the vast majority of rows to survive cleaning"
    for col in ["latitude", "longitude", "month", "event_cause", "priority"]:
        assert df[col].isna().sum() == 0, f"unexpected nulls in {col}"


def test_mixed_datetime_formats_do_not_silently_drop_rows():
    """Regression test for a real bug found during development: pandas'
    vectorised to_datetime locks onto one format per column and silently
    coerces rows with a different format to NaT - which used to make every
    feedback-logged row disappear during retraining. format='mixed' fixes it."""
    df, _ = build_processed_dataset(DATA_CSV)
    raw = pd.read_csv(DATA_CSV, low_memory=False)
    # allow a very small tolerance for rows that are genuinely unparseable
    # (e.g. a handful of malformed source rows), not a large silent drop
    assert len(df) >= int(len(raw) * 0.99)


def test_impact_index_bounds_and_monotonicity():
    df, _ = build_processed_dataset(DATA_CSV)
    durations = historical_duration_distribution(df)
    low = compute_impact_index(0.05, 10, "Low", False, durations)
    high = compute_impact_index(0.95, 500, "High", True, durations)
    assert 0 <= low["score"] <= 100
    assert 0 <= high["score"] <= 100
    assert high["score"] > low["score"], "more severe inputs must score higher"
    assert low["band"] in ("Low", "Medium")
    assert high["band"] in ("High", "Critical")


def test_forecast_engine_returns_well_formed_result():
    engine = ForecastEngine(ROOT)
    result = engine.forecast(
        event_cause="vehicle_breakdown", lat=12.9352, lon=77.6146,  # Silk Board area
        event_type="unplanned", when=pd.Timestamp("2026-07-01 09:00"),
    )
    assert 0 <= result["closure_probability"] <= 1
    assert 0 <= result["impact_index"]["score"] <= 100
    assert result["manpower"]["personnel"] >= 1
    assert result["barricades"]["barricade_points"] >= 0
    assert isinstance(result["similar_incidents"], pd.DataFrame)


def test_forecast_engine_handles_a_point_far_from_any_historical_data():
    """A brand-new planned-event location should still degrade gracefully
    (nearest-neighbour context + diversion fallback), not crash."""
    engine = ForecastEngine(ROOT)
    result = engine.forecast(
        event_cause="public_event", lat=13.20, lon=77.85,  # well outside core BLR
        event_type="planned", when=pd.Timestamp("2026-08-15 18:00"),
    )
    assert result["context"]["corridor"] is not None
    assert result["diversion"]["mode"] in ("live_osm", "historical_fallback")


def test_priority_override_is_respected():
    engine = ForecastEngine(ROOT)
    result = engine.forecast(
        event_cause="construction", lat=12.9716, lon=77.5946,
        event_type="planned", when=pd.Timestamp("2026-07-01 11:00"),
        priority_override="Low",
    )
    assert result["priority"]["used"] == "Low"


def test_offset_point_bearings_are_correct():
    lat, lon = 12.97, 77.59
    n_lat, n_lon = offset_point(lat, lon, 100, 0)      # due north
    e_lat, e_lon = offset_point(lat, lon, 100, 90)     # due east
    assert n_lat > lat and abs(n_lon - lon) < 1e-9
    assert abs(e_lat - lat) < 1e-9 and e_lon > lon


def test_bearing_between_matches_offset_point():
    """Round-trip check: the bearing computed back from a point we placed
    with offset_point should match the bearing we placed it at."""
    lat, lon = 12.97, 77.59
    for true_bearing in [0, 45, 90, 135, 180, 225, 270, 315]:
        plat, plon = offset_point(lat, lon, 200, true_bearing)
        computed = bearing_between(lat, lon, plat, plon)
        assert abs(computed - true_bearing) < 0.5, f"{true_bearing} -> {computed}"


def test_compute_deployment_points_contract():
    lat, lon = 12.97, 77.59
    dep = compute_deployment_points(lat, lon, n_barricades=4, n_personnel=10)
    assert len(dep["barricades"]) == 4
    assert len(dep["officers"]) == 10
    # no osmnx installed in the standard test environment -> must gracefully
    # fall back rather than raise
    assert dep["mode"] in ("geometric_fallback", "live_osm")
    # zero-barricade edge case (common for low-severity unplanned incidents)
    dep0 = compute_deployment_points(lat, lon, n_barricades=0, n_personnel=2)
    assert dep0["barricades"] == []
    assert len(dep0["officers"]) == 2


def test_geocode_address_never_raises():
    """Empty input must short-circuit to None with no network call at all -
    deterministic regardless of environment. A real query's OUTCOME depends
    on network/Nominatim availability (None in this sandbox, which has no
    internet; a real dict when run somewhere with internet), so we only
    assert it never raises - that's the property the Streamlit form
    actually depends on to show a friendly warning instead of crashing."""
    assert geocode_address("") is None
    assert geocode_address("   ") is None
    try:
        result = geocode_address("Silk Board Junction")
    except Exception as e:
        raise AssertionError(f"geocode_address must never raise, got {e!r}")
    assert result is None or (isinstance(result, dict) and "lat" in result and "lon" in result)


def test_feedback_round_trip(tmp_root=None):
    import tempfile
    import shutil
    tmp = Path(tempfile.mkdtemp())
    try:
        (tmp / "feedback").mkdir()
        log = feedback_store.log_outcome(
            tmp, event_cause="accident", event_type="unplanned", lat=12.9, lon=77.6,
            address="test", priority="High", corridor="Test Corridor",
            police_station="Test PS", start_datetime="2026-06-01T10:00:00Z",
            actual_end_datetime="2026-06-01T10:40:00Z", requires_road_closure=False,
            manpower_used=2, forecast_impact_score=40.0, escalated_beyond_forecast=False,
        )
        assert len(log) == 1
        reloaded = feedback_store.load_feedback_log(tmp)
        assert len(reloaded) == 1
        assert reloaded.iloc[0]["event_cause"] == "accident"
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    tests = [(name, fn) for name, fn in list(globals().items())
              if name.startswith("test_") and callable(fn)]
    passed, failed = 0, 0
    for name, fn in tests:
        try:
            fn()
            print(f"PASS  {name}")
            passed += 1
        except Exception as e:
            print(f"FAIL  {name}: {e}")
            failed += 1
    print(f"\n{passed} passed, {failed} failed")
    sys.exit(1 if failed else 0)
