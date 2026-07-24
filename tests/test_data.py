import pandas as pd
import pytest

from pv_weather.data import BASE_COLUMNS, generate_demo_data, validate_hourly_data


def test_demo_data_has_canonical_schema_and_hourly_order():
    data = generate_demo_data("2024-01-01", "2024-02-01")
    assert list(data.columns) == BASE_COLUMNS
    assert data["timestamp_utc"].is_monotonic_increasing
    assert data["timestamp_utc"].diff().dropna().eq(pd.Timedelta(hours=1)).all()
    normalized = data["pv_generation_mwh"] / data["installed_pv_capacity_mw"]
    assert normalized.between(0, 1).all()
    assert normalized.nunique() > 100


def test_validation_reports_missing_columns():
    with pytest.raises(ValueError, match="Fehlende Pflichtspalten"):
        validate_hourly_data(pd.DataFrame({"timestamp_utc": ["2024-01-01"]}))
