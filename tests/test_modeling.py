import pandas as pd

from pv_weather.data import generate_demo_data
from pv_weather.features import MODEL_FEATURES, TARGET, add_features
from pv_weather.modeling import predict_yield, train_yield_model


def test_temporal_model_and_prediction_interval():
    data = generate_demo_data("2023-01-01", "2023-06-30")
    bundle = train_yield_model(data)
    scenario = data.tail(2).drop(
        columns=["installed_pv_capacity_mw", "pv_generation_mwh"]
    )
    result = predict_yield(bundle, scenario)

    assert bundle.metrics["test_rows"] > 0
    assert bundle.metrics["model_mae"] < bundle.metrics["baseline_mae"]
    assert result["normalized_pv_prediction"].between(0, 1.2).all()
    assert (result["lower_80"] <= result["upper_80"]).all()
    assert set(bundle.feature_importance) == set(bundle.feature_names)


def test_training_and_evaluation_use_only_daylight_without_mutating_source():
    data = generate_demo_data("2024-01-01", "2024-02-15", seed=17)
    data["global_radiation_j_cm2"] = 100.0
    data["solar_zenith_angle_deg"] = 45.0

    # The thresholds are intentionally strict: equality is not daylight here.
    data.loc[data.index[0], "global_radiation_j_cm2"] = 10.0
    data.loc[data.index[1], "solar_zenith_angle_deg"] = 90.0
    data.loc[data.index[2], "global_radiation_j_cm2"] = 10.001
    data.loc[data.index[2], "solar_zenith_angle_deg"] = 89.999
    original = data.copy(deep=True)

    featured = add_features(data)
    eligible = featured.dropna(subset=[*MODEL_FEATURES, TARGET])
    eligible = eligible[
        (eligible["global_radiation_j_cm2"] > 10)
        & (eligible["solar_zenith_angle_deg"] < 90)
    ].sort_values("timestamp_utc")
    expected_split = int(len(eligible) * 0.8)

    bundle = train_yield_model(data)

    assert bundle.metrics["train_rows"] == expected_split
    assert bundle.metrics["test_rows"] == len(eligible) - expected_split
    assert bundle.split_timestamp == pd.Timestamp(
        eligible["timestamp_utc"].iloc[expected_split]
    )
    pd.testing.assert_frame_equal(data, original)
