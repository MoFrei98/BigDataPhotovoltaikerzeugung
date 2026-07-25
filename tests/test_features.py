import numpy as np

from pv_weather.data import generate_demo_data
from pv_weather.features import (
    MONOTONIC_CONSTRAINTS,
    MODEL_FEATURES,
    TARGET,
    add_features,
    estimate_module_temperature,
)


def test_features_create_normalized_target_and_thermal_stress():
    data = generate_demo_data("2024-06-01", "2024-06-10")
    featured = add_features(data)
    expected = data["pv_generation_mwh"] / data["installed_pv_capacity_mw"]
    assert np.allclose(featured[TARGET], expected)
    assert (featured["thermal_stress_c"] >= 0).all()
    assert "installed_pv_capacity_mw" not in MODEL_FEATURES
    assert "estimated_module_temperature_c" in MODEL_FEATURES
    assert set(MONOTONIC_CONSTRAINTS) == set(MODEL_FEATURES)


def test_module_temperature_rises_with_radiation_and_falls_with_wind():
    low_wind = estimate_module_temperature([25], [280], [1])[0]
    high_wind = estimate_module_temperature([25], [280], [8])[0]
    no_sun = estimate_module_temperature([25], [0], [1])[0]
    assert low_wind > high_wind > no_sun
