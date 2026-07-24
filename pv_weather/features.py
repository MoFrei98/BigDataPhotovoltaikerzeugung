"""Feature engineering for normalized photovoltaic generation."""

from __future__ import annotations

import numpy as np
import pandas as pd


TARGET = "normalized_pv_generation"

MODEL_FEATURES = [
    "temperature_c",
    "relative_humidity_pct",
    "global_radiation_j_cm2",
    "diffuse_radiation_j_cm2",
    "sunshine_duration_min",
    "solar_zenith_angle_deg",
    "cloud_cover_oktas",
    "wind_speed_m_s",
    "estimated_module_temperature_c",
    "thermal_stress_c",
    "radiation_thermal_interaction",
    "diffuse_share",
    "hour_sin",
    "hour_cos",
    "month_sin",
    "month_cos",
]


def estimate_module_temperature(
    air_temperature_c: pd.Series | np.ndarray,
    global_radiation_j_cm2: pd.Series | np.ndarray,
    wind_speed_m_s: pd.Series | np.ndarray,
) -> np.ndarray:
    """Approximate module temperature with a simple NOCT-style wind adjustment."""
    irradiance_w_m2 = np.asarray(global_radiation_j_cm2, dtype=float) / 0.36
    wind = np.maximum(np.asarray(wind_speed_m_s, dtype=float), 0)
    return np.asarray(air_temperature_c, dtype=float) + (
        0.03125 * irradiance_w_m2 / (1 + 0.12 * wind)
    )


def add_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Add calendar, radiation-composition and thermal-stress features."""
    data = frame.copy()
    timestamp = pd.to_datetime(data["timestamp_utc"], utc=True, errors="coerce")
    local = timestamp.dt.tz_convert("Europe/Berlin")
    data["hour"] = local.dt.hour
    data["month"] = local.dt.month
    data["season"] = pd.cut(
        data["month"],
        bins=[0, 2, 5, 8, 11, 12],
        labels=["Winter", "Frühling", "Sommer", "Herbst", "Winter_2"],
        include_lowest=True,
    ).astype(str).replace("Winter_2", "Winter")

    data["hour_sin"] = np.sin(2 * np.pi * data["hour"] / 24)
    data["hour_cos"] = np.cos(2 * np.pi * data["hour"] / 24)
    data["month_sin"] = np.sin(2 * np.pi * (data["month"] - 1) / 12)
    data["month_cos"] = np.cos(2 * np.pi * (data["month"] - 1) / 12)

    safe_global = data["global_radiation_j_cm2"].replace(0, np.nan)
    data["diffuse_share"] = (
        data["diffuse_radiation_j_cm2"] / safe_global
    ).clip(0, 1).fillna(0)
    data["estimated_module_temperature_c"] = estimate_module_temperature(
        data["temperature_c"],
        data["global_radiation_j_cm2"],
        data["wind_speed_m_s"],
    )
    data["thermal_stress_c"] = (
        data["estimated_module_temperature_c"] - 25
    ).clip(lower=0)
    data["radiation_thermal_interaction"] = (
        data["global_radiation_j_cm2"] * data["thermal_stress_c"]
    )

    if {"pv_generation_mwh", "installed_pv_capacity_mw"}.issubset(data.columns):
        data[TARGET] = (
            data["pv_generation_mwh"] / data["installed_pv_capacity_mw"].replace(0, np.nan)
        )
        data.loc[~data[TARGET].between(0, 1.2), TARGET] = np.nan
    return data
