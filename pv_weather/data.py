"""Load, validate and generate the canonical hourly PV-weather panel."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


BASE_COLUMNS = [
    "timestamp_utc",
    "temperature_c",
    "relative_humidity_pct",
    "global_radiation_j_cm2",
    "diffuse_radiation_j_cm2",
    "sunshine_duration_min",
    "solar_zenith_angle_deg",
    "cloud_cover_oktas",
    "wind_speed_m_s",
    "installed_pv_capacity_mw",
    "pv_generation_mwh",
]

NUMERIC_COLUMNS = BASE_COLUMNS[1:]


def validate_hourly_data(frame: pd.DataFrame) -> pd.DataFrame:
    """Validate units and schema, collapse duplicates and sort by UTC hour."""
    missing = [column for column in BASE_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"Fehlende Pflichtspalten: {', '.join(missing)}")

    data = frame[BASE_COLUMNS].copy()
    data["timestamp_utc"] = pd.to_datetime(data["timestamp_utc"], utc=True, errors="coerce")
    for column in NUMERIC_COLUMNS:
        data[column] = pd.to_numeric(data[column], errors="coerce")

    data = data.dropna(subset=["timestamp_utc"]).sort_values("timestamp_utc")
    data = data.groupby("timestamp_utc", as_index=False)[NUMERIC_COLUMNS].mean()

    nonnegative = [
        "relative_humidity_pct",
        "global_radiation_j_cm2",
        "diffuse_radiation_j_cm2",
        "sunshine_duration_min",
        "solar_zenith_angle_deg",
        "cloud_cover_oktas",
        "wind_speed_m_s",
        "installed_pv_capacity_mw",
        "pv_generation_mwh",
    ]
    for column in nonnegative:
        data.loc[data[column] < 0, column] = np.nan

    data.loc[data["relative_humidity_pct"] > 100, "relative_humidity_pct"] = np.nan
    data.loc[data["cloud_cover_oktas"] > 8, "cloud_cover_oktas"] = np.nan
    data.loc[data["sunshine_duration_min"] > 60, "sunshine_duration_min"] = np.nan
    data.loc[~data["solar_zenith_angle_deg"].between(0, 180), "solar_zenith_angle_deg"] = np.nan
    data.loc[data["installed_pv_capacity_mw"] <= 0, "installed_pv_capacity_mw"] = np.nan

    capacity_factor = data["pv_generation_mwh"] / data["installed_pv_capacity_mw"]
    data.loc[capacity_factor > 1.2, "pv_generation_mwh"] = np.nan
    if data.empty:
        raise ValueError("Der Datensatz enthält nach der Bereinigung keine gültigen Zeitstempel.")
    return data


def generate_demo_data(
    start: str = "2022-01-01",
    end: str = "2024-12-31 23:00",
    seed: int = 42,
) -> pd.DataFrame:
    """Create transparent synthetic data that encode irradiation and thermal loss.

    These data make the app executable without downloads. They are not empirical
    evidence and must not be used to answer the research question substantively.
    """
    rng = np.random.default_rng(seed)
    timestamp = pd.date_range(start, end, freq="h", tz="UTC")
    n = len(timestamp)
    local = timestamp.tz_convert("Europe/Berlin")
    hour = local.hour.to_numpy()
    day_of_year = local.dayofyear.to_numpy()
    year = local.year.to_numpy()

    seasonal = np.sin(2 * np.pi * (day_of_year - 172) / 365.25)
    air_temperature = (
        11.5
        + 11.5 * seasonal
        + 4.5 * np.sin(2 * np.pi * (hour - 14) / 24)
        + rng.normal(0, 2.8, n)
    )

    cloud_noise = rng.normal(0, 1, n)
    cloud_state = np.zeros(n)
    for index in range(1, n):
        cloud_state[index] = 0.93 * cloud_state[index - 1] + cloud_noise[index]
    cloud_cover = np.clip(4.2 + 1.25 * cloud_state + rng.normal(0, 0.5, n), 0, 8)

    daylight_length = 12 + 4.3 * seasonal
    sunrise = 12 - daylight_length / 2
    daylight = np.clip(np.sin(np.pi * (hour - sunrise) / daylight_length), 0, None)
    transmission = np.clip(1 - 0.075 * cloud_cover, 0.25, 1)
    irradiance_w_m2 = np.clip(
        930 * daylight * transmission + rng.normal(0, 18, n) * (daylight > 0),
        0,
        None,
    )
    global_radiation = irradiance_w_m2 * 0.36
    diffuse_share = np.clip(0.14 + 0.075 * cloud_cover, 0.12, 0.82)
    diffuse_radiation = global_radiation * diffuse_share
    sunshine_duration = np.clip(60 * daylight * (1 - cloud_cover / 9), 0, 60)
    solar_zenith = np.clip(90 - 62 * daylight, 18, 110)

    humidity = np.clip(
        76 - 1.15 * air_temperature + 2.2 * cloud_cover + rng.normal(0, 7, n),
        20,
        100,
    )
    wind_noise = rng.normal(0, 1, n)
    wind_state = np.zeros(n)
    for index in range(1, n):
        wind_state[index] = 0.88 * wind_state[index - 1] + wind_noise[index]
    wind_speed = np.clip(3.4 + 0.7 * wind_state - 0.7 * seasonal, 0.2, 14)

    installed_capacity = np.select(
        [year == 2022, year == 2023],
        [59_300.0, 67_600.0],
        default=81_800.0,
    )
    module_temperature = air_temperature + (
        0.03125 * irradiance_w_m2 / (1 + 0.12 * wind_speed)
    )
    temperature_factor = np.clip(1 - 0.0042 * (module_temperature - 25), 0.76, 1.08)
    normalized_generation = np.clip(
        (irradiance_w_m2 / 1_000)
        * 0.91
        * temperature_factor
        + rng.normal(0, 0.012, n) * (daylight > 0),
        0,
        0.96,
    )
    pv_generation = normalized_generation * installed_capacity

    return validate_hourly_data(
        pd.DataFrame(
            {
                "timestamp_utc": timestamp,
                "temperature_c": air_temperature,
                "relative_humidity_pct": humidity,
                "global_radiation_j_cm2": global_radiation,
                "diffuse_radiation_j_cm2": diffuse_radiation,
                "sunshine_duration_min": sunshine_duration,
                "solar_zenith_angle_deg": solar_zenith,
                "cloud_cover_oktas": cloud_cover,
                "wind_speed_m_s": wind_speed,
                "installed_pv_capacity_mw": installed_capacity,
                "pv_generation_mwh": pv_generation,
            }
        )
    )


def load_project_data(
    processed_path: str | Path = "data/processed/hourly_pv_weather.csv",
) -> tuple[pd.DataFrame, str]:
    """Load prepared observations or fall back to labelled synthetic demo data."""
    path = Path(processed_path)
    if path.exists():
        return validate_hourly_data(pd.read_csv(path)), f"Realdaten: {path.as_posix()}"
    return generate_demo_data(), "Synthetische Demo – keine empirischen Ergebnisse"
