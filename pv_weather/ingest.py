"""Import manually downloaded SMARD generation/capacity and DWD weather data."""

from __future__ import annotations

import unicodedata
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from .data import validate_hourly_data


def _plain(text: str) -> str:
    normalized = unicodedata.normalize("NFKD", str(text))
    return "".join(char for char in normalized if not unicodedata.combining(char)).lower()


def _read_csv(path: Path) -> pd.DataFrame:
    first_line = path.read_text(encoding="utf-8-sig", errors="replace").splitlines()[0]
    separator = ";" if first_line.count(";") >= first_line.count(",") else ","
    return pd.read_csv(path, sep=separator, dtype=str, encoding="utf-8-sig")


def _numeric(series: pd.Series) -> pd.Series:
    text = series.astype(str).str.strip().replace({"": np.nan, "-": np.nan, "–": np.nan})
    if text.str.contains(",", regex=False, na=False).mean() > 0.05:
        text = text.str.replace(".", "", regex=False).str.replace(",", ".", regex=False)
    text = text.str.replace(r"[^0-9eE+\-.]", "", regex=True)
    return pd.to_numeric(text, errors="coerce")


def _local_market_timestamp(series: pd.Series) -> pd.Series:
    parsed = pd.to_datetime(series, dayfirst=True, errors="coerce")
    if getattr(parsed.dt, "tz", None) is not None:
        return parsed.dt.tz_convert("UTC")
    try:
        return parsed.dt.tz_localize(
            "Europe/Berlin", ambiguous="infer", nonexistent="shift_forward"
        ).dt.tz_convert("UTC")
    except ValueError:
        return parsed.dt.tz_localize(
            "Europe/Berlin", ambiguous=True, nonexistent="shift_forward"
        ).dt.tz_convert("UTC")


def read_smard_generation(directory: str | Path) -> pd.DataFrame:
    """Read realized PV generation and sum sub-hourly MWh to UTC hours."""
    directory = Path(directory)
    files = sorted(directory.glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"Keine SMARD-CSV-Dateien in {directory} gefunden.")

    rows: list[pd.DataFrame] = []
    for path in files:
        frame = _read_csv(path)
        timestamp_candidates = [
            column
            for column in frame.columns
            if any(token in _plain(column) for token in ("datum von", "date from", "timestamp"))
        ]
        timestamp_column = timestamp_candidates[0] if timestamp_candidates else frame.columns[0]
        value_columns = [
            column
            for column in frame.columns
            if column != timestamp_column
            and not any(token in _plain(column) for token in ("datum bis", "date to"))
        ]
        pv_columns = [column for column in value_columns if "photovoltaik" in _plain(column)]
        if not pv_columns and len(value_columns) == 1 and _plain(path.stem).startswith(
            ("pv_", "pv-generation", "pv_generation")
        ):
            pv_columns = value_columns
        for column in pv_columns:
            rows.append(
                pd.DataFrame(
                    {
                        "timestamp_utc": _local_market_timestamp(frame[timestamp_column]),
                        "pv_generation_mwh": _numeric(frame[column]),
                    }
                ).dropna()
            )

    if not rows:
        raise ValueError(
            "Keine Photovoltaik-Reihe erkannt. Einzeldateien bitte mit "
            "`pv_generation_` benennen."
        )
    generation = pd.concat(rows, ignore_index=True).drop_duplicates(
        "timestamp_utc", keep="last"
    )
    return (
        generation.set_index("timestamp_utc")["pv_generation_mwh"]
        .resample("h")
        .sum(min_count=1)
        .reset_index()
    )


def read_installed_capacity(path: str | Path) -> pd.DataFrame:
    """Read annual installed PV capacity in MW from a simple or SMARD-style CSV."""
    path = Path(path)
    files = sorted(path.glob("*.csv")) if path.is_dir() else [path]
    if not files or not all(file.exists() for file in files):
        raise FileNotFoundError(f"Keine Kapazitäts-CSV unter {path} gefunden.")

    annual_rows: list[pd.DataFrame] = []
    for file in files:
        frame = _read_csv(file)
        year_columns = [
            column
            for column in frame.columns
            if any(token in _plain(column) for token in ("jahr", "year", "datum", "date"))
        ]
        capacity_columns = [
            column
            for column in frame.columns
            if (
                "photovoltaik" in _plain(column)
                or "pv" in _plain(column)
                or "installierte" in _plain(column)
            )
            and ("mw" in _plain(column) or "leistung" in _plain(column) or "capacity" in _plain(column))
        ]
        if not year_columns:
            year_columns = [frame.columns[0]]
        if not capacity_columns and len(frame.columns) == 2:
            capacity_columns = [frame.columns[1]]
        if not capacity_columns:
            continue

        year_raw = frame[year_columns[0]].astype(str).str.extract(r"((?:19|20)\d{2})")[0]
        annual_rows.append(
            pd.DataFrame(
                {
                    "year": pd.to_numeric(year_raw, errors="coerce"),
                    "installed_pv_capacity_mw": _numeric(frame[capacity_columns[0]]),
                }
            ).dropna()
        )

    if not annual_rows:
        raise ValueError("Keine jährliche installierte PV-Leistung erkannt.")
    annual = pd.concat(annual_rows, ignore_index=True)
    annual["year"] = annual["year"].astype(int)
    return annual.groupby("year", as_index=False)["installed_pv_capacity_mw"].last()


def _load_station_weights(path: str | Path | None) -> dict[str, float]:
    if path is None or not Path(path).exists():
        return {}
    frame = pd.read_csv(path, dtype={"station_id": str})
    if not {"station_id", "weight"}.issubset(frame.columns):
        raise ValueError("Stationsgewichte benötigen die Spalten station_id und weight.")
    return {
        str(station).zfill(5): float(weight)
        for station, weight in zip(frame["station_id"], frame["weight"], strict=True)
        if pd.notna(weight) and float(weight) > 0
    }


def read_dwd_archives(
    directory: str | Path,
    station_weights_path: str | Path | None = None,
) -> pd.DataFrame:
    """Aggregate DWD station observations using equal or supplied PV weights."""
    directory = Path(directory)
    files = sorted(directory.glob("*.zip"))
    if not files:
        raise FileNotFoundError(f"Keine DWD-ZIP-Dateien in {directory} gefunden.")

    weights = _load_station_weights(station_weights_path)
    long_frames: list[pd.DataFrame] = []
    column_map = {
        "TT_TU": ("temperature_c", 1.0),
        "RF_TU": ("relative_humidity_pct", 1.0),
        "FG_LBERG": ("global_radiation_j_cm2", 1.0),
        "FG_STRAHL": ("global_radiation_j_cm2", 1.0),
        "FD_LBERG": ("diffuse_radiation_j_cm2", 1.0),
        "FD_STRAHL": ("diffuse_radiation_j_cm2", 1.0),
        "SD_LBERG": ("sunshine_duration_min", 1.0),
        "SD_STRAHL": ("sunshine_duration_min", 1.0),
        "SD_SO": ("sunshine_duration_min", 60.0),
        "ZENIT": ("solar_zenith_angle_deg", 1.0),
        "V_N": ("cloud_cover_oktas", 1.0),
        "N": ("cloud_cover_oktas", 1.0),
        "F": ("wind_speed_m_s", 1.0),
    }

    for archive_path in files:
        with zipfile.ZipFile(archive_path) as archive:
            product_files = [
                name
                for name in archive.namelist()
                if "produkt_" in name.lower() and name.lower().endswith(".txt")
            ]
            if not product_files:
                continue
            frame = pd.read_csv(archive.open(product_files[0]), sep=";", dtype=str)
        frame.columns = [column.strip() for column in frame.columns]
        if "MESS_DATUM" not in frame:
            continue
        raw_timestamp = frame["MESS_DATUM"].astype(str).str.strip()
        timestamp = pd.to_datetime(
            raw_timestamp,
            format="%Y%m%d%H:%M" if raw_timestamp.str.contains(":").any() else "%Y%m%d%H",
            errors="coerce",
            utc=True,
        ).dt.floor("h")
        if "STATIONS_ID" in frame:
            station = frame["STATIONS_ID"].astype(str).str.strip().str.zfill(5)
        else:
            station_id = next(
                (part for part in archive_path.stem.split("_") if part.isdigit()),
                "unknown",
            )
            station = pd.Series(station_id.zfill(5), index=frame.index)

        for source_column, (target_column, factor) in column_map.items():
            if source_column not in frame:
                continue
            parsed = pd.DataFrame(
                {
                    "timestamp_utc": timestamp,
                    "station_id": station,
                    "variable": target_column,
                    "value": _numeric(frame[source_column]).replace(-999, np.nan) * factor,
                }
            ).dropna(subset=["timestamp_utc", "value"])
            long_frames.append(parsed)

    if not long_frames:
        raise ValueError("Keine unterstützten Wettergrößen in den DWD-ZIPs erkannt.")
    long = pd.concat(long_frames, ignore_index=True).drop_duplicates(
        ["timestamp_utc", "station_id", "variable"], keep="last"
    )
    long["weight"] = long["station_id"].map(weights).fillna(1.0)
    long["weighted_value"] = long["value"] * long["weight"]
    aggregated = (
        long.groupby(["timestamp_utc", "variable"], as_index=False)
        .agg(weighted_sum=("weighted_value", "sum"), weight_sum=("weight", "sum"))
    )
    aggregated["value"] = aggregated["weighted_sum"] / aggregated["weight_sum"]
    return aggregated.pivot(
        index="timestamp_utc", columns="variable", values="value"
    ).reset_index()


def prepare_dataset(
    smard_dir: str | Path,
    capacity_path: str | Path,
    dwd_dir: str | Path,
    station_weights_path: str | Path | None = None,
) -> pd.DataFrame:
    """Join hourly generation, annual capacity and aggregated weather in UTC."""
    generation = read_smard_generation(smard_dir)
    capacity = read_installed_capacity(capacity_path)
    weather = read_dwd_archives(dwd_dir, station_weights_path)

    merged = generation.merge(weather, on="timestamp_utc", how="left", validate="one_to_one")
    merged["year"] = merged["timestamp_utc"].dt.year
    merged = merged.merge(capacity, on="year", how="left", validate="many_to_one").drop(
        columns="year"
    )
    return validate_hourly_data(merged)
