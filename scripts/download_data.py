"""Download and freeze UBA air-quality and DWD weather data for Potsdam."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import requests


ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
USER_AGENT = "UrbanAirQualityForecast university project/2.0"

START_YEAR = 2016
END_YEAR = 2025
UBA_STATION = "DEBB021"  # Potsdam-Zentrum, urban background
DWD_STATION = "03987"  # Potsdam

UBA_URL = (
    "https://luftdaten.umweltbundesamt.de/api-proxy/airquality/csv"
    "?date_from={year}-01-01&date_to={year}-12-31&time_from=1&time_to=24"
    f"&station={UBA_STATION}"
)

DWD_BASE = "https://opendata.dwd.de/climate_environment/CDC/observations_germany/climate/hourly"
DWD_FILES = {
    "air_temperature": (
        f"{DWD_BASE}/air_temperature/historical/"
        "stundenwerte_TU_03987_18930101_20251231_hist.zip"
    ),
    "wind": (
        f"{DWD_BASE}/wind/historical/"
        "stundenwerte_FF_03987_18930101_20251231_hist.zip"
    ),
    "sun": (
        f"{DWD_BASE}/sun/historical/"
        "stundenwerte_SD_03987_18930101_20251231_hist.zip"
    ),
    "precipitation": (
        f"{DWD_BASE}/precipitation/historical/"
        "stundenwerte_RR_03987_19950901_20251231_hist.zip"
    ),
    "solar": f"{DWD_BASE}/solar/stundenwerte_ST_03987_row.zip",
}


def download(url: str, destination: Path, refresh: bool) -> dict[str, object]:
    retrieved_now = refresh or not destination.exists()
    if retrieved_now:
        response = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=180)
        response.raise_for_status()
        destination.write_bytes(response.content)

    content = destination.read_bytes()
    return {
        "file": destination.relative_to(ROOT).as_posix(),
        "url": url,
        "bytes": len(content),
        "sha256": hashlib.sha256(content).hexdigest(),
        "retrieved_now": retrieved_now,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--refresh", action="store_true", help="Download all files again.")
    args = parser.parse_args()

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    entries: list[dict[str, object]] = []

    for year in range(START_YEAR, END_YEAR + 1):
        entries.append(
            download(
                UBA_URL.format(year=year),
                RAW_DIR / f"uba_airquality_{UBA_STATION}_{year}.csv",
                args.refresh,
            )
        )

    for variable, url in DWD_FILES.items():
        entries.append(download(url, RAW_DIR / f"dwd_{variable}_{DWD_STATION}.zip", args.refresh))

    manifest = {
        "manifest_created_at_utc": datetime.now(timezone.utc).isoformat(),
        "analysis_period": {"start": START_YEAR, "end": END_YEAR},
        "stations": {
            "air_quality": {
                "id": UBA_STATION,
                "name": "Potsdam-Zentrum",
                "type": "urban background",
                "latitude": 52.401956,
                "longitude": 13.063989,
            },
            "weather": {
                "id": DWD_STATION,
                "name": "Potsdam",
                "latitude": 52.3812,
                "longitude": 13.0622,
            },
        },
        "files": entries,
    }
    manifest_path = RAW_DIR / "download_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    downloaded = sum(bool(entry["retrieved_now"]) for entry in entries)
    total_bytes = sum(int(entry["bytes"]) for entry in entries)
    print(f"Prepared {len(entries)} files ({downloaded} downloaded now, {total_bytes / 1e6:.1f} MB).")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
