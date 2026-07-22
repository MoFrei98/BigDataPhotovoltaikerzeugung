"""Download and freeze HLNUG air-quality and DWD weather data for Frankfurt."""

from __future__ import annotations

import argparse
import calendar
import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import requests


ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
USER_AGENT = "FrankfurtUrbanAirQualityForecast university project/3.0"

START_YEAR = 2016
END_YEAR = 2025
AIR_STATION = "DEHE005"  # Frankfurt-Höchst, urban background
HLNUG_STATION = "0617"  # internal station identifier in the HLNUG data portal
DWD_STATION = "01420"  # Frankfurt/Main

HLNUG_PARAMETERS = "15,14,44,83,18"  # O3, NO2, PM10, PM2.5, SO2
HLNUG_URL = (
    "https://app.hlnug.de/json/lmw/getStationTableData/"
    f"{HLNUG_STATION}/{HLNUG_PARAMETERS}/{{start}}/{{end}}"
    "?pad=1&utc=1&valueType=2"
)

DWD_BASE = "https://opendata.dwd.de/climate_environment/CDC/observations_germany/climate/hourly"
DWD_FILES = {
    "air_temperature": (
        f"{DWD_BASE}/air_temperature/historical/"
        "stundenwerte_TU_01420_19810101_20251231_hist.zip"
    ),
    "wind": (
        f"{DWD_BASE}/wind/historical/"
        "stundenwerte_FF_01420_19670101_20251231_hist.zip"
    ),
    "sun": (
        f"{DWD_BASE}/sun/historical/"
        "stundenwerte_SD_01420_19510101_20251231_hist.zip"
    ),
    "precipitation": (
        f"{DWD_BASE}/precipitation/historical/"
        "stundenwerte_RR_01420_19950901_20251231_hist.zip"
    ),
    "solar": f"{DWD_BASE}/solar/stundenwerte_ST_01420_row.zip",
}


def fetch(url: str) -> requests.Response:
    for attempt in range(5):
        response = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=180)
        if response.status_code not in {429, 500, 502, 503, 504}:
            response.raise_for_status()
            return response
        if attempt == 4:
            response.raise_for_status()
        time.sleep(2**attempt)
    raise RuntimeError("Download retry loop ended unexpectedly.")


def download(url: str, destination: Path, refresh: bool) -> dict[str, object]:
    retrieved_now = refresh or not destination.exists()
    if retrieved_now:
        destination.write_bytes(fetch(url).content)

    content = destination.read_bytes()
    return {
        "file": destination.relative_to(ROOT).as_posix(),
        "url": url,
        "bytes": len(content),
        "sha256": hashlib.sha256(content).hexdigest(),
        "retrieved_now": retrieved_now,
    }


def download_hlnug_year(year: int, destination: Path, refresh: bool) -> dict[str, object]:
    """Fetch monthly chunks because the portal limits a response to 1,100 rows."""
    retrieved_now = refresh or not destination.exists()
    request_urls: list[str] = []
    if retrieved_now:
        annual_payload: dict[str, object] = {"meta": {}, "data": {}}
        for month in range(1, 13):
            start = int(datetime(year, month, 1, tzinfo=timezone.utc).timestamp())
            last_day = calendar.monthrange(year, month)[1]
            end = int(datetime(year, month, last_day, 23, 59, 59, tzinfo=timezone.utc).timestamp())
            url = HLNUG_URL.format(start=start, end=end)
            request_urls.append(url)
            payload = fetch(url).json()
            annual_payload["meta"].update(payload.get("meta", {}))
            annual_payload["data"].update(payload.get("data", {}))
        destination.write_text(
            json.dumps(annual_payload, ensure_ascii=False, separators=(",", ":")),
            encoding="utf-8",
        )
    else:
        for month in range(1, 13):
            start = int(datetime(year, month, 1, tzinfo=timezone.utc).timestamp())
            last_day = calendar.monthrange(year, month)[1]
            end = int(datetime(year, month, last_day, 23, 59, 59, tzinfo=timezone.utc).timestamp())
            request_urls.append(HLNUG_URL.format(start=start, end=end))

    content = destination.read_bytes()
    return {
        "file": destination.relative_to(ROOT).as_posix(),
        "url": "https://app.hlnug.de/json/lmw/getStationTableData/",
        "request_urls": request_urls,
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
            download_hlnug_year(
                year,
                RAW_DIR / f"hlnug_airquality_{AIR_STATION}_{year}.json",
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
                "id": AIR_STATION,
                "portal_id": HLNUG_STATION,
                "name": "Frankfurt-Höchst",
                "type": "urban background",
                "provider": "HLNUG",
                "latitude": 50.10175,
                "longitude": 8.542517,
            },
            "weather": {
                "id": DWD_STATION,
                "name": "Frankfurt/Main",
                "latitude": 50.0259,
                "longitude": 8.5213,
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
