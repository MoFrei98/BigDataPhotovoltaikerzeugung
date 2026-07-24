"""Download official SMARD and DWD data for the PV weather model."""

from __future__ import annotations

import csv
import html
import json
import math
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urljoin
from urllib.request import Request, urlopen
from zoneinfo import ZoneInfo


SMARD_BASE_URL = "https://www.smard.de"
SMARD_PV_FILTER = 4068
SMARD_PV_CAPACITY_DATA_ID = 188
DWD_BASE_URL = (
    "https://opendata.dwd.de/climate_environment/CDC/"
    "observations_germany/climate/hourly"
)
USER_AGENT = "pv-weather-germany/0.1 (academic data download)"

DWD_DIRECTORIES = {
    "solar": f"{DWD_BASE_URL}/solar/",
    "air_temperature": f"{DWD_BASE_URL}/air_temperature/historical/",
    "cloudiness": f"{DWD_BASE_URL}/cloudiness/historical/",
    "wind": f"{DWD_BASE_URL}/wind/historical/",
}


@dataclass(frozen=True)
class DwdStation:
    station_id: str
    latitude: float
    longitude: float
    name: str
    start_date: str
    end_date: str
    elevation_m: int = 0


def _request_bytes(
    url: str,
    *,
    data: bytes | None = None,
    content_type: str | None = None,
    attempts: int = 3,
) -> bytes:
    headers = {"User-Agent": USER_AGENT}
    if content_type:
        headers["Content-Type"] = content_type
    request = Request(url, data=data, headers=headers)
    last_error: Exception | None = None
    for attempt in range(attempts):
        try:
            with urlopen(request, timeout=90) as response:
                return response.read()
        except (HTTPError, URLError, TimeoutError) as exc:
            last_error = exc
            if attempt + 1 < attempts:
                time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"Download fehlgeschlagen: {url}") from last_error


def _request_json(url: str) -> dict:
    return json.loads(_request_bytes(url).decode("utf-8"))


def _utc_milliseconds(year: int, month: int = 1, day: int = 1) -> int:
    return int(datetime(year, month, day, tzinfo=timezone.utc).timestamp() * 1_000)


def _berlin_milliseconds(year: int, month: int = 1, day: int = 1) -> int:
    return int(
        datetime(
            year, month, day, tzinfo=ZoneInfo("Europe/Berlin")
        ).timestamp()
        * 1_000
    )


def download_smard_generation(
    start_year: int,
    end_year: int,
    output_dir: str | Path,
) -> Path:
    """Download hourly realized PV generation from the public SMARD chart API."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    start_ms = _utc_milliseconds(start_year)
    end_ms = _utc_milliseconds(end_year + 1)

    index_url = (
        f"{SMARD_BASE_URL}/app/chart_data/{SMARD_PV_FILTER}/DE/index_hour.json"
    )
    timestamps = _request_json(index_url)["timestamps"]
    block_starts = [
        int(timestamp)
        for timestamp in timestamps
        if int(timestamp) < end_ms and int(timestamp) + 8 * 86_400_000 > start_ms
    ]
    if not block_starts:
        raise ValueError(
            f"SMARD bietet keine PV-Stundenwerte für {start_year}–{end_year} an."
        )

    def fetch_block(block_start: int) -> list:
        url = (
            f"{SMARD_BASE_URL}/app/chart_data/{SMARD_PV_FILTER}/DE/"
            f"{SMARD_PV_FILTER}_DE_hour_{block_start}.json"
        )
        return _request_json(url).get("series", [])

    rows: dict[int, float] = {}
    with ThreadPoolExecutor(max_workers=6) as executor:
        series_blocks = executor.map(fetch_block, block_starts)
    for series in series_blocks:
        for timestamp, value in series:
            timestamp = int(timestamp)
            if start_ms <= timestamp < end_ms and value is not None:
                rows[timestamp] = float(value)

    if not rows:
        raise ValueError("Die SMARD-Antwort enthielt keine verwendbaren PV-Werte.")

    path = output_dir / f"pv_generation_smard_{start_year}_{end_year}.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["timestamp_utc", "pv_generation_mwh"])
        for timestamp, value in sorted(rows.items()):
            iso_time = datetime.fromtimestamp(
                timestamp / 1_000, tz=timezone.utc
            ).isoformat()
            writer.writerow([iso_time, f"{value:.6f}"])
    return path


def _find_smard_capacity_module(configuration: dict) -> int:
    for category in configuration.get("main", []):
        for subcategory in category.get("sub", []):
            for modules in subcategory.get("module", {}).values():
                for module in modules:
                    if int(module.get("data_id", -1)) == SMARD_PV_CAPACITY_DATA_ID:
                        return int(module["id"])
    raise ValueError("SMARD-Modul für installierte PV-Leistung nicht gefunden.")


def _parse_smard_capacity_csv(payload: bytes) -> list[tuple[int, float]]:
    text = payload.decode("cp1252", errors="replace")
    reader = csv.DictReader(text.splitlines(), delimiter=";")
    rows: list[tuple[int, float]] = []
    for row in reader:
        if not row:
            continue
        date_value = next(
            (value for key, value in row.items() if key and "Datum von" in key),
            None,
        )
        capacity_value = next(
            (value for key, value in row.items() if key and "Photovoltaik" in key),
            None,
        )
        if not date_value or not capacity_value:
            continue
        year_match = re.search(r"(?:19|20)\d{2}", date_value)
        if not year_match:
            continue
        number = capacity_value.strip().replace(".", "").replace(",", ".")
        rows.append((int(year_match.group()), float(number)))
    return rows


def download_smard_capacity(
    start_year: int,
    end_year: int,
    output_path: str | Path,
) -> Path:
    """Download annual installed PV capacity from the SMARD download service."""
    configuration = _request_json(
        f"{SMARD_BASE_URL}/app/chart_configuration/market_data_configuration.json"
    )
    module_id = _find_smard_capacity_module(configuration)
    request_body = {
        "request_form": [
            {
                "moduleIds": [module_id],
                "region": "DE",
                "resolution": "year",
                "format": "CSV",
                "timestamp_from": _berlin_milliseconds(start_year),
                "timestamp_to": _berlin_milliseconds(end_year + 1),
                "type": "discrete",
                "language": "de",
            }
        ]
    }
    payload = _request_bytes(
        f"{SMARD_BASE_URL}/nip-download-manager/nip/download/market-data",
        data=json.dumps(request_body).encode("utf-8"),
        content_type="application/json",
    )
    rows = [
        row
        for row in _parse_smard_capacity_csv(payload)
        if start_year <= row[0] <= end_year
    ]
    missing = sorted(set(range(start_year, end_year + 1)) - {year for year, _ in rows})
    if missing:
        raise ValueError(
            "SMARD liefert keine installierte PV-Leistung für: "
            + ", ".join(map(str, missing))
        )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["year", "installed_pv_capacity_mw"])
        for year, capacity in sorted(rows):
            writer.writerow([year, f"{capacity:.3f}"])
    return output_path


def _archive_entries(
    directory_url: str,
    start_date: str,
    end_date: str,
    *,
    solar: bool = False,
) -> dict[str, str]:
    listing = _request_bytes(directory_url).decode("utf-8", errors="replace")
    links = {
        html.unescape(link)
        for link in re.findall(r'href=["\']([^"\']+\.zip)["\']', listing)
    }
    entries: dict[str, str] = {}
    for link in sorted(links):
        filename = Path(link).name
        if solar:
            match = re.fullmatch(r"stundenwerte_ST_(\d{5})_row\.zip", filename)
        else:
            match = re.fullmatch(
                r"stundenwerte_[A-Za-z]+_(\d{5})_(\d{8})_(\d{8})_hist\.zip",
                filename,
            )
        if not match:
            continue
        if not solar and not (
            match.group(2) <= start_date and match.group(3) >= end_date
        ):
            continue
        entries[match.group(1)] = urljoin(directory_url, link)
    return entries


def _parse_dwd_stations(payload: bytes) -> dict[str, DwdStation]:
    text = payload.decode("latin-1", errors="replace")
    stations: dict[str, DwdStation] = {}
    pattern = re.compile(
        r"^(\d{5})\s+(\d{8})\s+(\d{8})\s+(-?\d+)\s+"
        r"(-?\d+(?:\.\d+)?)\s+(-?\d+(?:\.\d+)?)\s+(.+?)\s{2,}",
        re.MULTILINE,
    )
    for match in pattern.finditer(text):
        (
            station_id,
            start_date,
            end_date,
            elevation,
            latitude,
            longitude,
            name,
        ) = match.groups()
        stations[station_id] = DwdStation(
            station_id=station_id,
            latitude=float(latitude),
            longitude=float(longitude),
            name=name.strip(),
            start_date=start_date,
            end_date=end_date,
            elevation_m=int(elevation),
        )
    return stations


def _distance_km(first: DwdStation, second: DwdStation) -> float:
    lat1, lat2 = math.radians(first.latitude), math.radians(second.latitude)
    delta_lat = lat2 - lat1
    delta_lon = math.radians(second.longitude - first.longitude)
    value = (
        math.sin(delta_lat / 2) ** 2
        + math.cos(lat1) * math.cos(lat2) * math.sin(delta_lon / 2) ** 2
    )
    return 6_371 * 2 * math.asin(math.sqrt(value))


def select_spread_stations(
    stations: list[DwdStation],
    count: int,
) -> list[DwdStation]:
    """Select stations greedily so the sample spans Germany geographically."""
    if count < 1:
        raise ValueError("Die Stationsanzahl muss mindestens 1 sein.")
    if len(stations) <= count:
        return sorted(stations, key=lambda station: station.station_id)

    center_latitude, center_longitude = 51.0, 10.0
    first = min(
        stations,
        key=lambda station: (
            (station.latitude - center_latitude) ** 2
            + (station.longitude - center_longitude) ** 2
        ),
    )
    selected = [first]
    remaining = {station.station_id: station for station in stations if station != first}
    while len(selected) < count:
        next_station = max(
            remaining.values(),
            key=lambda station: min(
                _distance_km(station, chosen) for chosen in selected
            ),
        )
        selected.append(next_station)
        del remaining[next_station.station_id]
    return sorted(selected, key=lambda station: station.station_id)


def discover_dwd_downloads(
    start_year: int,
    end_year: int,
    station_count: int = 16,
) -> tuple[list[DwdStation], list[tuple[str, str]]]:
    """Find geographically spread DWD stations with all required variables."""
    start_date = f"{start_year}0101"
    end_date = f"{end_year}1231"
    archives = {
        product: _archive_entries(
            url,
            start_date,
            end_date,
            solar=product == "solar",
        )
        for product, url in DWD_DIRECTORIES.items()
    }
    common_ids = set.intersection(
        *(set(product_archives) for product_archives in archives.values())
    )
    metadata = _parse_dwd_stations(
        _request_bytes(
            f"{DWD_DIRECTORIES['solar']}ST_Stundenwerte_Beschreibung_Stationen.txt"
        )
    )
    candidates = [
        metadata[station_id]
        for station_id in common_ids
        if station_id in metadata
        and metadata[station_id].start_date <= start_date
        and metadata[station_id].end_date >= end_date
        and metadata[station_id].elevation_m <= 800
    ]
    if not candidates:
        raise ValueError(
            "Keine DWD-Station deckt den gewählten Zeitraum mit allen "
            "benötigten Wettergrößen vollständig ab."
        )

    selected = select_spread_stations(candidates, min(station_count, len(candidates)))
    downloads = [
        (url, Path(url).name)
        for station in selected
        for product in DWD_DIRECTORIES
        if (url := archives[product].get(station.station_id))
    ]
    return selected, downloads


def download_dwd_archives(
    start_year: int,
    end_year: int,
    output_dir: str | Path,
    station_count: int = 16,
    workers: int = 6,
) -> tuple[list[DwdStation], list[Path]]:
    """Download all required DWD archives for the selected stations."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stations, downloads = discover_dwd_downloads(
        start_year, end_year, station_count=station_count
    )

    def fetch(item: tuple[str, str]) -> Path:
        url, filename = item
        destination = output_dir / filename
        if not destination.exists() or destination.stat().st_size == 0:
            destination.write_bytes(_request_bytes(url))
        return destination

    paths: list[Path] = []
    with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
        futures = {executor.submit(fetch, item): item for item in downloads}
        for future in as_completed(futures):
            paths.append(future.result())
    return stations, sorted(paths)
