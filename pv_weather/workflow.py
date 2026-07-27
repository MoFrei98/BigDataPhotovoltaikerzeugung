"""End-to-end workflow for refreshing the app with official real data."""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path

from .download import (
    DWD_BASE_URL,
    SMARD_BASE_URL,
    download_dwd_archives,
    download_smard_capacity,
    download_smard_generation,
)
from .ingest import prepare_dataset
from .modeling import YieldModelBundle, train_yield_model


MIN_DOWNLOAD_YEAR = 2015


@dataclass(frozen=True)
class RealDataUpdate:
    """Result of a completed download, preparation and training run."""

    start_year: int
    end_year: int
    row_count: int
    station_count: int
    processed_path: Path
    processed_version: int
    model: YieldModelBundle


def _notify(callback: Callable[[str], None] | None, message: str) -> None:
    if callback is not None:
        callback(message)


def refresh_real_data(
    project_root: str | Path,
    start_year: int,
    end_year: int,
    *,
    station_count: int = 16,
    workers: int = 6,
    on_progress: Callable[[str], None] | None = None,
) -> RealDataUpdate:
    """Download official data, build the panel and validate it by training."""
    latest_complete_year = date.today().year - 1
    if start_year < MIN_DOWNLOAD_YEAR:
        raise ValueError(f"Das früheste auswählbare Jahr ist {MIN_DOWNLOAD_YEAR}.")
    if end_year > latest_complete_year:
        raise ValueError(
            f"Das späteste auswählbare Jahr ist {latest_complete_year}."
        )
    if start_year > end_year:
        raise ValueError("Das Startjahr darf nicht nach dem Endjahr liegen.")
    if station_count < 1 or workers < 1:
        raise ValueError("Stations- und Worker-Anzahl müssen mindestens 1 sein.")

    root = Path(project_root)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    raw_run_dir = (
        root
        / "data"
        / "raw"
        / "app_downloads"
        / f"{start_year}_{end_year}"
        / run_id
    )
    smard_dir = raw_run_dir / "smard"
    capacity_path = raw_run_dir / "capacity" / "installed_pv_capacity.csv"
    dwd_dir = raw_run_dir / "dwd"

    _notify(
        on_progress,
        f"1/5 SMARD-PV-Erzeugung für {start_year}–{end_year} wird geladen …",
    )
    generation_path = download_smard_generation(start_year, end_year, smard_dir)

    _notify(on_progress, "2/5 Installierte PV-Leistung wird von SMARD geladen …")
    download_smard_capacity(start_year, end_year, capacity_path)

    _notify(
        on_progress,
        f"3/5 DWD-Wetterdaten für {station_count} räumlich verteilte "
        "Stationen werden geladen …",
    )
    stations, archives = download_dwd_archives(
        start_year,
        end_year,
        dwd_dir,
        station_count=station_count,
        workers=workers,
    )

    _notify(on_progress, "4/5 Das gemeinsame stündliche Datenpanel wird erzeugt …")
    prepared = prepare_dataset(smard_dir, capacity_path, dwd_dir)

    _notify(
        on_progress,
        "5/5 Das Prognosemodell wird mit PV-relevanten Tageslichtstunden und "
        "einem zeitlichen Test neu trainiert …",
    )
    model = train_yield_model(prepared)

    output_path = root / "data" / "processed" / "hourly_pv_weather.csv"
    manifest = {
        "downloaded_at_utc": datetime.now(timezone.utc).isoformat(),
        "start_year": start_year,
        "end_year": end_year,
        "sources": {"smard": SMARD_BASE_URL, "dwd": DWD_BASE_URL},
        "smard_generation_file": str(generation_path.relative_to(root)),
        "capacity_file": str(capacity_path.relative_to(root)),
        "dwd_station_ids": [station.station_id for station in stations],
        "dwd_archives": [str(path.relative_to(root)) for path in archives],
        "processed_file": str(output_path.relative_to(root)),
        "processed_rows": len(prepared),
        "model_metrics": model.metrics,
    }
    manifest_path = root / "data" / "raw" / "download_manifest.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    temporary_output = output_path.with_name(f".{output_path.name}.{run_id}.tmp")
    temporary_manifest = manifest_path.with_name(f".{manifest_path.name}.{run_id}.tmp")
    output_backup = output_path.with_name(f".{output_path.name}.{run_id}.bak")
    manifest_backup = manifest_path.with_name(f".{manifest_path.name}.{run_id}.bak")
    try:
        prepared.to_csv(temporary_output, index=False)
        temporary_manifest.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        if output_path.exists():
            output_path.replace(output_backup)
        if manifest_path.exists():
            manifest_path.replace(manifest_backup)

        temporary_output.replace(output_path)
        temporary_manifest.replace(manifest_path)
    except Exception:
        output_path.unlink(missing_ok=True)
        manifest_path.unlink(missing_ok=True)
        if output_backup.exists():
            output_backup.replace(output_path)
        if manifest_backup.exists():
            manifest_backup.replace(manifest_path)
        raise
    else:
        output_backup.unlink(missing_ok=True)
        manifest_backup.unlink(missing_ok=True)
    finally:
        temporary_output.unlink(missing_ok=True)
        temporary_manifest.unlink(missing_ok=True)

    return RealDataUpdate(
        start_year=start_year,
        end_year=end_year,
        row_count=len(prepared),
        station_count=len(stations),
        processed_path=output_path,
        processed_version=output_path.stat().st_mtime_ns,
        model=model,
    )
