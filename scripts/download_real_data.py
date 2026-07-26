"""Download real SMARD/DWD data, prepare the panel and validate model training."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pv_weather.download import (  # noqa: E402
    DWD_BASE_URL,
    SMARD_BASE_URL,
    download_dwd_archives,
    download_smard_capacity,
    download_smard_generation,
)
from pv_weather.ingest import prepare_dataset  # noqa: E402
from pv_weather.modeling import train_yield_model  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Offizielle SMARD- und DWD-Daten laden, aufbereiten und "
            "das PV-Modell testweise trainieren."
        )
    )
    parser.add_argument("--start-year", type=int, default=2022)
    parser.add_argument("--end-year", type=int, default=2024)
    parser.add_argument(
        "--stations",
        type=int,
        default=16,
        help="Anzahl räumlich verteilter DWD-Stationen (Standard: 16).",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=6,
        help="Parallele DWD-Downloads (Standard: 6).",
    )
    parser.add_argument(
        "--skip-training",
        action="store_true",
        help="Nur herunterladen und das Stundenpanel erzeugen.",
    )
    args = parser.parse_args()

    if args.start_year > args.end_year:
        parser.error("--start-year darf nicht nach --end-year liegen.")
    if args.stations < 1:
        parser.error("--stations muss mindestens 1 sein.")

    smard_dir = ROOT / "data" / "raw" / "smard"
    capacity_path = (
        ROOT / "data" / "raw" / "capacity" / "installed_pv_capacity.csv"
    )
    dwd_dir = ROOT / "data" / "raw" / "dwd"
    output_path = ROOT / "data" / "processed" / "hourly_pv_weather.csv"

    print(f"1/4 SMARD-PV-Erzeugung {args.start_year}–{args.end_year} wird geladen …")
    generation_path = download_smard_generation(
        args.start_year, args.end_year, smard_dir
    )
    print(f"    {generation_path}")

    print("2/4 Installierte PV-Leistung von SMARD wird geladen …")
    download_smard_capacity(args.start_year, args.end_year, capacity_path)
    print(f"    {capacity_path}")

    print(
        f"3/4 DWD-Wetterdaten für {args.stations} räumlich verteilte "
        "Stationen werden geladen …"
    )
    stations, archives = download_dwd_archives(
        args.start_year,
        args.end_year,
        dwd_dir,
        station_count=args.stations,
        workers=args.workers,
    )
    print("    Stationen: " + ", ".join(
        f"{station.station_id} ({station.name})" for station in stations
    ))
    print(f"    {len(archives)} ZIP-Archive in {dwd_dir}")

    print("4/4 Gemeinsames Stundenpanel wird erzeugt …")
    prepared = prepare_dataset(smard_dir, capacity_path, dwd_dir)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    prepared.to_csv(output_path, index=False)
    print(f"    {len(prepared):,} Zeilen in {output_path}")

    manifest = {
        "downloaded_at_utc": datetime.now(timezone.utc).isoformat(),
        "start_year": args.start_year,
        "end_year": args.end_year,
        "sources": {
            "smard": SMARD_BASE_URL,
            "dwd": DWD_BASE_URL,
        },
        "smard_generation_file": str(generation_path.relative_to(ROOT)),
        "capacity_file": str(capacity_path.relative_to(ROOT)),
        "dwd_station_ids": [station.station_id for station in stations],
        "dwd_archives": [path.name for path in archives],
        "processed_file": str(output_path.relative_to(ROOT)),
    }
    manifest_path = ROOT / "data" / "raw" / "download_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"    Herkunftsprotokoll: {manifest_path}")

    if not args.skip_training:
        print(
            "Modell wird mit PV-relevanten Tageslichtstunden und zeitlichem "
            "80/20-Split testweise trainiert …"
        )
        bundle = train_yield_model(prepared)
        metrics = bundle.metrics
        print(
            "    "
            f"Training: {int(metrics['train_rows']):,} Stunden | "
            f"Test: {int(metrics['test_rows']):,} Stunden"
        )
        print(
            "    "
            f"MAE: {metrics['model_mae']:.4f} | "
            f"RMSE: {metrics['model_rmse']:.4f} | "
            f"R²: {metrics['model_r2']:.4f}"
        )
        print(
            "Fertig. `streamlit run app.py` verwendet ab jetzt automatisch "
            "die Realdaten und trainiert das Modell beim Start."
        )
    else:
        print(
            "Fertig. `streamlit run app.py` trainiert das Modell beim Start "
            "automatisch mit den Realdaten."
        )


if __name__ == "__main__":
    main()
