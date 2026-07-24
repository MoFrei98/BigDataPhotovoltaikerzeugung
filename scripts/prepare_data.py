"""Build the canonical hourly PV-weather panel from SMARD and DWD downloads."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pv_weather.ingest import prepare_dataset  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smard-dir", type=Path, default=ROOT / "data" / "raw" / "smard")
    parser.add_argument(
        "--capacity",
        type=Path,
        default=ROOT / "data" / "raw" / "capacity" / "installed_pv_capacity.csv",
    )
    parser.add_argument("--dwd-dir", type=Path, default=ROOT / "data" / "raw" / "dwd")
    parser.add_argument(
        "--station-weights",
        type=Path,
        default=ROOT / "data" / "raw" / "station_weights.csv",
        help="Optional CSV with station_id and positive weight.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "data" / "processed" / "hourly_pv_weather.csv",
    )
    args = parser.parse_args()

    weights = args.station_weights if args.station_weights.exists() else None
    prepared = prepare_dataset(args.smard_dir, args.capacity, args.dwd_dir, weights)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    prepared.to_csv(args.output, index=False)

    normalized = prepared["pv_generation_mwh"] / prepared["installed_pv_capacity_mw"]
    print(f"Gespeichert: {args.output}")
    print(f"Zeilen: {len(prepared):,}")
    print(f"Zeitraum: {prepared['timestamp_utc'].min()} bis {prepared['timestamp_utc'].max()}")
    print(f"Median normierte PV-Erzeugung: {normalized.median():.1%}")
    print("Vollständigkeit (%):")
    print(prepared.notna().mean().mul(100).round(1).to_string())


if __name__ == "__main__":
    main()
