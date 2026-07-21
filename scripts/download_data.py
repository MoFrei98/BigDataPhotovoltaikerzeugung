"""Download and freeze the OWID source datasets used by the project."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import requests


ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
USER_AGENT = "BigDataCo2Emissions university project/1.0"

DATASETS = {
    "co-emissions-per-capita": {
        "csv": "https://ourworldindata.org/grapher/co-emissions-per-capita.csv?v=1&csvType=full&useColumnShortNames=true",
        "metadata": "https://ourworldindata.org/grapher/co-emissions-per-capita.metadata.json?v=1&csvType=full&useColumnShortNames=true",
    },
    "annual-co2-emissions-per-country": {
        "csv": "https://ourworldindata.org/grapher/annual-co2-emissions-per-country.csv?v=1&csvType=full&useColumnShortNames=true",
        "metadata": "https://ourworldindata.org/grapher/annual-co2-emissions-per-country.metadata.json?v=1&csvType=full&useColumnShortNames=true",
    },
    "gdp-worldbank-constant-usd": {
        "csv": "https://ourworldindata.org/grapher/gdp-worldbank-constant-usd.csv?v=1&csvType=full&useColumnShortNames=true",
        "metadata": "https://ourworldindata.org/grapher/gdp-worldbank-constant-usd.metadata.json?v=1&csvType=full&useColumnShortNames=true",
    },
}


def download(url: str, destination: Path, refresh: bool) -> dict[str, object]:
    retrieved_now = refresh or not destination.exists()
    if retrieved_now:
        response = requests.get(
            url,
            headers={"User-Agent": USER_AGENT},
            timeout=120,
        )
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
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="Download again even when a local copy exists.",
    )
    args = parser.parse_args()

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, object] = {
        "manifest_created_at_utc": datetime.now(timezone.utc).isoformat(),
        "datasets": {},
    }

    for slug, urls in DATASETS.items():
        entries = []
        entries.append(download(urls["csv"], RAW_DIR / f"{slug}.csv", args.refresh))
        entries.append(
            download(
                urls["metadata"],
                RAW_DIR / f"{slug}.metadata.json",
                args.refresh,
            )
        )
        manifest["datasets"][slug] = entries

    manifest_path = RAW_DIR / "download_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    downloaded_files = sum(
        entry["retrieved_now"]
        for entries in manifest["datasets"].values()
        for entry in entries
    )
    print(
        f"Prepared {len(DATASETS)} datasets in {RAW_DIR} "
        f"({downloaded_files} of {len(DATASETS) * 2} files downloaded now)"
    )
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
