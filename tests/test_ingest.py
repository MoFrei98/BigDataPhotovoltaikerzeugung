import zipfile
from pathlib import Path

import pandas as pd

from pv_weather.ingest import (
    prepare_dataset,
    read_installed_capacity,
    read_smard_generation,
)


def _write_generation(directory: Path) -> None:
    rows = [
        "Datum von;Datum bis;Photovoltaik [MWh] Originalauflösungen",
        "01.07.2024 12:00;01.07.2024 12:15;1.000,0",
        "01.07.2024 12:15;01.07.2024 12:30;1.100,0",
        "01.07.2024 12:30;01.07.2024 12:45;1.200,0",
        "01.07.2024 12:45;01.07.2024 13:00;1.300,0",
    ]
    (directory / "pv_generation_2024.csv").write_text("\n".join(rows), encoding="utf-8")


def _write_weather_zip(directory: Path, station: str, temperature: float) -> None:
    header = (
        "STATIONS_ID;MESS_DATUM;TT_TU;RF_TU;FG_LBERG;FD_LBERG;"
        "SD_LBERG;ZENIT;V_N;F;eor"
    )
    row = (
        f"{station};2024070110:00;{temperature};55;280;70;"
        "48;30;2;3.0;eor"
    )
    path = directory / f"stundenwerte_TEST_{station}_row.zip"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr(f"produkt_test_{station}.txt", f"{header}\n{row}\n")


def test_smard_generation_sums_quarter_hours(tmp_path):
    _write_generation(tmp_path)
    hourly = read_smard_generation(tmp_path)
    assert hourly["pv_generation_mwh"].iloc[0] == 4_600
    assert pd.Timestamp(hourly["timestamp_utc"].iloc[0]).tz is not None


def test_capacity_and_weighted_weather_join(tmp_path):
    smard = tmp_path / "smard"
    dwd = tmp_path / "dwd"
    smard.mkdir()
    dwd.mkdir()
    _write_generation(smard)
    _write_weather_zip(dwd, "00001", 20)
    _write_weather_zip(dwd, "00002", 28)

    capacity = tmp_path / "capacity.csv"
    capacity.write_text("year,installed_pv_capacity_mw\n2024,92000\n", encoding="utf-8")
    weights = tmp_path / "weights.csv"
    weights.write_text("station_id,weight\n00001,3\n00002,1\n", encoding="utf-8")

    annual = read_installed_capacity(capacity)
    assert annual["installed_pv_capacity_mw"].iloc[0] == 92_000

    prepared = prepare_dataset(smard, capacity, dwd, weights)
    assert prepared["temperature_c"].iloc[0] == 22
    assert prepared["installed_pv_capacity_mw"].iloc[0] == 92_000
    assert prepared["global_radiation_j_cm2"].iloc[0] == 280
