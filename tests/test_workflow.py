from datetime import date

import pandas as pd
import pytest

from pv_weather.data import generate_demo_data
from pv_weather.workflow import MIN_DOWNLOAD_YEAR, refresh_real_data


def test_refresh_real_data_rejects_invalid_years(tmp_path):
    with pytest.raises(ValueError, match=str(MIN_DOWNLOAD_YEAR)):
        refresh_real_data(tmp_path, MIN_DOWNLOAD_YEAR - 1, MIN_DOWNLOAD_YEAR)

    with pytest.raises(ValueError, match="Startjahr"):
        refresh_real_data(tmp_path, 2020, 2019)

    with pytest.raises(ValueError, match=str(date.today().year - 1)):
        refresh_real_data(tmp_path, 2020, date.today().year)


def test_refresh_real_data_replaces_panel_only_after_training(tmp_path, monkeypatch):
    prepared = generate_demo_data("2024-01-01", "2024-02-15", seed=7)
    calls: list[str] = []

    def fake_generation(start_year, end_year, output_dir):
        path = output_dir / "generation.csv"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("generation", encoding="utf-8")
        return path

    def fake_capacity(start_year, end_year, output_path):
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text("capacity", encoding="utf-8")
        return output_path

    class Station:
        station_id = "00001"

    def fake_weather(start_year, end_year, output_dir, **kwargs):
        path = output_dir / "weather.zip"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"weather")
        return [Station()], [path]

    class Model:
        metrics = {"model_mae": 0.01}

    monkeypatch.setattr("pv_weather.workflow.download_smard_generation", fake_generation)
    monkeypatch.setattr("pv_weather.workflow.download_smard_capacity", fake_capacity)
    monkeypatch.setattr("pv_weather.workflow.download_dwd_archives", fake_weather)
    monkeypatch.setattr(
        "pv_weather.workflow.prepare_dataset",
        lambda *args, **kwargs: prepared,
    )
    monkeypatch.setattr(
        "pv_weather.workflow.train_yield_model",
        lambda frame: Model(),
    )

    result = refresh_real_data(
        tmp_path,
        2020,
        2021,
        station_count=1,
        on_progress=calls.append,
    )

    stored = pd.read_csv(result.processed_path)
    assert len(stored) == len(prepared)
    assert result.row_count == len(prepared)
    assert result.station_count == 1
    assert len(calls) == 5
    assert (tmp_path / "data" / "raw" / "download_manifest.json").exists()


def test_refresh_real_data_keeps_existing_panel_when_training_fails(
    tmp_path,
    monkeypatch,
):
    existing_path = tmp_path / "data" / "processed" / "hourly_pv_weather.csv"
    existing_path.parent.mkdir(parents=True)
    existing_path.write_text("existing panel", encoding="utf-8")
    prepared = generate_demo_data("2024-01-01", "2024-02-15", seed=8)

    def fake_generation(start_year, end_year, output_dir):
        path = output_dir / "generation.csv"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("generation", encoding="utf-8")
        return path

    def fake_capacity(start_year, end_year, output_path):
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text("capacity", encoding="utf-8")
        return output_path

    monkeypatch.setattr("pv_weather.workflow.download_smard_generation", fake_generation)
    monkeypatch.setattr("pv_weather.workflow.download_smard_capacity", fake_capacity)
    monkeypatch.setattr(
        "pv_weather.workflow.download_dwd_archives",
        lambda *args, **kwargs: ([], []),
    )
    monkeypatch.setattr(
        "pv_weather.workflow.prepare_dataset",
        lambda *args, **kwargs: prepared,
    )
    monkeypatch.setattr(
        "pv_weather.workflow.train_yield_model",
        lambda frame: (_ for _ in ()).throw(ValueError("training failed")),
    )

    with pytest.raises(ValueError, match="training failed"):
        refresh_real_data(tmp_path, 2020, 2021)

    assert existing_path.read_text(encoding="utf-8") == "existing panel"
