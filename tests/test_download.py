from pv_weather.download import (
    DwdStation,
    _find_smard_capacity_module,
    _parse_smard_capacity_csv,
    select_spread_stations,
)


def test_finds_smard_pv_capacity_module():
    configuration = {
        "main": [
            {
                "sub": [
                    {
                        "module": {
                            "default": [
                                {"id": 1004068, "data_id": 4068},
                                {"id": 3000188, "data_id": 188},
                            ]
                        }
                    }
                ]
            }
        ]
    }
    assert _find_smard_capacity_module(configuration) == 3000188


def test_parses_german_smard_capacity_csv():
    payload = (
        "Datum von;Datum bis;Photovoltaik [MW] Originalauflösungen\r\n"
        "01.01.2022;01.01.2023;57.744,00\r\n"
        "01.01.2023;01.01.2024;63.066,00\r\n"
    ).encode("cp1252")
    assert _parse_smard_capacity_csv(payload) == [
        (2022, 57_744.0),
        (2023, 63_066.0),
    ]


def test_station_selection_spreads_locations():
    stations = [
        DwdStation("00001", 51.0, 10.0, "Mitte", "20000101", "20241231"),
        DwdStation("00002", 55.0, 8.0, "Nordwest", "20000101", "20241231"),
        DwdStation("00003", 48.0, 13.0, "Südost", "20000101", "20241231"),
        DwdStation("00004", 51.1, 10.1, "Mitte 2", "20000101", "20241231"),
    ]
    selected = select_spread_stations(stations, 3)
    assert {station.station_id for station in selected} == {
        "00001",
        "00002",
        "00003",
    }
