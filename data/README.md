# Datenablage

Große Rohdaten werden nicht versioniert. Ohne lokale Dateien starten App und
Notebook mit eindeutig markierten synthetischen Demodaten.

## 1. Realisierte Photovoltaikerzeugung aus SMARD

Im [SMARD-Downloadcenter](https://www.smard.de/home/downloadcenter/download-marktdaten/)
auswählen:

- Kategorie: Stromerzeugung
- Datensatz: Realisierte Erzeugung – Photovoltaik
- Region: Deutschland
- Auflösung: Stunde, alternativ Originalauflösung
- Format: CSV

Exporte nach `data/raw/smard/` kopieren. Kombinierte Dateien werden über eine
Spalte mit „Photovoltaik“ erkannt. Bei einer einzelnen generischen Wertespalte
muss der Dateiname mit `pv_generation_` beginnen. Viertelstunden-MWh werden zu
Stundenwerten summiert.

SMARD-Daten stehen unter CC BY 4.0. Namensnennung:
„Bundesnetzagentur | SMARD.de“.

## 2. Installierte PV-Leistung

Die [installierte Erzeugungsleistung](https://www.smard.de/page/en/wiki-article/5884/6038/installed-generation-capacity)
wird jährlich in MW benötigt. Datei:

`data/raw/capacity/installed_pv_capacity.csv`

Minimalschema:

```csv
year,installed_pv_capacity_mw
2022,59300
2023,67600
2024,81800
```

Auch deutsche Spaltennamen mit Jahr, Photovoltaik und MW werden erkannt. Der
Jahreswert wird allen Stunden desselben Jahres zugeordnet. Diese Näherung bildet
den unterjährigen Ausbau nicht exakt ab und muss in der Interpretation genannt
werden.

## 3. DWD-Wetterdaten

ZIP-Dateien mehrerer, deutschlandweit verteilter Stationen nach
`data/raw/dwd/` kopieren:

- [Solarstrahlung](https://opendata.dwd.de/climate_environment/CDC/observations_germany/climate/hourly/solar/)
- [Temperatur und Feuchte](https://opendata.dwd.de/climate_environment/CDC/observations_germany/climate/hourly/air_temperature/historical/)
- [Bewölkung](https://opendata.dwd.de/climate_environment/CDC/observations_germany/climate/hourly/cloudiness/historical/)
- [Wind](https://opendata.dwd.de/climate_environment/CDC/observations_germany/climate/hourly/wind/historical/)

Unterstützte DWD-Felder sind unter anderem `TT_TU`, `RF_TU`, `FG_LBERG`,
`FD_LBERG`, `SD_LBERG`, `ZENIT`, `V_N` und `F`. Strahlung bleibt als
Stundensumme in J/cm², Sonnenscheindauer wird in Minuten und Bewölkung in
Achteln (0–8) gespeichert.

Mindestens eine geeignete Station je Bundesland ist anzustreben. Ohne
Gewichtedatei wird je Stunde das arithmetische Mittel aller verfügbaren
Stationen gebildet.

Optional kann `data/raw/station_weights.csv` hinterlegt werden:

```csv
station_id,weight
00433,0.08
01048,0.05
...
```

Die Gewichte können den regionalen Anteilen der installierten PV-Leistung
entsprechen. Je Stunde werden sie über die tatsächlich verfügbaren Stationen
neu normiert.

## 4. Panel erzeugen

```powershell
python scripts/prepare_data.py
```

Ergebnis: `data/processed/hourly_pv_weather.csv`

```text
timestamp_utc
temperature_c
relative_humidity_pct
global_radiation_j_cm2
diffuse_radiation_j_cm2
sunshine_duration_min
solar_zenith_angle_deg
cloud_cover_oktas
wind_speed_m_s
installed_pv_capacity_mw
pv_generation_mwh
```

Die Zielvariable wird reproduzierbar berechnet als:

```text
normalized_pv_generation =
    pv_generation_mwh / (installed_pv_capacity_mw × 1 h)
```

Zeitstempel werden in UTC gespeichert; Stunde und Monat werden für das Modell
in deutscher Ortszeit erzeugt.
