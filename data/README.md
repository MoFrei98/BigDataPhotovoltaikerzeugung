# Daten

## Rohdaten

`raw/` enthält unveränderte amtliche Dateien für 2016–2025:

- zehn jährliche CSV-Dateien der UBA-Luftdaten-API v4 für die Station
  **DEBB021 Potsdam-Zentrum**;
- DWD-ZIP-Dateien der Station **03987 Potsdam** für stündliche Temperatur und
  Feuchte, Wind, Niederschlag, Sonnenscheindauer und Solarstrahlung.

`raw/download_manifest.json` hält Quell-URL, Abrufzeit, Dateigröße und
SHA-256-Prüfsumme fest. Die Rohdateien werden im Notebook nicht verändert.

## Zusammenführung

Die Luftqualitätsstation liegt rund 2,3 km von der Wetterstation entfernt. Nach
dieser räumlichen Zuordnung werden die Reihen stündlich verbunden. UBA-CSV-
Zeitstempel liegen in MEZ/MESZ vor und werden nach UTC konvertiert; DWD-
Zeitstempel liegen in UTC vor. Für die Analyse folgt eine Aggregation auf
lokale Kalendertage.

## Verarbeitete Daten

- `processed/potsdam_daily_air_weather.csv`: tägliches Analysepanel
- `processed/model_comparison.csv`: Testmetriken 2024–2025
- `processed/feature_importance.csv`: Permutationswichtigkeit
- `processed/analysis_summary.json`: zentrale reproduzierte Ergebnisse

## Quellen und Lizenzen

- Umweltbundesamt mit Daten der Messnetze der Länder und des Bundes:
  https://luftdaten.umweltbundesamt.de/
- Deutscher Wetterdienst, Climate Data Center (CC BY 4.0):
  https://opendata.dwd.de/climate_environment/CDC/observations_germany/climate/hourly/
- UBA-Luftqualitätsindex:
  https://www.umweltbundesamt.de/themen/luft/luftqualitaet/der-luftqualitaetsindex-lqi

Hinweis: Daten des laufenden Jahres können beim UBA vorläufig sein. Der lokale
Stand ist über das Manifest nachvollziehbar.
