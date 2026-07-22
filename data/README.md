# Daten

## Rohdaten

`raw/` enthält lokal eingefrorene amtliche Daten für 2016–2025:

- zehn jährliche JSON-Dateien des HLNUG-Messdatenportals für die Station
  **DEHE005 Frankfurt-Höchst**; wegen der Portalgrenze von 1.100 Zeilen werden
  zwölf Monatsabfragen ohne inhaltliche Veränderung zu einer Jahresdatei vereinigt;
- unveränderte DWD-ZIP-Dateien der Station **01420 Frankfurt/Main** für stündliche Temperatur und
  Feuchte, Wind, Niederschlag, Sonnenscheindauer und Solarstrahlung.

`raw/download_manifest.json` hält Quell-URL, Abrufzeit, Dateigröße und
SHA-256-Prüfsumme fest. Die Rohdateien werden im Notebook nicht verändert.

## Zusammenführung

Die Luftqualitätsstation liegt rund 8,6 km von der Wetterstation entfernt. Nach
dieser räumlichen Zuordnung werden die Reihen stündlich in UTC verbunden. Für
die Analyse folgt eine Aggregation auf lokale Kalendertage in Europe/Berlin.

## Verarbeitete Daten

- `processed/frankfurt_daily_air_weather.csv`: tägliches Analysepanel
- `processed/model_comparison.csv`: Testmetriken 2024–2025
- `processed/feature_importance.csv`: Permutationswichtigkeit
- `processed/analysis_summary.json`: zentrale reproduzierte Ergebnisse

## Quellen und Lizenzen

- HLNUG-Messdatenportal, Station Frankfurt-Höchst:
  https://www.hlnug.de/messwerte/datenportal/messstelle/2/1/0617/6/1/1748048400
- Deutscher Wetterdienst, Climate Data Center (CC BY 4.0):
  https://opendata.dwd.de/climate_environment/CDC/observations_germany/climate/hourly/
- UBA-Luftqualitätsindex:
  https://www.umweltbundesamt.de/themen/luft/luftqualitaet/der-luftqualitaetsindex-lqi

Der lokale Datenstand ist über das Manifest nachvollziehbar.
