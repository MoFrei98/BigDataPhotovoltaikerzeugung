# Daten

## Rohdaten

Die Dateien in `raw/` werden unverändert von Our World in Data (OWID)
heruntergeladen. Zu jeder CSV-Datei wird die zugehörige Metadaten-JSON
gespeichert. `raw/download_manifest.json` dokumentiert URL, Abrufzeitpunkt,
Dateigröße und SHA-256-Prüfsumme.

Verwendete Reihen:

- `co-emissions-per-capita`: territoriale CO₂-Emissionen pro Kopf
- `annual-co2-emissions-per-country`: territoriale CO₂-Gesamtemissionen
- `gdp-worldbank-constant-usd`: reales Gesamt-BIP in konstanten US-Dollar

CO₂ umfasst fossile Brennstoffe und industrielle Prozesse, schließt
Landnutzungsänderungen aus und ordnet internationale Luftfahrt und Schifffahrt
keinem einzelnen Staat zu. Das BIP ist inflationsbereinigt. Für die These werden
nur Veränderungen innerhalb eines Landes betrachtet; Wechselkurseffekte und
Preisniveauvergleiche zwischen Ländern sind daher nicht Gegenstand der Analyse.

## Verarbeitete Daten

Das Notebook erzeugt reproduzierbar:

- `processed/eu27_panel.csv`
- `processed/decoupling_results.csv`
- `processed/germany_forecast_2030.csv`
- `processed/forecast_model_comparison.csv`

Die Rohdaten werden nie überschrieben.

## Quellen

- https://ourworldindata.org/grapher/co-emissions-per-capita
- https://ourworldindata.org/grapher/annual-co2-emissions-per-country
- https://ourworldindata.org/grapher/gdp-worldbank-constant-usd
- https://ourworldindata.org/co2-and-greenhouse-gas-emissions

