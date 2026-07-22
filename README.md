# Hitze und städtische Luftqualität

Dieses Projekt untersucht für Frankfurt am Main (2016–2025), ob hohe Temperaturen die
Luftqualität insgesamt verschlechtern oder vor allem den dominierenden
Luftschadstoff verändern. Stündliche Messwerte von HLNUG und Deutschem
Wetterdienst werden über Zeit und räumlich benachbarte Stationen
verbunden.

## Forschungsfrage und These

> Verschlechtert Hitze die Luftqualität insgesamt, oder verändert sie vor allem
> die Zusammensetzung der Luftschadstoffe?

**These:** Hohe Temperaturen erhöhen insbesondere die Ozonbelastung, während
die Stickstoffdioxidbelastung gleichzeitig sinken kann. Dadurch verändert sich
an heißen Tagen der dominierende Luftschadstoff.

Die Auswertung bestätigt den starken Ozonanstieg. Bei NO₂ zeigt Frankfurt-Höchst
jedoch ebenfalls höhere Tagesmaxima an heißen Tagen. Die These wird deshalb nur
teilweise unterstützt.

## Prognoseziel

Ein Mehrklassenmodell prognostiziert, welcher UBA-LQI-Schadstoff am Folgetag
dominiert: O₃, NO₂, PM₁₀, PM₂,₅ oder SO₂. Das Notebook vergleicht Baseline,
logistische Regression und Random Forest auf einem zeitlich getrennten Test
(2024–2025). Eine interaktive Slider-Oberfläche erlaubt Wetter- und
Belastungsszenarien.

## Daten

- HLNUG-Messdatenportal: Station DEHE005 Frankfurt-Höchst
- DWD Climate Data Center: Station 01420 Frankfurt/Main
- Zeitraum: 2016–2025
- Wettermerkmale: Temperatur, Feuchte, Wind, Niederschlag, Sonnenscheindauer,
  Global- und Diffusstrahlung
- Verbindungslogik: HLNUG- und DWD-Zeit in UTC; räumliche Stationsdistanz
  rund 8,6 km

## Projektstruktur

```text
notebooks/      ausgeführtes Jupyter-Notebook mit Slider-Prognose
data/raw/       jährliche HLNUG-JSON- und unveränderte DWD-ZIP-Dateien plus Manifest
data/processed/ Tagespanel, Modellvergleich, Merkmalswichtigkeit, Ergebnis-JSON
figures/        reproduzierbare Abbildungen
models/         gespeichertes Prognosemodell
presentation/   kurze Ergebnispräsentation
scripts/        Download- und Notebook-Erzeugung
archive/        frühere Projektstände
```

## Reproduzieren

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python scripts/download_data.py
python scripts/create_notebook.py
jupyter lab notebooks/hitze_luftqualitaet_und_prognose.ipynb
```

Danach alle Zellen von oben nach unten ausführen. Mit
`python scripts/download_data.py --refresh` werden die amtlichen Rohdaten
bewusst neu eingefroren. Das Download-Manifest dokumentiert URLs, Abrufzeit,
Dateigröße und SHA-256-Prüfsumme.

## Methodische Grenzen

Die Fallstudie umfasst einen urbanen Hintergrundstandort und zeigt
Zusammenhänge, keine Kausalität. Verkehr, Ferntransport und Ozonvorläufer werden
nicht direkt modelliert. Historisch gemessenes Folgetagswetter ersetzt beim
Training eine reale Wetterprognose; die ausgewiesene Modellgüte ist deshalb für
echte Prognosefehler eher optimistisch. Das Modell ist keine amtliche Warnung.

## KI-Nutzung

Der Einsatz generativer KI und die menschlichen Kontrollschritte sind in
[`KI_NUTZUNG.md`](KI_NUTZUNG.md) dokumentiert.
