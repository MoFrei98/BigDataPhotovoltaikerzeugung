# PV Weather Predictor Germany

Streamlit-App und Jupyter-Notebook untersuchen, unter welchen meteorologischen
Bedingungen die auf die installierte Leistung normierte Photovoltaikerzeugung in
Deutschland am höchsten ist.

## Forschungsfrage und These

> Unter welchen meteorologischen Bedingungen ist die auf die installierte
> Leistung normierte Photovoltaikerzeugung in Deutschland am höchsten?

**These:** Die höchste normierte Photovoltaikerzeugung tritt bei hoher
Sonneneinstrahlung und moderaten Temperaturen auf. Bei sehr hohen Temperaturen
nimmt die Erzeugung trotz starker Einstrahlung durch thermische Verluste wieder
ab.

**Zielgruppe:** Energieanalystinnen und Energieanalysten sowie Fachkräfte in
Planung und Optimierung erneuerbarer Energiesysteme.

## Zielvariable

Für jede Stunde wird der Kapazitätsfaktor berechnet:

```text
normierte PV-Erzeugung =
    PV-Erzeugung [MWh] / (installierte PV-Leistung [MW] × 1 h)
```

Ein Wert von 0,50 bedeutet, dass die installierte Leistung in dieser Stunde im
Mittel zu 50 % ausgeschöpft wurde.

## App starten

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
streamlit run app.py
```

Ohne lokale Rohdaten startet die App mit einer klar markierten synthetischen
Demo. Sie testet den Workflow, liefert aber keine empirischen Ergebnisse.

Die App bietet:

- Prognose der normierten PV-Erzeugung
- Einordnung in niedrig, mittel oder hoch
- Schätzung von MW und MWh für eine wählbare installierte Leistung
- geschätzte Modultemperatur
- Temperatur-Sensitivitätskurve bei gleicher Einstrahlung
- Vergleich mit den Wetterbedingungen der besten beobachteten Stunden
- Permutation Importance der wichtigsten Einflussmerkmale

## Realdaten

Der vollständige Download von den offiziellen SMARD- und DWD-Webseiten, die
Aufbereitung und ein Test-Training lassen sich mit einem Befehl starten:

```powershell
python scripts/download_real_data.py --start-year 2022 --end-year 2024
```

Das Skript lädt die reale deutsche PV-Erzeugung und die installierte PV-Leistung
von SMARD. Aus den historischen DWD-Archiven wählt es automatisch räumlich
verteilte Stationen mit Solarstrahlung, Temperatur/Feuchte, Bewölkung und Wind.
Es erzeugt `data/processed/hourly_pv_weather.csv` und validiert anschließend das
Modell mit dem zeitlichen 80/20-Split.

Danach verwendet die App automatisch die Realdaten:

```powershell
streamlit run app.py
```

Bereits manuell heruntergeladene Dateien können weiterhin mit
`python scripts/prepare_data.py` verarbeitet werden. Weitere Details und die
optionale Gewichtung nach regionaler PV-Leistung stehen in
[`data/README.md`](data/README.md).

## Notebook

Das vollständig ausführbare Notebook wird aus derselben Kernlogik erzeugt, die
auch die App verwendet:

```powershell
python scripts/create_notebook.py --execute
jupyter lab notebooks/pv_wetter_deutschland.ipynb
```

## Methodik

- einheitliche Stundenauflösung und UTC-Zeitstempel
- Kalendermerkmale in Europe/Berlin
- deutschlandweite Mittelwerte oder optionale PV-gewichtete Stationsmittel
- jährliche Zuordnung der installierten Leistung
- geschätzte Modultemperatur über eine einfache NOCT-artige Näherung mit
  Windkorrektur
- Gradient Boosting gegen Median-Baseline
- zeitlich zusammenhängender 80/20-Test statt zufälligem Mischen
- empirisches 80%-Intervall aus Testresiduen

## Projektstruktur

```text
app.py                   Streamlit-App
pv_weather/              Daten-, Feature-, Import- und Modelllogik
scripts/prepare_data.py  SMARD-/DWD-Aufbereitung
scripts/create_notebook.py
notebooks/               ausgeführte Analyse
data/                    lokale Datenablage und Schema
tests/                   automatisierte Kernprüfungen
```

## Grenzen und Lizenz

DWD-Stationsmittel sind eine Annäherung an das deutschlandweite
PV-Flottenwetter. Eine Gewichtung nach regional installierter Leistung ist
methodisch stärker, benötigt aber belastbare regionale Kapazitätsgewichte. Die
jährliche Kapazitätszuordnung ignoriert den Ausbau innerhalb eines Jahres. Die
Modultemperatur wird nicht gemessen, sondern aus Lufttemperatur, Einstrahlung
und Wind angenähert. Die Auswertung zeigt Zusammenhänge, keine Kausalität und
keine Ertragsgarantie.

SMARD-Marktdaten stehen unter CC BY 4.0. Namensnennung:
„Bundesnetzagentur | SMARD.de“.
