# Entkopplung von Wirtschaftswachstum und CO₂-Emissionen in Europa

Dieses Projekt untersucht, ob seit 2005 in der Mehrheit der heutigen EU-27-
Staaten eine **absolute Entkopplung** stattgefunden hat: reales Gesamt-BIP steigt,
während territoriale CO₂-Gesamtemissionen sinken. Anschließend werden die
deutschen CO₂-Emissionen bis 2030 mit transparenten Zeitreihen-Baselines
prognostiziert.

## Forschungsfrage und These

> Für mehr als 50 % der EU-27-Staaten mit vollständigen Daten gilt zwischen
> 2005 und dem letzten gemeinsamen verfügbaren Jahr: Das reale Gesamt-BIP ist
> gestiegen und die territorialen CO₂-Emissionen sind gesunken.

Die primäre Entscheidung basiert auf dem Vergleich der Endpunkte. Eine
log-lineare Trendanalyse über den gesamten Zeitraum dient als Robustheitscheck.

## Projektstruktur

```text
notebooks/     ausgeführter Jupyter-Bericht
data/raw/      unveränderte OWID-Daten und Metadaten
data/processed/ abgeleitete Tabellen
figures/       exportierte Abbildungen
presentation/  kurze Ergebnispräsentation
scripts/       reproduzierbarer Download und Artefakt-Erstellung
archive/       unverändertes altes Iris-/ML-Projekt
```

## Installation und Ausführung

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python scripts/download_data.py
jupyter lab
```

Danach `notebooks/co2_entkopplung_und_prognose.ipynb` öffnen und alle Zellen
von oben nach unten ausführen. Die Rohdaten werden nur heruntergeladen, wenn
lokale Kopien fehlen. Mit `python scripts/download_data.py --refresh` kann ein
neuer Datenstand bewusst eingefroren werden.

## Definitionen

- **Territorial:** Emissionen werden dem Land zugerechnet, in dem sie entstehen.
- **CO₂:** fossile Brennstoffe und industrielle Prozesse, ohne Landnutzungsänderungen.
- **Reales BIP:** inflationsbereinigtes Gesamt-BIP, nicht BIP pro Kopf.
- **Absolute Entkopplung:** BIP-Veränderung > 0 % und CO₂-Veränderung < 0 %.

## Ergebnisse

Die belastbaren Kennzahlen stehen im ausgeführten Notebook und in der
Präsentation. Datenstand, gemeinsames Endjahr, Modellgüte und Unsicherheiten
werden dort automatisch aus den eingefrorenen Quelldateien berechnet.

## Datenquellen

- Global Carbon Budget (2025), aufbereitet von Our World in Data
- World Development Indicators, aufbereitet von Our World in Data
- Vollständige URLs, Metadaten und Prüfsummen: `data/raw/download_manifest.json`

## KI-Nutzung

Die Verwendung generativer KI wird transparent in `KI_NUTZUNG.md` dokumentiert.
Alle Definitionen, Datenquellen, Berechnungen, Visualisierungen und Aussagen
müssen vor der Abgabe fachlich durch den Verfasser geprüft werden.
