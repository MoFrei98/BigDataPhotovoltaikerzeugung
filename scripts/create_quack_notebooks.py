"""Create one executable project notebook for each QUA³CK phase."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_DIR = ROOT / "notebooks"

NOTEBOOK_FILES = {
    "Q": "01_Q_Fragestellung.ipynb",
    "U": "02_U_Datenverstaendnis.ipynb",
    "A3": "03_A3_Algorithmen_Features_Hyperparameter.ipynb",
    "C": "04_C_Schlussfolgern_und_Vergleichen.ipynb",
    "K": "05_K_Wissenstransfer.ipynb",
}


def markdown(source: str) -> dict[str, Any]:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": source.strip() + "\n",
    }


def code(source: str) -> dict[str, Any]:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source.strip() + "\n",
    }


def notebook(cells: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {
                "name": "python",
                "version": "3",
            },
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


SETUP_CELL = r"""
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from IPython.display import display

ROOT = Path.cwd().resolve()
if ROOT.name == "notebooks":
    ROOT = ROOT.parent
sys.path.insert(0, str(ROOT))

pd.set_option("display.max_columns", 30)
"""


def build_q_notebook() -> dict[str, Any]:
    return notebook(
        [
            markdown(
                r"""
# Q - Question (Fragestellung)

## QUA³CK-Phase

Die Q-Phase übersetzt ein fachliches Problem in eine präzise
Data-Science-Fragestellung. Nach der bereitgestellten QUA³CK-Unterlage werden
hier insbesondere **Problem, Zielgruppe, Erfolgsmetriken und Deployment-Ziel**
festgelegt.

## Umsetzung im Projekt

**Projekt:** Analyse und Prognose der normierten Photovoltaikerzeugung in
Deutschland anhand meteorologischer Bedingungen.

**Forschungsfrage:** Unter welchen meteorologischen Bedingungen ist die auf die
installierte Leistung normierte Photovoltaikerzeugung in Deutschland am
höchsten?

**These:** Die höchste normierte PV-Erzeugung tritt bei hoher Sonneneinstrahlung
und moderaten Temperaturen auf. Bei sehr hohen Temperaturen nimmt die Erzeugung
trotz starker Einstrahlung durch thermische Verluste wieder ab.

**Zielgruppe:** Energieanalystinnen und Energieanalysten sowie Fachkräfte in der
Planung und Optimierung erneuerbarer Energiesysteme.
"""
            ),
            markdown(
                r"""
## Problemabgrenzung und Zielvariable

Absolute PV-Erzeugung steigt mit dem Ausbau der installierten Leistung. Für
einen fairen Vergleich zwischen Jahren wird daher der stündliche
Kapazitätsfaktor verwendet:

\[
\text{normierte PV-Erzeugung} =
\frac{\text{PV-Erzeugung [MWh]}}
     {\text{installierte PV-Leistung [MW]} \cdot 1\text{ h}}
\]

Ein Wert von 0,50 bedeutet, dass die installierte Leistung in dieser Stunde im
Mittel zu 50 % ausgeschöpft wurde.

### Projektumfang

- Deutschlandweite, stündliche Betrachtung
- Wetterdaten aus DWD-Stationen und PV-Daten von SMARD
- deskriptive Analyse und überwachte Regression
- interaktive Bereitstellung als Streamlit-App

### Nicht Bestandteil

- Vorhersage einzelner Anlagen
- Kausaler Nachweis thermischer Verluste
- Ertragsgarantie oder Netzbetriebsprognose
- Vollständige Abbildung von Neigung, Ausrichtung, Schnee, Verschattung,
  Abregelung und technischer Verfügbarkeit
"""
            ),
            code(SETUP_CELL),
            code(
                r"""
project_definition = pd.DataFrame(
    [
        ("Problem", "Meteorologische Bedingungen hoher normierter PV-Erzeugung bestimmen"),
        ("Zielvariable", "PV-Erzeugung [MWh] / installierte Leistung [MW] / 1 h"),
        ("Analyseebene", "Deutschland, stündlich, UTC"),
        ("Zielgruppe", "Energieanalyse und Planung erneuerbarer Energiesysteme"),
        ("Ergebnis", "Nachvollziehbare Analyse, Prognosemodell und Streamlit-App"),
    ],
    columns=["Element", "Festlegung im PV-Projekt"],
)
display(project_definition)
"""
            ),
            markdown(
                r"""
## Erfolgs- und Abnahmekriterien

Die Kriterien verbinden fachlichen Nutzen, Modellgüte und technische
Reproduzierbarkeit. Die Modellgüte wird später in Phase C auf zeitlich
nachgelagerten Testdaten bestimmt.
"""
            ),
            code(
                r"""
acceptance_criteria = pd.DataFrame(
    [
        ("Daten", "Amtliche SMARD-/DWD-Daten oder klar markierte Demo", "Quelle wird im Notebook ausgegeben"),
        ("Zielvariable", "Plausible normierte Erzeugung", "Werte außerhalb 0 bis 1,2 werden verworfen"),
        ("Validierung", "Keine zufällige Mischung von Vergangenheit und Zukunft", "Zeitlicher 80/20-Split"),
        ("Modell", "Besser als einfache Referenz", "MAE des Modells < MAE der Median-Baseline"),
        ("Nutzbarkeit", "Szenarien interaktiv untersuchen", "Streamlit-App in app.py"),
        ("Reproduzierbarkeit", "Logik nicht im Notebook duplizieren", "Gemeinsame Funktionen in pv_weather/"),
    ],
    columns=["Bereich", "Abnahmekriterium", "Operationalisierung"],
)
display(acceptance_criteria)
"""
            ),
            markdown(
                r"""
## Technische Verankerung der Q-Phase

| Festlegung | Umsetzung im Projekt |
|---|---|
| Forschungsfrage und These | `README.md`, Notebook-Dokumentation und App-Texte |
| Zielvariable | `pv_weather/features.py` (`TARGET`, `add_features`) |
| Zielgruppe und Anwendung | `README.md` und `app.py` |
| Datenquellen | `pv_weather/download.py` und `data/README.md` |
| Deployment-Ziel | Streamlit-Anwendung in `app.py` |
| Qualitätskriterien | zeitlicher Test und Baseline in `pv_weather/modeling.py` |
"""
            ),
            code(
                r"""
required_artifacts = [
    "README.md",
    "app.py",
    "pv_weather/features.py",
    "pv_weather/modeling.py",
    "pv_weather/download.py",
]
artifact_check = pd.DataFrame(
    {
        "Artefakt": required_artifacts,
        "vorhanden": [(ROOT / path).exists() for path in required_artifacts],
    }
)
display(artifact_check)
assert artifact_check["vorhanden"].all(), "Mindestens ein definiertes Projektartefakt fehlt."
"""
            ),
            markdown(
                r"""
## Ergebnis und Übergabe an U

Die Forschungsfrage ist als Regressions- und Analyseproblem operationalisiert.
Für die U-Phase müssen nun Datenstruktur, Datenqualität, Verteilungen,
Zusammenhänge und mögliche Verzerrungen geprüft werden. Erst danach ist eine
fundierte Modellwahl zulässig.
"""
            ),
        ]
    )


def build_u_notebook() -> dict[str, Any]:
    return notebook(
        [
            markdown(
                r"""
# U - Understanding the Data (Datenverständnis)

## QUA³CK-Phase

Die U-Phase untersucht Struktur, Qualität, Verteilungen, Anomalien und
Zusammenhänge der Daten. Die Erkenntnisse begründen Datenbereinigung,
Merkmalsbildung und Algorithmuswahl.

## Umsetzung im Projekt

Das Projekt verbindet stündliche PV-Erzeugung und installierte PV-Leistung von
SMARD mit Wetterbeobachtungen räumlich verteilter DWD-Stationen. Der finale
Datensatz liegt auf Deutschlandebene in UTC vor.
"""
            ),
            code(SETUP_CELL),
            code(
                r"""
import matplotlib.pyplot as plt
import seaborn as sns

from pv_weather import TARGET, add_features, load_project_data
from pv_weather.modeling import select_pv_relevant_hours

sns.set_theme(style="whitegrid")

data, source = load_project_data(
    ROOT / "data" / "processed" / "hourly_pv_weather.csv"
)
featured = add_features(data)
daylight = select_pv_relevant_hours(featured)

print(source)
print(f"Zeitraum: {data['timestamp_utc'].min()} bis {data['timestamp_utc'].max()}")
print(f"Vollständiges Panel: {len(data):,} Stunden")
print(f"PV-relevante Tageslichtstunden: {len(daylight):,}")
display(data.head())
"""
            ),
            markdown(
                r"""
## Datenherkunft und Datenfluss

| Daten | Quelle | Projektumsetzung |
|---|---|---|
| PV-Erzeugung | Bundesnetzagentur / SMARD | Download und Einlesen in `download.py` / `ingest.py` |
| Installierte PV-Leistung | Bundesnetzagentur / SMARD | jährliche Zuordnung zur Normierung |
| Strahlung, Temperatur, Feuchte, Wolken, Wind | Deutscher Wetterdienst | Stationsauswahl, Stundenaggregation und Mittelung |
| Gemeinsames Stundenpanel | projektintern | `data/processed/hourly_pv_weather.csv` |

Der automatische End-to-End-Ablauf ist in `pv_weather/workflow.py` gekapselt.
Fehlen lokale Realdaten, stellt `load_project_data` einen ausdrücklich als
synthetisch markierten Demodatensatz bereit.
"""
            ),
            markdown("## Schema- und Qualitätsprüfung"),
            code(
                r"""
quality = pd.DataFrame(
    {
        "Datentyp": data.dtypes.astype(str),
        "Fehlend (n)": data.isna().sum(),
        "Fehlend (%)": data.isna().mean().mul(100).round(2),
        "Eindeutige Werte": data.nunique(),
    }
)
display(quality)

checks = pd.Series(
    {
        "Zeitstempel monoton": data["timestamp_utc"].is_monotonic_increasing,
        "Doppelte UTC-Stunden": int(data["timestamp_utc"].duplicated().sum()),
        "Ungültige Zielwerte": int(
            (~featured[TARGET].between(0, 1.2) & featured[TARGET].notna()).sum()
        ),
        "Negative Globalstrahlung": int((data["global_radiation_j_cm2"] < 0).sum()),
        "Feuchte außerhalb 0-100 %": int(
            (~data["relative_humidity_pct"].between(0, 100)
             & data["relative_humidity_pct"].notna()).sum()
        ),
    },
    name="Prüfergebnis",
)
display(checks.to_frame())
"""
            ),
            markdown(
                r"""
Nullwerte sind nicht automatisch fehlend: Globalstrahlung, Diffusstrahlung,
Sonnenscheindauer und PV-Erzeugung dürfen nachts physikalisch korrekt null sein.
Für die Modellierung werden nur Stunden mit Globalstrahlung über 10 J/cm² und
einem Sonnenzenit unter 90° verwendet; das vollständige Panel bleibt für
Datenkontrolle und Exploration erhalten.
"""
            ),
            code(
                r"""
numeric_summary = featured[
    [
        "temperature_c",
        "relative_humidity_pct",
        "global_radiation_j_cm2",
        "cloud_cover_oktas",
        "wind_speed_m_s",
        "estimated_module_temperature_c",
        TARGET,
    ]
].describe(percentiles=[0.01, 0.25, 0.5, 0.75, 0.99]).T
display(numeric_summary.round(3))
"""
            ),
            markdown("## Explorative Zusammenhänge"),
            code(
                r"""
plot_data = daylight.dropna(
    subset=["global_radiation_j_cm2", "temperature_c", TARGET]
).sample(min(8000, len(daylight)), random_state=42)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
scatter = axes[0].scatter(
    plot_data["global_radiation_j_cm2"],
    plot_data["temperature_c"],
    c=plot_data[TARGET] * 100,
    cmap="YlOrRd",
    s=14,
    alpha=0.45,
)
fig.colorbar(scatter, ax=axes[0], label="Normierte PV-Erzeugung (%)")
axes[0].set(
    title="Einstrahlung, Temperatur und Erzeugung",
    xlabel="Globalstrahlung (J/cm²)",
    ylabel="Lufttemperatur (°C)",
)

corr_columns = [
    "temperature_c",
    "relative_humidity_pct",
    "global_radiation_j_cm2",
    "cloud_cover_oktas",
    "wind_speed_m_s",
    "estimated_module_temperature_c",
    "diffuse_share",
    TARGET,
]
sns.heatmap(
    daylight[corr_columns].corr(),
    cmap="RdBu_r",
    center=0,
    vmin=-1,
    vmax=1,
    ax=axes[1],
)
axes[1].set_title("Lineare Korrelationen in Tageslichtstunden")
plt.tight_layout()
plt.show()
"""
            ),
            code(
                r"""
high_radiation_limit = daylight["global_radiation_j_cm2"].quantile(0.75)
strong_sun = daylight[
    daylight["global_radiation_j_cm2"] >= high_radiation_limit
].copy()
strong_sun["Temperaturklasse"] = pd.cut(
    strong_sun["temperature_c"],
    [-np.inf, 15, 25, 30, np.inf],
    labels=["< 15 °C", "15-25 °C", "25-30 °C", ">= 30 °C"],
)
temperature_summary = (
    strong_sun.groupby("Temperaturklasse", observed=True)[TARGET]
    .agg(Stunden="count", Mittelwert="mean", Median="median")
)
temperature_summary[["Mittelwert", "Median"]] *= 100
display(temperature_summary.round(2))
"""
            ),
            markdown(
                r"""
## Interpretation und Grenzen der U-Phase

- Globalstrahlung ist fachlich und empirisch der dominante Treiber.
- Der Temperatureffekt darf nicht aus einer unkontrollierten Gesamt-Korrelation
  abgeleitet werden, weil starke Einstrahlung zugleich Module erwärmt und die
  Erzeugung erhöht.
- Temperaturklassen bei ähnlich starker Einstrahlung liefern nur deskriptive
  Hinweise, keinen Kausalnachweis.
- Das ungewichtete Stationsmittel kann Regionen mit vielen Stationen
  übergewichten und regionale PV-Leistungsunterschiede glätten.
- Jährliche Kapazitätswerte bilden den unterjährigen Ausbau nur näherungsweise
  ab.

## Technische Verankerung der U-Phase

- `pv_weather/download.py`: automatischer Bezug amtlicher Rohdaten
- `pv_weather/ingest.py`: Einlesen, Zeitvereinheitlichung und Aggregation
- `pv_weather/data.py`: Schema, Plausibilisierung und Daten-Fallback
- `pv_weather/features.py`: Zielvariable und abgeleitete Analysemerkmale
- `data/processed/hourly_pv_weather.csv`: kanonisches Stundenpanel

## Übergabe an A³

Die A³-Phase erhält eine kontinuierliche Zielvariable, einen zeitlich geordneten
Datensatz und die fachliche Anforderung, Strahlungs- und Temperatureffekte
gemeinsam abzubilden, ohne Kalendermerkmale als versteckte Stellvertreter zu
verwenden.
"""
            ),
        ]
    )


def build_a3_notebook() -> dict[str, Any]:
    return notebook(
        [
            markdown(
                r"""
# A³ - Algorithms, Adapting Features, Adjusting Hyperparameters

## QUA³CK-Phase

Die iterative A³-Schleife umfasst:

1. **Algorithm selection** - geeignete Algorithmen auswählen,
2. **Adapting features** - Merkmale fachlich begründet anpassen,
3. **Adjusting hyperparameters** - Modellparameter abstimmen.

## Umsetzung im Projekt

Das PV-Projekt vergleicht eine einfache Median-Baseline mit einem
`HistGradientBoostingRegressor`. Das nichtlineare Modell kann Wechselwirkungen
zwischen Einstrahlung, Temperatur, Bewölkung, Feuchte und Wind abbilden.
Monotoniebedingungen stabilisieren die interaktive Szenarioanalyse.
"""
            ),
            code(SETUP_CELL),
            code(
                r"""
import matplotlib.pyplot as plt

from pv_weather import (
    MODEL_FEATURES,
    TARGET,
    add_features,
    load_project_data,
    train_yield_model,
)
from pv_weather.features import MONOTONIC_CONSTRAINTS
from pv_weather.modeling import (
    MAX_MODEL_SOLAR_ZENITH_DEG,
    MIN_MODEL_GLOBAL_RADIATION_J_CM2,
)

data, source = load_project_data(
    ROOT / "data" / "processed" / "hourly_pv_weather.csv"
)
featured = add_features(data)
print(source)
"""
            ),
            markdown("## A1 - Algorithm selection"),
            code(
                r"""
algorithm_choice = pd.DataFrame(
    [
        (
            "Median-Baseline",
            "Referenz ohne Wetterwissen",
            "Prüft, ob ML überhaupt Mehrwert liefert",
        ),
        (
            "Histogram Gradient Boosting",
            "Nichtlineare Regression mit Interaktionen und Monotoniebedingungen",
            "Produktives Projektmodell",
        ),
    ],
    columns=["Algorithmus", "Eigenschaft", "Rolle im Projekt"],
)
display(algorithm_choice)
"""
            ),
            markdown(
                r"""
Der Testdatensatz besteht aus den neuesten 20 % der verwendbaren
PV-relevanten Stunden. Eine zufällige Aufteilung würde Vergangenheit und
Zukunft vermischen und die zeitliche Übertragbarkeit zu optimistisch bewerten.
"""
            ),
            markdown("## A2 - Adapting features"),
            code(
                r"""
feature_contract = pd.DataFrame(
    [
        ("temperature_c", "°C", "Lufttemperatur", "direkt"),
        ("relative_humidity_pct", "%", "Feuchte", "direkt"),
        ("global_radiation_j_cm2", "J/cm²", "Energieangebot der Sonne", "direkt"),
        ("cloud_cover_oktas", "Achtel", "Bewölkung", "direkt"),
        ("wind_speed_m_s", "m/s", "Kühlung", "direkt"),
        (
            "estimated_module_temperature_c",
            "°C",
            "NOCT-artige Modultemperaturnäherung",
            "aus Temperatur, Strahlung und Wind",
        ),
        (
            "diffuse_share",
            "Anteil",
            "Zusammensetzung der Strahlung",
            "Diffusstrahlung / Globalstrahlung",
        ),
    ],
    columns=["Merkmal", "Einheit", "Fachliche Rolle", "Entstehung"],
)
display(feature_contract)
assert feature_contract["Merkmal"].tolist() == MODEL_FEATURES

preview_columns = [
    "temperature_c",
    "global_radiation_j_cm2",
    "wind_speed_m_s",
    "estimated_module_temperature_c",
    "diffuse_share",
    "thermal_stress_c",
    TARGET,
]
display(featured[preview_columns].head())
"""
            ),
            markdown(
                r"""
Uhrzeit, Monat und Sonnenstand werden zwar für Exploration und Filterung
berechnet, sind aber bewusst keine Eingaben des meteorologischen
Prognosemodells. Die installierte Leistung dient ausschließlich zur Bildung
der Zielvariable und zur späteren absoluten Skalierung.
"""
            ),
            markdown("## A3 - Adjusting hyperparameters"),
            code(
                r"""
final_hyperparameters = pd.Series(
    {
        "learning_rate": 0.07,
        "max_iter": 190,
        "max_leaf_nodes": 31,
        "min_samples_leaf": 30,
        "l2_regularization": 1.0,
        "random_state": 42,
        "test_fraction": 0.2,
        "minimale Globalstrahlung": MIN_MODEL_GLOBAL_RADIATION_J_CM2,
        "maximaler Sonnenzenit": MAX_MODEL_SOLAR_ZENITH_DEG,
    },
    name="Projektwert",
)
display(final_hyperparameters.to_frame())

constraint_names = {-1: "nicht steigend", 0: "frei", 1: "nicht fallend"}
constraints = pd.Series(
    {
        feature: constraint_names[value]
        for feature, value in MONOTONIC_CONSTRAINTS.items()
    },
    name="Isolierte Modellrichtung",
)
display(constraints.to_frame())
"""
            ),
            markdown(
                r"""
Die Parameter sind die aktuell im Projekt festgelegte Konfiguration. Eine
vollständige Hyperparametersuche mit separatem Validierungsfenster ist ein
sinnvoller Ausbaupunkt; der Testzeitraum darf dabei nicht zur Optimierung
verwendet werden.
"""
            ),
            code(
                r"""
bundle = train_yield_model(data)

experiment_result = pd.DataFrame(
    {
        "MAE": [
            bundle.metrics["baseline_mae"],
            bundle.metrics["model_mae"],
        ],
        "RMSE": [
            bundle.metrics["baseline_rmse"],
            bundle.metrics["model_rmse"],
        ],
        "R²": [
            bundle.metrics["baseline_r2"],
            bundle.metrics["model_r2"],
        ],
    },
    index=["Median-Baseline", "HistGradientBoosting"],
)
display(experiment_result.round(4))
print(f"Zeitlicher Test ab: {bundle.split_timestamp}")
"""
            ),
            code(
                r"""
importance = (
    pd.Series(bundle.feature_importance, name="Permutation Importance")
    .sort_values()
)
importance.plot.barh(figsize=(9, 4.5), color="#E19A18")
plt.title("Merkmalsbeitrag im zeitlichen Test")
plt.xlabel("Zunahme des Fehlers nach Permutation")
plt.tight_layout()
plt.show()
"""
            ),
            markdown(
                r"""
## Technische Verankerung der A³-Phase

| A³-Schritt | Umsetzung |
|---|---|
| Algorithm selection | `DummyRegressor` und `HistGradientBoostingRegressor` in `pv_weather/modeling.py` |
| Adapting features | `add_features` und `estimate_module_temperature` in `pv_weather/features.py` |
| Adjusting hyperparameters | explizite Modellkonfiguration in `train_yield_model` |
| Reproduzierbarkeit | feste Zufallszahl, zentrale Merkmalsliste und gemeinsame Kernlogik |
| Experimentauswertung | Metriken und Permutationswichtigkeit im `YieldModelBundle` |

## Übergabe an C

Die C-Phase bewertet Baseline und Modell quantitativ sowie qualitativ. Sie
prüft außerdem, ob die Ergebnisse die Forschungsfrage belastbar beantworten
und welche Einschränkungen kommuniziert werden müssen.
"""
            ),
        ]
    )


def build_c_notebook() -> dict[str, Any]:
    return notebook(
        [
            markdown(
                r"""
# C - Conclude and Compare (Schlussfolgern und Vergleichen)

## QUA³CK-Phase

Phase C vergleicht experimentelle Ergebnisse anhand definierter quantitativer
und qualitativer Kriterien. Ziel ist nicht nur das kleinste Fehlermaß, sondern
eine begründete Entscheidung über Modellnutzen, Interpretierbarkeit,
Wartbarkeit und Grenzen.

## Umsetzung im Projekt

Das Gradient-Boosting-Modell wird gegen eine Median-Baseline auf einem
zeitlich nachgelagerten Testfenster verglichen. MAE, RMSE und R² bilden den
quantitativen Vergleich; Residuen, fachliche Plausibilität und
Implementierungsaufwand ergänzen die Entscheidung.
"""
            ),
            code(SETUP_CELL),
            code(
                r"""
import matplotlib.pyplot as plt
import seaborn as sns

from pv_weather import TARGET, add_features, load_project_data, train_yield_model
from pv_weather.modeling import select_pv_relevant_hours

sns.set_theme(style="whitegrid")
data, source = load_project_data(
    ROOT / "data" / "processed" / "hourly_pv_weather.csv"
)
bundle = train_yield_model(data)
print(source)
print(f"Zeitlicher Test ab: {bundle.split_timestamp}")
"""
            ),
            markdown("## Quantitativer Modellvergleich"),
            code(
                r"""
comparison = pd.DataFrame(
    {
        "MAE (Prozentpunkte)": [
            bundle.metrics["baseline_mae"] * 100,
            bundle.metrics["model_mae"] * 100,
        ],
        "RMSE (Prozentpunkte)": [
            bundle.metrics["baseline_rmse"] * 100,
            bundle.metrics["model_rmse"] * 100,
        ],
        "R²": [
            bundle.metrics["baseline_r2"],
            bundle.metrics["model_r2"],
        ],
    },
    index=["Median-Baseline", "HistGradientBoosting"],
)
display(comparison.round(3))

mae_improvement = (
    1 - bundle.metrics["model_mae"] / bundle.metrics["baseline_mae"]
)
print(f"Relative MAE-Verbesserung gegenüber der Baseline: {mae_improvement:.1%}")
"""
            ),
            code(
                r"""
residuals = pd.Series(bundle.residuals, name="Residuum")
fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
sns.histplot(residuals * 100, bins=40, kde=True, ax=axes[0], color="#E19A18")
axes[0].axvline(0, color="black", linewidth=1)
axes[0].set(
    title="Verteilung der Testresiduen",
    xlabel="Beobachtung minus Prognose (Prozentpunkte)",
)
axes[1].scatter(
    np.arange(len(residuals)),
    residuals * 100,
    s=9,
    alpha=0.35,
    color="#376B5B",
)
axes[1].axhline(0, color="black", linewidth=1)
axes[1].set(
    title="Residuen in zeitlicher Reihenfolge",
    xlabel="Position im Testzeitraum",
    ylabel="Residuum (Prozentpunkte)",
)
plt.tight_layout()
plt.show()
"""
            ),
            markdown("## Qualitativer Vergleich"),
            code(
                r"""
qualitative_comparison = pd.DataFrame(
    [
        ("Vorhersagequalität", "niedrig", "höher; nichtlineare Zusammenhänge"),
        ("Interpretierbarkeit", "sehr hoch", "mittel; Permutationswichtigkeit verfügbar"),
        ("Trainingsaufwand", "sehr gering", "moderat"),
        ("Szenario-Plausibilität", "keine Wetterreaktion", "Monotoniebedingungen"),
        ("Wartbarkeit", "sehr einfach", "zentral in pv_weather/modeling.py gekapselt"),
    ],
    columns=["Kriterium", "Median-Baseline", "HistGradientBoosting"],
)
display(qualitative_comparison)
"""
            ),
            markdown("## Schlussfolgerung zur Forschungsfrage"),
            code(
                r"""
featured = add_features(data)
daylight = select_pv_relevant_hours(featured).dropna(subset=[TARGET])
high_radiation_limit = daylight["global_radiation_j_cm2"].quantile(0.75)
strong_sun = daylight[
    daylight["global_radiation_j_cm2"] >= high_radiation_limit
].copy()
strong_sun["Temperaturklasse"] = pd.cut(
    strong_sun["temperature_c"],
    [-np.inf, 15, 25, 30, np.inf],
    labels=["< 15 °C", "15-25 °C", "25-30 °C", ">= 30 °C"],
)
thesis_evidence = (
    strong_sun.groupby("Temperaturklasse", observed=True)[TARGET]
    .agg(Stunden="count", Mittelwert="mean", Median="median")
)
thesis_evidence[["Mittelwert", "Median"]] *= 100
display(thesis_evidence.round(2))

best_class = thesis_evidence["Median"].idxmax()
print(
    f"Bei starker Einstrahlung (>= {high_radiation_limit:.1f} J/cm²) "
    f"hat die Klasse {best_class} den höchsten beobachteten Median."
)
"""
            ),
            markdown(
                r"""
Die Tabelle liefert einen deskriptiven Hinweis zur These. Ein niedrigerer
Median bei sehr hohen Temperaturen wäre mit thermischen Verlusten vereinbar,
beweist sie aber nicht. Gleichzeitig verändern sich weitere Wetter- und
Betriebsbedingungen. Die modellierte Temperatur-Sensitivität ist ebenfalls
eine kontrollierte Modellreaktion und kein experimenteller Kausalnachweis.

## Modellentscheidung

Das Gradient-Boosting-Modell wird für die Anwendung gewählt, **wenn sein MAE im
zeitlichen Test unter dem MAE der Median-Baseline liegt**. Die Baseline bleibt
als dauerhafter Kontrollpunkt erhalten.
"""
            ),
            code(
                r"""
model_selected = bundle.metrics["model_mae"] < bundle.metrics["baseline_mae"]
decision = "HistGradientBoosting auswählen" if model_selected else "Modell nicht freigeben"
print(decision)
assert model_selected, "Das Modell übertrifft die Baseline im aktuellen Test nicht."
"""
            ),
            markdown(
                r"""
## Grenzen und Risiken

- Das DWD-Stationsmittel ist nicht vollständig nach regionaler PV-Leistung
  gewichtet.
- Die geschätzte Modultemperatur ist keine direkte Messung.
- Jahreskapazitäten ignorieren unterjährigen Ausbau.
- Seltene sehr heiße Stunden besitzen größere statistische Unsicherheit.
- Anlagenneigung, Ausrichtung, Schnee, Verschattung, Abregelung und technische
  Verfügbarkeit fehlen.
- Testresiduen liefern ein empirisches 80-%-Intervall, aber keine vollständige
  probabilistische Prognose.

## Technische Verankerung der C-Phase

- `pv_weather/modeling.py`: zeitlicher Split, Baseline, Metriken und Residuen
- `tests/test_modeling.py`: automatisierte Modellverträge
- `app.py`: Darstellung von Modellgüte, Szenarien und Grenzen
- `notebooks/pv_wetter_deutschland.ipynb`: zusammenhängende Gesamtdokumentation

## Übergabe an K

Die K-Phase bereitet Entscheidung, Kernergebnisse, Einschränkungen und
Nutzungsanleitung zielgruppengerecht auf und überführt die gemeinsame
Kernlogik in die Streamlit-Anwendung.
"""
            ),
        ]
    )


def build_k_notebook() -> dict[str, Any]:
    return notebook(
        [
            markdown(
                r"""
# K - Knowledge Transfer (Wissenstransfer)

## QUA³CK-Phase

Die K-Phase dokumentiert und kommuniziert Ergebnisse und überführt sie in
nutzbare Artefakte. Im Projekt umfasst das die reproduzierbare
Notebook-Dokumentation, gemeinsame Python-Kernlogik, Tests, README und eine
interaktive Streamlit-App.

## Umsetzung im Projekt

Die Zielgruppe kann meteorologische Szenarien einstellen, eine normierte
PV-Prognose samt empirischem 80-%-Intervall ansehen, den thermischen Effekt bei
konstanter Einstrahlung untersuchen und das Szenario mit besonders
ertragreichen beobachteten Stunden vergleichen.
"""
            ),
            code(SETUP_CELL),
            code(
                r"""
from pv_weather import (
    estimate_module_temperature,
    load_project_data,
    predict_yield,
    train_yield_model,
)

data, source = load_project_data(
    ROOT / "data" / "processed" / "hourly_pv_weather.csv"
)
bundle = train_yield_model(data)
print(source)
"""
            ),
            markdown("## Zielgruppengerechte Projektzusammenfassung"),
            code(
                r"""
summary = pd.Series(
    {
        "Forschungsfrage": (
            "Unter welchen meteorologischen Bedingungen ist die normierte "
            "PV-Erzeugung in Deutschland am höchsten?"
        ),
        "Daten": "Stündliche SMARD-PV-Daten und DWD-Wetterdaten",
        "Zielvariable": "PV-Erzeugung / installierte PV-Leistung / 1 h",
        "Modell": "HistGradientBoosting mit meteorologischen Merkmalen",
        "Validierung": f"Zeitlicher Test ab {bundle.split_timestamp:%d.%m.%Y}",
        "Modell-MAE": f"{bundle.metrics['model_mae'] * 100:.2f} Prozentpunkte",
        "Baseline-MAE": f"{bundle.metrics['baseline_mae'] * 100:.2f} Prozentpunkte",
        "Anwendung": "Interaktive Streamlit-App",
    },
    name="PV-Wetter-Projekt",
)
display(summary.to_frame())
"""
            ),
            markdown("## Beispiel für den Transfer vom Modell zur Anwendung"),
            code(
                r"""
scenario = pd.DataFrame(
    {
        "timestamp_utc": [pd.Timestamp("2024-01-01", tz="UTC")],
        "temperature_c": [25.0],
        "relative_humidity_pct": [55.0],
        "global_radiation_j_cm2": [270.0],
        "diffuse_radiation_j_cm2": [78.3],
        "sunshine_duration_min": [38.2],
        "cloud_cover_oktas": [2.0],
        "wind_speed_m_s": [3.0],
    }
)
prediction = predict_yield(bundle, scenario).iloc[0]
module_temperature = estimate_module_temperature(
    scenario["temperature_c"],
    scenario["global_radiation_j_cm2"],
    scenario["wind_speed_m_s"],
)[0]

result = pd.Series(
    {
        "Normierte Prognose (%)": prediction["normalized_pv_prediction"] * 100,
        "Untere 80-%-Grenze (%)": prediction["lower_80"] * 100,
        "Obere 80-%-Grenze (%)": prediction["upper_80"] * 100,
        "Geschätzte Modultemperatur (°C)": module_temperature,
    },
    name="Beispielszenario",
)
display(result.round(2).to_frame())
"""
            ),
            markdown(
                r"""
Die installierte PV-Leistung ist kein Modellmerkmal. Sie skaliert die normierte
Prognose nachträglich zu MW beziehungsweise MWh pro Stunde. Dadurch bleibt die
meteorologische Aussage unabhängig vom Ausbaugrad.
"""
            ),
            markdown("## Artefakt- und Kommunikationslandkarte"),
            code(
                r"""
artifacts = pd.DataFrame(
    [
        ("app.py", "Interaktive Prognose und Ergebnisvermittlung", "Zielgruppe"),
        ("README.md", "Projektüberblick, Installation und Grenzen", "Portfolio / Entwicklung"),
        ("notebooks/", "Nachvollziehbare QUA³CK-Dokumentation", "Lehre / Prüfung"),
        ("pv_weather/", "Wiederverwendbare Kernlogik", "Entwicklung"),
        ("tests/", "Automatisierte Qualitätsverträge", "Entwicklung / Wartung"),
        ("data/README.md", "Datenquellen, Schema und Aufbereitung", "Reproduzierbarkeit"),
        ("KI_NUTZUNG.md", "Transparenz zur KI-Unterstützung", "Prüfung / Portfolio"),
    ],
    columns=["Artefakt", "Transferfunktion", "Adressaten"],
)
artifacts["vorhanden"] = artifacts["Artefakt"].map(lambda path: (ROOT / path).exists())
display(artifacts)
"""
            ),
            markdown(
                r"""
## Reproduzierbare Nutzung

Im Projektverzeichnis:

```powershell
# Abhängigkeiten
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt

# Realdaten laden und Modell prüfen
python scripts/download_real_data.py --start-year 2020 --end-year 2025

# QUA³CK-Notebooks neu erzeugen und ausführen
python scripts/create_quack_notebooks.py --execute

# Tests
python -m pytest

# Anwendung
streamlit run app.py
```

Ohne lokale Rohdaten nutzt die Anwendung klar markierte synthetische
Demodaten. Diese demonstrieren den Workflow, sind aber kein empirischer Befund.
"""
            ),
            markdown("## Checkliste für Wissenstransfer und Bereitstellung"),
            code(
                r"""
transfer_checklist = pd.DataFrame(
    [
        ("Forschungsfrage und Zielgruppe dokumentiert", True),
        ("Datenquelle und Demo-Status sichtbar", True),
        ("Modell gegen Baseline validiert", bundle.metrics["model_mae"] < bundle.metrics["baseline_mae"]),
        ("Unsicherheitsintervall kommuniziert", True),
        ("Grenzen und fehlende Einflussgrößen dokumentiert", True),
        ("Streamlit-App vorhanden", (ROOT / "app.py").exists()),
        ("Automatisierte Tests vorhanden", (ROOT / "tests").exists()),
        ("Reproduzierbare Notebook-Erzeugung vorhanden", (ROOT / "scripts" / "create_quack_notebooks.py").exists()),
    ],
    columns=["Kriterium", "erfüllt"],
)
display(transfer_checklist)
assert transfer_checklist["erfüllt"].all(), "Die Wissenstransfer-Checkliste ist unvollständig."
"""
            ),
            markdown(
                r"""
## Kernaussage und verantwortungsvolle Kommunikation

Hohe Einstrahlung ist der wichtigste beobachtete Treiber hoher normierter
PV-Erzeugung. Temperaturunterschiede müssen bei vergleichbarer Einstrahlung
bewertet werden. Beobachtete Gruppenunterschiede und Modellreaktionen sind
keine Kausalnachweise und keine Ertragsgarantie.

Für einen produktiven Dauerbetrieb wären zusätzlich Monitoring von
Datenqualität und Model Drift, Modellversionierung, wiederholbares
Experiment-Tracking und eine dokumentierte Freigabe neuer Modellversionen
erforderlich.

## Technische Verankerung der K-Phase

- `app.py`: interaktive Bereitstellung
- `README.md`: Portfolio- und Nutzungsdokumentation
- `scripts/create_quack_notebooks.py`: reproduzierbare Phasendokumentation
- `pv_weather/workflow.py`: kontrollierter Daten- und Trainingsablauf
- `tests/`: Qualitätskontrolle
- `KI_NUTZUNG.md`: Transparenz
"""
            ),
        ]
    )


def build_notebooks() -> dict[str, dict[str, Any]]:
    return {
        NOTEBOOK_FILES["Q"]: build_q_notebook(),
        NOTEBOOK_FILES["U"]: build_u_notebook(),
        NOTEBOOK_FILES["A3"]: build_a3_notebook(),
        NOTEBOOK_FILES["C"]: build_c_notebook(),
        NOTEBOOK_FILES["K"]: build_k_notebook(),
    }


def execute_notebook(path: Path) -> None:
    try:
        import nbformat
        from nbclient import NotebookClient
    except ImportError as exc:
        raise RuntimeError(
            "Für --execute werden nbformat und nbclient benötigt. "
            "Installieren Sie die Notebook-Entwicklungsabhängigkeiten."
        ) from exc

    loaded = nbformat.read(path, as_version=4)
    NotebookClient(
        loaded,
        timeout=600,
        kernel_name="python3",
        resources={"metadata": {"path": str(ROOT)}},
    ).execute()
    nbformat.write(loaded, path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Erzeugt je QUA³CK-Phase ein separates Projektnotebook."
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Führt alle erzeugten Notebooks aus und speichert die Ausgaben.",
    )
    args = parser.parse_args()

    NOTEBOOK_DIR.mkdir(parents=True, exist_ok=True)
    created_paths: list[Path] = []
    for filename, content in build_notebooks().items():
        path = NOTEBOOK_DIR / filename
        path.write_text(
            json.dumps(content, ensure_ascii=False, indent=1) + "\n",
            encoding="utf-8",
        )
        created_paths.append(path)

    if args.execute:
        for path in created_paths:
            execute_notebook(path)

    for path in created_paths:
        print(f"Notebook gespeichert: {path}")


if __name__ == "__main__":
    main()
