"""Generate the auditable notebook for PV weather and thermal losses."""

from __future__ import annotations

import argparse
from pathlib import Path

import nbformat as nbf
from nbclient import NotebookClient


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "notebooks" / "pv_wetter_deutschland.ipynb"


def markdown(text: str):
    return nbf.v4.new_markdown_cell(text.strip())


def code(text: str):
    return nbf.v4.new_code_cell(text.strip())


def build_notebook():
    cells = [
        markdown(
            r"""
# Einfluss meteorologischer Bedingungen auf die Photovoltaikerzeugung in Deutschland

**Forschungsfrage:** Unter welchen meteorologischen Bedingungen ist die auf die
installierte Leistung normierte Photovoltaikerzeugung in Deutschland am höchsten?

**These:** Die höchste normierte Photovoltaikerzeugung tritt bei hoher
Sonneneinstrahlung und moderaten Temperaturen auf. Bei sehr hohen Temperaturen
nimmt die Erzeugung trotz starker Einstrahlung wieder ab.

**Ziel:** Ein stündliches Modell prognostiziert den deutschen PV-Kapazitätsfaktor
und identifiziert Bedingungen mit erhöhtem thermischem Verlust.

Das Notebook verwendet dieselben Funktionen wie die Streamlit-App. Fehlen reale
SMARD-/DWD-Daten, läuft es mit eindeutig markierten synthetischen Demodaten.
Diese sind kein empirischer Befund.
"""
        ),
        markdown(
            """
## 1. Setup und Datenstatus

SMARD liefert realisierte Photovoltaikerzeugung und die installierte Leistung.
DWD-Stationsdaten werden deutschlandweit aggregiert. Gespeichert wird in UTC,
Kalendermerkmale werden in Europe/Berlin erzeugt.
"""
        ),
        code(
            r"""
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from IPython.display import display

ROOT = Path.cwd().resolve()
if ROOT.name == "notebooks":
    ROOT = ROOT.parent
sys.path.insert(0, str(ROOT))

from pv_weather import (
    MODEL_FEATURES,
    TARGET,
    add_features,
    estimate_module_temperature,
    load_project_data,
    predict_yield,
    train_yield_model,
)

sns.set_theme(style="whitegrid", context="notebook")
pd.set_option("display.max_columns", 30)

data, source = load_project_data(ROOT / "data" / "processed" / "hourly_pv_weather.csv")
IS_DEMO = source.startswith("Synthetische")
print(source)
print(f"{len(data):,} Stunden | {data.timestamp_utc.min()} bis {data.timestamp_utc.max()}")
if IS_DEMO:
    print("ACHTUNG: Alle folgenden Zahlen illustrieren nur den Workflow.")
display(data.head())
"""
        ),
        markdown(
            """
## 2. Datenqualität und Zielvariable

Der stündliche Kapazitätsfaktor ist dimensionslos:

`PV-Erzeugung [MWh] / (installierte PV-Leistung [MW] × 1 h)`

Jahreswerte der installierten Leistung sind eine Näherung, weil der Ausbau
innerhalb des Jahres nicht vollständig abgebildet wird.
"""
        ),
        code(
            r"""
hourly = add_features(data)
quality = pd.DataFrame({
    "dtype": data.dtypes.astype(str),
    "fehlend_n": data.isna().sum(),
    "fehlend_pct": data.isna().mean().mul(100).round(2),
    "eindeutig": data.nunique(),
})
display(quality)
print(f"Doppelte UTC-Stunden: {data['timestamp_utc'].duplicated().sum()}")
print(f"Ungültige Zielwerte: {(~hourly[TARGET].between(0, 1.2) & hourly[TARGET].notna()).sum()}")
display(hourly[["timestamp_utc", "pv_generation_mwh", "installed_pv_capacity_mw", TARGET]].head())
"""
        ),
        markdown(
            """
## 3. Wetteraggregation und Feature Engineering

Kernmerkmale sind Global-/Diffusstrahlung, Sonnenscheindauer, Sonnenzenit,
Lufttemperatur, Feuchte, Bewölkung und Wind. Bei echten Daten werden Stationen
gleich oder mit regionalen PV-Gewichten gemittelt.

Die Modultemperatur ist nicht gemessen. Sie wird transparent mit einer
NOCT-artigen Näherung aus Lufttemperatur, Strahlung und Wind geschätzt:

`T_modul ≈ T_luft + 0,03125 × Einstrahlung_W/m² / (1 + 0,12 × Wind_m/s)`
"""
        ),
        code(
            r"""
selected = [
    "timestamp_utc", "hour", "month", "season",
    "global_radiation_j_cm2", "temperature_c", "wind_speed_m_s",
    "estimated_module_temperature_c", "thermal_stress_c", TARGET,
]
display(hourly[selected].head(24).tail())
print("Modellmerkmale:")
print("\n".join(f"• {feature}" for feature in MODEL_FEATURES))
"""
        ),
        markdown(
            """
## 4. Zusammenhang von Einstrahlung und normierter PV-Erzeugung

Nachtstunden werden für diese Darstellung entfernt. Farbe zeigt die geschätzte
Modultemperatur. Eine Punktwolke beschreibt Zusammenhänge, keine Kausalität.
"""
        ),
        code(
            r"""
daylight = hourly[
    (hourly["global_radiation_j_cm2"] > 10) & hourly[TARGET].notna()
].copy()
sample = daylight.sample(min(6000, len(daylight)), random_state=42)

plt.figure(figsize=(11, 5.5))
scatter = plt.scatter(
    sample["global_radiation_j_cm2"],
    sample[TARGET] * 100,
    c=sample["estimated_module_temperature_c"],
    cmap="YlOrRd",
    s=12,
    alpha=.35,
)
plt.colorbar(scatter, label="geschätzte Modultemperatur (°C)")
plt.xlabel("Globalstrahlung (J/cm²)")
plt.ylabel("normierte PV-Erzeugung (%)")
plt.title("Einstrahlung, Modultemperatur und PV-Kapazitätsfaktor")
plt.show()
"""
        ),
        markdown(
            """
## 5. Test der Temperaturthese bei starker Einstrahlung

Um Einstrahlung grob zu kontrollieren, werden nur Stunden im oberen
Strahlungsquartil verglichen. Saison, Sonnenstand und Wetter bleiben dennoch
mögliche Störfaktoren.
"""
        ),
        code(
            r"""
high_radiation_limit = daylight["global_radiation_j_cm2"].quantile(.75)
strong_sun = daylight[daylight["global_radiation_j_cm2"] >= high_radiation_limit].copy()
strong_sun["temperature_class"] = pd.cut(
    strong_sun["temperature_c"],
    [-30, 15, 25, 30, 60],
    labels=["< 15 °C", "15–25 °C", "25–30 °C", "≥ 30 °C"],
)
temperature_summary = strong_sun.groupby("temperature_class", observed=True)[TARGET].agg(
    Stunden="size", Mittelwert="mean", Median="median"
)
temperature_summary[["Mittelwert", "Median"]] *= 100
display(temperature_summary.round(2))

sns.boxplot(
    data=strong_sun.sample(min(8000, len(strong_sun)), random_state=42),
    x="temperature_class",
    y=TARGET,
    color="#F2B134",
    showfliers=False,
)
plt.xlabel("Lufttemperaturklasse")
plt.ylabel("normierte PV-Erzeugung")
plt.title(f"Oberes Strahlungsquartil ab {high_radiation_limit:.0f} J/cm²")
plt.show()
"""
        ),
        markdown(
            """
## 6. Wetterbedingungen der besten Stunden

Die Top-2%-Stunden beschreiben ein beobachtungsnahes Optimum. Sie sind kein
technisch garantiertes Anlagenoptimum und hängen von Messnetz, Zeitraum und
räumlicher Aggregation ab.
"""
        ),
        code(
            r"""
top = hourly.nlargest(max(50, int(len(hourly) * .02)), TARGET)
optimum = top[
    [
        TARGET, "global_radiation_j_cm2", "diffuse_radiation_j_cm2",
        "temperature_c", "estimated_module_temperature_c",
        "cloud_cover_oktas", "wind_speed_m_s", "relative_humidity_pct",
        "hour", "month",
    ]
].median().to_frame("Median Top 2 %")
display(optimum.round(2))
"""
        ),
        markdown(
            """
## 7. Zeitlich ehrliches Prognosemodell

Die ersten 80 % der vollständigen Beobachtungen bilden das Training, die
neuesten 20 % einen zusammenhängenden Test. Das verhindert zufälliges
Vermischen von Vergangenheit und Zukunft.
"""
        ),
        code(
            r"""
bundle = train_yield_model(data)
metrics = pd.DataFrame(
    {
        "Gradient Boosting": [
            bundle.metrics["model_mae"] * 100,
            bundle.metrics["model_rmse"] * 100,
            bundle.metrics["model_r2"],
        ],
        "Median-Baseline": [
            bundle.metrics["baseline_mae"] * 100,
            bundle.metrics["baseline_rmse"] * 100,
            bundle.metrics["baseline_r2"],
        ],
    },
    index=["MAE (Prozentpunkte)", "RMSE (Prozentpunkte)", "R²"],
)
print(f"Zeitlicher Test beginnt: {bundle.split_timestamp}")
display(metrics.round(3))
"""
        ),
        markdown(
            """
## 8. Wichtigste Einflussmerkmale

Permutation Importance misst den Anstieg des Testfehlers beim zufälligen
Vertauschen eines Merkmals. Korrelierte Wettergrößen teilen sich Bedeutung;
die Werte sind keine kausalen Effekte.
"""
        ),
        code(
            r"""
labels = {
    "global_radiation_j_cm2": "Globalstrahlung",
    "solar_zenith_angle_deg": "Sonnenzenit",
    "estimated_module_temperature_c": "geschätzte Modultemperatur",
    "temperature_c": "Lufttemperatur",
    "cloud_cover_oktas": "Bewölkung",
    "wind_speed_m_s": "Wind",
    "relative_humidity_pct": "Feuchte",
    "radiation_thermal_interaction": "Strahlung × thermischer Stress",
}
importance = (
    pd.Series(bundle.feature_importance)
    .sort_values(ascending=False)
    .head(12)
    .rename(index=labels)
)
display(importance.to_frame("Permutation Importance"))
importance.sort_values().plot.barh(figsize=(9, 5), color="#F2B134")
plt.xlabel("Anstieg des MAE")
plt.title("Einflussmerkmale im zeitlichen Test")
plt.show()
"""
        ),
        markdown(
            """
## 9. Szenario und thermischer Vergleich

Zwei Szenarien unterscheiden sich nur in der Lufttemperatur. Das Modell nutzt
ansonsten identische Wetter- und Kalenderwerte.
"""
        ),
        code(
            r"""
def scenario(temperature):
    radiation = 280.0
    wind = 3.0
    return pd.DataFrame({
        "timestamp_utc": [pd.Timestamp("2024-07-01 11:00", tz="UTC")],
        "temperature_c": [temperature],
        "relative_humidity_pct": [50.0],
        "global_radiation_j_cm2": [radiation],
        "diffuse_radiation_j_cm2": [radiation * .25],
        "sunshine_duration_min": [50.0],
        "solar_zenith_angle_deg": [30.0],
        "cloud_cover_oktas": [1.5],
        "wind_speed_m_s": [wind],
    })

comparison = pd.concat(
    [
        predict_yield(bundle, scenario(22)).assign(Szenario="22 °C Luft"),
        predict_yield(bundle, scenario(36)).assign(Szenario="36 °C Luft"),
    ],
    ignore_index=True,
)
comparison["Prognose (%)"] = comparison["normalized_pv_prediction"] * 100
display(comparison[["Szenario", "Prognose (%)", "lower_80", "upper_80"]])
"""
        ),
        markdown(
            """
## 10. Fazit und Grenzen

Die Forschungsfrage wird über den Zusammenhang zwischen Einstrahlung,
geschätzter Modultemperatur und normierter PV-Erzeugung beantwortet. Eine
empirische Schlussfolgerung ist nur zulässig, wenn oben Realdaten ausgewiesen
sind.

Wichtige Grenzen:

- Stationsmittel approximieren das Wetter der räumlich verteilten PV-Flotte.
- Regionale PV-Gewichte sind stärker als reine Stationsmittel.
- Jährliche Kapazitätswerte bilden unterjährigen Ausbau nicht ab.
- Lufttemperatur ist nicht Modultemperatur; letztere wird nur angenähert.
- Anlagenneigung, Ausrichtung, Schnee, Verschattung, Abregelung und technische
  Verfügbarkeit fehlen.
- Gruppenvergleiche und Feature Importance sind keine Kausalnachweise.
"""
        ),
    ]
    notebook = nbf.v4.new_notebook(cells=cells)
    notebook.metadata.kernelspec = {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    }
    notebook.metadata.language_info = {"name": "python", "version": "3"}
    return notebook


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute", action="store_true", help="Run all cells before saving.")
    args = parser.parse_args()

    notebook = build_notebook()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    if args.execute:
        NotebookClient(
            notebook,
            timeout=600,
            kernel_name="python3",
            resources={"metadata": {"path": str(ROOT)}},
        ).execute()
    nbf.write(notebook, OUTPUT)
    print(f"Notebook gespeichert: {OUTPUT}")


if __name__ == "__main__":
    main()
