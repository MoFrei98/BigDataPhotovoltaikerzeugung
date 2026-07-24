"""Photovoltaikeffizienz Rechner – Streamlit application."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st


ROOT = Path(__file__).resolve().parent

from pv_weather import (  # noqa: E402
    TARGET,
    add_features,
    estimate_module_temperature,
    load_project_data,
    predict_yield,
    train_yield_model,
)


st.set_page_config(
    page_title="Photovoltaikeffizienz Rechner",
    page_icon="☀️",
    layout="wide",
)

st.markdown(
    """
    <style>
    .stApp {background:
      radial-gradient(circle at 92% 3%, rgba(255,190,51,.23), transparent 27rem),
      linear-gradient(180deg, #FBFAF5 0%, #F2F6F2 100%);}
    .block-container {padding-top:4.25rem; max-width:1240px;}
    .eyebrow {letter-spacing:.14em; text-transform:uppercase; color:#567064;
      font-size:.78rem; font-weight:750; margin-bottom:.5rem;}
    .hero-title {font-size:clamp(2.35rem,5vw,4.7rem); line-height:.95;
      letter-spacing:-.055em; color:#173C34; font-weight:850; max-width:1000px;}
    .hero-copy {font-size:1.12rem; color:#4A625A; max-width:820px; margin:1rem 0 1.3rem;}
    .source-pill {display:inline-block; border:1px solid #B9C9C1; border-radius:999px;
      padding:.3rem .75rem; background:rgba(255,255,255,.72); font-size:.82rem;}
    div[data-testid="stMetric"] {background:rgba(255,255,255,.78); border:1px solid #D8E2DD;
      border-radius:16px; padding:1rem 1.1rem; box-shadow:0 8px 30px rgba(23,60,52,.06);}
    div[data-testid="stMetricValue"] {color:#173C34;}
    .insight {border-left:4px solid #F2B134; background:#FFF9E8; padding:1rem 1.1rem;
      border-radius:0 12px 12px 0; color:#314F46;}
    .method-note {background:#E8F1ED; padding:1rem 1.1rem; border-radius:12px; color:#314F46;}
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(show_spinner=False)
def get_data() -> tuple[pd.DataFrame, str]:
    return load_project_data(ROOT / "data" / "processed" / "hourly_pv_weather.csv")


@st.cache_resource(show_spinner="PV-Modell wird zeitlich validiert …")
def get_model(data: pd.DataFrame):
    return train_yield_model(data)


try:
    data, source_label = get_data()
    bundle = get_model(data)
except (OSError, ValueError) as exc:
    st.error(f"Daten oder Modell konnten nicht vorbereitet werden: {exc}")
    st.stop()

featured = add_features(data)
is_demo = source_label.startswith("Synthetische")

st.markdown(
    '<div class="eyebrow">Deutschland · Meteorologie × Photovoltaik</div>',
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="hero-title">Photovoltaikeffizienz<br>Rechner</div>',
    unsafe_allow_html=True,
)
st.markdown(
    """
    <div class="hero-copy">
    Unter welchen Wetterbedingungen schöpft Deutschlands Photovoltaikflotte ihre
    installierte Leistung am besten aus – und wann begrenzen hohe geschätzte
    Modultemperaturen den Ertrag trotz starker Einstrahlung?
    </div>
    """,
    unsafe_allow_html=True,
)
pill_color = "#8A4B08" if is_demo else "#176B4D"
st.markdown(
    f'<span class="source-pill" style="color:{pill_color}">{source_label}</span>',
    unsafe_allow_html=True,
)
if is_demo:
    st.warning(
        "Demo-Modus: Die Daten bilden den erwarteten Strahlungs- und Temperatureffekt "
        "synthetisch ab. Sie sind kein empirischer Befund. Realdaten unter "
        "`data/processed/hourly_pv_weather.csv` schalten die App automatisch um.",
        icon="🧪",
    )


def solar_zenith(day: date, hour: int, latitude_deg: float = 51.0) -> float:
    day_of_year = pd.Timestamp(day).dayofyear
    declination = np.deg2rad(23.44 * np.sin(2 * np.pi * (284 + day_of_year) / 365))
    latitude = np.deg2rad(latitude_deg)
    hour_angle = np.deg2rad(15 * (hour - 12))
    cosine = (
        np.sin(latitude) * np.sin(declination)
        + np.cos(latitude) * np.cos(declination) * np.cos(hour_angle)
    )
    return float(np.rad2deg(np.arccos(np.clip(cosine, -1, 1))))


def scenario_frame(
    selected_day: date,
    hour: int,
    radiation: float,
    temperature: float,
    cloud_cover: float,
    wind_speed: float,
    humidity: float,
) -> pd.DataFrame:
    timestamp = (
        pd.Timestamp(selected_day)
        .replace(hour=hour)
        .tz_localize("Europe/Berlin", ambiguous=True, nonexistent="shift_forward")
        .tz_convert("UTC")
    )
    diffuse_share = np.clip(0.14 + 0.075 * cloud_cover, 0.12, 0.82)
    sunshine = np.clip(60 * (radiation / 330) * (1 - cloud_cover / 9), 0, 60)
    return pd.DataFrame(
        {
            "timestamp_utc": [timestamp],
            "temperature_c": [temperature],
            "relative_humidity_pct": [humidity],
            "global_radiation_j_cm2": [radiation],
            "diffuse_radiation_j_cm2": [radiation * diffuse_share],
            "sunshine_duration_min": [sunshine],
            "solar_zenith_angle_deg": [solar_zenith(selected_day, hour)],
            "cloud_cover_oktas": [cloud_cover],
            "wind_speed_m_s": [wind_speed],
        }
    )


def yield_level(value: float) -> str:
    if value < 0.20:
        return "niedrig"
    if value < 0.50:
        return "mittel"
    return "hoch"


def percent(value: float, digits: int = 1) -> str:
    return f"{value * 100:.{digits}f} %".replace(".", ",")


def prediction_chart(predicted: float, lower: float, upper: float) -> alt.LayerChart:
    """Visualise the prediction and its interval on a compact 0–100 % scale."""
    predicted_pct = float(np.clip(predicted * 100, 0, 100))
    lower_pct = float(np.clip(lower * 100, 0, 100))
    upper_pct = float(np.clip(upper * 100, 0, 100))

    bands = pd.DataFrame(
        {
            "Start": [0, 20, 50],
            "Ende": [20, 50, 100],
            "Bereich": ["Niedrig", "Mittel", "Hoch"],
        }
    )
    interval = pd.DataFrame({"Start": [lower_pct], "Ende": [upper_pct]})
    point = pd.DataFrame(
        {
            "Prognose": [predicted_pct],
            "Beschriftung": [f"{predicted_pct:.1f} %".replace(".", ",")],
        }
    )

    x_scale = alt.Scale(domain=[0, 100], nice=False)
    background = (
        alt.Chart(bands)
        .mark_bar(size=28, cornerRadius=7)
        .encode(
            x=alt.X(
                "Start:Q",
                scale=x_scale,
                title="Normierte PV-Erzeugung (%)",
                axis=alt.Axis(values=[0, 20, 40, 60, 80, 100], grid=False),
            ),
            x2="Ende:Q",
            y=alt.value(52),
            color=alt.Color(
                "Bereich:N",
                scale=alt.Scale(
                    domain=["Niedrig", "Mittel", "Hoch"],
                    range=["#E9EEEB", "#DDE9E2", "#CDE5D6"],
                ),
                legend=None,
            ),
            tooltip=[
                "Bereich:N",
                alt.Tooltip("Start:Q", format=".0f"),
                alt.Tooltip("Ende:Q", format=".0f"),
            ],
        )
    )
    uncertainty = (
        alt.Chart(interval)
        .mark_rule(color="#E19A18", strokeWidth=7, strokeCap="round")
        .encode(
            x=alt.X("Start:Q", scale=x_scale),
            x2="Ende:Q",
            y=alt.value(52),
            tooltip=[
                alt.Tooltip("Start:Q", title="Untere Grenze", format=".1f"),
                alt.Tooltip("Ende:Q", title="Obere Grenze", format=".1f"),
            ],
        )
    )
    marker = (
        alt.Chart(point)
        .mark_point(color="#173C34", filled=True, shape="diamond", size=230)
        .encode(
            x=alt.X("Prognose:Q", scale=x_scale),
            y=alt.value(52),
            tooltip=[alt.Tooltip("Prognose:Q", title="Prognose", format=".1f")],
        )
    )
    label = (
        alt.Chart(point)
        .mark_text(color="#173C34", dy=-25, fontSize=14, fontWeight="bold")
        .encode(
            x=alt.X("Prognose:Q", scale=x_scale),
            y=alt.value(52),
            text="Beschriftung:N",
        )
    )
    return (background + uncertainty + marker + label).properties(height=105)


optimal_rows = featured.nlargest(max(50, int(len(featured) * 0.02)), TARGET)
optimal_defaults = {
    column: float(optimal_rows[column].median())
    for column in [
        "temperature_c",
        "relative_humidity_pct",
        "global_radiation_j_cm2",
        "cloud_cover_oktas",
        "wind_speed_m_s",
    ]
}

tab_prediction, tab_temperature, tab_optimum, tab_analysis, tab_method = st.tabs(
    ["Prognose", "Thermischer Effekt", "Optimale Bedingungen", "Datenanalyse", "Modell & Daten"]
)

with tab_prediction:
    st.subheader("Meteorologisches Szenario")
    inputs, output = st.columns([1.05, 0.95], gap="large")
    with inputs:
        left, right = st.columns(2)
        with left:
            selected_day = st.date_input("Datum", value=date(2024, 7, 1))
            selected_hour = st.slider("Uhrzeit", 0, 23, 13)
            radiation = st.slider("Globalstrahlung (J/cm²)", 0.0, 360.0, 270.0, 5.0)
            temperature = st.slider("Lufttemperatur (°C)", -10.0, 42.0, 25.0, 0.5)
        with right:
            cloud_cover = st.slider("Bewölkungsgrad (Achtel)", 0.0, 8.0, 2.0, 0.5)
            wind_speed = st.slider("Windgeschwindigkeit (m/s)", 0.0, 15.0, 3.0, 0.5)
            humidity = st.slider("Relative Luftfeuchtigkeit (%)", 10, 100, 55)
            installed_capacity = st.number_input(
                "Installierte PV-Leistung (MW)", 1_000, 200_000, 90_000, 1_000
            )

        scenario = scenario_frame(
            selected_day,
            selected_hour,
            radiation,
            temperature,
            cloud_cover,
            wind_speed,
            humidity,
        )
        prediction = predict_yield(bundle, scenario).iloc[0]
        predicted_yield = float(prediction["normalized_pv_prediction"])
        module_temperature = float(
            estimate_module_temperature(
                np.array([temperature]),
                np.array([radiation]),
                np.array([wind_speed]),
            )[0]
        )

    with output:
        st.markdown("##### Modellergebnis")
        lower, upper = float(prediction["lower_80"]), float(prediction["upper_80"])
        st.metric(
            "Prognostizierte normierte PV-Erzeugung",
            percent(predicted_yield),
            yield_level(predicted_yield).upper(),
        )
        st.altair_chart(
            prediction_chart(predicted_yield, lower, upper),
            width="stretch",
        )
        st.caption(
            "Der dunkle Marker zeigt die Prognose, die orange Linie das empirische "
            "80-%-Intervall."
        )
        a, b = st.columns(2)
        interval_label = f"{lower * 100:.1f}–{upper * 100:.1f} %".replace(".", ",")
        a.metric("Empirisches 80%-Intervall", interval_label)
        b.metric(
            "Geschätzte Modultemperatur",
            f"{module_temperature:.1f} °C".replace(".", ","),
        )
        average_power_mw = predicted_yield * installed_capacity
        st.metric(
            "Geschätzte Erzeugung bei gewählter Leistung",
            f"{average_power_mw:,.0f} MW".replace(",", "."),
            f"{average_power_mw:,.0f} MWh in einer Stunde".replace(",", "."),
        )

        if radiation >= 240 and module_temperature > 45:
            message = (
                "Hohe Einstrahlung, aber thermischer Stress: Der geschätzte "
                "Modultemperaturbereich kann den zusätzlichen Ertrag begrenzen."
            )
        elif radiation >= 220 and 15 <= temperature <= 27:
            message = (
                "Günstige Kombination: starke Einstrahlung trifft auf moderate "
                "Luft- und Modultemperaturen."
            )
        elif radiation < 100 or cloud_cover >= 6:
            message = "Strahlung beziehungsweise Bewölkung begrenzen dieses Szenario deutlich."
        else:
            message = "Gemischte Wetterlage ohne ausgeprägtes Strahlungs- oder Temperatursignal."
        st.markdown(f'<div class="insight">{message}</div>', unsafe_allow_html=True)

with tab_temperature:
    st.subheader("Ertrag bei gleicher Einstrahlung, variierter Temperatur")
    st.caption(
        "Alle übrigen Eingaben aus dem Prognose-Tab bleiben konstant. "
        "Die Kurve zeigt eine Modellreaktion, keinen isolierten kausalen Effekt."
    )
    temperatures = np.linspace(-5, 42, 95)
    curve_frames = [
        scenario_frame(
            selected_day,
            selected_hour,
            radiation,
            float(temp),
            cloud_cover,
            wind_speed,
            humidity,
        )
        for temp in temperatures
    ]
    curve_input = pd.concat(curve_frames, ignore_index=True)
    curve = predict_yield(bundle, curve_input)
    curve_data = pd.DataFrame(
        {
            "Lufttemperatur": temperatures,
            "Normierte Erzeugung": curve["normalized_pv_prediction"] * 100,
            "Modultemperatur": estimate_module_temperature(
                temperatures,
                np.full_like(temperatures, radiation),
                np.full_like(temperatures, wind_speed),
            ),
        }
    )
    temperature_line = (
        alt.Chart(curve_data)
        .mark_line(color="#E19A18", strokeWidth=3)
        .encode(
            x=alt.X("Lufttemperatur:Q", title="Lufttemperatur (°C)"),
            y=alt.Y("Normierte Erzeugung:Q", title="Prognose (%)"),
            tooltip=[
                alt.Tooltip("Lufttemperatur:Q", format=".1f"),
                alt.Tooltip("Modultemperatur:Q", format=".1f"),
                alt.Tooltip("Normierte Erzeugung:Q", format=".1f"),
            ],
        )
        .properties(height=400)
    )
    st.altair_chart(temperature_line, width="stretch")
    best_index = int(curve_data["Normierte Erzeugung"].idxmax())
    best_row = curve_data.iloc[best_index]
    current_row = curve_data.iloc[int(np.abs(temperatures - temperature).argmin())]
    p1, p2, p3 = st.columns(3)
    p1.metric("Bestes Modell-Szenario", f"{best_row['Lufttemperatur']:.1f} °C")
    p2.metric("Prognose dort", f"{best_row['Normierte Erzeugung']:.1f} %")
    p3.metric(
        "Differenz zum aktuellen Szenario",
        f"{best_row['Normierte Erzeugung'] - current_row['Normierte Erzeugung']:+.1f} Prozentpunkte",
    )

with tab_optimum:
    st.subheader("Aktuelles Szenario versus beobachtungsnahes Optimum")
    st.caption(
        "„Optimal“ entspricht den Medianbedingungen der besten 2 % Stunden im "
        "geladenen Datensatz, nicht einem technisch garantierten Anlagenoptimum."
    )
    optimum_scenario = scenario_frame(
        date(2024, 6, 21),
        13,
        optimal_defaults["global_radiation_j_cm2"],
        optimal_defaults["temperature_c"],
        optimal_defaults["cloud_cover_oktas"],
        optimal_defaults["wind_speed_m_s"],
        optimal_defaults["relative_humidity_pct"],
    )
    optimum_prediction = float(
        predict_yield(bundle, optimum_scenario)["normalized_pv_prediction"].iloc[0]
    )
    comparison = pd.DataFrame(
        {
            "Szenario": ["Aktuell", "Top-2%-Bedingungen"],
            "Normierte Erzeugung": [predicted_yield * 100, optimum_prediction * 100],
        }
    )
    bars = (
        alt.Chart(comparison)
        .mark_bar(cornerRadiusTopLeft=7, cornerRadiusTopRight=7, size=70)
        .encode(
            x=alt.X("Szenario:N", title=None),
            y=alt.Y("Normierte Erzeugung:Q", title="Prognose (%)"),
            color=alt.Color(
                "Szenario:N",
                scale=alt.Scale(
                    domain=["Aktuell", "Top-2%-Bedingungen"],
                    range=["#6F9185", "#F2B134"],
                ),
                legend=None,
            ),
            tooltip=["Szenario", alt.Tooltip("Normierte Erzeugung:Q", format=".1f")],
        )
        .properties(height=360)
    )
    st.altair_chart(bars, width="stretch")
    conditions = pd.DataFrame(
        {
            "Merkmal": ["Globalstrahlung", "Lufttemperatur", "Bewölkung", "Wind", "Feuchte"],
            "Top-2%-Median": [
                f"{optimal_defaults['global_radiation_j_cm2']:.0f} J/cm²",
                f"{optimal_defaults['temperature_c']:.1f} °C",
                f"{optimal_defaults['cloud_cover_oktas']:.1f}/8",
                f"{optimal_defaults['wind_speed_m_s']:.1f} m/s",
                f"{optimal_defaults['relative_humidity_pct']:.0f} %",
            ],
        }
    )
    st.dataframe(conditions, hide_index=True, width="stretch")

with tab_analysis:
    st.subheader("Forschungsfrage explorativ prüfen")
    daylight = featured[
        (featured["global_radiation_j_cm2"] > 10) & featured[TARGET].notna()
    ].copy()
    scatter_sample = daylight.sample(min(6_000, len(daylight)), random_state=42)
    scatter_sample["Normierte Erzeugung (%)"] = scatter_sample[TARGET] * 100
    scatter = (
        alt.Chart(scatter_sample)
        .mark_circle(opacity=0.28, size=25)
        .encode(
            x=alt.X("global_radiation_j_cm2:Q", title="Globalstrahlung (J/cm²)"),
            y=alt.Y("Normierte Erzeugung (%):Q"),
            color=alt.Color(
                "estimated_module_temperature_c:Q",
                title="geschätzte Modultemperatur (°C)",
                scale=alt.Scale(scheme="yelloworangered"),
            ),
            tooltip=[
                alt.Tooltip("global_radiation_j_cm2:Q", format=".1f"),
                alt.Tooltip("temperature_c:Q", format=".1f"),
                alt.Tooltip("estimated_module_temperature_c:Q", format=".1f"),
                alt.Tooltip("Normierte Erzeugung (%):Q", format=".1f"),
            ],
        )
        .properties(height=390)
    )
    st.altair_chart(scatter, width="stretch")

    high_radiation_limit = daylight["global_radiation_j_cm2"].quantile(0.75)
    strong_sun = daylight[daylight["global_radiation_j_cm2"] >= high_radiation_limit].copy()
    strong_sun["Temperaturklasse"] = pd.cut(
        strong_sun["temperature_c"],
        [-30, 15, 25, 30, 60],
        labels=["< 15 °C", "15–25 °C", "25–30 °C", "≥ 30 °C"],
    )
    temperature_summary = (
        strong_sun.groupby("Temperaturklasse", observed=True)[TARGET]
        .agg(["count", "mean", "median"])
        .mul({"count": 1, "mean": 100, "median": 100})
        .rename(columns={"count": "Stunden", "mean": "Mittelwert (%)", "median": "Median (%)"})
        .round(2)
    )
    st.markdown(
        f"##### Stunden im oberen Strahlungsquartil (≥ {high_radiation_limit:.0f} J/cm²)"
    )
    st.dataframe(temperature_summary, width="stretch")
    st.markdown(
        '<div class="method-note"><b>Interpretation:</b> Die gemeinsame Variation '
        "von Jahreszeit, Sonnenstand, Bewölkung, räumlicher Aggregation und Temperatur "
        "verhindert eine kausale Deutung einfacher Gruppenmittel. Das Modell erfasst "
        "nicht beobachtete Modultemperaturen nur über eine physikalische Näherung.</div>",
        unsafe_allow_html=True,
    )

with tab_method:
    st.subheader("Modellgüte, Einflussmerkmale und Daten")
    metrics = bundle.metrics
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("MAE Modell", f"{metrics['model_mae'] * 100:.2f} Prozentpunkte")
    m2.metric("RMSE Modell", f"{metrics['model_rmse'] * 100:.2f} Prozentpunkte")
    m3.metric("R² Modell", f"{metrics['model_r2']:.3f}")
    improvement = 1 - metrics["model_mae"] / metrics["baseline_mae"]
    m4.metric("MAE ggü. Median-Baseline", f"{improvement:+.1%}")
    st.caption(
        f"Zeitlicher Test ab {bundle.split_timestamp:%d.%m.%Y}; "
        "keine zufällige Mischung von Vergangenheit und Zukunft."
    )

    labels = {
        "global_radiation_j_cm2": "Globalstrahlung",
        "solar_zenith_angle_deg": "Sonnenzenit",
        "estimated_module_temperature_c": "geschätzte Modultemperatur",
        "temperature_c": "Lufttemperatur",
        "cloud_cover_oktas": "Bewölkung",
        "diffuse_radiation_j_cm2": "Diffusstrahlung",
        "sunshine_duration_min": "Sonnenscheindauer",
        "relative_humidity_pct": "Luftfeuchtigkeit",
        "wind_speed_m_s": "Windgeschwindigkeit",
        "thermal_stress_c": "thermischer Stress",
        "radiation_thermal_interaction": "Strahlung × thermischer Stress",
        "diffuse_share": "Diffusanteil",
        "hour_sin": "Stunde (sin)",
        "hour_cos": "Stunde (cos)",
        "month_sin": "Monat (sin)",
        "month_cos": "Monat (cos)",
    }
    importance = (
        pd.DataFrame(
            {
                "Merkmal": [labels.get(name, name) for name in bundle.feature_importance],
                "Bedeutung": list(bundle.feature_importance.values()),
            }
        )
        .sort_values("Bedeutung", ascending=False)
        .head(10)
    )
    importance_chart = (
        alt.Chart(importance)
        .mark_bar(color="#F2B134", cornerRadiusEnd=4)
        .encode(
            x=alt.X("Bedeutung:Q", title="Permutation Importance (MAE-Anstieg)"),
            y=alt.Y("Merkmal:N", sort="-x", title=None),
            tooltip=["Merkmal", alt.Tooltip("Bedeutung:Q", format=".4f")],
        )
        .properties(height=330)
    )
    st.altair_chart(importance_chart, width="stretch")

    q1, q2, q3 = st.columns(3)
    q1.metric("Stunden", f"{len(data):,}".replace(",", "."))
    q2.metric("Beginn", f"{data['timestamp_utc'].min():%d.%m.%Y}")
    q3.metric("Ende", f"{data['timestamp_utc'].max():%d.%m.%Y}")
    st.markdown(
        """
        **Reproduzierbarer Ablauf**

        1. SMARD-PV-Erzeugung, jährliche PV-Leistung und DWD-ZIPs in `data/raw/` ablegen.
        2. `python scripts/prepare_data.py` erzeugt das stündliche Panel.
        3. `python scripts/create_notebook.py --execute` aktualisiert das Notebook.
        4. `streamlit run app.py` startet diese App.
        """
    )
    st.markdown(
        '<div class="method-note"><b>Zielvariable:</b> PV-Erzeugung in MWh geteilt '
        "durch installierte PV-Leistung in MW und eine Stunde. Die installierte "
        "Leistung wird damit zur Normierung verwendet, nicht als Wetterprädiktor.</div>",
        unsafe_allow_html=True,
    )
    preview = data.tail(200).copy()
    preview["timestamp_utc"] = preview["timestamp_utc"].astype(str)
    st.dataframe(preview, width="stretch", hide_index=True)
    st.download_button(
        "Datenausschnitt als CSV",
        preview.to_csv(index=False).encode("utf-8"),
        "pv_weather_preview.csv",
        "text/csv",
    )
    st.markdown(
        """
        **Amtliche Quellen:** [SMARD – Marktdaten](https://www.smard.de/home/downloadcenter/download-marktdaten/)
        · [DWD – Solarstrahlung](https://opendata.dwd.de/climate_environment/CDC/observations_germany/climate/hourly/solar/)
        · [DWD – Temperatur](https://opendata.dwd.de/climate_environment/CDC/observations_germany/climate/hourly/air_temperature/historical/)
        · [DWD – Bewölkung](https://opendata.dwd.de/climate_environment/CDC/observations_germany/climate/hourly/cloudiness/historical/)
        · [DWD – Wind](https://opendata.dwd.de/climate_environment/CDC/observations_germany/climate/hourly/wind/historical/)
        """
    )

st.divider()
st.caption(
    "Photovoltaikeffizienz Rechner · Forschungsprototyp, keine Ertragsgarantie · "
    "SMARD-Namensnennung bei Realdaten: Bundesnetzagentur | SMARD.de"
)
