"""Photovoltaikeffizienz Rechner – Streamlit application."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
from streamlit_option_menu import option_menu


ROOT = Path(__file__).resolve().parent
PROCESSED_DATA_PATH = ROOT / "data" / "processed" / "hourly_pv_weather.csv"
MODEL_SOURCE_PATHS = (
    ROOT / "pv_weather" / "features.py",
    ROOT / "pv_weather" / "modeling.py",
)
SCENARIO_REFERENCE_DAY = date(2024, 7, 1)
SCENARIO_REFERENCE_HOUR = 13
MODULE_TEMPERATURE_HELP = (
    "Die Modultemperatur wird nicht gemessen, sondern NOCT-artig geschätzt: "
    "T_modul ≈ T_luft + 0,03125 × (Globalstrahlung / 0,36) "
    "/ (1 + 0,12 × Wind). Globalstrahlung in J/cm² wird dabei durch 0,36 "
    "in eine mittlere Einstrahlung in W/m² umgerechnet. Deshalb steigt die "
    "geschätzte Modultemperatur bei stärkerer Einstrahlung automatisch mit an; "
    "sie ist in diesem Diagramm keine unabhängige Messgröße."
)
from pv_weather import (  # noqa: E402
    TARGET,
    add_features,
    estimate_module_temperature,
    load_project_data,
    predict_yield,
    train_yield_model,
)
from pv_weather.workflow import MIN_DOWNLOAD_YEAR, refresh_real_data  # noqa: E402


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
    div[data-testid="stMetricValue"] {color:#173C34;
      font-size:clamp(1.55rem,2.4vw,2.25rem); white-space:normal;}
    div[data-testid="stMetricValue"] > div {overflow:visible; text-overflow:clip;
      white-space:normal;}
    .insight {border-left:4px solid #F2B134; background:#FFF9E8; padding:1rem 1.1rem;
      border-radius:0 12px 12px 0; color:#314F46;}
    .method-note {background:#E8F1ED; padding:1rem 1.1rem; border-radius:12px; color:#314F46;}
    </style>
    """,
    unsafe_allow_html=True,
)

# CSS für die tab-ähnliche option_menu-Navigation, angepasst an das bestehende Farbschema.
OPTION_MENU_STYLES = {
    "container": {
        "padding": "0",
        "background-color": "transparent",
        "border-bottom": "2px solid #D8E2DD",
        "margin-bottom": "1.5rem",
    },
    "icon": {"color": "#567064", "font-size": "15px"},
    "nav-link": {
        "font-size": "0.95rem",
        "font-weight": "600",
        "color": "#4A625A",
        "text-align": "center",
        "margin": "0 4px",
        "padding": "0.7rem 1.1rem",
        "border-radius": "10px 10px 0 0",
        "--hover-color": "#EEF3F0",
    },
    "nav-link-selected": {
        "background-color": "#FFF9E8",
        "color": "#173C34",
        "border-bottom": "3px solid #F2B134",
        "font-weight": "750",
    },
}


@st.cache_data(show_spinner=False)
def get_data(data_version: int | None) -> tuple[pd.DataFrame, str]:
    """Load data; the file timestamp invalidates a previously cached demo."""
    del data_version
    return load_project_data(PROCESSED_DATA_PATH)


@st.cache_resource(show_spinner="PV-Modell wird zeitlich validiert …")
def get_model(data: pd.DataFrame, model_version: tuple[int, ...]):
    del model_version
    return train_yield_model(data)


try:
    data_version = (
        PROCESSED_DATA_PATH.stat().st_mtime_ns
        if PROCESSED_DATA_PATH.exists()
        else None
    )
    data, source_label = get_data(data_version)
    model_version = tuple(path.stat().st_mtime_ns for path in MODEL_SOURCE_PATHS)
    downloaded_model = st.session_state.get("_downloaded_real_data_model")
    if (
        downloaded_model
        and downloaded_model["data_version"] == data_version
        and downloaded_model["model_version"] == model_version
    ):
        bundle = downloaded_model["bundle"]
    else:
        bundle = get_model(data, model_version)
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
    radiation: float,
    temperature: float,
    cloud_cover: float,
    wind_speed: float,
    humidity: float,
) -> pd.DataFrame:
    timestamp = (
        pd.Timestamp(selected_day)
        .replace(hour=SCENARIO_REFERENCE_HOUR)
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
            "solar_zenith_angle_deg": [
                solar_zenith(selected_day, SCENARIO_REFERENCE_HOUR)
            ],
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


def chart_interpretation(text: str) -> None:
    with st.expander("Interpretation zum Diagramm", expanded=False):
        st.markdown(text)


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

# ---------------------------------------------------------------------------
# Navigation: option_menu (horizontal, auf der Hauptseite) sieht wie echte
# Tabs aus, gibt im Gegensatz zu st.tabs() aber den aktiven Wert zurück.
# Damit weiß das Skript, welche Ansicht aktiv ist, und kann das
# Meteorologische Szenario gezielt nur dort in der Sidebar einblenden.
# ---------------------------------------------------------------------------
PAGES = [
    "Prognose",
    "Thermischer Effekt",
    "Optimale Bedingungen",
    "Datenexploration",
    "Über die App",
]
SCENARIO_PAGES = {"Prognose", "Thermischer Effekt", "Optimale Bedingungen"}
PAGE_ICONS = ["sliders", "thermometer-half", "trophy", "bar-chart-line", "gear"]

page = option_menu(
    menu_title=None,
    options=PAGES,
    icons=PAGE_ICONS,
    orientation="horizontal",
    styles=OPTION_MENU_STYLES,
)

scenario = prediction = None
predicted_yield = module_temperature = None
radiation = temperature = None
cloud_cover = wind_speed = humidity = installed_capacity = None

TEMPERATURE_VALUE_KEY = "_scenario_temperature_c"
ACTIVE_TEMPERATURE_SLIDER_KEY = "_active_temperature_slider"
THERMAL_TEMPERATURE_SLIDER_KEY = "_thermal_temperature_slider"

if TEMPERATURE_VALUE_KEY not in st.session_state:
    st.session_state[TEMPERATURE_VALUE_KEY] = 25.0


def remember_scenario_temperature() -> None:
    st.session_state[TEMPERATURE_VALUE_KEY] = float(
        st.session_state[ACTIVE_TEMPERATURE_SLIDER_KEY]
    )


if page in SCENARIO_PAGES:
    st.sidebar.markdown("### Meteorologisches Szenario")
    radiation = st.sidebar.slider("Globalstrahlung (J/cm²)", 0.0, 360.0, 270.0, 5.0)
    if page == "Thermischer Effekt":
        st.session_state[THERMAL_TEMPERATURE_SLIDER_KEY] = float(
            st.session_state[TEMPERATURE_VALUE_KEY]
        )
        temperature = st.sidebar.slider(
            "Lufttemperatur (°C)",
            min_value=-10.0,
            max_value=42.0,
            step=0.5,
            key=THERMAL_TEMPERATURE_SLIDER_KEY,
            disabled=True,
            help=(
                "Im Tab „Thermischer Effekt“ wird die Lufttemperatur automatisch "
                "über die gesamte Kurve variiert."
            ),
        )
    else:
        if ACTIVE_TEMPERATURE_SLIDER_KEY not in st.session_state:
            st.session_state[ACTIVE_TEMPERATURE_SLIDER_KEY] = float(
                st.session_state[TEMPERATURE_VALUE_KEY]
            )
        temperature = st.sidebar.slider(
            "Lufttemperatur (°C)",
            min_value=-10.0,
            max_value=42.0,
            step=0.5,
            key=ACTIVE_TEMPERATURE_SLIDER_KEY,
            on_change=remember_scenario_temperature,
        )
    cloud_cover = st.sidebar.slider("Bewölkungsgrad (Achtel)", 0.0, 8.0, 2.0, 0.5)
    wind_speed = st.sidebar.slider("Windgeschwindigkeit (m/s)", 0.0, 15.0, 3.0, 0.5)
    humidity = st.sidebar.slider("Relative Luftfeuchtigkeit (%)", 10, 100, 55)
    if page == "Prognose":
        with st.sidebar.container(border=True):
            st.markdown("#### Leistungsskalierung")
            installed_capacity = st.slider(
                "Installierte PV-Leistung (MW)", 1_000, 200_000, 90_000, 1_000
            )
            st.caption(
                "Skaliert nur die absolute Erzeugung in MW/MWh, nicht die "
                "normierte Modellprognose."
            )

    scenario = scenario_frame(
        SCENARIO_REFERENCE_DAY,
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

# ---------------------------------------------------------------------------
# Seiteninhalte
# ---------------------------------------------------------------------------

if page == "Prognose":
    lower, upper = float(prediction["lower_80"]), float(prediction["upper_80"])
    st.metric(
        "Prognostizierte normierte PV-Erzeugung",
        percent(predicted_yield),
        yield_level(predicted_yield).upper(),
    )
    st.markdown("##### Prognose und Unsicherheitsbereich")
    st.altair_chart(
        prediction_chart(predicted_yield, lower, upper),
        width="stretch",
    )
    chart_interpretation(
        "Der dunkle Marker kennzeichnet die Modellprognose. Die orange Linie zeigt "
        "den aus den Abweichungen im Testzeitraum abgeleiteten 80-%-"
        "Unsicherheitsbereich. Eine größere Spannweite weist auf eine unsicherere "
        "Schätzung hin. Die farbigen Bereiche ordnen die prognostizierte normierte "
        "Erzeugung als niedrig, mittel oder hoch ein."
    )
    a, b = st.columns(2)
    interval_label = f"{lower * 100:.1f}–{upper * 100:.1f} %".replace(".", ",")
    a.metric("Empirisches 80%-Intervall", interval_label)
    b.metric(
        "Geschätzte Modultemperatur",
        f"{module_temperature:.1f} °C".replace(".", ","),
        help=MODULE_TEMPERATURE_HELP,
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
            "Lufttemperaturen."
        )
    elif radiation < 100 or cloud_cover >= 6:
        message = "Strahlung beziehungsweise Bewölkung begrenzen dieses Szenario deutlich."
    else:
        message = "Gemischte Wetterlage ohne ausgeprägtes Strahlungs- oder Temperatursignal."
    st.markdown(f'<div class="insight">{message}</div>', unsafe_allow_html=True)

elif page == "Thermischer Effekt":
    st.subheader("Ertrag bei konstanter Einstrahlung und variierter Temperatur")
    st.caption(
        "Alle übrigen Eingaben aus der Szenario-Seitenleiste bleiben konstant. "
        "Die Kurve zeigt eine Modellreaktion, keinen isolierten kausalen Effekt."
    )
    temperatures = np.linspace(-5, 42, 95)
    curve_frames = [
        scenario_frame(
            SCENARIO_REFERENCE_DAY,
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
            y=alt.Y(
                "Normierte Erzeugung:Q",
                title="Normierte PV-Erzeugung (%)",
            ),
            tooltip=[
                alt.Tooltip("Lufttemperatur:Q", format=".1f"),
                alt.Tooltip("Modultemperatur:Q", format=".1f"),
                alt.Tooltip("Normierte Erzeugung:Q", format=".1f"),
            ],
        )
    )
    st.markdown("##### Temperatur-Sensitivität bei gleicher Einstrahlung")
    st.altair_chart(
        temperature_line.properties(height=400),
        width="stretch",
    )
    chart_interpretation(
        "Die Linie zeigt, wie sich die normierte Modellprognose verändert, wenn "
        "nur die Lufttemperatur variiert. Alle übrigen Szenarioeingaben bleiben "
        "konstant. Sinkt die Kurve bei höheren Temperaturen, berücksichtigt das "
        "Modell einen möglichen thermischen Leistungsverlust. Die Darstellung allein "
        "belegt jedoch noch keine Ursache-Wirkungs-Beziehung."
    )

elif page == "Optimale Bedingungen":
    st.subheader("Aktuelles Szenario versus beobachtungsnahes Optimum")
    st.caption(
        "„Optimal“ entspricht den Medianbedingungen der besten 2 % Stunden im "
        "geladenen Datensatz, nicht einem technisch garantierten Anlagenoptimum."
    )
    optimum_scenario = scenario_frame(
        date(2024, 6, 21),
        optimal_defaults["global_radiation_j_cm2"],
        optimal_defaults["temperature_c"],
        optimal_defaults["cloud_cover_oktas"],
        optimal_defaults["wind_speed_m_s"],
        optimal_defaults["relative_humidity_pct"],
    )
    optimum_prediction = float(
        predict_yield(bundle, optimum_scenario)["normalized_pv_prediction"].iloc[0]
    )
    current_prediction_pct = predicted_yield * 100
    optimum_prediction_pct = optimum_prediction * 100
    prediction_difference = current_prediction_pct - optimum_prediction_pct
    o1, o2, o3 = st.columns(3)
    o1.metric(
        "Aktuelles Szenario",
        f"{current_prediction_pct:.1f} %".replace(".", ","),
    )
    o2.metric(
        "Top-2-%-Bedingungen",
        f"{optimum_prediction_pct:.1f} %".replace(".", ","),
    )
    o3.metric(
        "Differenz zu den Top-2-%-Bedingungen",
        f"{prediction_difference:+.1f}".replace(".", ",") + " %-Pkt.",
        help=(
            "Aktuelles Szenario minus Top-2-%-Bedingungen. Ein negativer Wert "
            "bedeutet, dass das aktuelle Szenario darunter liegt."
        ),
    )

    condition_specs = [
        {
            "Merkmal": "Globalstrahlung",
            "Einheit": "J/cm²",
            "Minimum": 0.0,
            "Maximum": 360.0,
            "Aktuell": radiation,
            "Top-2-%-Median": optimal_defaults["global_radiation_j_cm2"],
            "Nachkommastellen": 0,
        },
        {
            "Merkmal": "Lufttemperatur",
            "Einheit": "°C",
            "Minimum": -10.0,
            "Maximum": 42.0,
            "Aktuell": temperature,
            "Top-2-%-Median": optimal_defaults["temperature_c"],
            "Nachkommastellen": 1,
        },
        {
            "Merkmal": "Bewölkung",
            "Einheit": "/8",
            "Minimum": 0.0,
            "Maximum": 8.0,
            "Aktuell": cloud_cover,
            "Top-2-%-Median": optimal_defaults["cloud_cover_oktas"],
            "Nachkommastellen": 1,
        },
        {
            "Merkmal": "Windgeschwindigkeit",
            "Einheit": "m/s",
            "Minimum": 0.0,
            "Maximum": 15.0,
            "Aktuell": wind_speed,
            "Top-2-%-Median": optimal_defaults["wind_speed_m_s"],
            "Nachkommastellen": 1,
        },
        {
            "Merkmal": "Relative Luftfeuchtigkeit",
            "Einheit": "%",
            "Minimum": 10.0,
            "Maximum": 100.0,
            "Aktuell": humidity,
            "Top-2-%-Median": optimal_defaults["relative_humidity_pct"],
            "Nachkommastellen": 0,
        },
    ]

    condition_charts = []
    for spec in condition_specs:
        condition_rows = []
        for scenario_name in ["Aktuell", "Top-2-%-Median"]:
            value = float(spec[scenario_name])
            digits = int(spec["Nachkommastellen"])
            number = f"{value:.{digits}f}".replace(".", ",")
            unit = str(spec["Einheit"])
            display_value = (
                f"{number}{unit}" if unit == "/8" else f"{number} {unit}"
            )
            condition_rows.append(
                {
                    "Szenario": scenario_name,
                    "Wert": value,
                    "Wertanzeige": display_value,
                }
            )
        condition_values = pd.DataFrame(condition_rows)
        condition_base = alt.Chart(condition_values).encode(
            x=alt.X(
                "Wert:Q",
                title=str(spec["Einheit"]),
                scale=alt.Scale(
                    domain=[float(spec["Minimum"]), float(spec["Maximum"])]
                ),
            ),
            y=alt.Y(
                "Szenario:N",
                title=None,
                sort=["Aktuell", "Top-2-%-Median"],
                axis=alt.Axis(labelLimit=180, labelPadding=8),
            ),
            color=alt.Color(
                "Szenario:N",
                scale=alt.Scale(
                    domain=["Aktuell", "Top-2-%-Median"],
                    range=["#6F9185", "#F2B134"],
                ),
                legend=alt.Legend(
                    title=None,
                    orient="top",
                    direction="horizontal",
                ),
            ),
            tooltip=[
                alt.Tooltip("Szenario:N"),
                alt.Tooltip("Wertanzeige:N", title=str(spec["Merkmal"])),
            ],
        )
        condition_points = condition_base.mark_circle(size=150)
        condition_labels = condition_base.mark_text(
            dy=-14,
            color="#173C34",
            fontSize=12,
            fontWeight="bold",
        ).encode(text="Wertanzeige:N")
        condition_charts.append(
            (condition_points + condition_labels).properties(
                height=86,
                title=str(spec["Merkmal"]),
            )
        )

    condition_comparison = (
        alt.vconcat(*condition_charts, spacing=12)
        .resolve_scale(color="shared")
        .configure_axis(labelFontSize=12, titleFontSize=12)
        .configure_title(
            anchor="start",
            color="#173C34",
            fontSize=14,
            fontWeight="bold",
        )
    )
    st.markdown("##### Wetterbedingungen im direkten Vergleich")
    st.altair_chart(condition_comparison, width="stretch")
    chart_interpretation(
        "Grün zeigt die aktuellen Reglerwerte, Gelb den Median der besten 2 % "
        "beobachteten Stunden. Jede Wettergröße besitzt wegen ihrer unterschiedlichen "
        "Einheit eine eigene, am jeweiligen Reglerbereich orientierte Skala. Deshalb "
        "lassen sich die horizontalen Abstände nur innerhalb derselben Wettergröße "
        "vergleichen. Die Gegenüberstellung beschreibt typische sehr gute "
        "Beobachtungen, aber kein technisch garantiertes Anlagenoptimum."
    )

elif page == "Datenexploration":
    st.subheader("Forschungsfrage explorativ prüfen")
    st.markdown("##### Umfang des ausgewerteten Datensatzes")
    q1, q2, q3 = st.columns(3)
    q1.metric(
        "Stunden",
        f"{len(data):,}".replace(",", "."),
    )
    q2.metric(
        "Beginn",
        f"{data['timestamp_utc'].min():%d.%m.%Y}",
    )
    q3.metric(
        "Ende",
        f"{data['timestamp_utc'].max():%d.%m.%Y}",
    )
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
                title="Geschätzte Modultemperatur (°C)",
                scale=alt.Scale(scheme="yelloworangered"),
                legend=alt.Legend(
                    orient="bottom",
                    direction="horizontal",
                    gradientLength=360,
                    gradientThickness=14,
                    titleLimit=320,
                    labelLimit=80,
                    titleFontSize=12,
                    labelFontSize=11,
                ),
            ),
            tooltip=[
                alt.Tooltip("global_radiation_j_cm2:Q", format=".1f"),
                alt.Tooltip("temperature_c:Q", format=".1f"),
                alt.Tooltip("estimated_module_temperature_c:Q", format=".1f"),
                alt.Tooltip("Normierte Erzeugung (%):Q", format=".1f"),
            ],
        )
        .properties(height=390)
        .configure_axis(labelFontSize=12, titleFontSize=13)
    )
    st.markdown(
        "##### Einstrahlung, Modultemperatur und PV-Erzeugung",
        help=MODULE_TEMPERATURE_HELP,
    )
    st.altair_chart(scatter, width="stretch")
    chart_interpretation(
        "Jeder Punkt repräsentiert eine Tageslichtstunde. Die horizontale Position "
        "zeigt die Globalstrahlung, die vertikale Position die normierte "
        "PV-Erzeugung. Die Farbe kennzeichnet die geschätzte Modultemperatur. "
        "Starke Einstrahlung erhöht häufig sowohl die Erzeugung als auch die "
        "Modultemperatur; daraus folgt nicht, dass Wärme die Leistung verbessert. "
        "Der Temperatureffekt lässt sich erst bei vergleichbarer Einstrahlung "
        "sinnvoll beurteilen, beispielsweise im Tab „Thermischer Effekt“."
    )

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
        .reset_index()
    )
    temperature_summary["Stunden (Beschriftung)"] = temperature_summary[
        "Stunden"
    ].map(lambda value: f"{int(value):,}".replace(",", "."))
    strong_sun["Normierte Erzeugung (%)"] = strong_sun[TARGET] * 100
    temperature_boxplot = (
        alt.Chart(strong_sun)
        .mark_boxplot(
            extent=1.5,
            size=54,
            color="#F2B134",
            median=alt.MarkConfig(color="#173C34", strokeWidth=2),
        )
        .encode(
            x=alt.X(
                "Temperaturklasse:N",
                title="Lufttemperaturklasse",
                sort=["< 15 °C", "15–25 °C", "25–30 °C", "≥ 30 °C"],
                axis=alt.Axis(labelAngle=0, labelLimit=140),
            ),
            y=alt.Y(
                "Normierte Erzeugung (%):Q",
                title="Normierte PV-Erzeugung (%)",
                scale=alt.Scale(zero=True),
            ),
        )
        .properties(height=360)
        .configure_axis(labelFontSize=12, titleFontSize=13)
    )

    temperature_order = ["< 15 °C", "15–25 °C", "25–30 °C", "≥ 30 °C"]
    count_bars = (
        alt.Chart(temperature_summary)
        .mark_bar(color="#6F9185", cornerRadiusTopLeft=6, cornerRadiusTopRight=6)
        .encode(
            x=alt.X(
                "Temperaturklasse:N",
                title="Lufttemperaturklasse",
                sort=temperature_order,
                axis=alt.Axis(labelAngle=0, labelLimit=140),
            ),
            y=alt.Y(
                "Stunden:Q",
                title="Anzahl beobachteter Stunden",
                scale=alt.Scale(zero=True),
            ),
            tooltip=[
                alt.Tooltip("Temperaturklasse:N", title="Temperaturklasse"),
                alt.Tooltip("Stunden:Q", title="Stunden", format=".0f"),
                alt.Tooltip("Mittelwert (%):Q", title="Mittelwert", format=".2f"),
                alt.Tooltip("Median (%):Q", title="Median", format=".2f"),
            ],
        )
    )
    count_labels = (
        alt.Chart(temperature_summary)
        .mark_text(dy=-10, color="#173C34", fontSize=13, fontWeight="bold")
        .encode(
            x=alt.X("Temperaturklasse:N", sort=temperature_order),
            y=alt.Y("Stunden:Q"),
            text="Stunden (Beschriftung):N",
        )
    )
    temperature_counts_chart = (
        (count_bars + count_labels)
        .properties(height=360)
        .configure_axis(labelFontSize=12, titleFontSize=13)
    )

    temperature_chart_column, count_chart_column = st.columns(2, gap="large")
    with temperature_chart_column:
        st.markdown(
            f"##### Temperaturvergleich bei starker Einstrahlung "
            f"(≥ {high_radiation_limit:.0f} J/cm²)"
        )
        st.caption(
            "Die Boxen enthalten die mittleren 50 % der Werte; die dunkle Linie "
            "kennzeichnet den Median."
        )
        st.altair_chart(temperature_boxplot, width="stretch")
        chart_interpretation(
            "Die dunkle Linie in jeder Box ist der Median, die Box enthält die "
            "mittleren 50 % der beobachteten Werte. Je größer die Box und je weiter "
            "die äußeren Begrenzungslinien auseinanderliegen, desto stärker schwankt "
            "die Erzeugung innerhalb der Temperaturklasse. Da gleichzeitig weitere "
            "Wetterbedingungen variieren, belegen Unterschiede zwischen den Klassen "
            "allein keine direkte Temperaturwirkung."
        )

    with count_chart_column:
        st.markdown("##### Verfügbare Stunden je Temperaturklasse")
        st.caption(
            "Die Zahlen zeigen die Datengrundlage des links dargestellten "
            "Temperaturvergleichs."
        )
        st.altair_chart(temperature_counts_chart, width="stretch")
        chart_interpretation(
            "Jeder Balken zeigt, wie viele Stunden mit starker Einstrahlung für die "
            "jeweilige Temperaturklasse vorliegen. Die genaue Anzahl steht über dem "
            "Balken; Mittelwert und Median der normierten Erzeugung erscheinen beim "
            "Überfahren. Ergebnisse aus Klassen mit wenigen Beobachtungen – "
            "insbesondere ab 30 °C – sind stärker von einzelnen Stunden abhängig und "
            "daher mit größerer Vorsicht zu interpretieren."
        )

    valid_temperature_summary = temperature_summary.dropna(
        subset=["Median (%)", "Stunden"]
    )
    if not valid_temperature_summary.empty:
        best_temperature_row = valid_temperature_summary.loc[
            valid_temperature_summary["Median (%)"].idxmax()
        ]
        best_temperature_class = str(best_temperature_row["Temperaturklasse"])
        best_temperature_median = float(best_temperature_row["Median (%)"])
        best_median_display = f"{best_temperature_median:.2f}".replace(".", ",")
        result_text = (
            f"Für Stunden mit mindestens {high_radiation_limit:.0f} J/cm² erreicht "
            f"die Temperaturklasse {best_temperature_class} den höchsten beobachteten "
            f"Median der normierten PV-Erzeugung ({best_median_display} %)."
        )

        hot_rows = valid_temperature_summary[
            valid_temperature_summary["Temperaturklasse"].astype("object")
            == "≥ 30 °C"
        ]
        if not hot_rows.empty:
            hot_row = hot_rows.iloc[0]
            hot_median = float(hot_row["Median (%)"])
            hot_hours = int(hot_row["Stunden"])
            median_gap = best_temperature_median - hot_median
            if median_gap > 0.05:
                median_gap_display = f"{median_gap:.2f}".replace(".", ",")
                hot_hours_display = f"{hot_hours:,}".replace(",", ".")
                result_text += (
                    f" Ab 30 °C liegt der Median um {median_gap_display} "
                    "Prozentpunkte darunter; für diese Klasse stehen "
                    f"{hot_hours_display} Stunden zur Verfügung."
                )
            elif median_gap < -0.05:
                median_gap_display = f"{abs(median_gap):.2f}".replace(".", ",")
                hot_hours_display = f"{hot_hours:,}".replace(",", ".")
                result_text += (
                    f" Ab 30 °C liegt der Median um {median_gap_display} "
                    f"Prozentpunkte darüber; ein Rückgang ist in dieser gruppierten "
                    f"Auswertung somit nicht erkennbar. Die Klasse umfasst "
                    f"{hot_hours_display} Stunden."
                )
            else:
                hot_hours_display = f"{hot_hours:,}".replace(",", ".")
                result_text += (
                    f" Ab 30 °C unterscheidet sich der Median kaum. Für diese Klasse "
                    f"stehen {hot_hours_display} Stunden zur Verfügung."
                )
        result_text += (
            " Die Auswertung beschreibt gemeinsame Beobachtungen und belegt allein "
            "noch keine direkte Temperaturwirkung."
        )
        st.markdown("##### Kurzfazit zur Forschungsfrage")
        st.info(result_text, icon="📌")

elif page == "Über die App":
    st.subheader("Modellgüte, Methodik und Daten")
    metrics = bundle.metrics
    m1, m2, m3, m4 = st.columns(4)
    m1.metric(
        "MAE Modell",
        f"{metrics['model_mae'] * 100:.2f}".replace(".", ",") + " %-Pkt.",
        help=(
            "MAE steht für „Mean Absolute Error“, also mittlerer absoluter Fehler. "
            "Der Wert gibt an, um wie viele Prozentpunkte die Prognose im Durchschnitt "
            "vom tatsächlich beobachteten Wert abweicht. Kleiner ist besser."
        ),
    )
    m2.metric(
        "RMSE Modell",
        f"{metrics['model_rmse'] * 100:.2f}".replace(".", ",") + " %-Pkt.",
        help=(
            "RMSE steht für „Root Mean Squared Error“. Diese Fehlerkennzahl gewichtet "
            "große Abweichungen stärker als kleine. Kleiner ist besser. Liegt der RMSE "
            "deutlich über dem MAE, gibt es einzelne größere Prognosefehler."
        ),
    )
    m3.metric(
        "R² Modell",
        f"{metrics['model_r2']:.3f}".replace(".", ","),
        help=(
            "R² heißt Bestimmtheitsmaß. Ein Wert nahe 1 bedeutet, dass das Modell "
            "einen großen Anteil der beobachteten Schwankungen im Testzeitraum erklärt. "
            "R² ist keine Trefferquote und bedeutet nicht „Prozent richtige Prognosen“."
        ),
    )
    improvement = 1 - metrics["model_mae"] / metrics["baseline_mae"]
    m4.metric(
        "Fehlerreduktion gegenüber Vergleich",
        f"{improvement:+.1%}".replace(".", ","),
        help=(
            "Verglichen wird mit einer sehr einfachen Prognose, die immer den mittleren "
            "typischen Wert der Trainingsdaten verwendet. Ein positiver Wert bedeutet, "
            "dass das Modell einen entsprechend kleineren durchschnittlichen Fehler hat."
        ),
    )
    st.caption(
        f"Zeitlicher Test ab {bundle.split_timestamp:%d.%m.%Y}; "
        "keine zufällige Mischung von Vergangenheit und Zukunft."
    )

    with st.expander("Reproduzierbarer technischer Ablauf"):
        st.markdown(
            """
            1. SMARD-PV-Erzeugung, jährliche PV-Leistung und DWD-ZIPs in
               `data/raw/` ablegen.
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
    st.markdown("### Datengrundlage und Realdaten")
    pill_color = "#8A4B08" if is_demo else "#176B4D"
    source_display = (
        "Synthetische Demodaten" if is_demo else "Realdaten aus SMARD und DWD"
    )
    st.markdown(
        f'<span class="source-pill" style="color:{pill_color}">{source_display}</span>',
        unsafe_allow_html=True,
    )
    if is_demo:
        st.warning(
            "Demo-Modus: Die Daten bilden den erwarteten Strahlungs- und "
            "Temperatureffekt synthetisch ab. Sie sind kein empirischer Befund.",
            icon="🧪",
        )

    update_summary = st.session_state.get("_real_data_update_summary")
    if update_summary:
        trained_rows = f"{update_summary['row_count']:,}".replace(",", ".")
        st.success(
            f"Realdaten für {update_summary['start_year']}–"
            f"{update_summary['end_year']} wurden geladen. Das Modell wurde mit "
            f"{trained_rows} Stunden neu trainiert.",
            icon="✅",
        )

    latest_download_year = date.today().year - 1
    default_start_year = max(MIN_DOWNLOAD_YEAR, latest_download_year - 5)
    with st.expander(
        "Realdaten herunterladen und Modell neu trainieren",
        expanded=is_demo,
    ):
        st.markdown(
            "Die App lädt die realisierte PV-Erzeugung und installierte Leistung "
            "von SMARD sowie Wetterdaten räumlich verteilter DWD-Stationen. "
            "Anschließend wird das gemeinsame Stundenpanel erzeugt und das Modell "
            "automatisch neu trainiert."
        )
        selected_start_year, selected_end_year = st.slider(
            "Auswertungszeitraum",
            min_value=MIN_DOWNLOAD_YEAR,
            max_value=latest_download_year,
            value=(default_start_year, latest_download_year),
            step=1,
        )
        st.caption(
            f"Wählbar sind vollständige Kalenderjahre von {MIN_DOWNLOAD_YEAR} bis "
            f"{latest_download_year}. Je nach Zeitraum und Verbindung kann der "
            "Vorgang mehrere Minuten dauern."
        )
        if st.button(
            "Download und Training starten",
            type="primary",
            icon=":material/download:",
        ):
            download_status = st.status(
                "Realdaten werden vorbereitet …",
                expanded=True,
            )
            try:
                update = refresh_real_data(
                    ROOT,
                    selected_start_year,
                    selected_end_year,
                    on_progress=download_status.write,
                )
            except Exception as exc:
                download_status.update(
                    label="Download oder Training fehlgeschlagen",
                    state="error",
                    expanded=True,
                )
                st.error(
                    "Der bisherige Datensatz bleibt erhalten. "
                    f"Fehlerdetails: {exc}"
                )
            else:
                st.session_state["_downloaded_real_data_model"] = {
                    "data_version": update.processed_version,
                    "model_version": model_version,
                    "bundle": update.model,
                }
                st.session_state["_real_data_update_summary"] = {
                    "start_year": update.start_year,
                    "end_year": update.end_year,
                    "row_count": update.row_count,
                    "station_count": update.station_count,
                }
                get_data.clear()
                download_status.update(
                    label="Download und Modelltraining abgeschlossen",
                    state="complete",
                    expanded=False,
                )
                st.rerun()

    st.warning(
        """
        **Grenzen des ungewichteten Stationsmittels**

        Ein ungewichteter Mittelwert über alle Wetterstationen erzeugt zwar pro
        Stunde einen deutschlandweiten Wetterwert, hat aber mehrere Probleme:

        - Regionen mit vielen Messstationen erhalten automatisch mehr Gewicht.
        - Die installierte PV-Leistung ist regional sehr unterschiedlich verteilt.
        - Die Wetterbedingungen können sich zwischen Nord- und Süddeutschland
          stark unterscheiden.
        - Extremwerte und regionale Unterschiede werden durch den Mittelwert
          „glattgebügelt“.
        - Unterschiedlich viele fehlende Stationswerte können die Zusammensetzung
          und damit den Mittelwert im Zeitverlauf verändern.
        """,
        icon="⚠️",
    )

    st.markdown("##### Vorschau des verwendeten aufbereiteten Datensatzes")
    if is_demo:
        st.caption(
            "Die Tabelle zeigt die letzten 200 Zeilen des aktuell verwendeten "
            "synthetischen Demodatensatzes."
        )
    else:
        st.caption(
            "Die Tabelle zeigt die letzten 200 Zeilen aus "
            "`data/processed/hourly_pv_weather.csv`. Aus diesem aufbereiteten "
            "Datensatz stammen auch die Trainings- und Testdaten des Modells."
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
