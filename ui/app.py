from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Tuple

import geopandas as gpd
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yaml

from src.inference import FloodPredictor
from src.postprocess_notebook_compat import export_all_notebook_artefacts


CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "system_config.yaml"


@st.cache_data
def load_config(config_path: Path) -> Dict[str, Any]:
    """
    Load the system configuration from a YAML file.

    Parameters
    ----------
    config_path : Path
        Path to the YAML configuration file.

    Returns
    -------
    Dict[str, Any]
        Parsed configuration dictionary.
    """
    with config_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


@st.cache_resource
def get_predictor() -> FloodPredictor:
    """
    Lazily construct the FloodPredictor and load PyTorch models once per session.
    """
    config = load_config(CONFIG_PATH)
    return FloodPredictor(config, config_path=CONFIG_PATH)


@st.cache_data
def get_available_members() -> List[str]:
    """
    Return the list of available ICON ensemble member names.
    """
    predictor = get_predictor()
    members = predictor._loader.list_icon_members()  # type: ignore[attr-defined]
    names: List[str] = [
        p.stem.replace("rain_kriged_ICON_ENS_", "").replace("_", " ") for p in members
    ]
    return names


@st.cache_data
def run_predictions_all_members() -> pd.DataFrame:
    """
    Run the full ensemble prediction once and cache the results.

    Returns
    -------
    pandas.DataFrame
        Concatenated predictions for all members.
    """
    predictor = get_predictor()
    return predictor.run_all_members()


@st.cache_data
def load_rain_for_member(member_name: str) -> pd.DataFrame:
    """
    Load raw ICON rain data for a specific ensemble member.

    Parameters
    ----------
    member_name : str
        Human-readable member identifier (e.g. ``'2026011200 mem01'``).
    """
    predictor = get_predictor()
    all_members = predictor._loader.load_all_icon_members()  # type: ignore[attr-defined]
    return all_members[member_name]


@st.cache_data
def get_station_locations() -> pd.DataFrame:
    """
    Extract station locations (lat/lon) for the configured station cells.

    The coordinates are derived from the UGRID GeoParquet geometry and
    transformed to WGS84 for mapping.
    """
    config = load_config(CONFIG_PATH)
    data_paths = config.get("data_paths", {})
    stations_cfg = config.get("stations", {})
    ugrid_path = Path(str(data_paths.get("ugrid_parquet_path", ""))).expanduser()

    station_cells: List[int] = list(stations_cfg.get("station_cells", []))
    station_names: Dict[str, str] = stations_cfg.get("station_names", {})

    if not ugrid_path.is_file():
        return pd.DataFrame(columns=["ID", "name", "lat", "lon"])

    gdf = gpd.read_parquet(ugrid_path)
    gdf = gdf[gdf["ID"].isin(station_cells)].copy()
    if gdf.empty:
        return pd.DataFrame(columns=["ID", "name", "lat", "lon"])

    # Transform to WGS84 for lat/lon
    gdf = gdf.to_crs("EPSG:4326")
    gdf["lat"] = gdf.geometry.y
    gdf["lon"] = gdf.geometry.x
    gdf["name"] = gdf["ID"].astype(str).map(station_names).fillna(gdf["ID"].astype(str))

    return gdf[["ID", "name", "lat", "lon"]].reset_index(drop=True)


def _build_header() -> None:
    """Render the main header section."""
    st.markdown(
        """
        <style>
        .main-header {
            font-size: 2.2rem;
            font-weight: 700;
            color: #0f172a;
            margin-bottom: 0.2rem;
        }
        .sub-header {
            font-size: 0.95rem;
            color: #64748b;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="main-header">Israel National Flood Forecasting Center</div>
        <div class="sub-header">
            Real-time monitoring of flood risk based on ICON numerical weather prediction and
            hydrological response models.
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.write("")  # spacing


def _sidebar_controls(config: Dict[str, Any], member_names: List[str]) -> Dict[str, Any]:
    """
    Render sidebar controls for forecast selection and alert thresholds.

    Parameters
    ----------
    config : Dict[str, Any]
        Global configuration dictionary.

    Returns
    -------
    Dict[str, Any]
        Dictionary of user-selected options.
    """
    ui_cfg = config.get("ui", {})

    st.sidebar.title("Controls")

    selected_forecast = st.sidebar.selectbox(
        "Active ICON ensemble member",
        options=member_names,
        index=0 if member_names else None,
    )

    default_threshold = float(ui_cfg.get("default_alert_prob_threshold", 0.6))
    alert_threshold = st.sidebar.slider(
        "Flood probability alert threshold",
        min_value=0.1,
        max_value=0.99,
        value=default_threshold,
        step=0.01,
    )

    st.sidebar.markdown("---")
    run_clicked = st.sidebar.button("Run / Update Forecast", type="primary")
    st.sidebar.markdown("**Display options**")
    show_confidence = st.sidebar.checkbox("Show confidence intervals", value=True)

    return {
        "forecast": selected_forecast,
        "alert_threshold": alert_threshold,
        "show_confidence": show_confidence,
        "run_clicked": run_clicked,
    }


def _kpi_section(predictions: pd.DataFrame, alert_threshold: float) -> None:
    """Render the top-level KPI cards based on real predictions."""
    col1, col2, col3 = st.columns(3)

    if predictions.empty:
        col1.metric(label="Stations at high risk", value="0")
        col2.metric(label="Max predicted discharge [m³/s]", value="–")
        col3.metric(label="Max event probability", value="–")
        return

    # Aggregate over time and ensemble: max probability per station
    agg = (
        predictions.groupby("ID")
        .agg(
            max_prob=("prob_event", "max"),
            max_q=("q_final", "max"),
        )
        .reset_index()
    )

    stations_at_risk = int((agg["max_prob"] >= alert_threshold).sum())
    max_q = float(agg["max_q"].max())
    max_prob = float(agg["max_prob"].max())

    col1.metric(label="Stations at high risk", value=str(stations_at_risk))
    col2.metric(label="Max predicted discharge [m³/s]", value=f"{max_q:.2f}")
    col3.metric(label="Max event probability", value=f"{max_prob:.2f}")


def _map_section(config: Dict[str, Any], predictions: pd.DataFrame, alert_threshold: float) -> None:
    """
    Render an interactive station map.

    Currently uses synthetic example data; will be wired to real station
    metadata and predictions during the inference implementation.
    """
    ui_cfg = config.get("ui", {})
    center_lat = float(ui_cfg.get("map_center_lat", 31.5))
    center_lon = float(ui_cfg.get("map_center_lon", 35.0))

    st.subheader("Spatial overview of station flood risk")

    stations_df = get_station_locations()
    if stations_df.empty or predictions.empty:
        st.info("No station locations or predictions available for mapping.")
        return

    # Compute risk metric per station (max probability over time and ensemble)
    risk = (
        predictions.groupby("ID")["prob_event"]
        .max()
        .reset_index()
        .rename(columns={"prob_event": "max_prob"})
    )
    stations = stations_df.merge(risk, on="ID", how="left")
    stations["max_prob"] = stations["max_prob"].fillna(0.0)

    def _risk_color(p: float) -> str:
        if p >= alert_threshold:
            return "red"
        if p >= 0.5 * alert_threshold:
            return "orange"
        return "green"

    stations["color"] = stations["max_prob"].apply(_risk_color)

    # Basic scatter map using Streamlit's built-in map (ignores color but keeps positions),
    # while we provide risk summary in the table below.
    st.map(
        data=stations[["lat", "lon"]],
        zoom=ui_cfg.get("map_zoom_start", 8),
    )

    st.dataframe(
        stations[["ID", "name", "max_prob"]],
        use_container_width=True,
        hide_index=True,
    )


def _hydrograph_section(
    predictions: pd.DataFrame,
    member_name: str,
    station_id: int,
    show_confidence: bool,
) -> None:
    """
    Render an example hydrograph panel with rain and discharge.

    Parameters
    ----------
    show_confidence : bool
        Whether to display shaded confidence intervals.
    """
    st.subheader("Station hydrograph: rain rate vs. predicted discharge")

    if predictions.empty:
        st.info("No predictions available to plot.")
        return

    # Filter predictions to the selected member and station
    station_pred = predictions[
        (predictions["member"] == member_name) & (predictions["ID"] == station_id)
    ].copy()
    if station_pred.empty:
        st.info("No prediction data for the selected member/station.")
        return

    station_pred = station_pred.sort_values("time")

    # Load corresponding rain data and aggregate over station cell(s)
    rain_df = load_rain_for_member(member_name)
    rain_station = rain_df[rain_df["ID"] == station_id].copy()
    if "time" in rain_station.columns:
        rain_station = (
            rain_station.groupby("time")["rainrate"].mean().reset_index()
        )

    # Align time axis
    times = station_pred["time"]
    rain_aligned = rain_station.set_index("time").reindex(times, method="nearest")
    rain_rate = rain_aligned["rainrate"].fillna(0.0).to_numpy()
    discharge = station_pred["q_final"].to_numpy()

    fig = go.Figure()

    fig = go.Figure()

    # Rain as bars
    fig.add_bar(
        x=times,
        y=rain_rate,
        name="Rain rate [mm/h]",
        marker_color="rgba(37, 99, 235, 0.7)",
        yaxis="y2",
    )

    # Discharge as line
    fig.add_trace(
        go.Scatter(
            x=times,
            y=discharge,
            name="Predicted discharge [m³/s]",
            mode="lines",
            line=dict(color="rgba(220, 38, 38, 0.9)", width=3),
            yaxis="y1",
        )
    )

    if show_confidence:
        upper = discharge * 1.15
        lower = discharge * 0.85
        fig.add_trace(
            go.Scatter(
                x=list(times) + list(times[::-1]),
                y=list(upper) + list(lower[::-1]),
                fill="toself",
                fillcolor="rgba(239, 68, 68, 0.18)",
                line=dict(color="rgba(0,0,0,0)"),
                hoverinfo="skip",
                showlegend=True,
                name="Discharge CI",
                yaxis="y1",
            )
        )

    fig.update_layout(
        template="plotly_white",
        height=420,
        margin=dict(l=40, r=40, t=40, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1.0),
        xaxis=dict(title="Time", gridcolor="rgba(148, 163, 184, 0.3)"),
        yaxis=dict(title="Discharge [m³/s]", gridcolor="rgba(148, 163, 184, 0.3)"),
        yaxis2=dict(
            title="Rain rate [mm/h]",
            overlaying="y",
            side="right",
            showgrid=False,
        ),
    )

    st.plotly_chart(fig, use_container_width=True)


def main() -> None:
    """Run the Streamlit dashboard application."""
    st.set_page_config(
        page_title="Israel Flood Forecasting Center",
        page_icon="🌧️",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    config = load_config(CONFIG_PATH)
    member_names = get_available_members()

    _build_header()
    selections = _sidebar_controls(config, member_names)

    predictions: pd.DataFrame
    if "predictions" not in st.session_state or selections["run_clicked"]:
        with st.spinner("Running ensemble flood forecast..."):
            predictions = run_predictions_all_members()
        st.session_state["predictions"] = predictions

        # Export notebook-compatible artefacts on each fresh run
        try:
            export_all_notebook_artefacts(predictions, config)
        except Exception as exc:  # noqa: BLE001
            st.warning(f"Failed to export notebook-compatible artefacts: {exc}")
    else:
        predictions = st.session_state["predictions"]

    _kpi_section(predictions, alert_threshold=selections["alert_threshold"])

    # Station selector for hydrograph
    stations_df = get_station_locations()
    if not stations_df.empty:
        default_station = int(stations_df["ID"].iloc[0])
        station_id = int(
            st.selectbox(
                "Select station cell for hydrograph",
                options=list(stations_df["ID"]),
                format_func=lambda x: stations_df.set_index("ID").loc[x, "name"],
                index=list(stations_df["ID"]).index(default_station),
            )
        )
    else:
        station_id = 0

    col_map, col_ts = st.columns([1.1, 1.3])
    with col_map:
        _map_section(config, predictions, alert_threshold=selections["alert_threshold"])
    with col_ts:
        if station_id != 0:
            _hydrograph_section(
                predictions=predictions,
                member_name=selections["forecast"],
                station_id=station_id,
                show_confidence=selections["show_confidence"],
            )
        else:
            st.info("No station selected for hydrograph display.")


if __name__ == "__main__":
    main()

