from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import json
import numpy as np
import pandas as pd
import plotly.graph_objects as go


def _get_paths_from_config(config: Mapping[str, Any]) -> dict[str, Path]:
    data_paths = config.get("data_paths", {}) or {}
    if not isinstance(data_paths, dict):
        raise ValueError("Configuration key 'data_paths' must be a mapping.")

    basin_folder = Path(str(data_paths.get("basin_folder", "."))).expanduser()
    icon_export_dir = Path(
        str(data_paths.get("icon_predictions_export_dir", basin_folder / "output" / "icon_predictions" / "predictions_export"))
    ).expanduser()
    combined_model_dir = basin_folder / "output" / "model" / "TWO_STAGE_COMBINED_HYDROGRAPH"

    return {
        "basin_folder": basin_folder,
        "icon_export_dir": icon_export_dir,
        "combined_model_dir": combined_model_dir,
    }


def export_cell_summary(predictions: pd.DataFrame, config: Mapping[str, Any]) -> Path:
    """
    Export a cell-level summary CSV compatible with the original notebook.

    The file is written as:
        icon_predictions/predictions_export/cell_summary.csv
    under the configured basin folder.
    """
    paths = _get_paths_from_config(config)
    out_dir = paths["icon_export_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)

    if predictions.empty:
        # Still create an empty CSV so downstream scripts don't fail.
        out_path = out_dir / "cell_summary.csv"
        empty = pd.DataFrame(columns=["ID", "name", "max_prob", "max_q", "mean_prob", "mean_q"])
        empty.to_csv(out_path, index=False)
        return out_path

    stations_cfg = config.get("stations", {}) or {}
    if not isinstance(stations_cfg, dict):
        stations_cfg = {}
    station_names = stations_cfg.get("station_names", {}) or {}

    agg = (
        predictions.groupby("ID")
        .agg(
            max_prob=("prob_event", "max"),
            max_q=("q_final", "max"),
            mean_prob=("prob_event", "mean"),
            mean_q=("q_final", "mean"),
        )
        .reset_index()
    )
    agg["ID"] = agg["ID"].astype(int)
    agg["name"] = agg["ID"].astype(str).map(station_names).fillna(agg["ID"].astype(str))
    # Reorder columns to be stable
    agg = agg[["ID", "name", "max_prob", "max_q", "mean_prob", "mean_q"]]

    out_path = out_dir / "cell_summary.csv"
    agg.to_csv(out_path, index=False)
    return out_path


def export_uncertainty_json(predictions: pd.DataFrame, config: Mapping[str, Any]) -> Path:
    """
    Export a JSON file with basic ensemble uncertainty statistics per cell/time.

    This approximates the ``uncertainty_analysis.json`` artefact from the
    research notebook by providing mean/std and selected quantiles of
    ``q_final`` across ensemble members for each (ID, time) pair.
    """
    paths = _get_paths_from_config(config)
    out_dir = paths["combined_model_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)

    out_path = out_dir / "uncertainty_analysis.json"

    if predictions.empty:
        with out_path.open("w", encoding="utf-8") as f:
            json.dump({"entries": []}, f)
        return out_path

    # Aggregate over ensemble members
    grouped = predictions.groupby(["ID", "time"])
    stats = grouped["q_final"].agg(
        mean_q="mean",
        std_q="std",
        p10=lambda x: np.percentile(x, 10),
        p50=lambda x: np.percentile(x, 50),
        p90=lambda x: np.percentile(x, 90),
    ).reset_index()

    # Convert time to ISO for JSON serialisation
    stats["time"] = pd.to_datetime(stats["time"]).dt.strftime("%Y-%m-%dT%H:%M:%S")

    records = stats.to_dict(orient="records")
    payload = {"entries": records}
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False)

    return out_path


def export_hydrograph_html_reports(predictions: pd.DataFrame, config: Mapping[str, Any]) -> dict[str, Path]:
    """
    Export Plotly HTML reports mirroring the notebook artefacts:

    - hydrographs_stage1_events_plotly.html
    - hydrographs_predicted_vs_actual_plotly.html

    The content is not an exact pixel-perfect clone of the notebook,
    but provides comparable diagnostics in the expected locations.
    """
    paths = _get_paths_from_config(config)
    out_dir = paths["combined_model_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)

    files: dict[str, Path] = {}
    if predictions.empty:
        # Still create empty placeholder HTML files
        for name in [
            "hydrographs_stage1_events_plotly.html",
            "hydrographs_predicted_vs_actual_plotly.html",
        ]:
            path = out_dir / name
            path.write_text("<html><body><h3>No data available</h3></body></html>", encoding="utf-8")
            files[name] = path
        return files

    # Choose one representative station (first configured station if available)
    stations_cfg = config.get("stations", {}) or {}
    station_cells = list(stations_cfg.get("station_cells", [])) if isinstance(stations_cfg, dict) else []
    if station_cells:
        station_id = int(station_cells[0])
    else:
        station_id = int(predictions["ID"].iloc[0])

    # Use median ensemble (by member name) for the hydrograph line
    members_sorted = sorted(predictions["member"].unique())
    mid_idx = len(members_sorted) // 2
    member_ref = members_sorted[mid_idx]

    df_station = predictions[
        (predictions["ID"] == station_id) & (predictions["member"] == member_ref)
    ].copy()
    df_station = df_station.sort_values("time")

    # 1) Stage1-style event probabilities over time
    fig_events = go.Figure()
    fig_events.add_trace(
        go.Scatter(
            x=df_station["time"],
            y=df_station["prob_event"],
            mode="lines",
            name="Event probability",
        )
    )
    fig_events.update_layout(
        template="plotly_white",
        title=f"Stage 1 event probabilities – station {station_id} (member {member_ref})",
        xaxis_title="Time",
        yaxis_title="Probability of event",
        yaxis=dict(range=[0.0, 1.0]),
    )
    events_path = out_dir / "hydrographs_stage1_events_plotly.html"
    fig_events.write_html(str(events_path), include_plotlyjs="cdn")
    files["hydrographs_stage1_events_plotly.html"] = events_path

    # 2) Predicted discharge over time (no observations available here)
    fig_q = go.Figure()
    fig_q.add_trace(
        go.Scatter(
            x=df_station["time"],
            y=df_station["q_final"],
            mode="lines",
            name="Predicted discharge (q_final)",
        )
    )
    fig_q.update_layout(
        template="plotly_white",
        title=f"Predicted discharge – station {station_id} (member {member_ref})",
        xaxis_title="Time",
        yaxis_title="Discharge [m³/s]",
    )
    q_path = out_dir / "hydrographs_predicted_vs_actual_plotly.html"
    fig_q.write_html(str(q_path), include_plotlyjs="cdn")
    files["hydrographs_predicted_vs_actual_plotly.html"] = q_path

    return files


def export_all_notebook_artefacts(predictions: pd.DataFrame, config: Mapping[str, Any]) -> dict[str, Path]:
    """
    Convenience wrapper to export all notebook-compatible artefacts in one call.
    """
    paths: dict[str, Path] = {}
    paths["cell_summary"] = export_cell_summary(predictions, config)
    paths["uncertainty_json"] = export_uncertainty_json(predictions, config)
    html_files = export_hydrograph_html_reports(predictions, config)
    paths.update(html_files)
    return paths

