"""
Page 3: Rain–Discharge Merge. Dedicated to processing discharge data, aligning
time series to kriged rain timestamps, and merging to produce the final
ML training dataset (rain_with_discharge_{year}.parquet) exactly as in
MY_LAST_JUPYTER.ipynb. Includes Hydrological Dashboard: Merged Data Preview,
Hydrograph (discharge + rain per event), and Storm Animation (spatial time series).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import time

import matplotlib.animation as mpl_animation
import matplotlib.pyplot as plt
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yaml

from src.data_merger import (
    DEFAULT_AREA_NAME_MAPPING,
    merge_rain_and_discharge_for_years,
    process_discharge_for_years,
)


CONFIG_PATH = Path(__file__).resolve().parents[2] / "configs" / "system_config.yaml"


def _load_config() -> Dict[str, Any]:
    with CONFIG_PATH.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _format_frame_label(frame_name: str) -> str:
    """Format frame name (ISO timestamp) for animation slider label."""
    if not frame_name:
        return ""
    try:
        return pd.Timestamp(frame_name).strftime("%m-%d %H:%M")
    except Exception:
        return frame_name[:16] if len(frame_name) > 16 else frame_name


def _discover_years_from_kriged(kriged_folder: Path) -> List[int]:
    if not kriged_folder.is_dir():
        return []
    years: List[int] = []
    for p in sorted(kriged_folder.glob("rain_kriged_*.parquet")):
        stem = p.stem  # rain_kriged_2015
        digits = "".join(c for c in stem if c.isdigit())
        if len(digits) == 4:
            try:
                years.append(int(digits))
            except ValueError:
                continue
    return sorted(set(years))


def _load_events_for_hydrograph(events_csv_path: Path) -> pd.DataFrame:
    """Load and clean events CSV (notebook logic: corr_st/corr_ed, corrected only, 48h extension)."""
    events = pd.read_csv(events_csv_path)
    for col in ["st", "ed", "corr_st", "corr_ed"]:
        if col in events.columns:
            events[col] = pd.to_datetime(events[col], errors="coerce")
    if "corr_st" in events.columns and "st" in events.columns:
        events["start_date"] = events["corr_st"].fillna(events["st"])
    elif "st" in events.columns:
        events["start_date"] = events["st"]
    else:
        events["start_date"] = pd.NaT
    if "corr_ed" in events.columns and "ed" in events.columns:
        events["end_date"] = events["corr_ed"].fillna(events["ed"])
    elif "ed" in events.columns:
        events["end_date"] = events["ed"]
    else:
        events["end_date"] = pd.NaT
    events = events.dropna(subset=["start_date", "end_date"]).reset_index(drop=True)
    if "status" in events.columns:
        events = events[events["status"].str.lower() == "corrected"].copy()
    events["original_start"] = events["start_date"].copy()
    events["extended_start"] = events["original_start"] - pd.Timedelta(hours=48)
    return events


def _build_hydrograph_figure(
    discharge_by_time: pd.DataFrame,
    rain_by_time: pd.DataFrame,
    extended_start: pd.Timestamp,
    original_start: pd.Timestamp,
    event_end: pd.Timestamp,
    event_label: str,
) -> go.Figure:
    """Build interactive Plotly figure: discharge (left y), rain (right y), vertical lines and shaded regions."""
    fig = go.Figure()
    # Discharge
    if not discharge_by_time.empty and discharge_by_time["discharge"].notna().any():
        valid = discharge_by_time.dropna(subset=["discharge"])
        fig.add_trace(
            go.Scatter(
                x=valid["time"],
                y=valid["discharge"],
                name="Discharge (mean)",
                line=dict(color="steelblue", width=2),
                fill="tozeroy",
                fillcolor="rgba(70, 130, 180, 0.3)",
                mode="lines",
            )
        )
        max_q = valid["discharge"].max()
        max_time = valid.loc[valid["discharge"].idxmax(), "time"]
        fig.add_trace(
            go.Scatter(
                x=[max_time],
                y=[max_q],
                mode="markers",
                marker=dict(symbol="circle-open", size=14, color="crimson", line=dict(width=2)),
                name=f"Peak: {max_q:.3f} m³/s",
            )
        )
    # Rain (secondary y)
    if not rain_by_time.empty and rain_by_time["rainrate"].max() > 0:
        fig.add_trace(
            go.Scatter(
                x=rain_by_time["time"],
                y=rain_by_time["rainrate"],
                name="Rain rate (mean)",
                line=dict(color="red", width=1.5, dash="dash"),
                fill="tozeroy",
                fillcolor="rgba(220, 20, 60, 0.2)",
                mode="lines",
                yaxis="y2",
            )
        )
    # Vertical lines and shaded regions (use numeric time for Plotly datetime axis)
    def _ts_ms(ts: pd.Timestamp) -> float:
        return pd.Timestamp(ts).value / 1e6  # nanoseconds to milliseconds

    for ts, color, dash, name in [
        (extended_start, "gray", "dot", "Extended start (48h before)"),
        (original_start, "green", "solid", "Event start"),
        (event_end, "red", "solid", "Event end"),
    ]:
        fig.add_vline(x=_ts_ms(ts), line_dash=dash, line_color=color, line_width=2, annotation_text=name)
    fig.add_vrect(
        x0=_ts_ms(extended_start),
        x1=_ts_ms(original_start),
        fillcolor="gray",
        opacity=0.1,
        line_width=0,
    )
    fig.add_vrect(
        x0=_ts_ms(original_start),
        x1=_ts_ms(event_end),
        fillcolor="gold",
        opacity=0.12,
        line_width=0,
    )
    fig.update_layout(
        title=dict(
            text=event_label,
            font=dict(size=14),
        ),
        xaxis=dict(title="Time", gridcolor="rgba(128,128,128,0.2)"),
        yaxis=dict(
            title="Discharge (m³/s)",
            side="left",
            gridcolor="rgba(128,128,128,0.2)",
            rangemode="tozero",
        ),
        yaxis2=dict(
            title="Rain rate (mm/10min)",
            side="right",
            overlaying="y",
            showgrid=False,
            rangemode="tozero",
        ),
        template="plotly_white",
        height=420,
        margin=dict(l=60, r=60, t=60, b=50),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        hovermode="x unified",
    )
    return fig


def _create_storm_animation_video(
    df_window: pd.DataFrame,
    unique_times: List[pd.Timestamp],
    station_info: List[Dict[str, Any]],
    year: int,
    output_dir: Path,
) -> Path:
    """
    Create a Matplotlib FuncAnimation over the focused time window and save to disk
    as MP4 (preferred) or GIF. Returns the saved file path.
    """
    output_dir = output_dir.expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    if df_window.empty or not unique_times:
        raise ValueError("No data available for animation.")

    # Prepare static ranges for a stable view
    lon_min, lon_max = df_window["lon"].min(), df_window["lon"].max()
    lat_min, lat_max = df_window["lat"].min(), df_window["lat"].max()
    rain_min, rain_max = float(df_window["rainrate"].min()), float(
        df_window["rainrate"].max() or 1e-9
    )
    if df_window["discharge"].notna().any():
        flow_max = float(df_window["discharge"].max())
    else:
        flow_max = 1.0

    fig, ax = plt.subplots(figsize=(8, 6), dpi=100)

    # Initial empty scatters
    sc_rain = ax.scatter([], [], c=[], cmap="Reds", s=5, vmin=rain_min, vmax=rain_max, alpha=0.8)
    sc_stations = ax.scatter(
        [], [], c=[], cmap="Blues", s=150, marker="D", vmin=0, vmax=flow_max, alpha=0.9,
        edgecolor="black", linewidth=1.0,
    )

    ax.set_xlim(lon_min - 0.01, lon_max + 0.01)
    ax.set_ylim(lat_min - 0.01, lat_max + 0.01)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    title = ax.set_title(f"Rain & Discharge Animation – {year}", fontsize=12)
    ax.grid(True, alpha=0.3, linestyle="--")

    cbar_rain = plt.colorbar(sc_rain, ax=ax, fraction=0.046, pad=0.04)
    cbar_rain.set_label("Rain Rate (mm/10min)", fontsize=9)
    cbar_flow = plt.colorbar(sc_stations, ax=ax, fraction=0.046, pad=0.08)
    cbar_flow.set_label("Discharge (m³/s)", fontsize=9)

    station_ids = {s["cell_id"] for s in station_info}

    def update(frame_idx: int):
        current_time = unique_times[frame_idx]
        df_t = df_window[df_window["time"] == current_time]
        station_mask = df_t["ID"].isin(station_ids)
        df_st = df_t[station_mask]
        df_rn = df_t[~station_mask]

        if not df_rn.empty:
            sc_rain.set_offsets(df_rn[["lon", "lat"]].to_numpy())
            sc_rain.set_array(df_rn["rainrate"].to_numpy())
        else:
            sc_rain.set_offsets([])
            sc_rain.set_array([])

        if not df_st.empty:
            sc_stations.set_offsets(df_st[["lon", "lat"]].to_numpy())
            sc_stations.set_array(df_st["discharge"].fillna(0).to_numpy())
        else:
            sc_stations.set_offsets([])
            sc_stations.set_array([])

        title.set_text(
            f"Rain & Discharge Animation – {year}\nTime: {current_time.strftime('%Y-%m-%d %H:%M')}"
        )
        return sc_rain, sc_stations, title

    ani = mpl_animation.FuncAnimation(
        fig,
        update,
        frames=len(unique_times),
        interval=300,
        blit=False,
        repeat=True,
    )

    # Try MP4 first, then GIF if ffmpeg not available
    mp4_path = output_dir / f"storm_animation_{year}.mp4"
    gif_path = output_dir / f"storm_animation_{year}.gif"
    saved_path: Path
    try:
        writer = mpl_animation.FFMpegWriter(fps=3)
        ani.save(str(mp4_path), writer=writer)
        saved_path = mp4_path
    except Exception:
        writer = mpl_animation.PillowWriter(fps=3)
        ani.save(str(gif_path), writer=writer)
        saved_path = gif_path
    finally:
        plt.close(fig)

    return saved_path


def main() -> None:
    st.set_page_config(
        page_title="Rain–Discharge Merge",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.markdown(
        """
        <style>
        .main-header { font-size: 2rem; font-weight: 700; color: #0f172a; margin-bottom: 0.2rem; }
        .sub-header { font-size: 0.95rem; color: #64748b; }
        </style>
        <div class="main-header">Finalize Training Dataset (Merge Rain & Discharge)</div>
        <div class="sub-header">
            Process discharge data (resample to 10-min, align to kriged rain timestamps),
            then merge with kriged rain to produce the final ML training parquets.
            Logic is a strict 1:1 port from MY_LAST_JUPYTER.ipynb.
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.write("")

    base_cfg = _load_config()
    data_paths = base_cfg.get("data_paths", {}) or {}

    # Sidebar: paths and year selection
    st.sidebar.title("Paths & years")

    default_basin = str(data_paths.get("basin_folder", "")) or str(
        Path("/media/data-nvme/Darga_28_for_test_only")
    )
    basin_folder_str = st.sidebar.text_input(
        "Basin folder",
        value=default_basin,
        help="Basin root (info.txt, output/ugrid, output/rain, output/discharge).",
    )
    basin_folder = Path(basin_folder_str).expanduser()

    default_discharge_dir = str(
        data_paths.get("discharge_csv_dir", "/media/data-nvme/Q need update!!!/ProcessedDB")
    )
    discharge_csv_dir = st.sidebar.text_input(
        "Discharge CSV directory (ProcessedDB)",
        value=default_discharge_dir,
        help="Directory containing {AreaName}_ts_V1.csv files.",
    )
    area_name = st.sidebar.text_input(
        "Discharge area name",
        value=DEFAULT_AREA_NAME_MAPPING.get(
            basin_folder.name, "Northern Dead Sea"
        ),
        help="Used to build CSV path: {dir}/{area_name}_ts_V1.csv",
    )
    discharge_csv_path = Path(discharge_csv_dir).expanduser() / f"{area_name}_ts_V1.csv"
    if not discharge_csv_path.is_file():
        st.sidebar.caption(f"CSV path: `{discharge_csv_path}` (file may not exist yet)")

    default_metadata = str(
        data_paths.get("station_metadata_path", "")
    ) or "/media/data-nvme/Q need update!!!/Q_stations_metadata.xlsx"
    metadata_path_str = st.sidebar.text_input(
        "Station metadata (Excel)",
        value=default_metadata,
    )
    metadata_path = Path(metadata_path_str).expanduser()

    st.sidebar.markdown("---")
    st.sidebar.caption("Dashboard: hydrograph & animation")
    events_cfg = base_cfg.get("events", {}) or {}
    _ev_csv = events_cfg.get("events_library_csv") or data_paths.get("events_library_csv")
    default_events_csv = (
        str(_ev_csv).strip() if _ev_csv else ""
    ) or "/media/data-nvme/storm library/Northern Dead Sea_events_library_supervised_V3.csv"
    events_csv_str = st.sidebar.text_input(
        "Events library CSV (for hydrographs)",
        value=default_events_csv,
        help="Used in Hydrograph tab to select event and plot discharge + rain.",
    )
    events_csv_path = Path(events_csv_str).expanduser() if events_csv_str else None

    kriged_rain_folder = basin_folder / "output" / "rain" / "event_rain" / "intepulated_rain_on_ugrid"
    discharge_output_dir = basin_folder / "output" / "discharge"
    merge_output_dir = basin_folder / "output" / "rain_with_discharge"
    ugrid_path = basin_folder / "output" / "ugrid" / "final_ugrid.parquet"

    available_years = _discover_years_from_kriged(kriged_rain_folder)
    st.sidebar.markdown("---")
    if available_years:
        selected_years: List[int] = st.sidebar.multiselect(
            "Years to process (from kriged rain files)",
            options=available_years,
            default=available_years[-1:] if available_years else [],
        )
    else:
        selected_years = []
        st.sidebar.info("No kriged rain files found. Run Kriging on the Rain Grids page first.")

    # Execution section
    st.subheader("Execution")

    col_discharge, col_merge = st.columns(2)

    with col_discharge:
        st.markdown("**Step 1: Process discharge**")
        st.caption(
            "Load discharge CSV, resample 5-min → 10-min (mean), filter to exact "
            "rain timestamps from kriged files. Writes discharge_processed_{year}.parquet."
        )
        run_discharge = st.button("Run discharge processing", key="run_discharge")
        if run_discharge:
            if not discharge_csv_path.is_file():
                st.error(f"Discharge CSV not found: {discharge_csv_path}")
            elif not metadata_path.is_file():
                st.error(f"Metadata file not found: {metadata_path}")
            else:
                with st.spinner("Processing discharge (resample & align to rain timestamps)..."):
                    try:
                        n_stations, paths = process_discharge_for_years(
                            basin_folder=basin_folder,
                            discharge_csv_path=discharge_csv_path,
                            metadata_path=metadata_path,
                            kriged_rain_folder=kriged_rain_folder,
                            discharge_output_dir=discharge_output_dir,
                            years=selected_years or None,
                            force=False,
                        )
                        st.success(f"Discharge processing done. Stations in basin: {n_stations}.")
                        for p in paths:
                            st.code(str(p), language="text")
                    except Exception as exc:
                        st.error(f"Discharge processing failed: {exc}")

    with col_merge:
        st.markdown("**Step 2: Merge rain & discharge**")
        st.caption(
            "Map discharge stations to nearest UGRID cell, merge by exact time. "
            "Writes rain_with_discharge_{year}.parquet (final ML training dataset)."
        )
        run_merge = st.button("Run rain–discharge merge", key="run_merge", type="primary")
        if run_merge:
            if not ugrid_path.is_file():
                st.error(f"UGRID file not found: {ugrid_path}")
            else:
                with st.spinner("Merging kriged rain with discharge..."):
                    try:
                        result = merge_rain_and_discharge_for_years(
                            basin_folder=basin_folder,
                            ugrid_parquet_path=ugrid_path,
                            kriged_rain_folder=kriged_rain_folder,
                            discharge_folder=discharge_output_dir,
                            metadata_path=metadata_path,
                            output_dir=merge_output_dir,
                            years=selected_years or None,
                            force=False,
                        )
                        if result:
                            st.success("Merge completed.")
                            for yr, p in result.items():
                                st.write(f"**{yr}:** `{p.name}`")
                        else:
                            st.info("No merged files produced (e.g. no kriged rain for selected years).")
                    except Exception as exc:
                        st.error(f"Merge failed: {exc}")

    # -------------------------------------------------------------------------
    # Hydrological Dashboard (tabs: Preview, Hydrograph, Storm Animation)
    # -------------------------------------------------------------------------
    st.markdown("---")
    st.subheader("Hydrological Dashboard")
    st.caption("Inspect merged data, event hydrographs, and spatial storm animation.")

    merged_files = (
        sorted(merge_output_dir.glob("rain_with_discharge_*.parquet"))
        if merge_output_dir.is_dir()
        else []
    )
    has_merged = len(merged_files) > 0

    tab_preview, tab_hydrograph, tab_animation = st.tabs(
        ["Merged Data Preview", "Hydrograph", "Storm Animation"]
    )

    with tab_preview:
        if not has_merged:
            st.info(
                "No merged files yet. Run **Run rain–discharge merge** above, then refresh."
            )
        else:
            file_labels = [f.name for f in merged_files]
            preview_idx = st.selectbox(
                "Select merged file to preview",
                options=list(range(len(merged_files))),
                format_func=lambda i: file_labels[i],
                key="preview_file",
            )
            chosen = merged_files[preview_idx]
            df_preview = pd.read_parquet(chosen)
            if "time" in df_preview.columns:
                df_preview["time"] = pd.to_datetime(df_preview["time"], errors="coerce")
            st.write(f"**File:** `{chosen.name}` — **rows:** {len(df_preview):,}")
            preview_cols = [
                c
                for c in ["ID", "time", "rainrate", "discharge", "lon", "lat"]
                if c in df_preview.columns
            ]
            st.dataframe(
                df_preview[preview_cols].head(50),
                use_container_width=True,
                hide_index=True,
            )
            st.caption(
                "First 50 rows. Check that `rainrate` and `discharge` are time-aligned; "
                "NaNs in discharge are expected for cells without a station."
            )

    with tab_hydrograph:
        if not events_csv_path or not events_csv_path.is_file():
            st.info(
                "Set **Events library CSV (for hydrographs)** in the sidebar to a valid path, "
                "then select an event to plot discharge and rain over the extended period."
            )
        elif not has_merged:
            st.info("Run **Run rain–discharge merge** first so hydrograph data is available.")
        else:
            events_df = _load_events_for_hydrograph(events_csv_path)
            if events_df.empty:
                st.warning("No corrected events found in the CSV.")
            else:
                event_options = []
                for idx, ev in events_df.iterrows():
                    os = ev.get("original_start", pd.NaT)
                    ed = ev.get("end_date", pd.NaT)
                    os_str = os.strftime("%Y-%m-%d %H:%M") if pd.notna(os) else "?"
                    ed_str = ed.strftime("%Y-%m-%d %H:%M") if pd.notna(ed) else "?"
                    label = f"Event {idx}: {os_str} → {ed_str}"
                    event_options.append((idx, label))
                selected = st.selectbox(
                    "Select event",
                    options=[o[0] for o in event_options],
                    format_func=lambda i: next(l for k, l in event_options if k == i),
                    key="hydro_event",
                )
                ev = events_df.loc[selected]
                original_start = ev["original_start"]
                event_end = ev["end_date"]
                extended_start = ev["extended_start"]
                year = original_start.year
                rain_file = merge_output_dir / f"rain_with_discharge_{year}.parquet"
                if extended_start.year < year:
                    prev_file = merge_output_dir / f"rain_with_discharge_{extended_start.year}.parquet"
                    if prev_file.is_file():
                        rain_file = prev_file
                if not rain_file.is_file():
                    st.warning(f"No merged file for year {year} (or previous year).")
                else:
                    df_ev = pd.read_parquet(rain_file, columns=["ID", "time", "rainrate", "discharge"])
                    df_ev["time"] = pd.to_datetime(df_ev["time"])
                    extended_data = df_ev[
                        (df_ev["time"] >= extended_start) & (df_ev["time"] <= event_end)
                    ].copy()
                    if extended_data.empty:
                        st.warning("No rows in merged file for this event's time window.")
                    else:
                        discharge_data = extended_data[extended_data["discharge"].notna()]
                        if not discharge_data.empty:
                            discharge_by_time = (
                                discharge_data.groupby("time")
                                .agg({"discharge": "mean", "rainrate": "mean"})
                                .reset_index()
                            )
                        else:
                            discharge_by_time = pd.DataFrame(columns=["time", "discharge", "rainrate"])
                        rain_by_time = (
                            extended_data.groupby("time")["rainrate"].mean().reset_index()
                        )
                        rain_by_time.columns = ["time", "rainrate"]
                        event_label = (
                            f"Event {selected} – Extended: {extended_start.strftime('%Y-%m-%d %H:%M')} → "
                            f"{event_end.strftime('%Y-%m-%d %H:%M')}"
                        )
                        fig_h = _build_hydrograph_figure(
                            discharge_by_time,
                            rain_by_time,
                            extended_start,
                            original_start,
                            event_end,
                            event_label,
                        )
                        st.plotly_chart(fig_h, use_container_width=True)

    with tab_animation:
        # Require merged data and events CSV for event-based animation
        if not has_merged:
            st.info("Run **Run rain–discharge merge** first so merged files exist.")
        elif not events_csv_path or not events_csv_path.is_file():
            st.info(
                "Set **Events library CSV (for hydrographs)** in the sidebar to a valid path. "
                "The animation uses these events to pick the time window."
            )
        else:
            events_df = _load_events_for_hydrograph(events_csv_path)
            if events_df.empty:
                st.warning("No corrected events found in the events CSV.")
            else:
                # Event selector (same style as Hydrograph tab)
                event_options_anim = []
                for idx, ev in events_df.iterrows():
                    os = ev.get("original_start", pd.NaT)
                    ed = ev.get("end_date", pd.NaT)
                    os_str = os.strftime("%Y-%m-%d %H:%M") if pd.notna(os) else "?"
                    ed_str = ed.strftime("%Y-%m-%d %H:%M") if pd.notna(ed) else "?"
                    label = f"Event {idx}: {os_str} → {ed_str}"
                    event_options_anim.append((idx, label))
                selected_anim = st.selectbox(
                    "Select event for animation",
                    options=[o[0] for o in event_options_anim],
                    format_func=lambda i: next(l for k, l in event_options_anim if k == i),
                    key="anim_event",
                )
                ev_a = events_df.loc[selected_anim]
                original_start_a = ev_a["original_start"]
                event_end_a = ev_a["end_date"]
                extended_start_a = ev_a["extended_start"]
                year_anim = original_start_a.year

                # Determine which merged file to use (year or previous year for extended window)
                rain_file_anim = merge_output_dir / f"rain_with_discharge_{year_anim}.parquet"
                if extended_start_a.year < year_anim:
                    prev_file_anim = merge_output_dir / f"rain_with_discharge_{extended_start_a.year}.parquet"
                    if prev_file_anim.is_file():
                        rain_file_anim = prev_file_anim

                if not rain_file_anim.is_file():
                    st.warning(f"No merged file found for year {year_anim} (or previous year).")
                else:
                    df_full = pd.read_parquet(
                        rain_file_anim, columns=["ID", "time", "rainrate", "discharge", "lon", "lat"]
                    )
                    df_full["time"] = pd.to_datetime(df_full["time"], errors="coerce")

                    # Restrict to the event's extended window first
                    mask_event = (df_full["time"] >= extended_start_a) & (
                        df_full["time"] <= event_end_a
                    )
                    df_event = df_full.loc[mask_event].copy()
                    if df_event.empty:
                        st.warning("No merged rows found for this event's extended window.")
                    else:
                        # Peak detection strictly within this event window
                        discharge_event = df_event[df_event["discharge"].notna()]
                        discharge_event = discharge_event[discharge_event["discharge"] > 0]
                        if discharge_event.empty:
                            st.warning(
                                "No positive discharge found within this event. "
                                "Animation cannot focus on a meaningful peak."
                            )
                        else:
                            peak_row = discharge_event.loc[
                                discharge_event["discharge"].idxmax()
                            ]
                            peak_time = peak_row["time"]
                            st.info(f"Peak discharge for this event: {peak_time}")

                            # Dynamic window controls: hours before/after peak
                            c1, c2 = st.columns(2)
                            with c1:
                                hours_before = st.number_input(
                                    "Hours before peak",
                                    min_value=0.0,
                                    max_value=24.0,
                                    value=2.0,
                                    step=0.5,
                                    key="anim_hours_before",
                                )
                            with c2:
                                hours_after = st.number_input(
                                    "Hours after peak",
                                    min_value=0.0,
                                    max_value=24.0,
                                    value=2.0,
                                    step=0.5,
                                    key="anim_hours_after",
                                )

                            start_anim = peak_time - pd.Timedelta(hours=hours_before)
                            end_anim = peak_time + pd.Timedelta(hours=hours_after)
                            st.caption(
                                f"Animation window: {start_anim} → {end_anim} "
                                f"(±{hours_before} h / +{hours_after} h around peak)."
                            )

                            generate_anim = st.button(
                                "Generate Animation Video", key="gen_anim", type="primary"
                            )

                            mask = (df_event["time"] >= start_anim) & (
                                df_event["time"] <= end_anim
                            )
                            df_window = df_event.loc[mask].copy()
                            if df_window.empty:
                                st.warning("No data in the selected animation window.")
                            else:
                                unique_times = (
                                    sorted(df_window["time"].dropna().unique().tolist())
                                )
                                if not unique_times:
                                    st.warning(
                                        "No valid timestamps in the selected animation window."
                                    )
                                else:
                                    discharge_cells = (
                                        df_window[df_window["discharge"].notna()][
                                            "ID"
                                        ]
                                        .unique()
                                        .tolist()
                                    )
                                    station_info = []
                                    for cell_id in discharge_cells:
                                        row = df_window[df_window["ID"] == cell_id].iloc[0]
                                        station_info.append(
                                            {
                                                "cell_id": int(cell_id),
                                                "lat": row["lat"],
                                                "lon": row["lon"],
                                            }
                                        )

                                    if generate_anim:
                                        with st.spinner(
                                            "Rendering animation video (backend) ..."
                                        ):
                                            try:
                                                anim_dir = (
                                                    basin_folder / "output" / "animations"
                                                )
                                                video_path = _create_storm_animation_video(
                                                    df_window,
                                                    unique_times,
                                                    station_info,
                                                    year_anim,
                                                    anim_dir,
                                                )
                                                st.success(
                                                    f"Animation saved to `{video_path}`."
                                                )
                                                if video_path.suffix.lower() == ".mp4":
                                                    st.video(str(video_path))
                                                else:
                                                    st.image(str(video_path))
                                            except Exception as exc:
                                                st.error(
                                                    f"Animation creation failed: {exc}"
                                                )
                                    else:
                                        st.caption(
                                            "Click **Generate Animation Video** to "
                                            "pre-render a short movie for this event "
                                            "and custom peak window."
                                        )


if __name__ == "__main__":
    main()
