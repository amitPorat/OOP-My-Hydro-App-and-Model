from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import plotly.express as px
import streamlit as st
import yaml

from src.rain_builder import ICONPreprocessor, RmcompRainPreprocessor
from src.rain_kriging import run_kriging_for_years


CONFIG_PATH = Path(__file__).resolve().parents[2] / "configs" / "system_config.yaml"


def _load_config(config_path: Path) -> Dict[str, Any]:
    with config_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _build_header() -> None:
    st.markdown(
        """
        <style>
        .main-header {
            font-size: 2.1rem;
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
        <div class="main-header">RMcomp Rain Grids</div>
        <div class="sub-header">
            Generate yearly RMcomp rain grids on the UGRID and derive
            event-based rain cubes for supervised model training.
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.write("")


def _suggest_rmcomp_paths(basin_folder: Path) -> Dict[str, Path]:
    return {
        "basin_folder": basin_folder,
        "rain_output_dir": basin_folder / "output" / "rain",
        "event_rain_dir": basin_folder / "output" / "rain" / "event_rain",
        "final_ugrid_path": basin_folder / "output" / "ugrid" / "final_ugrid.parquet",
    }


def _discover_years(rmcomp_dir: Path) -> List[int]:
    if not rmcomp_dir.is_dir():
        return []
    years: List[int] = []
    for p in sorted(rmcomp_dir.glob("RM*.nc")):
        stem = p.stem
        # Expect patterns like RM2020, RM1999, etc.
        digits = "".join(ch for ch in stem if ch.isdigit())
        if len(digits) == 4:
            try:
                years.append(int(digits))
            except ValueError:
                continue
    return sorted(list(dict.fromkeys(years)))


def _build_rmcomp_config(
    base_cfg: Dict[str, Any],
    basin_folder: Path,
    rmcomp_dir: Path,
    rain_output_dir: Path,
    events_csv: Optional[str],
    antecedent_hours: float,
) -> Dict[str, Any]:
    cfg = dict(base_cfg)
    data_paths = dict(cfg.get("data_paths", {}) or {})
    suggested = _suggest_rmcomp_paths(basin_folder)

    data_paths.setdefault("basin_folder", str(basin_folder))
    data_paths.setdefault("info_path", str(basin_folder / "info.txt"))
    data_paths.setdefault(
        "final_ugrid_path", str(suggested["final_ugrid_path"])
    )
    data_paths["rmcomp_dir"] = str(rmcomp_dir)
    data_paths["rain_output_dir"] = str(rain_output_dir)
    cfg["data_paths"] = data_paths

    events_cfg = dict(cfg.get("events", {}) or {})
    if events_csv:
        events_cfg["events_library_csv"] = events_csv
    # Pass configurable antecedent window in hours down to the preprocessor.
    events_cfg["antecedent_hours"] = antecedent_hours
    cfg["events"] = events_cfg
    return cfg


def _summarise_rain_outputs(rain_root: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    yearly_records: List[Dict[str, Any]] = []
    event_records: List[Dict[str, Any]] = []

    if rain_root.is_dir():
        for p in sorted(rain_root.glob("rain_*.parquet")):
            yearly_records.append(
                {
                    "file": p.name,
                    "year": "".join(ch for ch in p.stem if ch.isdigit()),
                    "path": str(p),
                }
            )

        event_dir = rain_root / "event_rain"
        if event_dir.is_dir():
            for p in sorted(event_dir.glob("rain_events_*.parquet")):
                event_records.append(
                    {
                        "file": p.name,
                        "year": "".join(ch for ch in p.stem if ch.isdigit()),
                        "path": str(p),
                    }
                )

    df_yearly = pd.DataFrame(yearly_records)
    df_events = pd.DataFrame(event_records)
    return df_yearly, df_events


def main() -> None:
    st.set_page_config(
        page_title="Rain Grids (RMcomp)",
        page_icon="🌧️",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    base_cfg = _load_config(CONFIG_PATH)
    _build_header()

    st.sidebar.title("Basin & RMcomp paths")

    default_basin = str(
        base_cfg.get("data_paths", {}).get("basin_folder", "")
    ) or str(Path("/media/data-nvme/Darga_28_for_test_only"))
    basin_folder_str = st.sidebar.text_input(
        "Basin folder",
        value=default_basin,
        help="Root folder for the basin (contains info.txt, output/ugrid, raw/rmcomp, etc.).",
    )
    basin_folder = Path(basin_folder_str).expanduser()
    suggested = _suggest_rmcomp_paths(basin_folder)

    # Notebook default for RMcomp input path
    nb_rmcomp_default = Path("/media/data-ssd/RMcompBD")
    rmcomp_dir_str = st.sidebar.text_input(
        "RMcomp NetCDF directory",
        value=str(
            base_cfg.get("data_paths", {}).get("rmcomp_dir", nb_rmcomp_default)
        ),
    )
    rain_output_dir_str = st.sidebar.text_input(
        "Rain output directory",
        value=str(
            base_cfg.get("data_paths", {}).get(
                "rain_output_dir", suggested["rain_output_dir"]
            )
        ),
    )

    events_csv_default: Optional[str] = None
    events_cfg = base_cfg.get("events", {}) or {}
    if "events_library_csv" in events_cfg:
        events_csv_default = str(events_cfg["events_library_csv"])
    else:
        # Notebook default storm-library CSV for Northern Dead Sea
        nb_events_default = Path(
            "/media/data-nvme/storm library/Northern Dead Sea_events_library_supervised_V3.csv"
        )
        if nb_events_default.is_file():
            events_csv_default = str(nb_events_default)
        else:
            # Best-effort guess in basin folder
            guesses = list(basin_folder.glob("*events_library*.csv"))
            if guesses:
                events_csv_default = str(guesses[0])

    events_csv_str = st.sidebar.text_input(
        "Events library CSV",
        value=events_csv_default or "",
        help="CSV containing supervised event windows (start/end times).",
    )

    antecedent_hours = st.sidebar.number_input(
        "Antecedent lead-time (hours)",
        min_value=0.0,
        max_value=240.0,
        value=48.0,
        step=1.0,
        help="Number of hours to extend each event backwards in time (notebook default is 48).",
    )

    rmcomp_dir = Path(rmcomp_dir_str).expanduser()
    rain_output_dir = Path(rain_output_dir_str).expanduser()
    rain_output_dir.mkdir(parents=True, exist_ok=True)

    available_years = _discover_years(rmcomp_dir)
    st.sidebar.markdown("---")
    if available_years:
        year_options = available_years
        default_years = [available_years[-1]]
    else:
        year_options = []
        default_years = []
    selected_years: List[int] = st.sidebar.multiselect(
        "Years to process (from RM{year}.nc)",
        options=year_options,
        default=default_years,
    )

    col_actions, col_status = st.columns([1.1, 1.3])

    with col_actions:
        st.subheader("Processing controls")
        cfg_rm = _build_rmcomp_config(
            base_cfg,
            basin_folder=basin_folder,
            rmcomp_dir=rmcomp_dir,
            rain_output_dir=rain_output_dir,
            events_csv=events_csv_str or None,
            antecedent_hours=antecedent_hours,
        )

        if not selected_years:
            st.info("Select at least one year with available `RM{year}.nc` to enable processing.")

        generate_yearly = st.button(
            "Generate yearly RMcomp rain parquet", type="primary"
        )
        if generate_yearly and selected_years:
            with st.spinner("Generating yearly RMcomp rain fields on UGRID..."):
                try:
                    pre = RmcompRainPreprocessor(cfg_rm)
                    outputs: List[str] = []
                    for y in selected_years:
                        path = pre.run_year(int(y))
                        outputs.append(str(path))
                    st.success("Yearly RMcomp rain generation completed.")
                    st.write("Outputs:")
                    for p in outputs:
                        st.code(p, language="bash")
                except Exception as exc:  # noqa: BLE001
                    st.error(f"Yearly RMcomp preprocessing failed: {exc}")

        generate_events = st.button("Generate event-based RMcomp rain parquet")
        if generate_events and selected_years:
            with st.spinner("Slicing yearly RMcomp rain into event windows..."):
                try:
                    pre = RmcompRainPreprocessor(cfg_rm)
                    outputs: List[str] = []
                    for y in selected_years:
                        path = pre.run_events_for_year(int(y))
                        outputs.append(str(path))
                    st.success("Event-based RMcomp rain generation completed.")
                    st.write("Outputs:")
                    for p in outputs:
                        st.code(p, language="bash")
                except Exception as exc:  # noqa: BLE001
                    st.error(f"Event-based RMcomp preprocessing failed: {exc}")

        st.markdown("**Optional – Krige event rain onto UGRID**")
        st.caption(
            "Uses only event-sliced files (rain_events_{year}.parquet). "
            "Run \"Generate event-based RMcomp rain parquet\" first."
        )
        run_kriging = st.button("Run kriging onto UGRID (per year)")
        if run_kriging:
            with st.spinner("Running Ordinary Kriging of event rain onto UGRID cells..."):
                try:
                    basin_path = Path(cfg_rm.get("data_paths", {}).get("basin_folder", basin_folder_str))
                    kriged = run_kriging_for_years(basin_path, years=selected_years or None)
                    if kriged:
                        st.success("Kriging completed.")
                        for yr, p in kriged.items():
                            st.write(f"Year {yr}:")
                            st.code(str(p), language="bash")
                    else:
                        st.info("No kriged outputs were generated.")
                except Exception as exc:  # noqa: BLE001
                    st.error(f"Kriging failed: {exc}")

        # ------------------------------------------------------------------
        # ICON Forecast Processing
        # ------------------------------------------------------------------
        st.markdown("---")
        st.markdown("**ICON Forecast Processing**")
        st.caption(
            "Transform raw ICON ensemble NetCDF into 10-minute UGRID Parquet files "
            "matching the RMCOMP schema (ID, time, rainrate)."
        )
        icon_nc_path_str = st.text_input(
            "ICON NetCDF file path",
            value="",
            placeholder="/path/to/ICON_ENS_2026011200.nc",
            help="Full path to the ICON ensemble NetCDF file.",
        )
        process_icon = st.button("Process ICON Ensemble", type="primary", key="process_icon")
        if process_icon and icon_nc_path_str.strip():
            icon_nc_path = Path(icon_nc_path_str.strip()).expanduser()
            ugrid_path = basin_folder / "output" / "ugrid" / "final_ugrid.parquet"
            icon_output_dir = basin_folder / "output" / "rain" / "icon_forecasts"
            if not ugrid_path.is_file():
                st.error(f"UGRID file not found: {ugrid_path}. Run UGRID & Terrain page first.")
            elif not icon_nc_path.is_file():
                st.error(f"ICON file not found: {icon_nc_path}")
            else:
                progress_placeholder = st.empty()
                status_placeholder = st.empty()

                def icon_progress(member_done: int, total: int) -> None:
                    progress_placeholder.progress(member_done / total, text=f"Member {member_done}/{total}")
                    status_placeholder.caption(f"Processing RAINC_{member_done:02d} …")

                try:
                    preprocessor = ICONPreprocessor(
                        ugrid_path=ugrid_path,
                        output_dir=icon_output_dir,
                        progress_callback=icon_progress,
                    )
                    written = preprocessor.process_nc(icon_nc_path)
                    progress_placeholder.progress(1.0, text="Done.")
                    status_placeholder.empty()
                    if written:
                        st.success(f"Wrote {len(written)} Parquet file(s) to: `{icon_output_dir}`")
                        st.code("\n".join(str(p) for p in written[:5]) + ("\n..." if len(written) > 5 else ""), language="text")
                        preview_path = written[0]
                        df_preview = pd.read_parquet(preview_path)
                        st.markdown("**Preview (first member)**")
                        st.dataframe(df_preview.head(100), use_container_width=True, hide_index=True)
                        st.caption(f"Columns: {list(df_preview.columns)}; dtypes: ID={df_preview['ID'].dtype}, time={df_preview['time'].dtype}, rainrate={df_preview['rainrate'].dtype}")
                    else:
                        st.warning("No Parquet files were written (RAINC_01..20 not found or empty).")
                except Exception as exc:  # noqa: BLE001
                    st.error(f"ICON processing failed: {exc}")
                    raise

        st.markdown("---")
        st.info(
            "The Model Playground page expects these event-based RMcomp "
            "parquets as the primary training data source."
        )

    with col_status:
        st.subheader("Current RMcomp outputs")

        df_yearly, df_events = _summarise_rain_outputs(rain_output_dir)

        st.markdown("**Yearly rain parquet files**")
        if df_yearly.empty:
            st.info("No `rain_{year}.parquet` files found yet.")
        else:
            st.dataframe(
                df_yearly[["year", "file"]],
                use_container_width=True,
                hide_index=True,
            )

        st.markdown("---")
        st.markdown("**Event-based rain parquet files**")
        if df_events.empty:
            st.info("No `rain_events_{year}.parquet` files found yet.")
        else:
            st.dataframe(
                df_events[["year", "file"]],
                use_container_width=True,
                hide_index=True,
            )

        st.markdown("---")
        st.caption(
            "RMcomp products are historical, high-resolution rain estimates. "
            "They underpin supervised training, while ICON forecasts feed "
            "the operational Forecast Center."
        )

    # ------------------------------------------------------------------
    # Visual Verification & Sanity Check
    # ------------------------------------------------------------------
    st.markdown("---")
    st.subheader("Visual Verification & Sanity Check")

    event_dir = rain_output_dir / "event_rain"
    event_files = (
        sorted(event_dir.glob("rain_events_*.parquet"))
        if event_dir.is_dir()
        else []
    )

    if not event_files:
        st.info(
            "No event-based RMcomp files found in "
            f"`{event_dir}`. Generate them first, then refresh this page."
        )
        return

    # 1) File selector
    file_labels = [f.name for f in event_files]
    selected_idx = st.selectbox(
        "Select event-based RMcomp file",
        options=list(range(len(event_files))),
        format_func=lambda i: file_labels[i],
    )
    selected_file = event_files[selected_idx]

    # Load once per selection
    df_ev = pd.read_parquet(selected_file)
    st.write(f"Loaded `{selected_file.name}` with **{len(df_ev):,}** rows.")

    # 2) General stats
    if "event_id" in df_ev.columns:
        n_events = df_ev["event_id"].nunique()
        st.write(f"Unique events (`event_id`): **{n_events}**")
    else:
        st.warning("Column `event_id` not found in this file.")
        n_events = 0

    st.markdown("**Data preview (head):**")
    st.dataframe(df_ev.head(), use_container_width=True, hide_index=True)

    # Storm library validation report (CSV vs event-based parquet)
    if events_csv_str:
        try:
            pre_for_report = RmcompRainPreprocessor(
                _build_rmcomp_config(
                    base_cfg,
                    basin_folder=basin_folder,
                    rmcomp_dir=rmcomp_dir,
                    rain_output_dir=rain_output_dir,
                    events_csv=events_csv_str or None,
                    antecedent_hours=antecedent_hours,
                )
            )
            events_all = pre_for_report._load_and_clean_events()  # noqa: SLF001
            # Determine year from selected_file name (rain_events_{year}.parquet)
            year_str = "".join(ch for ch in selected_file.stem if ch.isdigit())
            year_int: Optional[int] = int(year_str) if year_str else None
            if year_int is not None:
                events_year = events_all[events_all["year"] == year_int]
                n_csv_events = len(events_year)
                n_parquet_events = n_events
                st.markdown("### Storm library validation (CSV vs parquet)")
                st.write(
                    f"Year **{year_int}** – CSV has **{n_csv_events}** valid events after cleaning; "
                    f"parquet contains **{n_parquet_events}** unique `event_id` values."
                )
        except Exception as exc:  # noqa: BLE001
            st.warning(f"Storm library validation could not be computed: {exc}")

    if n_events == 0:
        st.info("No events to inspect in this file.")
        return

    # Ensure time typed
    if "time" in df_ev.columns:
        df_ev["time"] = pd.to_datetime(df_ev["time"], errors="coerce")

    # 3) Temporal validation (antecedent extension)
    st.markdown("### Temporal validation (antecedent extension)")

    event_ids = sorted(df_ev["event_id"].dropna().unique())
    selected_event_id = st.selectbox(
        "Select event_id for inspection",
        options=event_ids,
    )

    sub_ev = df_ev[df_ev["event_id"] == selected_event_id].copy()
    if sub_ev.empty:
        st.warning("No rows found for the selected event.")
        return

    # Expect event_st / event_ed columns as in notebook
    event_st = sub_ev.get("event_st", pd.NaT).iloc[0]
    event_ed = sub_ev.get("event_ed", pd.NaT).iloc[0]
    t_min = sub_ev["time"].min()
    t_max = sub_ev["time"].max()

    col_t1, col_t2 = st.columns(2)
    with col_t1:
        st.markdown("**Event metadata (from CSV / cleaning):**")
        st.write(f"event_st: `{event_st}`")
        st.write(f"event_ed: `{event_ed}`")
        with col_t2:
            st.markdown("**Actual rain time span (after antecedent extension):**")
        st.write(f"time.min(): `{t_min}`")
        st.write(f"time.max(): `{t_max}`")

    # 4) Spatial map (rain snapshot)
    st.markdown("### Spatial snapshot (lon–lat rain field)")

    # Limit options to finite times for cleaner UI
    unique_times = sorted(sub_ev["time"].dropna().unique())
    if not unique_times:
        st.info("No valid timestamps found for this event.")
        return

    # For long events, allow selecting a subset via slider index
    idx_time = st.slider(
        "Select time index within event",
        min_value=0,
        max_value=len(unique_times) - 1,
        value=len(unique_times) // 2,
    )
    selected_time = unique_times[idx_time]
    st.write(f"Selected timestamp: `{selected_time}`")

    snap = sub_ev[sub_ev["time"] == selected_time].copy()
    if snap.empty:
        st.info("No rain samples at this exact timestamp.")
    else:
        if {"lon", "lat", "rainrate"}.issubset(snap.columns):
            fig = px.scatter(
                snap,
                x="lon",
                y="lat",
                color="rainrate",
                color_continuous_scale="Blues",
                title="RMcomp rain snapshot",
                labels={"rainrate": "rainrate"},
            )
            fig.update_layout(height=450)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning(
                "Expected columns `lon`, `lat`, `rainrate` are not all present; "
                "cannot render spatial snapshot."
            )

    # ------------------------------------------------------------------
    # Kriged rain visualization on UGRID
    # ------------------------------------------------------------------
    st.markdown("---")
    st.subheader("Kriged Rain on UGRID (UGRID cells)")

    basin_path = Path(base_cfg.get("data_paths", {}).get("basin_folder", basin_folder_str)).expanduser()
    kriged_dir = rain_output_dir / "event_rain" / "intepulated_rain_on_ugrid"
    kriged_files = (
        sorted(kriged_dir.glob("rain_kriged_*.parquet"))
        if kriged_dir.is_dir()
        else []
    )

    if not kriged_files:
        st.info(
            "No kriged rain files found. Run the kriging step above, then refresh this page."
        )
    else:
        kriged_labels = [p.name for p in kriged_files]
        k_idx = st.selectbox(
            "Select kriged rain file",
            options=list(range(len(kriged_files))),
            format_func=lambda i: kriged_labels[i],
        )
        kriged_file = kriged_files[k_idx]
        df_k = pd.read_parquet(kriged_file)
        if "time" in df_k.columns:
            df_k["time"] = pd.to_datetime(df_k["time"], errors="coerce")

        # Load UGRID centroids for mapping
        ugrid_path = basin_path / "output" / "ugrid" / "final_ugrid.parquet"
        if not ugrid_path.is_file():
            st.warning(f"UGRID file not found for visualization: {ugrid_path}")
        else:
            ugrid = pd.read_parquet(ugrid_path)
            if {"ID", "lon", "lat"}.issubset(ugrid.columns) and not df_k.empty:
                times_k = sorted(df_k["time"].dropna().unique())
                if times_k:
                    tk_idx = st.slider(
                        "Select time index for kriged snapshot",
                        min_value=0,
                        max_value=len(times_k) - 1,
                        value=len(times_k) // 2,
                    )
                    tk = times_k[tk_idx]
                    st.write(f"Kriged timestamp: `{tk}`")
                    snap_k = df_k[df_k["time"] == tk].copy()
                    merged = snap_k.merge(ugrid[["ID", "lon", "lat"]], on="ID", how="left")
                    if {"lon", "lat", "rainrate"}.issubset(merged.columns):
                        fig_k = px.scatter(
                            merged,
                            x="lon",
                            y="lat",
                            color="rainrate",
                            color_continuous_scale="Blues",
                            title="Kriged rain on UGRID cells",
                            labels={"rainrate": "rainrate"},
                        )
                        fig_k.update_layout(height=450)
                        st.plotly_chart(fig_k, use_container_width=True)
                    else:
                        st.warning("Merged kriged/UGRID data missing lon/lat/rainrate.")
                else:
                    st.info("No valid timestamps found in kriged file.")
            else:
                st.warning("UGRID or kriged file missing required columns for visualization.")

    # ------------------------------------------------------------------
    # Storm-library vs parquet validation
    # ------------------------------------------------------------------
    st.markdown("---")
    st.subheader("Storm Library vs. Parquet Validation")

    try:
        pre_for_validation = RmcompRainPreprocessor(cfg_rm)
        events_clean = pre_for_validation._load_and_clean_events()  # type: ignore[attr-defined]
    except Exception as exc:  # noqa: BLE001
        st.warning(f"Could not load/clean events for validation: {exc}")
        return

    if events_clean.empty:
        st.info("No valid/corrected events found in the storm library CSV.")
        return

    stem = selected_file.stem
    digits = "".join(ch for ch in stem if ch.isdigit())
    if len(digits) != 4:
        st.info("Could not infer year from selected event rain filename.")
        return
    year_sel = int(digits)

    events_year = events_clean[events_clean["year"] == year_sel].copy()
    n_events_csv = len(events_year)
    if "event_id" in df_ev.columns:
        n_events_parquet = df_ev["event_id"].nunique()
    else:
        n_events_parquet = 0

    st.write(
        f"For year **{year_sel}**: CSV after cleaning has **{n_events_csv}** events; "
        f"`rain_events_{year_sel}.parquet` has **{n_events_parquet}** unique `event_id`."
    )

    csv_ids = set(events_year.index.tolist())
    parquet_ids = set(df_ev["event_id"].unique().tolist()) if "event_id" in df_ev.columns else set()

    missing_in_parquet = sorted(csv_ids - parquet_ids)
    extra_in_parquet = sorted(parquet_ids - csv_ids)

    if not missing_in_parquet and not extra_in_parquet:
        st.success(
            f"Event IDs match perfectly for year {year_sel} between the storm library and parquet."
        )
    else:
        if missing_in_parquet:
            st.warning(
                f"{len(missing_in_parquet)} event(s) present in CSV but missing in parquet: "
                f"{missing_in_parquet[:10]}{' ...' if len(missing_in_parquet) > 10 else ''}"
            )
        if extra_in_parquet:
            st.warning(
                f"{len(extra_in_parquet)} event_id(s) present in parquet but not in CSV: "
                f"{extra_in_parquet[:10]}{' ...' if len(extra_in_parquet) > 10 else ''}"
            )


if __name__ == "__main__":
    main()

