from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List

import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
import yaml

from src.preprocess import UgridPreprocessor


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
        <div class="main-header">UGRID & Terrain Builder</div>
        <div class="sub-header">
            Generate the QuadTree UGRID mesh, attach terrain attributes, and
            derive D50 and Manning's n exactly as in the research notebook.
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.write("")


def _suggest_paths_for_basin(basin_folder: Path) -> Dict[str, Path]:
    """Mirror the conventions used in system_config.yaml for a given basin."""
    return {
        "basin_folder": basin_folder,
        "streams_path": basin_folder / "streams_clipped.shp",
        "info_path": basin_folder / "info.txt",
        "dem_input_path": basin_folder
        / "clipped_dem"
        / "DEM_clip_4326_100m_buffer.tif",
        "ugrid_output_dir": basin_folder / "output" / "ugrid",
        "dem_output_dir": basin_folder / "output" / "dem_output",
    }


def _override_data_paths(
    base_config: Dict[str, Any],
    basin_folder_str: str,
) -> Dict[str, Any]:
    cfg = deepcopy(base_config)
    data_paths: Dict[str, Any] = dict(cfg.get("data_paths", {}))

    basin_folder = Path(basin_folder_str).expanduser()
    suggested = _suggest_paths_for_basin(basin_folder)

    data_paths["basin_folder"] = str(suggested["basin_folder"])
    data_paths["streams_path"] = str(suggested["streams_path"])
    data_paths["info_path"] = str(suggested["info_path"])
    data_paths["dem_input_path"] = str(suggested["dem_input_path"])
    data_paths["ugrid_output_dir"] = str(suggested["ugrid_output_dir"])
    data_paths["dem_output_dir"] = str(suggested["dem_output_dir"])

    cfg["data_paths"] = data_paths
    return cfg


def _summarise_ugrid_outputs(basin_folder: Path) -> Dict[str, Any]:
    paths = _suggest_paths_for_basin(basin_folder)
    ugrid_dir = paths["ugrid_output_dir"]
    summary: Dict[str, Any] = {
        "output_dir": str(ugrid_dir),
        "exists": ugrid_dir.is_dir(),
        "files": [],
    }
    if ugrid_dir.is_dir():
        files: List[str] = []
        for p in sorted(ugrid_dir.glob("*")):
            files.append(p.name)
        summary["files"] = files
    return summary


def _load_ugrid_parquet(basin_folder: Path) -> gpd.GeoDataFrame | None:
    """Load UGRID GeoParquet with full postprocessed attributes if it exists."""
    paths = _suggest_paths_for_basin(basin_folder)
    ugrid_dir = paths["ugrid_output_dir"]
    # Prefer the postprocessed file with D50 and Manning's n; fall back to terrain-only.
    d50_path = ugrid_dir / "ugrid_cells_with_d50.parquet"
    terrain_path = ugrid_dir / "ugrid_cells_with_terrain.parquet"
    if d50_path.is_file():
        return gpd.read_parquet(d50_path)
    if terrain_path.is_file():
        return gpd.read_parquet(terrain_path)
    return None


def _terrain_columns(gdf: gpd.GeoDataFrame) -> List[str]:
    candidates = [
        "DEM_MEAN",
        "SLOPE_MEAN",
        "ASPECT_MEAN",
        "FLOWACC_MEAN",
        "FLOWDIR_MEAN",
        "RUGGED_MEAN",
        "AREA_2M",
        "STRM_ORDER",
        "STREAM_CELL",
        "D50MM",
        "MANNING_N",
    ]
    return [c for c in candidates if c in gdf.columns]


def main() -> None:
    st.set_page_config(
        page_title="UGRID & Terrain",
        page_icon="🧩",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    base_config = _load_config(CONFIG_PATH)

    _build_header()

    st.sidebar.title("Basin selection & paths")

    default_basin = str(base_config.get("data_paths", {}).get("basin_folder", "")) or str(
        Path("/media/data-nvme/Darga_28_for_test_only")
    )
    basin_folder_str = st.sidebar.text_input(
        "Basin folder",
        value=default_basin,
        help="Root folder for the basin (contains info.txt, streams_clipped.shp, output/, raw/ etc.).",
    )
    basin_folder = Path(basin_folder_str).expanduser()

    suggested = _suggest_paths_for_basin(basin_folder)

    cfg = _override_data_paths(
        base_config,
        basin_folder_str=basin_folder_str,
    )

    st.sidebar.markdown("---")
    st.sidebar.title("UGRID parameters")
    st.sidebar.caption(
        "Values are initialised from the Darga notebook configuration."
    )

    ugrid_cfg = base_config.get("preprocessing", {}).get("ugrid", {})
    quad_threshold = st.sidebar.number_input(
        "Quad threshold (points per cell)",
        min_value=1,
        max_value=50,
        value=int(ugrid_cfg.get("quad_threshold", 5)),
        step=1,
    )
    quad_max_depth = st.sidebar.number_input(
        "Quad max depth",
        min_value=4,
        max_value=20,
        value=int(ugrid_cfg.get("quad_max_depth", 12)),
        step=1,
    )
    quad_min_stream_order = st.sidebar.number_input(
        "Minimum stream order",
        min_value=1,
        max_value=10,
        value=int(ugrid_cfg.get("quad_min_stream_order", 3)),
        step=1,
    )
    quad_spacing = st.sidebar.number_input(
        "Stream densification spacing [m]",
        min_value=1,
        max_value=50,
        value=int(ugrid_cfg.get("quad_spacing", 4)),
        step=1,
    )

    col_inputs, col_steps, col_outputs = st.columns([1.1, 1.2, 1.1])

    with col_inputs:
        st.subheader("Required inputs")
        st.write(f"**Basin folder:** `{basin_folder}`")

        st.markdown("**Key files (expected):**")
        expected_files = {
            "info.txt": suggested["info_path"],
            "streams_clipped.shp": suggested["streams_path"],
            "DEM (clipped)": suggested["dem_input_path"],
        }
        for label, path in expected_files.items():
            exists = "✅" if path.exists() else "⚠️"
            st.write(f"{exists} `{label}` → `{path}`")

        st.markdown("---")
        st.markdown("**UGRID output dir (expected):**")
        st.code(str(suggested["ugrid_output_dir"]), language="bash")

    with col_steps:
        st.subheader("Processing steps")

        st.markdown("**Step 1 – Generate UGRID + terrain + hydraulics**")
        run_ugrid = st.button("Run UGRID preprocessing", type="primary")
        if run_ugrid:
            status_placeholder = st.empty()
            with st.spinner("Running UGRID preprocessing (mesh + terrain + hydraulics)..."):
                try:
                    ugrid_pre = UgridPreprocessor(cfg)
                    # Override UGRID hyperparameters to match UI selections / Darga notebook
                    ugrid_pre._ugrid_cfg.quad_threshold = int(quad_threshold)  # type: ignore[attr-defined]
                    ugrid_pre._ugrid_cfg.quad_max_depth = int(quad_max_depth)  # type: ignore[attr-defined]
                    ugrid_pre._ugrid_cfg.quad_min_stream_order = int(
                        quad_min_stream_order
                    )  # type: ignore[attr-defined]
                    ugrid_pre._ugrid_cfg.quad_spacing = int(quad_spacing)  # type: ignore[attr-defined]
                    status_placeholder.write("• Building QuadTree UGRID mesh and computing cell attributes...")
                    ugrid_parquet = ugrid_pre.run()
                    status_placeholder.write(
                        "• Sampling DEM & terrain derivatives, assigning D50, and computing Manning's n..."
                    )
                    st.success(f"UGRID completed: `{ugrid_parquet}`")
                except Exception as exc:  # noqa: BLE001
                    st.error(f"UGRID preprocessing failed: {exc}")

        st.markdown("---")
        st.info(
            "Once the UGRID & terrain pipeline completes successfully, the RMcomp "
            "Rain Grids and Model Playground pages can consume the prepared "
            "static products for this basin."
        )

    with col_outputs:
        st.subheader("Current outputs")

        ugrid_summary = _summarise_ugrid_outputs(basin_folder)
        st.markdown("**UGRID artefacts**")
        st.write(f"Directory: `{ugrid_summary['output_dir']}`")
        if not ugrid_summary["exists"]:
            st.warning("UGRID output directory does not exist yet.")
        else:
            files = ugrid_summary.get("files", [])
            if not files:
                st.info("No files found in UGRID output directory.")
            else:
                st.write("Files:")
                for name in files:
                    st.write(f"- `{name}`")

        st.markdown("---")
        st.subheader("UGRID & terrain preview")

        gdf = _load_ugrid_parquet(basin_folder)
        if gdf is None:
            st.info("No `ugrid_cells_with_terrain.parquet` found yet.")
        else:
            st.write(f"Cells: **{len(gdf)}**")
            terrain_cols = _terrain_columns(gdf)
            # Ensure D50MM and MANNING_N are explicitly visible if present.
            preview_cols: List[str] = ["ID"]
            for col in ["D50MM", "MANNING_N"]:
                if col in gdf.columns:
                    preview_cols.append(col)
            preview_cols += [c for c in terrain_cols if c not in preview_cols]
            st.markdown("**Sample of UGRID attributes (first 10 cells):**")
            st.dataframe(
                pd.DataFrame(gdf[preview_cols].head(10)),
                hide_index=True,
                use_container_width=True,
            )

            with st.expander("Map view (UGRID geometry)", expanded=False):
                try:
                    gdf_4326 = gdf.to_crs("EPSG:4326")
                    gdf_4326["lat"] = gdf_4326.geometry.centroid.y
                    gdf_4326["lon"] = gdf_4326.geometry.centroid.x
                    st.map(gdf_4326[["lat", "lon"]], zoom=10)
                except Exception as exc:  # noqa: BLE001
                    st.error(f"Failed to render map view: {exc}")

            with st.expander("Field-based plot (static)", expanded=False):
                field_options = _terrain_columns(gdf) or ["AREA_2M"]
                field = st.selectbox(
                    "Field to plot",
                    options=field_options,
                    index=field_options.index("AREA_2M")
                    if "AREA_2M" in field_options
                    else 0,
                )
                try:
                    fig, ax = plt.subplots(figsize=(6, 5))
                    gdf.plot(
                        column=field,
                        ax=ax,
                        cmap="viridis",
                        legend=True,
                        edgecolor="none",
                    )
                    ax.set_axis_off()
                    ax.set_title(f"UGRID – {field}")
                    st.pyplot(fig, clear_figure=True)
                except Exception as exc:  # noqa: BLE001
                    st.error(f"Failed to plot UGRID field '{field}': {exc}")


if __name__ == "__main__":
    main()

