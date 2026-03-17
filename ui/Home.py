from __future__ import annotations

from pathlib import Path

import streamlit as st


def main() -> None:
    """Landing page for the multi-page hydrological MLOps app."""
    st.set_page_config(
        page_title="Hydro Forecasting Platform",
        page_icon="🌧️",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.title("Hydrological Forecasting Platform")
    st.markdown(
        """
        This application is organised into four tightly-coupled pages:

        - **1 – UGRID & Terrain**: Build the QuadTree UGRID mesh, attach terrain, D50 and Manning's *n*
          exactly as in the research notebook.
        - **2 – Rain Grids (RMcomp)**: Project yearly RMcomp ``RM{year}.nc`` files onto the UGRID and
          derive event-based rain cubes for supervised learning.
        - **3 – Model Playground**: Train and calibrate the two-stage LSTM models on historical
          RMcomp event data, track NSE/loss, and promote runs to production.
        - **4 – Forecast Center**: Run operational ensemble forecasts using ICON data and the
          promoted production checkpoints.
        """
    )

    st.markdown(
        "Use the navigation menu on the left to open any of the four pages."
    )

    project_root = Path(__file__).resolve().parents[1]
    basins_dir = project_root / "configs" / "basins"
    if basins_dir.is_dir():
        basin_files = sorted(basins_dir.glob("*.yaml"))
        if basin_files:
            st.subheader("Available basins")
            for cfg in basin_files:
                st.write(f"- `{cfg.stem}` → `{cfg.name}`")


if __name__ == "__main__":
    main()

