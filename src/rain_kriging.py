from __future__ import annotations

"""
Exact port of the RMcomp → UGRID kriging logic from
MY_LAST_JUPYTER.ipynb (cell titled `# KRIGING`).

This module takes point-based RMcomp event rain
(`rain_events_{year}.parquet` with columns
`time, lat, lon, rainrate`) and interpolates it onto the
UGRID cell centroids (`final_ugrid.parquet` with `ID, lon, lat`)
using Ordinary Kriging (pykrige) with a spherical variogram
model, exactly as in the notebook.

The implementation here is intentionally sequential and follows
the same `for` loop + `try/except` structure as the notebook
cell, without any parallelisation or math changes.
"""

import logging
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd
from pykrige.ok import OrdinaryKriging


def ordinary_kriging(rain_df: pd.DataFrame, grid_lat: np.ndarray, grid_lon: np.ndarray) -> np.ndarray:
    """
    Exact port from notebook cell `# KRIGING` – ordinary_kriging.
    """
    OK = OrdinaryKriging(
        rain_df["lat"].values,
        rain_df["lon"].values,
        rain_df["rainrate"].values,
        variogram_model="spherical",
        verbose=False,
        enable_plotting=False,
    )
    z, _ = OK.execute("points", grid_lat, grid_lon)
    return np.asarray(z)


def run_kriging_for_years(
    basin_folder: Path,
    years: Optional[Iterable[int]] = None,
) -> Dict[int, Path]:
    """
    Run kriging for one or more years, mirroring the notebook logic.

    Parameters
    ----------
    basin_folder : Path
        Basin root folder (`folder` variable in the notebook).
    years : iterable of int, optional
        Years to process. If None, all years with
        `rain_events_{year}.parquet` present are processed.

    Returns
    -------
    Dict[int, Path]
        Mapping from year to the kriged parquet path
        (`rain_kriged_{year}.parquet`).
    """
    basin_folder = basin_folder.expanduser()
    ugrid_path = basin_folder / "output" / "ugrid" / "final_ugrid.parquet"
    event_rain_folder = basin_folder / "output" / "rain" / "event_rain"
    output_dir = event_rain_folder / "intepulated_rain_on_ugrid"
    output_dir.mkdir(parents=True, exist_ok=True)

    if not ugrid_path.is_file():
        raise FileNotFoundError(f"UGRID file not found for kriging: {ugrid_path}")
    if not event_rain_folder.is_dir():
        raise FileNotFoundError(
            f"Event rain folder not found for kriging: {event_rain_folder}"
        )

    # Load basin-shaped UGRID
    ugrid = pd.read_parquet(ugrid_path)
    grid_lon = ugrid["lon"].to_numpy()
    grid_lat = ugrid["lat"].to_numpy()
    grid_id = ugrid["ID"].to_numpy()

    rain_files = sorted(event_rain_folder.glob("rain_events_*.parquet"))
    if years is not None:
        years = set(int(y) for y in years)
        rain_files = [p for p in rain_files if int("".join(ch for ch in p.stem if ch.isdigit())) in years]

    outputs: Dict[int, Path] = {}

    for rain_file in rain_files:
        stem = rain_file.stem  # e.g. rain_events_2015
        digits = "".join(ch for ch in stem if ch.isdigit())
        if len(digits) != 4:
            continue
        year = int(digits)
        kriged_output_file = output_dir / f"rain_kriged_{year}.parquet"
        if kriged_output_file.exists():
            outputs[year] = kriged_output_file
            continue

        df_rain = pd.read_parquet(rain_file, columns=["time", "lat", "lon", "rainrate"])
        if df_rain.empty:
            continue
        all_times = sorted(df_rain["time"].unique())

        rows: List[Dict[str, object]] = []
        for t in all_times:
            sub = df_rain[df_rain["time"] == t]
            try:
                if len(sub) < 3:
                    z = np.full_like(grid_lon, sub["rainrate"].mean(), dtype=float)
                elif sub["rainrate"].std() < 1e-8:
                    z = np.full_like(grid_lon, sub["rainrate"].mean(), dtype=float)
                else:
                    z = ordinary_kriging(sub, grid_lat, grid_lon)
            except Exception as e:
                logging.warning("Kriging failed at %s: %s, fallback to mean", t, e)
                z = np.full_like(grid_lon, sub["rainrate"].mean(), dtype=float)

            # Clip negative rain values to zero, as in the notebook.
            z = np.clip(z, 0.0, None)

            for cid, rr in zip(grid_id, z, strict=False):
                rows.append(
                    {
                        "ID": int(cid),
                        "time": pd.Timestamp(t),
                        "rainrate": float(rr),
                    }
                )

        if rows:
            df_out = pd.DataFrame(rows)
            df_out["ID"] = df_out["ID"].astype("int32")
            df_out["rainrate"] = df_out["rainrate"].astype("float32")
            df_out = df_out.sort_values(["ID", "time"]).reset_index(drop=True)
            df_out.to_parquet(kriged_output_file, index=False)
            outputs[year] = kriged_output_file

    return outputs

