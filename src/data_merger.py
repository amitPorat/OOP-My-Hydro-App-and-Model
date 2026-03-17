"""
Exact 1:1 port of the discharge processing and rain-discharge merge workflow
from MY_LAST_JUPYTER.ipynb (cells after KRIGING: DISCHARGE DATA PROCESSING and
MERGE RAIN AND DISCHARGE).

No alterations to pandas/numpy logic: same resampling, time alignment,
merge-by-time mapping, and save steps as in the notebook.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from geopandas import GeoDataFrame
from scipy.spatial import cKDTree
from shapely import wkt
from shapely.geometry import Point

logger = logging.getLogger(__name__)

# Default area name mapping (notebook: basin folder name -> discharge CSV area name)
DEFAULT_AREA_NAME_MAPPING: Dict[str, str] = {
    "Darga_28_for_test_only": "Northern Dead Sea",
}


def process_discharge_for_years(
    basin_folder: Path,
    discharge_csv_path: Path,
    metadata_path: Path,
    kriged_rain_folder: Path,
    discharge_output_dir: Path,
    area_name_mapping: Optional[Mapping[str, str]] = None,
    years: Optional[Iterable[int]] = None,
    force: bool = False,
) -> Tuple[int, List[Path]]:
    """
    Port of the notebook DISCHARGE DATA PROCESSING cell.

    Steps (exact notebook logic):
    1. Load basin polygon from info.txt (line 1 WKT).
    2. Load discharge CSV; parse 'date' with format '%Y/%m/%d %H:%M:%S', dropna(subset=['date']).
    3. Get discharge_station_ids from CSV 'shd_id'.unique().
    4. Load station metadata (Excel); filter to stations with discharge data.
    5. Create stations_gdf with Point(lon, lat), crs EPSG:4326; sjoin with basin (within).
    6. For each year present in kriged rain files: load rain_kriged_{year}.parquet columns ['time'],
       rain_timestamps = set(rain_df['time'].unique()).
    7. Filter discharge to that year (date.dt.year == year). Per station: set_index('date').sort_index(),
       resample('10min').mean(), reset_index, rename to 'time' and 'discharge', then
       station_resampled[station_resampled['time'].isin(rain_timestamps)].
    8. Concat per-year, sort_values(['shd_id', 'time']). Save discharge_processed_{year}.parquet
       with columns shd_id (int32), time, discharge (float32), zstd compression.

    Returns
    -------
    num_stations : int
        Number of stations in basin with discharge data.
    paths_created : list of Path
        Paths to written discharge_processed_{year}.parquet files.
    """
    basin_folder = Path(basin_folder).expanduser()
    discharge_output_dir = Path(discharge_output_dir).expanduser()
    discharge_output_dir.mkdir(parents=True, exist_ok=True)
    kriged_rain_folder = Path(kriged_rain_folder).expanduser()
    info_path = basin_folder / "info.txt"

    if not info_path.is_file():
        raise FileNotFoundError(f"Basin info file not found: {info_path}")
    with open(info_path, "r", encoding="utf-8") as f:
        basin_wkt = f.readline().strip()
    basin_poly = wkt.loads(basin_wkt)
    basin_gdf = GeoDataFrame(geometry=[basin_poly], crs="EPSG:4326")

    if not Path(discharge_csv_path).expanduser().is_file():
        logger.warning("Discharge CSV not found: %s", discharge_csv_path)
        return 0, []

    discharge_df_all = pd.read_csv(Path(discharge_csv_path).expanduser())
    discharge_station_ids = sorted(discharge_df_all["shd_id"].unique())

    stations_df = pd.read_excel(Path(metadata_path).expanduser())
    if discharge_station_ids:
        stations_df = stations_df[stations_df["shd_id"].isin(discharge_station_ids)].copy()
    else:
        return 0, []

    if "lat" not in stations_df.columns or "lon" not in stations_df.columns:
        raise ValueError("Station metadata missing 'lat' or 'lon' columns")

    geometry_stations = [
        Point(lon, lat) for lon, lat in zip(stations_df["lon"], stations_df["lat"], strict=False)
    ]
    stations_gdf = GeoDataFrame(stations_df, geometry=geometry_stations, crs="EPSG:4326")
    stations_in_basin = stations_gdf.sjoin(basin_gdf, how="inner", predicate="within")
    stations_in_basin = stations_in_basin.drop(columns=["index_right"])

    num_stations = len(stations_in_basin)
    if num_stations == 0:
        return 0, []

    station_ids = stations_in_basin["shd_id"].tolist()

    # Parse date exactly as notebook
    discharge_df_all["date"] = pd.to_datetime(
        discharge_df_all["date"], format="%Y/%m/%d %H:%M:%S", errors="coerce"
    )
    discharge_df_all = discharge_df_all.dropna(subset=["date"])
    discharge_df = discharge_df_all[discharge_df_all["shd_id"].isin(station_ids)].copy()

    if discharge_df.empty:
        return num_stations, []

    kriged_files = sorted(kriged_rain_folder.glob("rain_kriged_*.parquet"))
    if not kriged_files:
        logger.warning("No kriged rain files found; cannot filter discharge to rain timestamps.")
        return num_stations, []

    years_set = set(years) if years is not None else None
    all_discharge_resampled_list: List[pd.DataFrame] = []

    for kriged_file in kriged_files:
        year = int(kriged_file.stem.split("_")[2])
        if years_set is not None and year not in years_set:
            continue

        rain_df = pd.read_parquet(kriged_file, columns=["time"])
        rain_df["time"] = pd.to_datetime(rain_df["time"])
        rain_timestamps = set(rain_df["time"].unique())

        discharge_year = discharge_df[discharge_df["date"].dt.year == year].copy()
        if discharge_year.empty:
            continue

        discharge_resampled_list: List[pd.DataFrame] = []
        for station_id in station_ids:
            station_discharge = discharge_year[discharge_year["shd_id"] == station_id].copy()
            if station_discharge.empty:
                continue
            station_discharge = station_discharge.set_index("date").sort_index()
            station_resampled = station_discharge[["rg_qms"]].resample("10min").mean()
            station_resampled = station_resampled.reset_index()
            time_col = station_resampled.columns[0]
            station_resampled = station_resampled.rename(
                columns={time_col: "time", "rg_qms": "discharge"}
            )
            station_resampled["shd_id"] = station_id
            station_resampled = station_resampled[
                station_resampled["time"].isin(rain_timestamps)
            ].copy()
            if not station_resampled.empty:
                discharge_resampled_list.append(station_resampled)

        if discharge_resampled_list:
            year_discharge = pd.concat(discharge_resampled_list, ignore_index=True)
            year_discharge = year_discharge.sort_values(["shd_id", "time"])
            all_discharge_resampled_list.append(year_discharge)

    if not all_discharge_resampled_list:
        return num_stations, []

    discharge_resampled = pd.concat(all_discharge_resampled_list, ignore_index=True)
    discharge_resampled = discharge_resampled.sort_values(["shd_id", "time"])
    discharge_resampled["year"] = discharge_resampled["time"].dt.year

    paths_created: List[Path] = []
    for yr in sorted(discharge_resampled["year"].unique()):
        out_file = discharge_output_dir / f"discharge_processed_{yr}.parquet"
        if out_file.exists() and not force:
            paths_created.append(out_file)
            continue
        year_data = discharge_resampled[discharge_resampled["year"] == yr].copy()
        if year_data.empty:
            continue
        year_data = year_data[["shd_id", "time", "discharge"]].copy()
        year_data["shd_id"] = year_data["shd_id"].astype("int32")
        year_data["discharge"] = year_data["discharge"].astype("float32")
        table = pa.Table.from_pandas(year_data, preserve_index=False)
        with pq.ParquetWriter(
            out_file, table.schema, compression="zstd", compression_level=3
        ) as writer:
            writer.write_table(table)
        paths_created.append(out_file)

    return num_stations, paths_created


def merge_rain_and_discharge_for_years(
    basin_folder: Path,
    ugrid_parquet_path: Path,
    kriged_rain_folder: Path,
    discharge_folder: Path,
    metadata_path: Path,
    output_dir: Path,
    years: Optional[Iterable[int]] = None,
    force: bool = False,
) -> Dict[int, Path]:
    """
    Port of the notebook MERGE RAIN AND DISCHARGE cell.

    Steps (exact notebook logic):
    1. Load UGRID (lon, lat, ID); cell_coords = column_stack([lon, lat]).
    2. Load station metadata; load basin polygon from info.txt; stations_gdf with Point(lon,lat);
       sjoin(stations_gdf, basin_gdf, how='inner', predicate='within').
    3. cKDTree(cell_coords); distances, indices = tree.query(station_coords, k=1);
       station_to_cell[station_id] = {cell_id, distance_km, cell_lon, cell_lat}.
    4. For each rain_kriged_{year}.parquet: load rain_df, rain_df['time'] = pd.to_datetime(rain_df['time']).
       Load discharge_processed_{year}.parquet; if missing, rain_df['discharge'] = np.nan (float32).
       Else: rain_df['discharge'] = np.nan; for each (station_id, cell_info): station_discharge =
       discharge_df[discharge_df['shd_id']==station_id]; discharge_dict = dict(zip(station_discharge['time'], station_discharge['discharge']));
       mask = rain_df['ID'] == cell_id; rain_df.loc[mask, 'discharge'] = rain_df.loc[mask, 'time'].map(discharge_dict).
       rain_df['discharge'] = rain_df['discharge'].astype('float32').
    5. rain_df_with_coords = rain_df.merge(ugrid_df[['ID','lon','lat']], on='ID', how='left');
       output_data = rain_df_with_coords[['ID','time','rainrate','discharge','lon','lat']];
       dtypes int32/float32; save rain_with_discharge_{year}.parquet with zstd.

    Returns
    -------
    Dict[int, Path]
        Year -> path to rain_with_discharge_{year}.parquet.
    """
    basin_folder = Path(basin_folder).expanduser()
    ugrid_parquet_path = Path(ugrid_parquet_path).expanduser()
    kriged_rain_folder = Path(kriged_rain_folder).expanduser()
    discharge_folder = Path(discharge_folder).expanduser()
    output_dir = Path(output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    info_path = basin_folder / "info.txt"

    if not ugrid_parquet_path.is_file():
        raise FileNotFoundError(f"UGRID file not found: {ugrid_parquet_path}")

    ugrid_df = pd.read_parquet(ugrid_parquet_path)
    if "lon" not in ugrid_df.columns or "lat" not in ugrid_df.columns:
        raise ValueError("UGRID must have 'lon' and 'lat' columns")
    cell_coords = np.column_stack([ugrid_df["lon"].values, ugrid_df["lat"].values])
    cell_ids = ugrid_df["ID"].values

    stations_df = pd.read_excel(Path(metadata_path).expanduser())
    if "lat" not in stations_df.columns or "lon" not in stations_df.columns:
        raise ValueError("Station metadata missing 'lat' or 'lon' columns")

    with open(info_path, "r", encoding="utf-8") as f:
        basin_wkt = f.readline().strip()
    basin_poly = wkt.loads(basin_wkt)
    basin_gdf = GeoDataFrame(geometry=[basin_poly], crs="EPSG:4326")

    geometry_stations = [
        Point(lon, lat) for lon, lat in zip(stations_df["lon"], stations_df["lat"], strict=False)
    ]
    stations_gdf = GeoDataFrame(stations_df, geometry=geometry_stations, crs="EPSG:4326")
    stations_in_basin = stations_gdf.sjoin(basin_gdf, how="inner", predicate="within")
    stations_in_basin = stations_in_basin.drop(columns=["index_right"])

    if stations_in_basin.empty:
        logger.warning("No stations in basin; merge will produce rain-only files.")
        station_to_cell: Dict[Any, Dict[str, Any]] = {}
    else:
        station_coords = np.column_stack(
            [stations_in_basin["lon"].values, stations_in_basin["lat"].values]
        )
        station_ids_arr = stations_in_basin["shd_id"].values
        tree = cKDTree(cell_coords)
        distances, indices = tree.query(station_coords, k=1)
        station_to_cell = {}
        for i, station_id in enumerate(station_ids_arr):
            nearest_cell_idx = indices[i]
            nearest_cell_id = cell_ids[nearest_cell_idx]
            distance_km = float(distances[i] * 111.0)
            station_to_cell[station_id] = {
                "cell_id": nearest_cell_id,
                "distance_km": distance_km,
                "cell_lon": float(cell_coords[nearest_cell_idx][0]),
                "cell_lat": float(cell_coords[nearest_cell_idx][1]),
            }

    kriged_rain_files = sorted(kriged_rain_folder.glob("rain_kriged_*.parquet"))
    years_set = set(years) if years is not None else None
    result: Dict[int, Path] = {}

    for rain_file in kriged_rain_files:
        year = int(rain_file.stem.split("_")[2])
        if years_set is not None and year not in years_set:
            continue
        output_file = output_dir / f"rain_with_discharge_{year}.parquet"
        if output_file.exists() and not force:
            result[year] = output_file
            continue

        rain_df = pd.read_parquet(rain_file)
        rain_df["time"] = pd.to_datetime(rain_df["time"])

        discharge_file = discharge_folder / f"discharge_processed_{year}.parquet"
        if not discharge_file.exists():
            rain_df["discharge"] = np.nan
            rain_df["discharge"] = rain_df["discharge"].astype("float32")
        else:
            discharge_df = pd.read_parquet(discharge_file)
            discharge_df["time"] = pd.to_datetime(discharge_df["time"])
            rain_df["discharge"] = np.nan
            for station_id, cell_info in station_to_cell.items():
                cell_id = cell_info["cell_id"]
                station_discharge = discharge_df[discharge_df["shd_id"] == station_id].copy()
                if station_discharge.empty:
                    continue
                cell_rain = rain_df[rain_df["ID"] == cell_id]
                if cell_rain.empty:
                    continue
                discharge_dict = dict(
                    zip(station_discharge["time"], station_discharge["discharge"], strict=False)
                )
                mask = rain_df["ID"] == cell_id
                rain_df.loc[mask, "discharge"] = rain_df.loc[mask, "time"].map(discharge_dict)
            rain_df["discharge"] = rain_df["discharge"].astype("float32")

        rain_df_with_coords = rain_df.merge(
            ugrid_df[["ID", "lon", "lat"]], on="ID", how="left"
        )
        output_data = rain_df_with_coords[
            ["ID", "time", "rainrate", "discharge", "lon", "lat"]
        ].copy()
        output_data["ID"] = output_data["ID"].astype("int32")
        output_data["rainrate"] = output_data["rainrate"].astype("float32")
        output_data["lon"] = output_data["lon"].astype("float32")
        output_data["lat"] = output_data["lat"].astype("float32")

        table = pa.Table.from_pandas(output_data, preserve_index=False)
        with pq.ParquetWriter(
            output_file, table.schema, compression="zstd", compression_level=3
        ) as writer:
            writer.write_table(table)
        result[year] = output_file

    return result
