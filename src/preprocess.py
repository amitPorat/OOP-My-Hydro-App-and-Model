from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List
import shutil

import geopandas as gpd
import numpy as np
import pandas as pd
import xarray as xr
import rioxarray as rxr
from rasterstats import zonal_stats
from scipy.spatial import cKDTree
from shapely import wkt
from shapely.geometry import Point, Polygon, box
from whitebox.whitebox_tools import WhiteboxTools


@dataclass
class UgridConfig:
    """
    Configuration for UGRID generation and terrain sampling.
    """

    crs_target: str = "EPSG:2039"
    quad_threshold: int = 5
    quad_max_depth: int = 12
    quad_min_stream_order: int = 3
    quad_spacing: int = 4
    dem_resolution: int = 4
    dem_nodata: int = -9999


class UgridPreprocessor:
    """
    Preprocessor for building the UGRID mesh and sampling terrain rasters.

    This is a modular port of the UGRID generation logic in the notebook:

    - Generate / load a QuadTree-based mesh clipped to the basin.
    - Attach stream-related attributes (``STRM_ORDER``, ``STREAM_CELL``).
    - Compute cell attributes (``AREA_2M``, ``X``, ``Y``).
    - Generate DEM derivatives using WhiteboxTools.
    - Sample terrain rasters into the UGRID polygons and persist as GeoParquet.
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        data_paths = config.get("data_paths", {})
        self._basin_folder = Path(str(data_paths.get("basin_folder", ""))).expanduser()
        self._streams_path = Path(str(data_paths.get("streams_path", ""))).expanduser()
        self._info_path = Path(str(data_paths.get("info_path", ""))).expanduser()
        self._dem_input_path = Path(str(data_paths.get("dem_input_path", ""))).expanduser()
        self._ugrid_output_dir = Path(str(data_paths.get("ugrid_output_dir", ""))).expanduser()
        self._dem_output_dir = Path(str(data_paths.get("dem_output_dir", ""))).expanduser()

        self._ugrid_output_dir.mkdir(parents=True, exist_ok=True)
        self._dem_output_dir.mkdir(parents=True, exist_ok=True)

        self._ugrid_path = self._ugrid_output_dir / "ugrid_cells.shp"
        self._ugrid_terrain_parquet = (
            self._ugrid_output_dir / "ugrid_cells_with_terrain.parquet"
        )
        self._ugrid_terrain_shp = (
            self._ugrid_output_dir / "ugrid_cells_with_terrain.shp"
        )

        self._ugrid_cfg = UgridConfig()

        self._terrain_files = {
            "DEM_MEAN": "DEM_2039_4m.tif",
            "SLOPE_MEAN": "slope.tif",
            "ASPECT_MEAN": "aspect.tif",
            "FLOWDIR_MEAN": "flowdir.tif",
            "FLOWACC_MEAN": "flowacc.tif",
            "RUGGED_MEAN": "ruggedness.tif",
        }

        self._wbt = WhiteboxTools()
        self._wbt.set_working_dir(str(self._dem_output_dir))
        self._wbt.set_verbose_mode(True)

    def _load_basin_polygon(self) -> gpd.GeoDataFrame:
        with self._info_path.open("r", encoding="utf-8") as f:
            basin_wkt = f.readline().strip()
        basin_poly = wkt.loads(basin_wkt)
        basin = gpd.GeoDataFrame(geometry=[basin_poly], crs="EPSG:4326").to_crs(
            self._ugrid_cfg.crs_target
        )
        return basin

    def _generate_ugrid_mesh(self) -> gpd.GeoDataFrame:
        """
        Generate the base UGRID mesh using a QuadTree over stream points.
        """
        basin = self._load_basin_polygon()

        streams = gpd.read_file(self._streams_path).to_crs(self._ugrid_cfg.crs_target)
        streams = streams[
            streams["STRM_ORDER"] >= self._ugrid_cfg.quad_min_stream_order
        ].copy()

        def densify_line(geom, spacing: int) -> List[Point]:
            n_points = max(int(geom.length // spacing), 2)
            return [
                geom.interpolate(i / (n_points - 1), normalized=True)
                for i in range(n_points)
            ]

        stream_points: List[Point] = []
        stream_orders: List[int] = []
        for _, row in streams.iterrows():
            pts = densify_line(row.geometry, spacing=self._ugrid_cfg.quad_spacing)
            stream_points.extend(pts)
            stream_orders.extend([row["STRM_ORDER"]] * len(pts))

        points_gdf = gpd.GeoDataFrame(
            {"STRM_ORDER": stream_orders},
            geometry=stream_points,
            crs=streams.crs,
        )

        def quad_subdivide(
            points: gpd.GeoDataFrame,
            box_coords: Iterable[float],
            threshold: int,
            depth: int,
            max_depth: int,
        ) -> List[Polygon]:
            minx, maxx, miny, maxy = box_coords
            bbox_poly = box(minx, miny, maxx, maxy)
            inside = points[points.geometry.within(bbox_poly)]
            if len(inside) <= threshold or depth >= max_depth:
                return [bbox_poly]
            xm = (minx + maxx) / 2.0
            ym = (miny + maxy) / 2.0
            boxes = [
                (minx, xm, miny, ym),
                (xm, maxx, miny, ym),
                (minx, xm, ym, maxy),
                (xm, maxx, ym, maxy),
            ]
            res: List[Polygon] = []
            for b in boxes:
                res.extend(quad_subdivide(inside, b, threshold, depth + 1, max_depth))
            return res

        bbox = basin.total_bounds
        box_coords = (bbox[0], bbox[2], bbox[1], bbox[3])
        quads = quad_subdivide(
            points_gdf,
            box_coords,
            threshold=self._ugrid_cfg.quad_threshold,
            depth=0,
            max_depth=self._ugrid_cfg.quad_max_depth,
        )

        quad_polys = [
            Polygon(
                [
                    (b.bounds[0], b.bounds[1]),
                    (b.bounds[2], b.bounds[1]),
                    (b.bounds[2], b.bounds[3]),
                    (b.bounds[0], b.bounds[3]),
                    (b.bounds[0], b.bounds[1]),
                ]
            )
            for b in quads
        ]

        quad_gdf = gpd.GeoDataFrame(geometry=quad_polys, crs=self._ugrid_cfg.crs_target)
        quad_clipped = gpd.overlay(quad_gdf, basin, how="intersection")

        # Assign attributes
        quad_clipped["ID"] = np.arange(1, len(quad_clipped) + 1)
        quad_clipped["STREAM_CELL"] = quad_clipped.intersects(
            streams.geometry.unary_union
        ).astype(int)

        quad_coords = np.array(
            [[geom.centroid.x, geom.centroid.y] for geom in quad_clipped.geometry]
        )
        point_coords = np.array([[p.x, p.y] for p in points_gdf.geometry])
        tree = cKDTree(point_coords)
        _, idx = tree.query(quad_coords)
        quad_clipped["STRM_ORDER"] = points_gdf.iloc[idx]["STRM_ORDER"].values
        quad_clipped["AREA_2M"] = quad_clipped.geometry.area.round(2)
        quad_clipped["X"] = quad_coords[:, 0].round(2)
        quad_clipped["Y"] = quad_coords[:, 1].round(2)

        return quad_clipped

    def _generate_dem_derivatives(self) -> None:
        """
        Generate DEM derivatives using WhiteboxTools and store them under dem_output_dir.
        """
        dem_output_path = self._dem_output_dir / self._terrain_files["DEM_MEAN"]

        # --- DEM workflow exactly as in the notebook ---
        # 1) Load clipped DEM in EPSG:4326
        dem_src = rxr.open_rasterio(self._dem_input_path, masked=True).squeeze()
        if dem_src.rio.crs is None:
            dem_src = dem_src.rio.write_crs("EPSG:4326", inplace=True)

        # 2) Reproject to target CRS with desired resolution
        dem_reproj = dem_src.rio.reproject(
            self._ugrid_cfg.crs_target,
            resolution=self._ugrid_cfg.dem_resolution,
        )

        # 3) Clip to basin polygon (already in target CRS)
        basin = self._load_basin_polygon()
        dem_clipped = dem_reproj.rio.clip(basin.geometry, basin.crs, drop=True)

        # 4) Save to dem_output_path (DEM_2039_4m.tif equivalent)
        dem_clipped.rio.to_raster(dem_output_path)

        dem_input = str(dem_output_path)

        # NOTE: WhiteboxTools Python API expects positional arguments in some versions;
        # using positional calls for maximum compatibility.
        # Slope
        self._wbt.slope(
            dem_input,
            str(self._dem_output_dir / self._terrain_files["SLOPE_MEAN"]),
        )
        # Aspect
        self._wbt.aspect(
            dem_input,
            str(self._dem_output_dir / self._terrain_files["ASPECT_MEAN"]),
        )
        # Flow direction (D8)
        self._wbt.d8_pointer(
            dem_input,
            str(self._dem_output_dir / self._terrain_files["FLOWDIR_MEAN"]),
        )
        # Flow accumulation
        self._wbt.d8_flow_accumulation(
            dem_input,
            str(self._dem_output_dir / self._terrain_files["FLOWACC_MEAN"]),
        )
        # Ruggedness index
        self._wbt.ruggedness_index(
            dem_input,
            str(self._dem_output_dir / self._terrain_files["RUGGED_MEAN"]),
        )

        # At this point, dem_output_path is a small, basin-clipped DEM

    def _sample_terrain(self, quad_clipped: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """
        Sample DEM derivatives into the UGRID polygons to create terrain attributes.
        """
        stats_config = {
            "DEM_MEAN": self._terrain_files["DEM_MEAN"],
            "SLOPE_MEAN": self._terrain_files["SLOPE_MEAN"],
            "ASPECT_MEAN": self._terrain_files["ASPECT_MEAN"],
            "FLOWACC_MEAN": self._terrain_files["FLOWACC_MEAN"],
            "FLOWDIR_MEAN": self._terrain_files["FLOWDIR_MEAN"],
            "RUGGED_MEAN": self._terrain_files["RUGGED_MEAN"],
        }

        for field_name, raster_file in stats_config.items():
            raster_path = self._dem_output_dir / raster_file
            zs = zonal_stats(
                quad_clipped,
                raster_path,
                stats=["mean"],
                nodata=self._ugrid_cfg.dem_nodata,
            )
            values = [
                row["mean"] if row["mean"] is not None else np.nan for row in zs
            ]
            quad_clipped[field_name] = values

        return quad_clipped

    def run(self) -> Path:
        """
        Run the full UGRID preprocessing pipeline.

        Returns
        -------
        Path
            Path to the final UGRID GeoParquet file with terrain attributes.
        """
        if self._ugrid_terrain_parquet.exists():
            # Ensure downstream artefacts (D50/Manning/final_ugrid) exist as well.
            self._ensure_postprocessing_products()
            return self._ugrid_terrain_parquet

        if self._ugrid_path.exists():
            quad_clipped = gpd.read_file(self._ugrid_path)
        else:
            quad_clipped = self._generate_ugrid_mesh()
            quad_clipped.to_file(self._ugrid_path)

        if not self._dem_output_dir.exists():
            self._dem_output_dir.mkdir(parents=True, exist_ok=True)

        self._generate_dem_derivatives()
        quad_clipped = self._sample_terrain(quad_clipped)

        # Persist as shapefile and parquet (terrain only)
        quad_clipped.to_file(self._ugrid_terrain_shp)
        quad_clipped.to_parquet(self._ugrid_terrain_parquet)

        # Run notebook-equivalent postprocessing: D50, Manning n, final_ugrid
        self._ensure_postprocessing_products()

        return self._ugrid_terrain_parquet

    # ------------------------------------------------------------------
    # Post-processing steps ported from the research notebook:
    # - Attach D50 (parent material)
    # - Compute Manning's n
    # - Build final_ugrid (lon/lat in EPSG:4326)
    # ------------------------------------------------------------------

    def _ensure_postprocessing_products(self) -> None:
        """
        Ensure that UGRID postprocessing artefacts from the notebook exist:

        - ugrid_cells_with_d50.parquet
        - ugrid_cells_with_d50.parquet including MANNING_N
        - final_ugrid.parquet (EPSG:4326 with lon/lat)
        """
        if not self._ugrid_terrain_parquet.exists():
            return

        basin_folder = self._basin_folder
        ugrid_dir = self._ugrid_output_dir

        # 1) D50 assignment
        ugrid_with_d50 = ugrid_dir / "ugrid_cells_with_d50.parquet"
        if not ugrid_with_d50.exists():
            self._assign_d50(
                terrain_parquet=self._ugrid_terrain_parquet,
                parent_path=basin_folder / "parent_material_with_d50.shp",
                output_path=ugrid_with_d50,
            )

        # 2) Manning n
        if ugrid_with_d50.exists():
            self._compute_manning_n(ugrid_with_d50)

        # 3) final_ugrid (lon/lat, EPSG:4326)
        final_ugrid = ugrid_dir / "final_ugrid.parquet"
        if not final_ugrid.exists() and ugrid_with_d50.exists():
            self._build_final_ugrid(ugrid_with_d50, final_ugrid)

    def _assign_d50(self, terrain_parquet: Path, parent_path: Path, output_path: Path) -> None:
        """
        Attach D50 values from a parent material shapefile to the UGRID cells.
        """
        if not terrain_parquet.exists() or not parent_path.exists():
            return

        ugrid = gpd.read_parquet(terrain_parquet)
        parent = gpd.read_file(parent_path)

        # Static mapping from parent material description (Hebrew) to D50MM (mm)
        d50_mapping = {
            "אבן גיר קשה ודולומיט": 0.5,
            "אבן גיר קשה, דולומיט וצור": 0.3,
            "אבן חול": 0.4,
            "אבן חול גירית (כורכר)": 0.35,
            "הליט": 0.01,
            "חוור אגמי ועדשות כבול": 0.005,
            "חוור הלשון": 0.01,
            "חול דיונרי עם משקעים סילטיים": 0.06,
            "חול דיונרי רצנטי": 0.1,
            "חול חופי קדום (חמרה) ואבן חול גירית בלויה": 0.12,
            "חול חופי רצנטי": 0.15,
            "חול יבשתי": 0.2,
            "חול יבשתי מכיל גבס": 0.15,
            "כבול": 0.001,
            "לס": 0.01,
            "משקעים אבניים וקונגלומרט": 1.0,
            "משקעים אבניים וקונגלומרט מכיל גבס": 0.8,
            "משקעים חוליים ואבניים שרובם הצטברו בהובלת מים": 0.03,
            "משקעים חרסיתיים שרובם הצטברו בהובלת מים": 0.015,
            "משקעים סיינים שרובם הצטברו בהובלת מים": 0.02,
            "סלעי פרץ (כגון בזלת) ופירוקלאסטיים (כגון טוף)": 0.4,
            "סלעי תהום (כגון גרניט)": 0.6,
            "קירטון": 0.02,
            "קירטון וחוור": 0.05,
            "קירטון ונארי": 0.03,
            "קירטון וצור": 0.1,
        }

        # Map descriptions to D50
        if "Descriptio" in parent.columns:
            parent["d50_mm"] = parent["Descriptio"].map(d50_mapping)
        else:
            return

        # Reproject parent to UGRID CRS if needed
        if ugrid.crs is not None and parent.crs is not None and ugrid.crs != parent.crs:
            parent = parent.to_crs(ugrid.crs)

        # Spatial join: assign D50 per cell; drop duplicates by ID
        joined = gpd.sjoin(
            ugrid,
            parent[["geometry", "d50_mm", "legend"]],
            how="left",
            predicate="intersects",
        )
        joined = joined.sort_values("ID").drop_duplicates(subset="ID")

        joined = joined.rename(
            columns={
                "d50_mm": "D50MM",
                "legend": "PARENT_MATERIAL",
            }
        )
        joined = joined.drop(columns=["index_right"], errors="ignore")

        joined.to_parquet(output_path)

    def _compute_manning_n(self, ugrid_with_d50_path: Path) -> None:
        """
        Compute Manning's n using D50, slope, stream order, and ruggedness,
        and store it in the same GeoParquet.
        """
        if not ugrid_with_d50_path.exists():
            return

        gdf = gpd.read_parquet(ugrid_with_d50_path)

        if "MANNING_N" in gdf.columns:
            return

        # Convert D50 from mm to m
        if "D50MM" not in gdf.columns:
            return

        gdf["D50_M"] = gdf["D50MM"] / 1000.0
        gdf["D50_M"] = gdf["D50_M"].replace(0, np.nan)

        # Prepare slope and stream order
        if "SLOPE_MEAN" in gdf.columns:
            gdf["SLOPE_MEAN"] = gdf["SLOPE_MEAN"].replace(0, 0.0001)
        else:
            gdf["SLOPE_MEAN"] = 0.0001

        if "STRM_ORDER" in gdf.columns:
            gdf["STRM_ORDER"] = gdf["STRM_ORDER"].replace(0, 1)
        else:
            gdf["STRM_ORDER"] = 1

        # 1. Grain roughness (Strickler)
        n_grain = 0.034 * np.power(gdf["D50_M"], 1.0 / 6.0)

        # 2. Form roughness (Jarrett) – slope in decimal
        n_slope = 0.32 * np.power(gdf["SLOPE_MEAN"] / 100.0, 0.38)

        # 3. Stream order correction
        order_corr = np.power(gdf["STRM_ORDER"], 0.16)

        # 4. Terrain correction using ruggedness
        if "RUGGED_MEAN" in gdf.columns:
            rugged_correction = np.where(
                gdf["RUGGED_MEAN"] > 30,
                0.02,
                np.where(gdf["RUGGED_MEAN"] > 15, 0.01, 0.0),
            )
        else:
            rugged_correction = 0.0

        gdf["MANNING_N"] = (n_grain + n_slope) / order_corr + rugged_correction

        gdf.to_parquet(ugrid_with_d50_path, index=False)

    def _build_final_ugrid(self, ugrid_with_d50_path: Path, final_path: Path) -> None:
        """
        Build final_ugrid.parquet in EPSG:4326 with lon/lat, as in the notebook.
        """
        if not ugrid_with_d50_path.exists():
            return

        gdf = gpd.read_parquet(ugrid_with_d50_path)

        # If geometry is already present and CRS known, reuse; otherwise reconstruct
        if "geometry" not in gdf.columns or gdf.geometry.is_empty.all():  # type: ignore[attr-defined]
            if "X" in gdf.columns and "Y" in gdf.columns:
                gdf["geometry"] = gdf.apply(lambda row: Point(row["X"], row["Y"]), axis=1)
                gdf = gpd.GeoDataFrame(gdf, geometry="geometry", crs="EPSG:2039")
            else:
                return
        else:
            if not isinstance(gdf, gpd.GeoDataFrame):
                gdf = gpd.GeoDataFrame(gdf, geometry="geometry")

        # Reproject to EPSG:4326 and extract lon/lat from cell centroids
        gdf = gdf.to_crs("EPSG:4326")
        centroids = gdf.geometry.centroid
        gdf["lon"] = centroids.x
        gdf["lat"] = centroids.y

        # Drop internal geometry/X/Y columns
        drop_cols = [c for c in ["X", "Y", "geometry"] if c in gdf.columns]
        gdf = gdf.drop(columns=drop_cols, errors="ignore")

        gdf.to_parquet(final_path, index=False)


class IconPreprocessor:
    """
    Preprocessor for ICON ensemble forecasts.

    In the notebooks, the ICON workflow performs:

    - Ingestion of raw ICON model output (NetCDF or gridded data).
    - Temporal interpolation from the model time step to 10-minute resolution.
    - Spatial interpolation / kriging onto the UGRID cells.
    - Validation and saving of ``rain_kriged_ICON_ENS_*.parquet`` files.

    This class defines the production interface and path handling for that
    workflow. The actual kriging logic should closely follow the notebook
    implementation, but for safety this module will currently raise a clear
    error if raw ICON files are present and no kriged outputs exist yet.
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        data_paths = config.get("data_paths", {})
        self._icon_raw_dir = Path(str(data_paths.get("icon_raw_dir", ""))).expanduser()
        self._icon_forecasts_dir = Path(
            str(data_paths.get("icon_forecasts_dir", ""))
        ).expanduser()
        self._basin_folder = Path(str(data_paths.get("basin_folder", ""))).expanduser()

        self._icon_forecasts_dir.mkdir(parents=True, exist_ok=True)

    def list_existing_kriged_members(self) -> List[Path]:
        """
        List already processed kriged ICON member files.
        """
        return sorted(
            self._icon_forecasts_dir.glob("rain_kriged_ICON_ENS_*.parquet")
        )

    def run(self) -> None:
        """
        Execute the ICON preprocessing workflow.

        Notes
        -----
        - If kriged outputs already exist in ``icon_forecasts_dir``, this
          method returns immediately.
        - If no kriged outputs are found but raw ICON files are present in
          ``icon_raw_dir``, this method raises a ``NotImplementedError`` to
          signal that the kriging logic must be implemented following the
          research notebook.
        """
        # If kriged outputs already exist, do nothing.
        existing = self.list_existing_kriged_members()
        if existing:
            return

        # Select ICON NetCDF file
        raw_files = sorted(self._icon_raw_dir.glob("*.nc"))
        if not raw_files:
            raise FileNotFoundError(
                f"No raw ICON NetCDF files found in {self._icon_raw_dir} and no "
                f"kriged outputs in {self._icon_forecasts_dir}."
            )

        icon_file_path = raw_files[0]

        # Basin info and UGRID path mirror the notebook conventions
        info_path = self._basin_folder / "info.txt"
        ugrid_path = (
            self._basin_folder / "output" / "ugrid" / "final_ugrid.parquet"
        )
        output_dir = self._icon_forecasts_dir
        output_dir.mkdir(parents=True, exist_ok=True)

        # Validation: basin info and UGRID
        if not info_path.exists():
            raise FileNotFoundError(f"Basin info file not found: {info_path}")
        if not ugrid_path.exists():
            raise FileNotFoundError(f"UGRID file not found: {ugrid_path}")

        # Load basin polygon
        with info_path.open("r", encoding="utf-8") as f:
            lines = f.readlines()
        basin_wkt = lines[0].strip()
        basin_poly = wkt.loads(basin_wkt)
        basin_gdf = gpd.GeoDataFrame(geometry=[basin_poly], crs="EPSG:4326")
        xmin, ymin, xmax, ymax = basin_gdf.total_bounds

        # Load ICON dataset
        ds = xr.open_dataset(icon_file_path)

        rainc_vars = [v for v in ds.data_vars.keys() if "RAINC" in v]
        if not rainc_vars:
            raise ValueError(
                f"No RAINC ensemble variables found in ICON file: {icon_file_path}"
            )

        lon = ds["lon"].values
        lat = ds["lat"].values

        # Spatial mask
        lon_mask = (lon >= xmin) & (lon <= xmax)
        lat_mask = (lat >= ymin) & (lat <= ymax)
        if not lon_mask.any() or not lat_mask.any():
            raise ValueError("No ICON grid points found within basin boundaries.")
        cropped_lon = lon[lon_mask]
        cropped_lat = lat[lat_mask]

        # Load UGRID lon/lat/ID from final_ugrid.parquet
        ugrid_df = pd.read_parquet(ugrid_path)
        grid_lon = ugrid_df["lon"].values
        grid_lat = ugrid_df["lat"].values
        grid_ids = ugrid_df["ID"].values

        # Time coordinates
        time_var = ds["time"]
        time_values = pd.to_datetime(time_var.values)
        time_delta_hours = (time_values[1] - time_values[0]).total_seconds() / 3600.0

        # Process all ensemble members individually, as in the notebook
        ensemble_members_to_process = rainc_vars
        all_output_paths: List[Path] = []

        for member_name in ensemble_members_to_process:
            rain_var = ds[member_name]
            member_suffix = member_name.replace("RAINC_", "mem")

            icon_filename = icon_file_path.stem
            output_filename = f"rain_kriged_{icon_filename}_{member_suffix}.parquet"
            output_path = output_dir / output_filename

            if output_path.exists():
                all_output_paths.append(output_path)
                continue

            # Crop to basin
            rain_cropped = rain_var.sel(lon=lon[lon_mask], lat=lat[lat_mask])

            # Convert cumulative rain (kg/m²) to rate (mm/hr)
            rain_rate = rain_cropped.diff(dim="time") / time_delta_hours
            rain_rate_first = xr.zeros_like(rain_cropped.isel(time=0))
            rain_rate = xr.concat([rain_rate_first, rain_rate], dim="time")
            rain_rate = rain_rate.assign_coords(time=time_values)

            # Temporal interpolation to 10 min, as in the notebook
            rain_rate = rain_rate.resample(time="10min").interpolate("linear")
            time_values_member = pd.to_datetime(rain_rate.time.values)

            # Spatial interpolation ICON grid → UGRID cells using OrdinaryKriging
            all_rain_data: List[Dict[str, Any]] = []
            member_rain_rate = rain_rate

            # Precompute 2D coordinate grid for cropped ICON domain
            lon_2d, lat_2d = np.meshgrid(cropped_lon, cropped_lat)

            for t_idx, t in enumerate(time_values_member):
                rain_t = member_rain_rate.isel(time=t_idx)
                rain_values_flat = rain_t.values.flatten()
                valid_mask = ~np.isnan(rain_values_flat)

                if valid_mask.sum() == 0:
                    interpolated_values = np.zeros(len(grid_ids), dtype=float)
                else:
                    rain_lon = lon_2d.flatten()[valid_mask]
                    rain_lat = lat_2d.flatten()[valid_mask]
                    rain_vals = rain_values_flat[valid_mask]
                    try:
                        OK = OrdinaryKriging(
                            rain_lon,
                            rain_lat,
                            rain_vals,
                            variogram_model="spherical",
                            verbose=False,
                            enable_plotting=False,
                        )
                        z, _ = OK.execute("points", grid_lat, grid_lon)
                        interpolated_values = np.asarray(z, dtype=float)
                    except Exception:
                        interpolated_values = np.zeros(len(grid_ids), dtype=float)

                for cell_idx, cell_id in enumerate(grid_ids):
                    all_rain_data.append(
                        {
                            "ID": int(cell_id),
                            "time": pd.Timestamp(t),
                            "rainrate": float(interpolated_values[cell_idx]),
                        }
                    )

            rain_df = pd.DataFrame(all_rain_data)
            rain_df["ID"] = rain_df["ID"].astype("int32")
            rain_df["rainrate"] = rain_df["rainrate"].astype("float32")
            rain_df["rainrate"] = rain_df["rainrate"].fillna(0.0)
            rain_df = rain_df.sort_values(["ID", "time"]).reset_index(drop=True)

            rain_df.to_parquet(output_path, index=False, engine="pyarrow")
            if not output_path.exists():
                raise RuntimeError(f"ICON kriged output was not created: {output_path}")

            all_output_paths.append(output_path)

        # Close dataset
        ds.close()


def build_preprocessors(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convenience factory to construct all preprocessing components.
    """
    return {
        "ugrid": UgridPreprocessor(config),
        "icon": IconPreprocessor(config),
    }

