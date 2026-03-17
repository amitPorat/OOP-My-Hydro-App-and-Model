from __future__ import annotations

import gc
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Tuple

import netCDF4 as nc
import numpy as np
import pandas as pd
import xarray as xr
from geopandas import GeoDataFrame
from scipy.interpolate import griddata
from shapely import wkt


@dataclass
class RmcompConfig:
    """
    Configuration for RMcomp historical rain preprocessing.

    This mirrors the structure and behaviour of the research notebook
    that worked with yearly RM{year}.nc files and an events library
    CSV. The class is intentionally conservative and focuses on
    filesystem orchestration, chunked NetCDF reading and event-based
    slicing without changing any hydrological logic downstream.
    """

    # Names of coordinate and rain variables inside RMcomp NetCDF files.
    # The IMS RMcomp product uses rotated coordinates rlon/rlat and
    # the RAINRATE variable, as in the notebook.
    lon_name: str = "rlon"
    lat_name: str = "rlat"
    time_name: str = "time"
    rain_name: str = "RAINRATE"

    # Event library column names – can be overridden via config if the
    # CSV uses different headers.
    event_id_column: str = "event_id"
    start_column: str = "start_time"
    end_column: str = "end_time"


class RmcompRainPreprocessor:
    """
    Preprocessor for RMcomp historical rain data.

    Responsibilities
    ----------------
    - Read basin BBOX from ``info.txt`` (first-line WKT).
    - Open yearly ``RM{year}.nc`` NetCDF with chunking.
    - Spatially subset the RMcomp grid to the basin envelope.
    - Sample rain intensities at UGRID cell centroids (nearest-neighbour)
      using the ``final_ugrid.parquet`` lon/lat coordinates.
    - Persist yearly cell-level rain time series as:
        ``output/rain/rain_{year}.parquet``.
    - Slice yearly data into event windows defined in the events
      library CSV and persist as:
        ``output/rain/event_rain/rain_events_{year}.parquet``.

    Notes
    -----
    - The exact physical units and transformations follow the RMcomp
      product; no scaling is applied here. The downstream feature
      engineering and models operate on these fields exactly as they
      appeared in the notebook.
    - All paths are provided via the configuration dictionary to keep
      this component environment-agnostic.
    """

    def __init__(self, config: Mapping[str, Any]) -> None:
        data_paths = config.get("data_paths", {})
        if not isinstance(data_paths, Mapping):
            raise ValueError("Configuration key 'data_paths' must be a mapping.")

        self._basin_folder = Path(str(data_paths.get("basin_folder", ""))).expanduser()
        self._rmcomp_dir = Path(str(data_paths.get("rmcomp_dir", ""))).expanduser()

        default_rain_root = self._basin_folder / "output" / "rain"
        self._rain_root = Path(
            str(data_paths.get("rain_output_dir", default_rain_root))
        ).expanduser()
        self._rain_root.mkdir(parents=True, exist_ok=True)

        self._event_rain_root = self._rain_root / "event_rain"
        self._event_rain_root.mkdir(parents=True, exist_ok=True)

        # Basin polygon / BBOX info from info.txt:
        # line 1: WKT polygon (used elsewhere),
        # line 2: explicit BBOX coordinates (used here, as in notebook).
        self._info_path = Path(str(data_paths.get("info_path", ""))).expanduser()
        if not self._info_path.is_file():
            # Fall back to standard basin layout if not explicitly configured.
            self._info_path = self._basin_folder / "info.txt"

        # UGRID with lon/lat centroids (final_ugrid.parquet from notebook flow)
        default_final_ugrid = (
            self._basin_folder / "output" / "ugrid" / "final_ugrid.parquet"
        )
        self._final_ugrid_path = Path(
            str(data_paths.get("final_ugrid_path", default_final_ugrid))
        ).expanduser()

        events_cfg = config.get("events", {})
        if not isinstance(events_cfg, Mapping):
            events_cfg = {}

        # Optional events library CSV (can be overridden at UI level).
        events_csv = events_cfg.get("events_library_csv")
        self._events_csv_path: Optional[Path]
        if events_csv:
            self._events_csv_path = Path(str(events_csv)).expanduser()
        else:
            # Heuristic default under basin folder.
            guessed = list(self._basin_folder.glob("*events_library*.csv"))
            self._events_csv_path = guessed[0] if guessed else None

        # Column names used by the events library. For RMcomp NetCDF we
        # strictly follow the notebook conventions: RAINRATE on rlat/rlon
        # with time in minutes since base_time. Only the event metadata
        # column names are configurable.
        self._rm_cfg = RmcompConfig()
        if "event_id_column" in events_cfg:
            self._rm_cfg.event_id_column = str(events_cfg["event_id_column"])
        if "start_column" in events_cfg:
            self._rm_cfg.start_column = str(events_cfg["start_column"])
        if "end_column" in events_cfg:
            self._rm_cfg.end_column = str(events_cfg["end_column"])

        # Antecedent extension in hours, mirroring the notebook default (48 h)
        # but configurable via `events.antecedent_hours` for UI control.
        self._antecedent_hours: float = float(events_cfg.get("antecedent_hours", 48))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run_year(self, year: int) -> Path:
        """
        Build yearly RMcomp rain parquet for a given year (point-based).

        Parameters
        ----------
        year : int
            Calendar year corresponding to ``RM{year}.nc``.

        Returns
        -------
        Path
            Path to ``rain_{year}.parquet``.
        """
        out_path = self._rain_root / f"rain_{year}.parquet"
        if out_path.is_file():
            return out_path

        nc_name = f"RM{year}.nc"
        nc_path = (self._rmcomp_dir / nc_name).expanduser()
        if not nc_path.is_file():
            raise FileNotFoundError(f"RMcomp NetCDF file not found: {nc_path}")

        # --- Load BBOX from line 2 of info.txt, exactly as in notebook ---
        xmin, ymin, xmax, ymax = self._load_bbox_from_info_line2()

        # --- Open NetCDF and read rotated coordinates and time axis ---
        dataset = nc.Dataset(str(nc_path), mode="r")

        # IMS RMcomp BD uses 2D rotated coordinates rlat/rlon on (y, x).
        var_names = dataset.variables.keys()
        if "rlat" not in var_names or "rlon" not in var_names:
            dataset.close()
            raise KeyError(
                "RMcomp dataset is missing expected coordinate variables 'rlat'/'rlon'. "
                f"Available variables: {list(var_names)}"
            )

        rlat = dataset.variables["rlat"][:]
        rlon = dataset.variables["rlon"][:]

        time_raw = dataset.variables[self._rm_cfg.time_name][:]
        time_units = dataset.variables[self._rm_cfg.time_name].units
        base_time = pd.to_datetime(time_units.split("since")[1].strip())
        time = base_time + pd.to_timedelta(time_raw, unit="m")

        # --- Spatial BBOX mask in rotated coordinates (rlon/rlat) ---
        mask = (rlon >= xmin) & (rlon <= xmax) & (rlat >= ymin) & (rlat <= ymax)
        ys, xs = np.where(mask)
        if ys.size == 0 or xs.size == 0:
            dataset.close()
            raise ValueError(
                f"No RMcomp grid points found in BBOX range for year {year}."
            )

        extracted: List[Tuple[pd.Timestamp, float, float, float]] = []
        num_steps = len(time)
        chunk_size = 720  # matches notebook default

        rain_var = dataset.variables[self._rm_cfg.rain_name]

        for start in range(0, num_steps, chunk_size):
            end = min(start + chunk_size, num_steps)
            rain_chunk = rain_var[start:end]  # shape: [time, y, x]

            for t_idx in range(rain_chunk.shape[0]):
                rain_slice = rain_chunk[t_idx]  # 2D [y, x]
                values = rain_slice[ys, xs]
                valid = np.isfinite(values)
                if not valid.any():
                    continue
                lat_vals = rlat[ys[valid], xs[valid]]
                lon_vals = rlon[ys[valid], xs[valid]]
                rain_vals = values[valid]
                ts = time[start + t_idx]
                for la, lo, rr in zip(lat_vals, lon_vals, rain_vals, strict=False):
                    extracted.append(
                        (pd.Timestamp(ts), float(la), float(lo), float(rr))
                    )

        dataset.close()

        df = pd.DataFrame(
            extracted, columns=["time", "lat", "lon", "rainrate"]
        )
        if not df.empty:
            df["rainrate"] = df["rainrate"].astype("float32")
            df = df.sort_values(["time", "lat", "lon"]).reset_index(drop=True)

        df.to_parquet(out_path, index=False, engine="pyarrow")
        return out_path

    def run_events_for_year(self, year: int) -> Path:
        """
        Slice yearly RMcomp rain into event windows defined in the events CSV.

        Parameters
        ----------
        year : int
            Calendar year processed previously via :meth:`run_year`.

        Returns
        -------
        Path
            Path to ``rain_events_{year}.parquet`` containing at least
            ``['ID', 'time', 'rainrate', event_id_column]``.
        """
        if self._events_csv_path is None:
            raise FileNotFoundError(
                "Events library CSV path is not configured. "
                "Set 'events.events_library_csv' in the configuration."
            )

        out_path = self._event_rain_root / f"rain_events_{year}.parquet"
        if out_path.is_file():
            return out_path

        rain_file = self._rain_root / f"rain_{year}.parquet"
        if not rain_file.is_file():
            # Ensure base rain exists for this year
            rain_file = self.run_year(year)

        # Load and clean events exactly as in the notebook (corr_st/corr_ed,
        # cancelled events, overlap resolution, discharge filtering, 48h
        # antecedent extension).
        events = self._load_and_clean_events()
        if events.empty:
            # No events to process at all
            empty_cols = ["time", "lat", "lon", "rainrate", "event_id"]
            pd.DataFrame(columns=empty_cols).to_parquet(
                out_path, index=False, engine="pyarrow"
            )
            return out_path

        # Use original_start year for grouping
        events_year = events[events["year"] == year].copy()
        if events_year.empty:
            # Mirror notebook behaviour: warn/skip but create no event file
            empty_cols = ["time", "lat", "lon", "rainrate", "event_id"]
            pd.DataFrame(columns=empty_cols).to_parquet(
                out_path, index=False, engine="pyarrow"
            )
            return out_path

        rain_df = pd.read_parquet(rain_file)
        rain_df["time"] = pd.to_datetime(rain_df["time"], errors="coerce")

        rain_base = self._rain_root
        event_rain_parts: List[pd.DataFrame] = []

        for _, ev in events_year.iterrows():
            start_date = ev["start_date"]
            end_date = ev["end_date"]

            # Handle events extended 48h into previous year
            event_start_year = start_date.year
            if event_start_year < year:
                prev_year_file = rain_base / f"rain_{event_start_year}.parquet"
                if prev_year_file.exists():
                    prev_rain = pd.read_parquet(prev_year_file)
                    prev_rain["time"] = pd.to_datetime(
                        prev_rain["time"], errors="coerce"
                    )
                    rain_combined = pd.concat(
                        [prev_rain, rain_df], ignore_index=True
                    )
                else:
                    rain_combined = rain_df
            else:
                rain_combined = rain_df

            df_cut = rain_combined[
                (rain_combined["time"] >= start_date)
                & (rain_combined["time"] <= end_date)
            ].copy()
            if df_cut.empty:
                continue

            # Match notebook event metadata
            df_cut["event_id"] = ev.name
            df_cut["event_st"] = ev["start_date"]
            df_cut["event_ed"] = ev["end_date"]
            df_cut["event_status"] = ev.get("status")
            df_cut["event_user"] = ev.get("user")

            event_rain_parts.append(df_cut)

        if event_rain_parts:
            all_event_data = pd.concat(event_rain_parts, ignore_index=True)
            all_event_data.to_parquet(out_path, index=False, engine="pyarrow")
        else:
            empty_cols = ["time", "lat", "lon", "rainrate", "event_id"]
            pd.DataFrame(columns=empty_cols).to_parquet(
                out_path, index=False, engine="pyarrow"
            )

        return out_path

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_bbox_from_info_line2(self) -> Tuple[float, float, float, float]:
        """
        Load BBOX from line 2 of info.txt, mirroring the notebook.

        Line 2 contains a sequence of coordinate pairs that define the
        polygon; we compute xmin/xmax/ymin/ymax from these points.
        """
        if not self._info_path.is_file():
            raise FileNotFoundError(f"Basin info file not found: {self._info_path}")
        with self._info_path.open("r", encoding="utf-8") as f:
            lines = f.readlines()
        if len(lines) < 2:
            raise ValueError(
                f"info.txt at {self._info_path} does not contain a BBOX line (line 2)."
            )
        bbox_line = (
            lines[1].strip().replace("(", "").replace(")", "")
        )
        bbox_parts = [
            float(coord.strip()) for coord in bbox_line.replace(",", " ").split()
        ]
        bbox_coords = list(zip(bbox_parts[::2], bbox_parts[1::2]))
        xmin = min(p[0] for p in bbox_coords)
        xmax = max(p[0] for p in bbox_coords)
        ymin = min(p[1] for p in bbox_coords)
        ymax = max(p[1] for p in bbox_coords)
        return xmin, ymin, xmax, ymax

    def _load_and_clean_events(self) -> pd.DataFrame:
        """
        Load and clean the events library exactly as in the notebook:

        - Parse st/ed and corr_st/corr_ed.
        - Use corrected dates when available (corr_st/corr_ed), otherwise st/ed.
        - Drop rows without valid dates.
        - Remove 'cancelled' events.
        - Resolve overlapping events by adjusting the second event's start
          date with a 10-minute buffer.
        - Keep only 'corrected' events.
        - Optionally filter events by discharge if discharge files exist.
        - Extend each event 48 hours backwards in time.
        - Add 'year' column from original_start.
        """
        events = pd.read_csv(self._events_csv_path)  # type: ignore[arg-type]

        # Convert dates
        for col in ["st", "ed", "corr_st", "corr_ed"]:
            if col in events.columns:
                events[col] = pd.to_datetime(events[col], errors="coerce")

        # Use corrected dates if available, otherwise original
        events["start_date"] = events.get("corr_st").fillna(events.get("st"))
        events["end_date"] = events.get("corr_ed").fillna(events.get("ed"))
        events = events.dropna(subset=["start_date", "end_date"]).reset_index(
            drop=True
        )

        # 1. Remove cancelled events
        if "status" in events.columns:
            cancelled_mask = events["status"].str.lower() == "cancelled"
            events = events[~cancelled_mask].copy()

        # 2. Resolve overlapping events
        overlaps: List[Dict[str, Any]] = []
        for i, ev1 in events.iterrows():
            for j, ev2 in events.iterrows():
                if i >= j:
                    continue
                if not (
                    ev1["end_date"] < ev2["start_date"]
                    or ev1["start_date"] > ev2["end_date"]
                ):
                    overlap_start = max(ev1["start_date"], ev2["start_date"])
                    overlap_end = min(ev1["end_date"], ev2["end_date"])
                    overlaps.append(
                        {
                            "event1_idx": i,
                            "event2_idx": j,
                            "event1_end": ev1["end_date"],
                            "event2_start": ev2["start_date"],
                            "overlap_start": overlap_start,
                            "overlap_end": overlap_end,
                        }
                    )

        if overlaps:
            events_to_adjust: Dict[int, pd.Timestamp] = {}
            for ov in overlaps:
                event2_idx = ov["event2_idx"]
                event1_end = ov["event1_end"]
                if event2_idx not in events_to_adjust:
                    events_to_adjust[event2_idx] = event1_end
                else:
                    if event1_end > events_to_adjust[event2_idx]:
                        events_to_adjust[event2_idx] = event1_end

            for idx, max_end in events_to_adjust.items():
                new_start = max_end + pd.Timedelta(minutes=10)
                events.loc[idx, "start_date"] = new_start

        # Filter only corrected events
        if "status" in events.columns:
            mask_corrected = events["status"].str.lower() == "corrected"
            events = events[mask_corrected].copy()

        # Filter false events by discharge, if available
        events, _ = self._filter_events_by_discharge(events)

        # Extend each event to include a configurable number of hours before start
        events["original_start"] = events["start_date"].copy()
        events["start_date"] = events["start_date"] - pd.Timedelta(
            hours=self._antecedent_hours
        )

        # Use original start year for grouping and file naming
        events["year"] = events["original_start"].dt.year
        return events

    def _filter_events_by_discharge(
        self, events: pd.DataFrame, threshold: float = 0.01
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Mirror notebook logic to filter events without discharge in the basin.
        """
        discharge_folder = self._basin_folder / "output" / "discharge"
        if not discharge_folder.exists():
            # Skip filtering if discharge folder doesn't exist
            return events.copy(), pd.DataFrame()

        events_with_discharge: List[int] = []
        events_without_discharge: List[int] = []

        for idx, ev in events.iterrows():
            year = ev["start_date"].year
            discharge_file = discharge_folder / f"discharge_processed_{year}.parquet"
            if not discharge_file.exists():
                events_with_discharge.append(idx)
                continue
            try:
                discharge_df = pd.read_parquet(discharge_file)
                discharge_df["time"] = pd.to_datetime(
                    discharge_df["time"], errors="coerce"
                )
                event_discharge = discharge_df[
                    (discharge_df["time"] >= ev["start_date"])
                    & (discharge_df["time"] <= ev["end_date"])
                ]
                if len(event_discharge) > 0:
                    max_discharge = event_discharge["discharge"].max()
                    if max_discharge > threshold:
                        events_with_discharge.append(idx)
                    else:
                        events_without_discharge.append(idx)
                else:
                    events_without_discharge.append(idx)
            except Exception:
                events_with_discharge.append(idx)

        filtered_events = events.loc[events_with_discharge].copy()
        removed_events = (
            events.loc[events_without_discharge].copy()
            if events_without_discharge
            else pd.DataFrame()
        )
        return filtered_events, removed_events


# ---------------------------------------------------------------------------
# ICON NetCDF → UGRID Parquet (RMCOMP schema)
# ---------------------------------------------------------------------------

class ICONPreprocessor:
    """
    Transform raw ICON ensemble NetCDF into 10-minute UGRID Parquet files
    matching the RMCOMP schema (ID int32, time datetime64[ns], rainrate float32).

    - Loads UGRID (final_ugrid.parquet) for target ID, lat, lon.
    - Converts cumulative RAINC_01..RAINC_20 to 30-min discrete depth (diff),
      then distributes each 30-min depth into three 10-min slots (divide by 3).
    - Interpolates from ICON 2D (lat, lon) to UGRID centroids via
      scipy.interpolate.griddata(method='nearest').
    - Writes one Parquet per member: rain_icon_<forecast_time>_mem<XX>.parquet.
    """

    def __init__(
        self,
        ugrid_path: Path,
        output_dir: Path,
        *,
        chunk_time: Optional[int] = None,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> None:
        """
        Parameters
        ----------
        ugrid_path : Path
            Path to final_ugrid.parquet (must have ID, lat, lon).
        output_dir : Path
            Directory for output Parquet files.
        chunk_time : int, optional
            Chunk size along time for xarray (default 60).
        progress_callback : callable(member_index, total_members), optional
            Called after each member is processed (e.g. for st.progress).
        """
        self._ugrid_path = Path(ugrid_path).expanduser()
        self._output_dir = Path(output_dir).expanduser()
        self._output_dir.mkdir(parents=True, exist_ok=True)
        self._chunk_time = chunk_time or 60
        self._progress_callback = progress_callback

    def _load_ugrid(self) -> pd.DataFrame:
        df = pd.read_parquet(self._ugrid_path)
        for c in ("ID", "lat", "lon"):
            if c not in df.columns:
                raise KeyError(f"UGRID parquet must have column '{c}'. Found: {list(df.columns)}")
        return df[["ID", "lat", "lon"]].copy()

    def _forecast_time_from_path(self, nc_path: Path) -> str:
        """Derive forecast time string from filename, e.g. ICON_ENS_2026011200.nc -> 2026011200."""
        stem = nc_path.stem
        digits = "".join(ch for ch in stem if ch.isdigit())
        return digits[:10] if len(digits) >= 10 else digits or "unknown"

    def process_nc(
        self,
        nc_path: Path,
        *,
        members: Optional[List[int]] = None,
    ) -> List[Path]:
        """
        Process an ICON NetCDF file and write one Parquet per ensemble member.

        Parameters
        ----------
        nc_path : Path
            Path to the ICON NetCDF (e.g. ICON_ENS_2026011200.nc).
        members : list of int, optional
            Member indices 1..20 to process (default all 1..20).

        Returns
        -------
        list of Path
            Paths to written Parquet files.
        """
        nc_path = Path(nc_path).expanduser()
        if not nc_path.is_file():
            raise FileNotFoundError(f"ICON NetCDF not found: {nc_path}")

        ugrid_df = self._load_ugrid()
        ids = ugrid_df["ID"].values.astype(np.int32)
        points_ugrid = ugrid_df[["lon", "lat"]].values  # (N, 2) for griddata

        forecast_time = self._forecast_time_from_path(nc_path)
        member_list = list(members or range(1, 21))
        total = len(member_list)
        written: List[Path] = []

        for k, mem in enumerate(member_list):
            var_name = f"RAINC_{mem:02d}"
            try:
                out_path = self._process_one_member(
                    nc_path=nc_path,
                    var_name=var_name,
                    forecast_time=forecast_time,
                    member_id=mem,
                    points_ugrid=points_ugrid,
                    ids=ids,
                )
                if out_path is not None:
                    written.append(out_path)
            finally:
                gc.collect()
            if self._progress_callback is not None:
                self._progress_callback(k + 1, total)

        return written

    def _process_one_member(
        self,
        nc_path: Path,
        var_name: str,
        forecast_time: str,
        member_id: int,
        points_ugrid: np.ndarray,
        ids: np.ndarray,
    ) -> Optional[Path]:
        """Load one RAINC var, convert to 10-min UGRID, write Parquet."""
        ds = xr.open_dataset(nc_path, chunks={"time": self._chunk_time})
        if var_name not in ds.data_vars:
            ds.close()
            return None

        da = ds[var_name]  # (time, lat, lon)
        time_coord = ds.coords["time"]
        lat = ds.coords["lat"].values
        lon = ds.coords["lon"].values
        # Build ICON grid points (lon, lat) for griddata
        lon_2d, lat_2d = np.meshgrid(lon, lat)
        points_icon = np.column_stack([lon_2d.ravel(), lat_2d.ravel()])

        # Cumulative -> 30-min discrete depth (mm). First timestep: use 0 or first value.
        accum = da.values
        if accum.shape[0] < 2:
            ds.close()
            return None
        depth_30min = np.zeros_like(accum, dtype=np.float32)
        depth_30min[0] = 0.0  # first step: no prior accumulation
        depth_30min[1:] = np.diff(accum, axis=0).astype(np.float32)
        depth_30min = np.maximum(depth_30min, 0.0)

        # 30-min -> 10-min: divide by 3 and replicate into three 10-min slots
        n_30 = depth_30min.shape[0]
        n_10 = n_30 * 3
        time_30 = pd.to_datetime(time_coord.values)
        time_10 = pd.date_range(
            start=time_30[0],
            periods=n_10,
            freq="10min",
        )
        depth_10min = np.zeros((n_10, depth_30min.shape[1], depth_30min.shape[2]), dtype=np.float32)
        for i in range(n_30):
            third = depth_30min[i] / 3.0
            depth_10min[i * 3] = third
            depth_10min[i * 3 + 1] = third
            depth_10min[i * 3 + 2] = third

        # Interpolate each 10-min slice to UGRID (nearest)
        n_cells = points_ugrid.shape[0]
        table = np.zeros((n_10 * n_cells, 3), dtype=np.float64)  # time_idx, ID, rainrate
        for t in range(n_10):
            values_icon = depth_10min[t].ravel()
            values_ugrid = griddata(
                points_icon,
                values_icon,
                points_ugrid,
                method="nearest",
                fill_value=0.0,
            )
            values_ugrid = np.nan_to_num(values_ugrid, nan=0.0).astype(np.float32)
            for c in range(n_cells):
                row = t * n_cells + c
                table[row, 0] = t
                table[row, 1] = ids[c]
                table[row, 2] = values_ugrid[c]

        ds.close()

        # Build DataFrame: ID (int32), time (datetime64[ns]), rainrate (float32)
        df = pd.DataFrame({
            "ID": np.tile(ids, n_10).astype(np.int32),
            "time": np.repeat(time_10, n_cells),
            "rainrate": table[:, 2].astype(np.float32),
        })
        df = df.sort_values(["time", "ID"]).reset_index(drop=True)

        out_path = self._output_dir / f"rain_icon_{forecast_time}_mem{member_id:02d}.parquet"
        df.to_parquet(out_path, index=False, engine="pyarrow")
        return out_path


def build_rain_preprocessor(config: Dict[str, Any]) -> RmcompRainPreprocessor:
    """
    Convenience factory to construct an RMcomp rain preprocessor.
    """
    return RmcompRainPreprocessor(config)

