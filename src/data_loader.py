from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import pandas as pd


class WeatherDataLoader:
    """
    Loader for ICON ensemble forecasts and precomputed UGRID products.

    This class encapsulates all filesystem access related to:

    - ICON ensemble rain forecasts stored as Parquet files
      (e.g. ``rain_kriged_ICON_ENS_*.parquet``).
    - UGRID cell attributes with terrain statistics
      (e.g. ``ugrid_cells_with_terrain.parquet``).
    - Basin-level configuration such as the base folder.

    All paths are supplied via the YAML configuration and are never
    hardcoded inside this module.
    """

    def __init__(self, config: Dict[str, object]) -> None:
        """
        Initialize the loader from a configuration dictionary.

        Parameters
        ----------
        config : Dict[str, object]
            Parsed configuration tree from ``system_config.yaml``.
        """
        data_cfg = config.get("data_paths", {})
        if not isinstance(data_cfg, dict):
            raise ValueError("Configuration key 'data_paths' must be a mapping.")

        self._basin_folder = Path(str(data_cfg.get("basin_folder", ""))).expanduser()
        self._icon_forecasts_dir = Path(
            str(data_cfg.get("icon_forecasts_dir", ""))
        ).expanduser()
        self._ugrid_parquet_path = Path(
            str(data_cfg.get("ugrid_parquet_path", ""))
        ).expanduser()

    @property
    def basin_folder(self) -> Path:
        """Return the base folder for the basin (e.g. Darga_28_for_test_only)."""
        return self._basin_folder

    @property
    def icon_forecasts_dir(self) -> Path:
        """Return the directory containing ICON ensemble Parquet files."""
        return self._icon_forecasts_dir

    @property
    def ugrid_parquet_path(self) -> Path:
        """Return the path to the UGRID GeoParquet file with terrain attributes."""
        return self._ugrid_parquet_path

    def list_icon_members(self) -> List[Path]:
        """
        List available ICON ensemble forecast member files.

        Returns
        -------
        List[Path]
            Sorted list of Parquet files matching the expected naming
            pattern inside ``icon_forecasts_dir``.

        Raises
        ------
        FileNotFoundError
            If the configured directory does not exist or contains
            no matching files.
        """
        if not self.icon_forecasts_dir.is_dir():
            raise FileNotFoundError(
                f"ICON forecasts directory does not exist: {self.icon_forecasts_dir}"
            )

        members = sorted(
            self.icon_forecasts_dir.glob("rain_kriged_ICON_ENS_*.parquet")
        )
        if not members:
            raise FileNotFoundError(
                f"No ICON forecast members found in {self.icon_forecasts_dir} "
                "with pattern 'rain_kriged_ICON_ENS_*.parquet'."
            )
        return members

    def load_icon_member(self, path: Path) -> pd.DataFrame:
        """
        Load a single ICON ensemble member Parquet file.

        Parameters
        ----------
        path : Path
            Path to the member Parquet file.

        Returns
        -------
        pandas.DataFrame
            Data frame with at least columns ``['ID', 'time', 'rainrate']``.

        Raises
        ------
        FileNotFoundError
            If the Parquet file does not exist.
        """
        if not path.is_file():
            raise FileNotFoundError(f"ICON member file not found: {path}")

        df = pd.read_parquet(path)
        if "time" in df.columns:
            df["time"] = pd.to_datetime(df["time"])
        return df

    def load_all_icon_members(self, members: Optional[Iterable[Path]] = None) -> Dict[str, pd.DataFrame]:
        """
        Load all ICON ensemble members into memory.

        Parameters
        ----------
        members : Iterable[Path], optional
            Optional explicit list of member paths. If omitted, all
            members returned by :meth:`list_icon_members` are used.

        Returns
        -------
        Dict[str, pandas.DataFrame]
            Mapping from a human-readable member name to its data
            frame. The default name follows the notebook convention
            (e.g. ``'2026011200 mem01'``).
        """
        resolved_members = list(members) if members is not None else self.list_icon_members()
        result: Dict[str, pd.DataFrame] = {}

        for member_path in resolved_members:
            df = self.load_icon_member(member_path)
            member_name = (
                member_path.stem.replace("rain_kriged_ICON_ENS_", "").replace("_", " ")
            )
            result[member_name] = df

        return result

    def load_ugrid(self) -> pd.DataFrame:
        """
        Load the UGRID cells with terrain attributes.

        Returns
        -------
        pandas.DataFrame
            UGRID table, typically with geometry and columns such as
            ``['ID', 'STREAM_CELL', 'STRM_ORDER', 'AREA_2M', 'X', 'Y',"
            " 'DEM_MEAN', 'SLOPE_MEAN', 'ASPECT_MEAN', 'FLOWACC_MEAN',"
            " 'FLOWDIR_MEAN', 'RUGGED_MEAN']``.

        Raises
        ------
        FileNotFoundError
            If the configured GeoParquet file does not exist.
        """
        if not self.ugrid_parquet_path.is_file():
            raise FileNotFoundError(
                f"UGRID parquet file not found: {self.ugrid_parquet_path}"
            )

        return pd.read_parquet(self.ugrid_parquet_path)

