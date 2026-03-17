from __future__ import annotations

from typing import Iterable, List

import numpy as np
import pandas as pd


class UgridBuilder:
    """
    Utility for attaching UGRID-based static features to ICON forecasts.

    In the notebooks, UGRID construction (quad-tree mesh, DEM
    processing, WhiteboxTools derivatives) is performed offline and
    written to GeoParquet. For operational inference we only need to
    load these precomputed attributes and merge them with dynamic ICON
    rain fields.
    """

    def __init__(self, ugrid_df: pd.DataFrame, terrain_features: Iterable[str]) -> None:
        """
        Initialize the builder.

        Parameters
        ----------
        ugrid_df : pandas.DataFrame
            UGRID table with at least columns ``['ID', 'X', 'Y']`` and
            the configured terrain feature columns (e.g.
            ``'DEM_MEAN', 'SLOPE_MEAN', 'FLOWDIR_MEAN', 'AREA_2M'``).
        terrain_features : Iterable[str]
            Names of terrain feature columns to keep when joining with
            ICON rain fields. These should mirror the ``terrain_cols``
            list in the notebooks.
        """
        self._ugrid_df = ugrid_df.copy()
        self._terrain_features: List[str] = list(terrain_features)

    @property
    def terrain_features(self) -> List[str]:
        """Return the list of terrain feature column names."""
        return list(self._terrain_features)

    def attach_static_features(self, rain_df: pd.DataFrame) -> pd.DataFrame:
        """
        Attach static UGRID features to a dynamic rain DataFrame.

        Parameters
        ----------
        rain_df : pandas.DataFrame
            ICON rain data with an ``'ID'`` column referencing UGRID
            cell identifiers.

        Returns
        -------
        pandas.DataFrame
            Data frame containing rain fields plus static terrain
            attributes and cell coordinates ``X, Y``.
        """
        merge_cols = ["ID"] + self._terrain_features + ["X", "Y"]

        missing = [c for c in merge_cols if c not in self._ugrid_df.columns]
        if missing:
            raise KeyError(
                f"UGRID table is missing required columns: {missing}. "
                "Ensure UGRID preprocessing has produced the expected fields."
            )

        return rain_df.merge(self._ugrid_df[merge_cols], on="ID", how="left")


def add_engineered_features(df: pd.DataFrame, include_lagged_discharge: bool = False) -> pd.DataFrame:
    """
    Add temporal engineered features used during training.

    This function is a faithful port of the ``add_engineered_features``
    helper defined in the research notebook. It ensures that the
    inference-time feature space matches what the scalers and LSTM
    models expect (e.g. cumulative rain, intensities, time encodings).

    Parameters
    ----------
    df : pandas.DataFrame
        Input data with at least ``['ID', 'time', 'rainrate', 'discharge']``.
    include_lagged_discharge : bool, optional
        Whether to create lagged discharge features. For ICON-only
        forecasting we typically use ``False``.

    Returns
    -------
    pandas.DataFrame
        Data frame with additional engineered feature columns.
    """
    df = df.copy()
    df = df.sort_values(["ID", "time"]).reset_index(drop=True)

    if include_lagged_discharge:
        df["discharge_lag1"] = df.groupby("ID")["discharge"].shift(1)
        df["discharge_lag1"] = df["discharge_lag1"].fillna(0.0)
        df["discharge_lag1_log"] = np.log1p(df["discharge_lag1"])

    # Cumulative rain features
    for window in [6, 12, 24, 36]:
        col_name = f"rain_cum_{window}t"
        rolling = (
            df.groupby("ID")["rainrate"]
            .rolling(window, min_periods=1)
            .sum()
            .reset_index(0, drop=True)
        )
        df[col_name] = rolling.fillna(0.0)

    # Rain intensity features
    df["rain_max_1h"] = (
        df.groupby("ID")["rainrate"]
        .rolling(6, min_periods=1)
        .max()
        .reset_index(0, drop=True)
        .fillna(0.0)
    )
    df["rain_max_2h"] = (
        df.groupby("ID")["rainrate"]
        .rolling(12, min_periods=1)
        .max()
        .reset_index(0, drop=True)
        .fillna(0.0)
    )
    df["rain_max_4h"] = (
        df.groupby("ID")["rainrate"]
        .rolling(24, min_periods=1)
        .max()
        .reset_index(0, drop=True)
        .fillna(0.0)
    )
    df["rain_mean_1h"] = (
        df.groupby("ID")["rainrate"]
        .rolling(6, min_periods=1)
        .mean()
        .reset_index(0, drop=True)
        .fillna(0.0)
    )

    # Rain rate dynamics
    df["rain_rate_of_change"] = (
        df.groupby("ID")["rainrate"].diff().fillna(0.0)
    )
    df["rain_acceleration"] = (
        df.groupby("ID")["rain_rate_of_change"].diff().fillna(0.0)
    )

    # Weighted cumulative rain (recent timesteps weighted more)
    decay_rate = 6  # Half-life of 6 timesteps
    df["rain_weighted_cum"] = 0.0
    for idx in df.groupby("ID").groups.values():
        group_df = df.loc[idx].copy()
        rain_values = group_df["rainrate"].values
        weighted_sum = np.zeros(len(rain_values))
        for i in range(len(rain_values)):
            window = min(i + 1, 24)
            weights = np.exp(-np.arange(window) / decay_rate)
            weighted_sum[i] = np.sum(
                rain_values[max(0, i - 23) : i + 1] * weights[-window:]
            )
        df.loc[idx, "rain_weighted_cum"] = weighted_sum

    # Time features
    df["hour"] = df["time"].dt.hour
    df["day_of_year"] = df["time"].dt.dayofyear
    df["month"] = df["time"].dt.month
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
    # Day-of-year: use 366 in leap years, 365 otherwise to keep [0,1] cycle correct
    days_in_year = np.where(df["time"].dt.is_leap_year, 366, 365)
    df["doy_sin"] = np.sin(2 * np.pi * df["day_of_year"] / days_in_year)
    df["doy_cos"] = np.cos(2 * np.pi * df["day_of_year"] / days_in_year)

    return df


