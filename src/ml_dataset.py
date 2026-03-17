"""
ML dataset utilities: notebook-exact feature engineering, group_features,
and make_sequences for the two-stage LSTM pipeline on merged Rain–Discharge data.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset, TensorDataset

from .features import add_engineered_features


def simulate_forecast_degradation(
    seq_array: np.ndarray,
    rain_indices: List[int],
    pristine_hours: float = 9.0,
    time_step_mins: int = 10,
    window: int = 5,
) -> np.ndarray:
    """
    Simulate ICON-style smoothing on rain features after a pristine (RMCOMP) window.

    First `pristine_hours` (in 10-min steps = pristine_hours * 6) timesteps are
    left unchanged. For the rest, apply a rolling mean to smooth rain peaks,
    simulating the spatial smoothing of the ICON model vs radar (RMCOMP).

    Parameters
    ----------
    seq_array : np.ndarray
        Shape (seq_len, num_features), float32. Single sequence.
    rain_indices : list of int
        Column indices of rain-related features to smooth.
    pristine_hours : float
        Hours of "pristine" (unaltered) lead time (e.g. 9 = 9h of RMCOMP).
    time_step_mins : int
        Timestep in minutes (e.g. 10 for 10-min resolution).
    window : int
        Rolling mean window size for smoothing (odd recommended).

    Returns
    -------
    np.ndarray
        Same shape as seq_array; only rain columns after pristine span are modified.
    """
    if not rain_indices:
        return seq_array
    seq_len, _ = seq_array.shape
    pristine_timesteps = int(pristine_hours * 60 / time_step_mins)
    pristine_timesteps = min(max(0, pristine_timesteps), seq_len)
    if pristine_timesteps >= seq_len:
        return seq_array.copy()
    window = max(1, min(window, seq_len - pristine_timesteps))
    kernel = np.ones(window, dtype=np.float32) / window
    out = np.asarray(seq_array, dtype=np.float32).copy()
    for col in rain_indices:
        segment = out[pristine_timesteps:, col]
        smoothed = np.convolve(segment, kernel, mode="same")
        out[pristine_timesteps:, col] = smoothed
    return out


class AugmentableSequenceDataset(Dataset[Tuple[torch.Tensor, torch.Tensor]]):
    """
    Wraps (X, y) tensors and optionally applies forecast degradation to X
    in __getitem__ (training only). Preserves tensor shapes.
    """

    def __init__(
        self,
        X: torch.Tensor,
        y: torch.Tensor,
        rain_indices: List[int],
        apply_forecast_augmentation: bool = False,
        pristine_hours: float = 9.0,
        time_step_mins: int = 10,
        window: int = 5,
    ) -> None:
        self.X = X
        self.y = y
        self.rain_indices = list(rain_indices)
        self.apply_forecast_augmentation = apply_forecast_augmentation
        self.pristine_hours = pristine_hours
        self.time_step_mins = time_step_mins
        self.window = window

    def __len__(self) -> int:
        return self.X.size(0)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.X[idx]  # (seq_len, num_features)
        if self.apply_forecast_augmentation and self.rain_indices:
            arr = x.numpy()
            degraded = simulate_forecast_degradation(
                arr,
                self.rain_indices,
                pristine_hours=self.pristine_hours,
                time_step_mins=self.time_step_mins,
                window=self.window,
            )
            x = torch.from_numpy(degraded).float()
        return x, self.y[idx]


# Canonical feature names produced by add_engineered_features (notebook-exact).
RAIN_FEATURES = [
    "rainrate",
    "rain_cum_6t",
    "rain_cum_12t",
    "rain_cum_24t",
    "rain_cum_36t",
    "rain_max_1h",
    "rain_max_2h",
    "rain_max_4h",
    "rain_mean_1h",
    "rain_rate_of_change",
    "rain_acceleration",
    "rain_weighted_cum",
]
DISCHARGE_FEATURES = ["discharge_lag1", "discharge_lag1_log"]
COORD_FEATURES = ["lon", "lat"]
TIME_FEATURES = ["hour_sin", "hour_cos", "month_sin", "month_cos", "doy_sin", "doy_cos"]

DEFAULT_TERRAIN = [
    "DEM_MEAN",
    "SLOPE_MEAN",
    "ASPECT_MEAN",
    "FLOWACC_MEAN",
    "FLOWDIR_MEAN",
    "RUGGED_MEAN",
    "AREA_2M",
]


def get_feature_cols(
    terrain_features: Optional[Sequence[str]] = None,
) -> Tuple[List[str], List[str]]:
    """
    Return (feature_cols_s1, feature_cols_s2) in notebook-exact order.
    Stage 1: rain + terrain + coord + time. Stage 2: same + discharge_lag.
    """
    terrain = list(terrain_features or DEFAULT_TERRAIN)
    s1 = RAIN_FEATURES + terrain + COORD_FEATURES + TIME_FEATURES
    s2 = s1 + DISCHARGE_FEATURES
    return s1, s2


def group_features(
    feature_cols: List[str],
    include_discharge: bool = True,
) -> Dict[str, List[int]]:
    """
    Map feature names to column indices for embedding groups (notebook-exact).
    Returns dict with keys: rain, discharge, terrain, coord, time.
    """
    rain = [i for i, f in enumerate(feature_cols) if f in RAIN_FEATURES or "rain" in f.lower()]
    discharge = (
        [i for i, f in enumerate(feature_cols) if f in DISCHARGE_FEATURES]
        if include_discharge
        else []
    )
    coord = [i for i, f in enumerate(feature_cols) if f in COORD_FEATURES or f in ("X", "Y")]
    time_feat = [i for i, f in enumerate(feature_cols) if f in TIME_FEATURES]
    terrain = [
        i
        for i, f in enumerate(feature_cols)
        if f not in {*RAIN_FEATURES, *DISCHARGE_FEATURES, *COORD_FEATURES, *TIME_FEATURES}
        and "rain" not in f.lower()
    ]
    return {
        "rain": rain,
        "discharge": discharge,
        "terrain": terrain,
        "coord": coord,
        "time": time_feat,
    }


def make_sequences(
    df: pd.DataFrame,
    feature_cols: List[str],
    target_col: str,
    seq_len: int,
    station_ids: Optional[Sequence[int]] = None,
    log_target: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build (X, y) sequences exactly as in the notebook: rolling windows per ID,
    each sample [t - seq_len + 1 : t + 1] for features, target at t.
    If log_target=True, y = np.log1p(raw_target) for Stage 2 regression (notebook-exact).
    """
    df = df.sort_values(["ID", "time"]).reset_index(drop=True)
    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        raise KeyError(f"make_sequences: missing columns {missing}")
    if target_col not in df.columns:
        raise KeyError(f"make_sequences: target column '{target_col}' not in dataframe")

    sequences: List[np.ndarray] = []
    targets: List[float] = []
    unique_ids = list(df["ID"].unique()) if station_ids is None else list(station_ids)

    for cell_id in unique_ids:
        cell_data = df[df["ID"] == cell_id].copy()
        cell_data = cell_data.sort_values("time").reset_index(drop=True)
        if len(cell_data) < seq_len:
            continue
        for t_idx in range(seq_len - 1, len(cell_data)):
            seq_slice = cell_data.iloc[t_idx - seq_len + 1 : t_idx + 1]
            feat_values = seq_slice[feature_cols].to_numpy(dtype=np.float32)
            y_val = float(cell_data.loc[t_idx, target_col])
            if not np.isfinite(y_val):
                continue
            if log_target:
                y_val = float(np.log1p(max(0.0, y_val)))
            sequences.append(feat_values)
            targets.append(y_val)

    if not sequences:
        return np.empty((0, seq_len, len(feature_cols)), dtype=np.float32), np.empty((0,), dtype=np.float32)

    X = np.stack(sequences, axis=0)
    y = np.asarray(targets, dtype=np.float32)
    return X, y


def load_merged_parquets_with_engineered_features(
    merge_dir: Path,
    years: List[int],
    ugrid_df: Optional[pd.DataFrame] = None,
    terrain_cols: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Load rain_with_discharge_{year}.parquet for given years, optionally join
    UGRID terrain, then add_engineered_features (notebook-exact).
    """
    parts: List[pd.DataFrame] = []
    for year in years:
        path = merge_dir / f"rain_with_discharge_{year}.parquet"
        if not path.is_file():
            continue
        df = pd.read_parquet(path)
        df["time"] = pd.to_datetime(df["time"])
        parts.append(df)
    if not parts:
        return pd.DataFrame()
    df = pd.concat(parts, ignore_index=True)
    if ugrid_df is not None and terrain_cols is not None:
        merge_cols = ["ID"] + [c for c in terrain_cols if c in ugrid_df.columns]
        if merge_cols:
            df = df.merge(ugrid_df[merge_cols], on="ID", how="left")
    df = add_engineered_features(df, include_lagged_discharge=True)
    return df


def build_dataloaders_from_merged_years(
    basin_folder: Path,
    train_years: List[int],
    val_years: List[int],
    seq_len: int,
    batch_size: int,
    discharge_threshold: float = 0.01,
    terrain_features: Optional[List[str]] = None,
    ugrid_path: Optional[Path] = None,
    apply_forecast_augmentation: bool = False,
    pristine_hours: float = 9.0,
    time_step_mins: int = 10,
    augmentation_window: int = 5,
    test_years: Optional[List[int]] = None,
) -> Tuple[
    DataLoader[Any],
    DataLoader[Any],
    DataLoader[Any],
    DataLoader[Any],
    List[str],
    List[str],
    Dict[str, List[int]],
    Dict[str, List[int]],
    StandardScaler,
    StandardScaler,
    Optional[DataLoader[Any]],
    Optional[DataLoader[Any]],
]:
    """
    Build Stage 1 and Stage 2 train/val (and optional test) DataLoaders from merged parquet years.
    StandardScaler is fitted ONLY on the training split; train, val, and test are
    transformed to prevent data leakage.
    Returns loaders, feature_cols_s1, feature_cols_s2, groups_s1, groups_s2, scaler_s1, scaler_s2,
    test_loader_s1, test_loader_s2 (None if test_years is empty or None).
    """
    merge_dir = basin_folder / "output" / "rain_with_discharge"
    ugrid_df = None
    if ugrid_path and ugrid_path.is_file():
        ugrid_df = pd.read_parquet(ugrid_path)
    terrain = list(terrain_features or DEFAULT_TERRAIN)
    # Only attach terrain columns that exist in UGRID
    if ugrid_df is not None:
        terrain = [c for c in terrain if c in ugrid_df.columns]
    feature_cols_s1, feature_cols_s2 = get_feature_cols(terrain)
    groups_s1 = group_features(feature_cols_s1, include_discharge=False)
    groups_s2 = group_features(feature_cols_s2, include_discharge=True)

    df_train = load_merged_parquets_with_engineered_features(
        merge_dir, train_years, ugrid_df=ugrid_df, terrain_cols=terrain
    )
    df_val = load_merged_parquets_with_engineered_features(
        merge_dir, val_years, ugrid_df=ugrid_df, terrain_cols=terrain
    )
    if df_train.empty or df_val.empty:
        raise ValueError("No data after loading merged parquets; check train_years and val_years.")

    # Ensure all feature columns exist (fill missing with 0)
    for c in feature_cols_s1:
        if c not in df_train.columns:
            df_train[c] = 0.0
        if c not in df_val.columns:
            df_val[c] = 0.0
    for c in feature_cols_s2:
        if c not in df_train.columns:
            df_train[c] = 0.0
        if c not in df_val.columns:
            df_val[c] = 0.0
    # Use only columns that exist and are in feature list
    exist_s1 = [c for c in feature_cols_s1 if c in df_train.columns]
    exist_s2 = [c for c in feature_cols_s2 if c in df_train.columns]
    if len(exist_s1) != len(feature_cols_s1) or len(exist_s2) != len(feature_cols_s2):
        feature_cols_s1 = exist_s1
        feature_cols_s2 = exist_s2
        groups_s1 = group_features(feature_cols_s1, include_discharge=False)
        groups_s2 = group_features(feature_cols_s2, include_discharge=True)

    df_train["event_flag"] = (df_train["discharge"] >= discharge_threshold).astype("float32")
    df_val["event_flag"] = (df_val["discharge"] >= discharge_threshold).astype("float32")

    X1_tr, y1_tr = make_sequences(df_train, feature_cols_s1, "event_flag", seq_len)
    X1_val, y1_val = make_sequences(df_val, feature_cols_s1, "event_flag", seq_len)
    # Stage 2: notebook uses log1p(discharge) as target for weighted_mse_loss in log-space
    X2_tr, y2_tr = make_sequences(df_train, feature_cols_s2, "discharge", seq_len, log_target=True)
    X2_val, y2_val = make_sequences(df_val, feature_cols_s2, "discharge", seq_len, log_target=True)

    n_feat_s1 = X1_tr.shape[2]
    n_feat_s2 = X2_tr.shape[2]
    scaler_s1 = StandardScaler()
    scaler_s2 = StandardScaler()
    scaler_s1.fit(X1_tr.reshape(-1, n_feat_s1))
    scaler_s2.fit(X2_tr.reshape(-1, n_feat_s2))
    X1_tr = scaler_s1.transform(X1_tr.reshape(-1, n_feat_s1)).reshape(X1_tr.shape).astype(np.float32)
    X1_val = scaler_s1.transform(X1_val.reshape(-1, n_feat_s1)).reshape(X1_val.shape).astype(np.float32)
    X2_tr = scaler_s2.transform(X2_tr.reshape(-1, n_feat_s2)).reshape(X2_tr.shape).astype(np.float32)
    X2_val = scaler_s2.transform(X2_val.reshape(-1, n_feat_s2)).reshape(X2_val.shape).astype(np.float32)

    X1_tr_t = torch.from_numpy(X1_tr).float()
    y1_tr_t = torch.from_numpy(y1_tr).float()
    X1_val_t = torch.from_numpy(X1_val).float()
    y1_val_t = torch.from_numpy(y1_val).float()
    X2_tr_t = torch.from_numpy(X2_tr).float()
    y2_tr_t = torch.from_numpy(y2_tr).float()
    X2_val_t = torch.from_numpy(X2_val).float()
    y2_val_t = torch.from_numpy(y2_val).float()

    if apply_forecast_augmentation:
        train_s1 = AugmentableSequenceDataset(
            X1_tr_t, y1_tr_t,
            rain_indices=groups_s1["rain"],
            apply_forecast_augmentation=True,
            pristine_hours=pristine_hours,
            time_step_mins=time_step_mins,
            window=augmentation_window,
        )
        train_s2 = AugmentableSequenceDataset(
            X2_tr_t, y2_tr_t,
            rain_indices=groups_s2["rain"],
            apply_forecast_augmentation=True,
            pristine_hours=pristine_hours,
            time_step_mins=time_step_mins,
            window=augmentation_window,
        )
    else:
        train_s1 = TensorDataset(X1_tr_t, y1_tr_t)
        train_s2 = TensorDataset(X2_tr_t, y2_tr_t)

    val_s1 = TensorDataset(X1_val_t, y1_val_t)
    val_s2 = TensorDataset(X2_val_t, y2_val_t)

    train_loader_s1 = DataLoader(train_s1, batch_size=batch_size, shuffle=True)
    val_loader_s1 = DataLoader(val_s1, batch_size=batch_size)
    train_loader_s2 = DataLoader(train_s2, batch_size=batch_size, shuffle=True)
    val_loader_s2 = DataLoader(val_s2, batch_size=batch_size)

    test_loader_s1: Optional[DataLoader[Any]] = None
    test_loader_s2: Optional[DataLoader[Any]] = None
    if test_years:
        df_test = load_merged_parquets_with_engineered_features(
            merge_dir, test_years, ugrid_df=ugrid_df, terrain_cols=terrain
        )
        if not df_test.empty:
            for c in feature_cols_s1:
                if c not in df_test.columns:
                    df_test[c] = 0.0
            for c in feature_cols_s2:
                if c not in df_test.columns:
                    df_test[c] = 0.0
            df_test["event_flag"] = (df_test["discharge"] >= discharge_threshold).astype("float32")
            X1_te, y1_te = make_sequences(df_test, feature_cols_s1, "event_flag", seq_len)
            X2_te, y2_te = make_sequences(df_test, feature_cols_s2, "discharge", seq_len, log_target=True)
            X1_te = scaler_s1.transform(X1_te.reshape(-1, n_feat_s1)).reshape(X1_te.shape).astype(np.float32)
            X2_te = scaler_s2.transform(X2_te.reshape(-1, n_feat_s2)).reshape(X2_te.shape).astype(np.float32)
            test_s1 = TensorDataset(torch.from_numpy(X1_te).float(), torch.from_numpy(y1_te).float())
            test_s2 = TensorDataset(torch.from_numpy(X2_te).float(), torch.from_numpy(y2_te).float())
            test_loader_s1 = DataLoader(test_s1, batch_size=batch_size)
            test_loader_s2 = DataLoader(test_s2, batch_size=batch_size)

    return (
        train_loader_s1,
        val_loader_s1,
        train_loader_s2,
        val_loader_s2,
        feature_cols_s1,
        feature_cols_s2,
        groups_s1,
        groups_s2,
        scaler_s1,
        scaler_s2,
        test_loader_s1,
        test_loader_s2,
    )
