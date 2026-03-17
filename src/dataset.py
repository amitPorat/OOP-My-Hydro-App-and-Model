from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

from .data_loader import WeatherDataLoader
from .features import UgridBuilder, add_engineered_features


@dataclass
class SplitConfig:
    """Configuration for chronological train/validation split."""

    val_fraction: float = 0.2


def _load_rmcomp_event_dataframe(config: Mapping[str, Any]) -> pd.DataFrame:
    data_paths = config.get("data_paths", {})
    if not isinstance(data_paths, Mapping):
        raise ValueError("Configuration key 'data_paths' must be a mapping.")

    basin_folder = Path(str(data_paths.get("basin_folder", ""))).expanduser()
    default_event_dir = basin_folder / "output" / "rain" / "event_rain"
    events_dir = Path(
        str(data_paths.get("rain_events_dir", default_event_dir))
    ).expanduser()

    if not events_dir.is_dir():
        raise FileNotFoundError(
            f"RMcomp event rain directory does not exist: {events_dir}"
        )

    parts: List[pd.DataFrame] = []
    for p in sorted(events_dir.glob("rain_events_*.parquet")):
        df = pd.read_parquet(p)
        parts.append(df)

    if not parts:
        raise FileNotFoundError(
            f"No RMcomp event rain parquet files found in {events_dir} "
            "(expected files named 'rain_events_{year}.parquet')."
        )

    df_all = pd.concat(parts, ignore_index=True)
    if "time" in df_all.columns:
        df_all["time"] = pd.to_datetime(df_all["time"])
    return df_all


def _attach_static_and_engineered_features(
    base_df: pd.DataFrame,
    loader: WeatherDataLoader,
) -> pd.DataFrame:
    """
    Attach UGRID static features and reproduce notebook feature engineering.

    The RMcomp event parquet is expected to contain at least:
    ``['ID', 'time', 'rainrate', 'discharge']`` with optional ``event_id``.
    """
    ugrid_df = loader.load_ugrid()
    terrain_features = loader._config.get("features", {}).get(  # type: ignore[attr-defined]
        "terrain_features", []
    )
    builder = UgridBuilder(ugrid_df, terrain_features=terrain_features or [])

    df = base_df.copy()
    if "time" in df.columns:
        df["time"] = pd.to_datetime(df["time"])
    df = builder.attach_static_features(df)

    if "discharge" not in df.columns:
        raise KeyError(
            "RMcomp event dataset is missing 'discharge' column. The training "
            "pipeline expects observed discharge for supervised learning."
        )

    df = add_engineered_features(df, include_lagged_discharge=True)
    return df


def _build_sequences(
    df: pd.DataFrame,
    feature_cols: List[str],
    target_col: str,
    seq_len: int,
    station_ids: Iterable[int] | None = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build rolling sequences for supervised training.

    This mirrors the temporal structure used in inference, but returns
    explicit (X, y) pairs for PyTorch training.
    """
    df = df.sort_values(["ID", "time"]).reset_index(drop=True)
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
            feat_values = seq_slice[feature_cols].to_numpy(dtype=float)
            y_val = float(cell_data.loc[t_idx, target_col])
            if not np.isfinite(y_val):
                continue
            sequences.append(feat_values)
            targets.append(y_val)

    if not sequences:
        return np.empty((0, seq_len, len(feature_cols))), np.empty((0,), dtype=float)

    X = np.stack(sequences, axis=0)
    y = np.asarray(targets, dtype=float)
    return X, y


def _split_train_val_chronological(
    X: np.ndarray,
    y: np.ndarray,
    split_cfg: SplitConfig,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n = X.shape[0]
    if n == 0:
        return X, X, y, y
    n_val = max(1, int(round(split_cfg.val_fraction * n)))
    n_train = max(1, n - n_val)
    X_train = X[:n_train]
    y_train = y[:n_train]
    X_val = X[n_train:]
    y_val = y[n_train:]
    return X_train, X_val, y_train, y_val


def build_dataloaders_from_rmcomp(
    config: Dict[str, Any],
    feature_cols_s1: List[str],
    feature_cols_s2: List[str],
) -> Tuple[
    DataLoader[Any],
    DataLoader[Any],
    DataLoader[Any],
    DataLoader[Any],
    StandardScaler,
    StandardScaler,
]:
    """
    Construct Stage 1 and Stage 2 DataLoaders backed by RMcomp event rain data.

    StandardScaler is fitted ONLY on the training split; .transform() is applied
    to train and validation to prevent data leakage. Returns (loaders, scaler_s1, scaler_s2).

    Stage 1 target is a binary flood-event indicator derived from discharge
    using the same threshold as used during inference. Stage 2 target is
    continuous discharge.
    """
    model_cfg = config.get("model", {}) or {}
    tr_cfg = config.get("training", {}) or {}
    inf_cfg = config.get("inference", {}) or {}
    stations_cfg = config.get("stations", {}) or {}

    seq_len = int(model_cfg.get("seq_len", 24))
    batch_size = int(tr_cfg.get("batch_size", 32))
    discharge_threshold = float(
        inf_cfg.get("discharge_threshold_m3s", 0.01)
    )
    val_fraction = float(tr_cfg.get("val_fraction", 0.2))
    split_cfg = SplitConfig(val_fraction=val_fraction)

    loader = WeatherDataLoader(config)  # reuses data_paths and UGRID config

    df_events = _load_rmcomp_event_dataframe(config)
    df_events["time"] = pd.to_datetime(df_events["time"])

    # Binary label for Stage 1: event if discharge exceeds threshold.
    if "discharge" not in df_events.columns:
        raise KeyError(
            "RMcomp event dataset is missing 'discharge' column. Cannot "
            "derive binary event labels for Stage 1."
        )
    df_events["event_flag"] = (df_events["discharge"] >= discharge_threshold).astype(
        "float32"
    )

    df_full = _attach_static_and_engineered_features(df_events, loader)

    station_cells: List[int] = list(stations_cfg.get("station_cells", []))
    station_ids = station_cells or None

    X1, y1 = _build_sequences(
        df_full,
        feature_cols=feature_cols_s1,
        target_col="event_flag",
        seq_len=seq_len,
        station_ids=station_ids,
    )
    X2, y2 = _build_sequences(
        df_full,
        feature_cols=feature_cols_s2,
        target_col="discharge",
        seq_len=seq_len,
        station_ids=station_ids,
    )

    X1_tr, X1_val, y1_tr, y1_val = _split_train_val_chronological(X1, y1, split_cfg)
    X2_tr, X2_val, y2_tr, y2_val = _split_train_val_chronological(X2, y2, split_cfg)

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

    train_s1 = TensorDataset(
        torch.from_numpy(X1_tr), torch.from_numpy(y1_tr).float()
    )
    val_s1 = TensorDataset(
        torch.from_numpy(X1_val), torch.from_numpy(y1_val).float()
    )
    train_s2 = TensorDataset(
        torch.from_numpy(X2_tr), torch.from_numpy(y2_tr).float()
    )
    val_s2 = TensorDataset(
        torch.from_numpy(X2_val), torch.from_numpy(y2_val).float()
    )

    train_loader_s1 = DataLoader(train_s1, batch_size=batch_size, shuffle=True)
    val_loader_s1 = DataLoader(val_s1, batch_size=batch_size)
    train_loader_s2 = DataLoader(train_s2, batch_size=batch_size, shuffle=True)
    val_loader_s2 = DataLoader(val_s2, batch_size=batch_size)

    return train_loader_s1, val_loader_s1, train_loader_s2, val_loader_s2, scaler_s1, scaler_s2

