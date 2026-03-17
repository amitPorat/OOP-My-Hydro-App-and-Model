"""
Training entry point: load config, apply UI overrides, run Stage 1 and Stage 2
in-process with optional ui_callback for live metric streaming.
"""

from __future__ import annotations

import uuid
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
import yaml
from torch.utils.data import DataLoader

from src.dataset import build_dataloaders_from_rmcomp
from src.experiment_tracker import ExperimentLogger
from src.ml_dataset import build_dataloaders_from_merged_years
from src.model import BinaryClassifierLSTM, ImprovedLSTMWithEmbeddings
from src.trainer import Stage1Trainer, Stage2Trainer, _group_features
from src.tracking import DEFAULT_DB_PATH


def load_config(path_or_dict: Path | str | Dict[str, Any]) -> Dict[str, Any]:
    if isinstance(path_or_dict, dict):
        return dict(path_or_dict)
    path = Path(path_or_dict)
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _deep_merge(base: Dict[str, Any], overrides: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(base)
    for k, v in overrides.items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def apply_overrides(
    config: Dict[str, Any],
    overrides: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    if not overrides:
        return config
    # Map flat UI keys into nested config
    flat = {}
    if "epochs" in overrides:
        flat.setdefault("training", {})["max_epochs"] = int(overrides["epochs"])
    if "batch_size" in overrides:
        flat.setdefault("training", {})["batch_size"] = int(overrides["batch_size"])
    if "learning_rate" in overrides:
        flat.setdefault("training", {})["learning_rate"] = float(overrides["learning_rate"])
    if "hidden_size" in overrides:
        h = int(overrides["hidden_size"])
        flat.setdefault("model", {})["stage1_binary"] = {
            **config.get("model", {}).get("stage1_binary", {}),
            "hidden_size": h,
        }
        flat.setdefault("model", {})["stage2_regression"] = {
            **config.get("model", {}).get("stage2_regression", {}),
            "hidden_size": h,
        }
    if "seq_len" in overrides:
        flat.setdefault("model", {})["seq_len"] = int(overrides["seq_len"])
    merged = _deep_merge(config, flat)
    # Merge again with any direct nested overrides
    if "training" in overrides:
        merged["training"] = _deep_merge(merged.get("training", {}), overrides["training"])
    if "model" in overrides:
        merged["model"] = _deep_merge(merged.get("model", {}), overrides["model"])
    return merged


def build_feature_cols(config: Dict[str, Any]) -> Tuple[List[str], List[str]]:
    terrain = list(config.get("features", {}).get("terrain_features", []) or [])
    if not terrain:
        terrain = ["DEM_MEAN", "SLOPE_MEAN", "ASPECT_MEAN", "FLOWACC_MEAN", "FLOWDIR_MEAN", "RUGGED_MEAN", "AREA_2M"]
    rain = [f"rain_{i}" for i in range(6)]
    coords = ["lon", "lat"]
    time_feat = ["hour_sin", "hour_cos", "month_sin", "month_cos", "doy_sin", "doy_cos"]
    feature_cols_s1 = rain + terrain + coords + time_feat
    discharge_lags = ["discharge_lag_1", "discharge_lag_2"]
    feature_cols_s2 = feature_cols_s1 + discharge_lags
    return feature_cols_s1, feature_cols_s2


def run_training(
    config_path_or_dict: Path | str | Dict[str, Any],
    overrides: Optional[Dict[str, Any]] = None,
    ui_callback: Optional[Callable[[int, float, float, Optional[float], Optional[float]], None]] = None,
    db_path: Optional[Path] = None,
) -> Dict[str, Any]:
    config = load_config(config_path_or_dict)
    config = apply_overrides(config, overrides)

    feature_cols_s1, feature_cols_s2 = build_feature_cols(config)
    groups_s1 = _group_features(feature_cols_s1, include_discharge=False)
    groups_s2 = _group_features(feature_cols_s2, include_discharge=True)

    model_cfg = config.get("model", {})
    seq_len = int(model_cfg.get("seq_len", 24))
    embed_dim = int(model_cfg.get("embed_dim", 32))
    s1_cfg = model_cfg.get("stage1_binary", {})
    s2_cfg = model_cfg.get("stage2_regression", {})
    tr_cfg = config.get("training", {})
    batch_size = int(tr_cfg.get("batch_size", 32))

    data_paths = config.get("data_paths", {})
    basin_folder = data_paths.get("basin_folder", ".")
    checkpoint_base = Path(basin_folder) / "output" / "model"
    dir_s1 = checkpoint_base / "TWO_STAGE_STAGE1_HYDROGRAPH"
    dir_s2 = checkpoint_base / "LSTM_IMPROVED"

    run_id_base = str(uuid.uuid4())
    run_id_s1 = run_id_base + "_stage1"
    run_id_s2 = run_id_base + "_stage2"
    basin_name = "default"
    if overrides and "basin_name" in overrides:
        basin_name = str(overrides["basin_name"])
    else:
        try:
            basin_name = Path(basin_folder).name or "default"
        except Exception:
            pass
    config_path_str = "config.yaml"
    if overrides and "config_path" in overrides:
        config_path_str = str(overrides["config_path"])
    elif isinstance(config_path_or_dict, (Path, str)):
        config_path_str = str(config_path_or_dict)
    db_path = db_path or DEFAULT_DB_PATH
    features_json = "[]"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Build RMcomp-based event dataloaders for both stages.
    (
        train_loader_s1,
        val_loader_s1,
        train_loader_s2,
        val_loader_s2,
        scaler_s1,
        scaler_s2,
    ) = build_dataloaders_from_rmcomp(config, feature_cols_s1, feature_cols_s2)
    model_s1 = BinaryClassifierLSTM(
        rain_indices=groups_s1["rain"],
        terrain_indices=groups_s1["terrain"],
        coord_indices=groups_s1["coord"],
        time_indices=groups_s1["time"],
        feature_cols=feature_cols_s1,
        hidden_size=int(s1_cfg.get("hidden_size", 256)),
        num_layers=int(s1_cfg.get("num_layers", 2)),
        dropout=float(s1_cfg.get("dropout", 0.2)),
        embed_dim=embed_dim,
    )
    trainer_s1 = Stage1Trainer(
        model=model_s1,
        train_loader=train_loader_s1,
        val_loader=val_loader_s1,
        config=config,
        device=device,
        checkpoint_dir=dir_s1,
        run_id=run_id_s1,
        basin_name=basin_name,
        config_path=config_path_str,
        features_json=features_json,
        ui_callback=ui_callback,
        db_path=db_path,
        feature_cols=feature_cols_s1,
        feature_groups=groups_s1,
    )
    path_s1 = trainer_s1.run()

    model_s2 = ImprovedLSTMWithEmbeddings(
        rain_indices=groups_s2["rain"],
        discharge_indices=groups_s2["discharge"],
        terrain_indices=groups_s2["terrain"],
        coord_indices=groups_s2["coord"],
        time_indices=groups_s2["time"],
        feature_cols=feature_cols_s2,
        hidden_size=int(s2_cfg.get("hidden_size", 128)),
        num_layers=int(s2_cfg.get("num_layers", 2)),
        dropout=float(s2_cfg.get("dropout", 0.2)),
        embed_dim=embed_dim,
    )
    trainer_s2 = Stage2Trainer(
        model=model_s2,
        train_loader=train_loader_s2,
        val_loader=val_loader_s2,
        config=config,
        device=device,
        checkpoint_dir=dir_s2,
        run_id=run_id_s2,
        basin_name=basin_name,
        config_path=config_path_str,
        features_json=features_json,
        ui_callback=ui_callback,
        db_path=db_path,
        feature_cols=feature_cols_s2,
        feature_groups=groups_s2,
    )
    path_s2 = trainer_s2.run()

    return {
        "run_id": run_id_base,
        "basin_name": basin_name,
        "stage1_path": str(path_s1),
        "stage2_path": str(path_s2),
    }


def run_training_playground(
    basin_folder: Path,
    experiment_name: str,
    train_years: List[int],
    val_years: List[int],
    epochs: int = 50,
    learning_rate: float = 1e-4,
    batch_size: int = 32,
    seq_len: int = 24,
    hidden_size_s1: int = 256,
    hidden_size_s2: int = 128,
    num_layers: int = 2,
    dropout: float = 0.2,
    embed_dim: int = 32,
    beta_weight: float = 15.0,
    global_qmax: float = 60.0,
    early_stop_patience: int = 7,
    lr_patience: int = 5,
    lr_factor: float = 0.5,
    lr_min: float = 1e-6,
    discharge_threshold: float = 0.01,
    experiment_notes: str = "",
    terrain_features: Optional[List[str]] = None,
    apply_forecast_augmentation: bool = False,
    pristine_hours: float = 9.0,
    time_step_mins: int = 10,
    augmentation_window: int = 5,
    ui_callback: Optional[Callable[[int, float, float, Optional[float], Optional[float]], None]] = None,
    test_years: Optional[List[int]] = None,
    experiment_logger: Optional[Any] = None,
    status_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """
    Run two-stage LSTM training from Model Playground: merged parquet by year,
    ExperimentLogger for config + per-epoch metrics, checkpoints saved in run dir.
    If experiment_logger is provided (e.g. from UI), it is used and config is assumed saved.
    status_path: if set, trainers write training_status.json each epoch for live UI polling.
    """
    basin_folder = Path(basin_folder).expanduser()
    logger = experiment_logger if experiment_logger is not None else ExperimentLogger(basin_folder, experiment_name)
    config = {
        "experiment_name": experiment_name,
        "experiment_notes": experiment_notes,
        "model_architecture": "LSTM (Baseline)",
        "epochs": epochs,
        "learning_rate": learning_rate,
        "batch_size": batch_size,
        "seq_len": seq_len,
        "hidden_size_s1": hidden_size_s1,
        "hidden_size_s2": hidden_size_s2,
        "num_layers": num_layers,
        "dropout": dropout,
        "embed_dim": embed_dim,
        "beta_weight": beta_weight,
        "global_qmax": global_qmax,
        "early_stop_patience": early_stop_patience,
        "lr_patience": lr_patience,
        "lr_factor": lr_factor,
        "lr_min": lr_min,
        "train_years": train_years,
        "val_years": val_years,
        "test_years": test_years or [],
        "discharge_threshold": discharge_threshold,
        "apply_forecast_augmentation": apply_forecast_augmentation,
        "pristine_hours": pristine_hours,
        "time_step_mins": time_step_mins,
        "augmentation_window": augmentation_window,
    }
    logger.save_config(config)

    trainer_config: Dict[str, Any] = {
        "training": {
            "max_epochs": epochs,
            "learning_rate": learning_rate,
            "batch_size": batch_size,
            "early_stopping_patience": early_stop_patience,
            "lr_scheduler": {"patience": lr_patience, "factor": lr_factor, "min_lr": lr_min},
            "use_weighted_mse_loss": True,
            "beta_weight": beta_weight,
            "global_qmax": global_qmax,
        },
        "model": {
            "seq_len": seq_len,
            "embed_dim": embed_dim,
            "stage1_binary": {
                "hidden_size": hidden_size_s1,
                "num_layers": num_layers,
                "dropout": dropout,
            },
            "stage2_regression": {
                "hidden_size": hidden_size_s2,
                "num_layers": num_layers,
                "dropout": dropout,
            },
        },
    }
    ugrid_path = basin_folder / "output" / "ugrid" / "final_ugrid.parquet"
    if not ugrid_path.is_file():
        ugrid_path = basin_folder / "output" / "ugrid" / "ugrid_cells_with_terrain.parquet"
    (
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
    ) = build_dataloaders_from_merged_years(
        basin_folder=basin_folder,
        train_years=train_years,
        val_years=val_years,
        seq_len=seq_len,
        batch_size=batch_size,
        discharge_threshold=discharge_threshold,
        terrain_features=terrain_features,
        ugrid_path=ugrid_path,
        apply_forecast_augmentation=apply_forecast_augmentation,
        pristine_hours=pristine_hours,
        time_step_mins=time_step_mins,
        augmentation_window=augmentation_window,
        test_years=test_years or [],
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_id_base = logger.run_id
    run_id_s1 = run_id_base + "_stage1"
    run_id_s2 = run_id_base + "_stage2"
    basin_name = basin_folder.name or "default"

    model_s1 = BinaryClassifierLSTM(
        rain_indices=groups_s1["rain"],
        terrain_indices=groups_s1["terrain"],
        coord_indices=groups_s1["coord"],
        time_indices=groups_s1["time"],
        feature_cols=feature_cols_s1,
        hidden_size=hidden_size_s1,
        num_layers=num_layers,
        dropout=dropout,
        embed_dim=embed_dim,
    )
    trainer_s1 = Stage1Trainer(
        model=model_s1,
        train_loader=train_loader_s1,
        val_loader=val_loader_s1,
        config=trainer_config,
        device=device,
        checkpoint_dir=logger.run_dir,
        run_id=run_id_s1,
        basin_name=basin_name,
        config_path="playground",
        features_json="[]",
        ui_callback=ui_callback,
        db_path=None,
        feature_cols=feature_cols_s1,
        feature_groups=groups_s1,
        experiment_logger=logger,
        scaler=scaler_s1,
        test_loader=test_loader_s1,
        status_path=status_path,
    )
    path_s1 = trainer_s1.run()

    model_s2 = ImprovedLSTMWithEmbeddings(
        rain_indices=groups_s2["rain"],
        discharge_indices=groups_s2["discharge"],
        terrain_indices=groups_s2["terrain"],
        coord_indices=groups_s2["coord"],
        time_indices=groups_s2["time"],
        feature_cols=feature_cols_s2,
        hidden_size=hidden_size_s2,
        num_layers=num_layers,
        dropout=dropout,
        embed_dim=embed_dim,
    )
    trainer_s2 = Stage2Trainer(
        model=model_s2,
        train_loader=train_loader_s2,
        val_loader=val_loader_s2,
        config=trainer_config,
        device=device,
        checkpoint_dir=logger.run_dir,
        run_id=run_id_s2,
        basin_name=basin_name,
        config_path="playground",
        features_json="[]",
        ui_callback=ui_callback,
        db_path=None,
        feature_cols=feature_cols_s2,
        feature_groups=groups_s2,
        experiment_logger=logger,
        scaler=scaler_s2,
        test_loader=test_loader_s2,
        status_path=status_path,
    )
    path_s2 = trainer_s2.run()

    out: Dict[str, Any] = {
        "run_id": run_id_base,
        "basin_name": basin_name,
        "stage1_path": str(path_s1),
        "stage2_path": str(path_s2),
        "run_dir": str(logger.run_dir),
    }
    if hasattr(trainer_s1, "test_loss") and trainer_s1.test_loss is not None:
        out["test_loss_s1"] = trainer_s1.test_loss
    if hasattr(trainer_s2, "test_loss") and trainer_s2.test_loss is not None:
        out["test_loss_s2"] = trainer_s2.test_loss
    if hasattr(trainer_s2, "test_nse") and trainer_s2.test_nse is not None:
        out["test_nse"] = trainer_s2.test_nse
    if hasattr(trainer_s2, "test_kge") and trainer_s2.test_kge is not None:
        out["test_kge"] = trainer_s2.test_kge
    return out


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/system_config.yaml"))
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--hidden-size", type=int, default=None)
    parser.add_argument("--seq-len", type=int, default=None)
    args = parser.parse_args()
    overrides = {}
    if args.epochs is not None:
        overrides["epochs"] = args.epochs
    if args.batch_size is not None:
        overrides["batch_size"] = args.batch_size
    if args.lr is not None:
        overrides["learning_rate"] = args.lr
    if args.hidden_size is not None:
        overrides["hidden_size"] = args.hidden_size
    if args.seq_len is not None:
        overrides["seq_len"] = args.seq_len
    result = run_training(args.config, overrides=overrides if overrides else None)
    print("Training complete:", result)
