"""
PyTorch training loops for Stage 1 (binary) and Stage 2 (regression) LSTM models.

Supports optional UI callbacks for live metric streaming and logs every run
to the experiment tracking database.
"""

from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import torch
from torch import Tensor, nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from .model import (
    BinaryClassifierLSTM,
    ImprovedLSTMWithEmbeddings,
    IdentityScaler,
    weighted_mse_loss,
)
from .tracking import DEFAULT_DB_PATH, ExperimentRecord, log_experiment


def nse(obs: np.ndarray, sim: np.ndarray) -> float:
    """Nash-Sutcliffe Efficiency. 1 = perfect; <0 worse than mean."""
    obs = np.asarray(obs, dtype=float).ravel()
    sim = np.asarray(sim, dtype=float).ravel()
    mask = np.isfinite(obs) & np.isfinite(sim)
    if mask.sum() < 2:
        return float("nan")
    o, s = obs[mask], sim[mask]
    ss_res = np.sum((o - s) ** 2)
    ss_tot = np.sum((o - o.mean()) ** 2)
    if ss_tot == 0:
        return float("nan")
    return float(1.0 - ss_res / ss_tot)


def kge(obs: np.ndarray, sim: np.ndarray) -> float:
    """Kling-Gupta Efficiency. 1 = perfect."""
    obs = np.asarray(obs, dtype=float).ravel()
    sim = np.asarray(sim, dtype=float).ravel()
    mask = np.isfinite(obs) & np.isfinite(sim)
    if mask.sum() < 2:
        return float("nan")
    o, s = obs[mask], sim[mask]
    r = np.corrcoef(o, s)[0, 1] if np.std(o) > 0 and np.std(s) > 0 else 0.0
    alpha = np.std(s, ddof=1) / np.std(o, ddof=1) if np.std(o, ddof=1) > 0 else 0.0
    beta = np.mean(s) / np.mean(o) if np.mean(o) != 0 else 0.0
    return float(1.0 - np.sqrt((r - 1) ** 2 + (alpha - 1) ** 2 + (beta - 1) ** 2))


UICallback = Callable[
    [int, float, float, Optional[float], Optional[float]],
    None,
]


def _group_features(feature_cols: List[str], include_discharge: bool = True) -> Dict[str, List[int]]:
    """Map feature names to indices for embedding groups (notebook-compatible)."""
    rain = [i for i, f in enumerate(feature_cols) if "rain" in f.lower()]
    discharge = (
        [i for i, f in enumerate(feature_cols) if "discharge" in f.lower() and "lag" in f.lower()]
        if include_discharge
        else []
    )
    coord = [
        i
        for i, f in enumerate(feature_cols)
        if f in ("lon", "lat", "lon_x", "lat_x", "lon_y", "lat_y", "X", "Y")
    ]
    time_feat = [i for i, f in enumerate(feature_cols) if f in ("hour_sin", "hour_cos", "month_sin", "month_cos", "doy_sin", "doy_cos")]
    terrain = [
        i
        for i, f in enumerate(feature_cols)
        if f not in {*rain, *discharge, *coord, *time_feat}
    ]
    return {
        "rain": rain,
        "discharge": discharge,
        "coord": coord,
        "time": time_feat,
        "terrain": terrain,
    }


class Stage1Trainer:
    """Train Stage 1 binary classifier; no NSE/KGE; logs to DB at end."""

    def __init__(
        self,
        model: BinaryClassifierLSTM,
        train_loader: DataLoader[Any],
        val_loader: DataLoader[Any],
        config: Dict[str, Any],
        device: torch.device,
        checkpoint_dir: Path,
        run_id: str,
        basin_name: str,
        config_path: str,
        features_json: str = "[]",
        ui_callback: Optional[UICallback] = None,
        db_path: Optional[Path] = None,
        feature_cols: Optional[List[str]] = None,
        feature_groups: Optional[Dict[str, List[int]]] = None,
        experiment_logger: Optional[Any] = None,
        scaler: Optional[Any] = None,
        test_loader: Optional[DataLoader[Any]] = None,
        status_path: Optional[Path] = None,
    ) -> None:
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = device
        self.checkpoint_dir = checkpoint_dir
        self.run_id = run_id
        self.basin_name = basin_name
        self.config_path = config_path
        self.features_json = features_json
        self.ui_callback = ui_callback
        self.db_path = db_path or DEFAULT_DB_PATH
        self.experiment_logger = experiment_logger
        self.test_loader = test_loader
        self.status_path = Path(status_path) if status_path else None
        self.test_loss: Optional[float] = None

        # Notebook-compatible metadata
        self.feature_cols: List[str] = list(feature_cols or [])
        self.feature_groups: Dict[str, List[int]] = feature_groups or {
            "rain": [],
            "terrain": [],
            "coord": [],
            "time": [],
        }
        self.scaler = scaler if scaler is not None else IdentityScaler()

        tr = config.get("training", {})
        self.epochs = int(tr.get("max_epochs", 50))
        self.lr = float(tr.get("learning_rate", 1e-4))
        self.patience = int(tr.get("early_stopping_patience", 7))
        self.lr_patience = int(tr.get("lr_scheduler", {}).get("patience", 5))
        self.lr_factor = float(tr.get("lr_scheduler", {}).get("factor", 0.5))
        self.lr_min = float(tr.get("lr_scheduler", {}).get("min_lr", 1e-6))

        model_cfg = config.get("model", {})
        s1 = model_cfg.get("stage1_binary", {})
        self.hidden_size = int(s1.get("hidden_size", 256))
        self.num_layers = int(s1.get("num_layers", 2))
        self.dropout = float(s1.get("dropout", 0.2))
        self.seq_len = int(model_cfg.get("seq_len", 24))
        self.embed_dim = int(model_cfg.get("embed_dim", 32))
        self.batch_size = int(tr.get("batch_size", 32))

        self.optimizer = Adam(self.model.parameters(), lr=self.lr)
        self.scheduler = ReduceLROnPlateau(
            self.optimizer,
            mode="min",
            factor=self.lr_factor,
            patience=self.lr_patience,
            min_lr=self.lr_min,
        )
        self.criterion = nn.BCEWithLogitsLoss()

    def _write_status(
        self,
        status: str,
        epoch: int,
        train_loss: float,
        val_loss: float,
        history: List[Dict[str, Any]],
    ) -> None:
        if self.status_path is None:
            return
        obj = {
            "status": status,
            "stage": "stage1",
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_nse": None,
            "val_kge": None,
            "history": history,
        }
        try:
            with self.status_path.open("w", encoding="utf-8") as f:
                json.dump(obj, f, indent=2)
        except Exception:
            pass

    def run(self) -> Path:
        checkpoint_dir = Path(self.checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        best_val_loss = float("inf")
        best_state: Optional[Dict[str, Any]] = None
        patience_counter = 0
        history: List[Dict[str, Any]] = []
        self._write_status("running", 0, 0.0, 0.0, history)

        for epoch in range(1, self.epochs + 1):
            self.model.train()
            train_losses: List[float] = []
            for batch in self.train_loader:
                X = batch[0].to(self.device)
                y = batch[1].to(self.device).float().unsqueeze(1)
                self.optimizer.zero_grad()
                logits = self.model(X)
                loss = self.criterion(logits, y)
                loss.backward()
                self.optimizer.step()
                train_losses.append(loss.item())

            train_loss = np.mean(train_losses)
            self.model.eval()
            val_losses: List[float] = []
            with torch.no_grad():
                for batch in self.val_loader:
                    X = batch[0].to(self.device)
                    y = batch[1].to(self.device).float().unsqueeze(1)
                    logits = self.model(X)
                    loss = self.criterion(logits, y)
                    val_losses.append(loss.item())
            val_loss = np.mean(val_losses)
            self.scheduler.step(val_loss)

            history.append({
                "stage": "stage1",
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_nse": None,
                "val_kge": None,
            })
            self._write_status("running", epoch, train_loss, val_loss, history)

            if self.ui_callback is not None:
                self.ui_callback(epoch, train_loss, val_loss, None, None)
            if self.experiment_logger is not None:
                self.experiment_logger.log_metrics(
                    epoch,
                    {"stage": "stage1", "train_loss": train_loss, "val_loss": val_loss},
                )

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1
            if patience_counter >= self.patience:
                if best_state is not None:
                    self.model.load_state_dict(best_state)
                break

        self.model.eval()
        if self.test_loader is not None:
            test_losses: List[float] = []
            with torch.no_grad():
                for batch in self.test_loader:
                    X = batch[0].to(self.device)
                    y = batch[1].to(self.device).float().unsqueeze(1)
                    logits = self.model(X)
                    loss = self.criterion(logits, y)
                    test_losses.append(loss.item())
            self.test_loss = float(np.mean(test_losses))
            if self.experiment_logger is not None:
                self.experiment_logger.log_metrics(
                    0,
                    {"stage": "stage1", "test_loss": self.test_loss},
                )
        if self.status_path is not None:
            self._write_status("done", epoch, train_loss, val_loss, history)

        ckpt_path = checkpoint_dir / "stage1_binary_classifier_hydrograph.pt"
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "scaler": self.scaler,
                "feature_cols": self.feature_cols,
                "rain_indices": self.feature_groups.get("rain", []),
                "terrain_indices": self.feature_groups.get("terrain", []),
                "coord_indices": self.feature_groups.get("coord", []),
                "time_indices": self.feature_groups.get("time", []),
                "model_params": {
                    "hidden_size": self.hidden_size,
                    "num_layers": self.num_layers,
                    "dropout": self.dropout,
                    "embed_dim": self.embed_dim,
                    "seq_len": self.seq_len,
                },
            },
            ckpt_path,
        )
        record = ExperimentRecord(
            run_id=self.run_id,
            basin=self.basin_name,
            stage="stage1",
            config_path=self.config_path,
            seq_len=self.seq_len,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=self.dropout,
            batch_size=self.batch_size,
            learning_rate=self.lr,
            epochs=epoch,
            features_json=self.features_json,
            val_loss=best_val_loss,
            val_nse=None,
            val_kge=None,
            checkpoint_path=str(ckpt_path),
        )
        log_experiment(record, db_path=self.db_path)
        return ckpt_path


class Stage2Trainer:
    """Train Stage 2 regression model; compute NSE/KGE on val; logs to DB at end."""

    def __init__(
        self,
        model: ImprovedLSTMWithEmbeddings,
        train_loader: DataLoader[Any],
        val_loader: DataLoader[Any],
        config: Dict[str, Any],
        device: torch.device,
        checkpoint_dir: Path,
        run_id: str,
        basin_name: str,
        config_path: str,
        features_json: str = "[]",
        ui_callback: Optional[UICallback] = None,
        db_path: Optional[Path] = None,
        feature_cols: Optional[List[str]] = None,
        feature_groups: Optional[Dict[str, List[int]]] = None,
        experiment_logger: Optional[Any] = None,
        scaler: Optional[Any] = None,
        test_loader: Optional[DataLoader[Any]] = None,
        status_path: Optional[Path] = None,
    ) -> None:
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = device
        self.checkpoint_dir = checkpoint_dir
        self.run_id = run_id
        self.basin_name = basin_name
        self.config_path = config_path
        self.features_json = features_json
        self.ui_callback = ui_callback
        self.db_path = db_path or DEFAULT_DB_PATH
        self.experiment_logger = experiment_logger
        self.test_loader = test_loader
        self.status_path = Path(status_path) if status_path else None
        self.test_loss: Optional[float] = None
        self.test_nse: Optional[float] = None
        self.test_kge: Optional[float] = None

        self.feature_cols: List[str] = list(feature_cols or [])
        self.feature_groups: Dict[str, List[int]] = feature_groups or {
            "rain": [],
            "discharge": [],
            "terrain": [],
            "coord": [],
            "time": [],
        }
        self.scaler = scaler if scaler is not None else IdentityScaler()

        tr = config.get("training", {})
        self.epochs = int(tr.get("max_epochs", 50))
        self.lr = float(tr.get("learning_rate", 1e-4))
        self.patience = int(tr.get("early_stopping_patience", 7))
        self.lr_patience = int(tr.get("lr_scheduler", {}).get("patience", 5))
        self.lr_factor = float(tr.get("lr_scheduler", {}).get("factor", 0.5))
        self.lr_min = float(tr.get("lr_scheduler", {}).get("min_lr", 1e-6))

        model_cfg = config.get("model", {})
        s2 = model_cfg.get("stage2_regression", {})
        self.hidden_size = int(s2.get("hidden_size", 128))
        self.num_layers = int(s2.get("num_layers", 2))
        self.dropout = float(s2.get("dropout", 0.2))
        self.seq_len = int(model_cfg.get("seq_len", 24))
        self.embed_dim = int(model_cfg.get("embed_dim", 32))
        self.batch_size = int(tr.get("batch_size", 32))

        self.optimizer = Adam(self.model.parameters(), lr=self.lr)
        self.scheduler = ReduceLROnPlateau(
            self.optimizer,
            mode="min",
            factor=self.lr_factor,
            patience=self.lr_patience,
            min_lr=self.lr_min,
        )
        self.use_weighted_mse = bool(
            config.get("training", {}).get("use_weighted_mse_loss", True)
        )
        self.beta_weight = float(config.get("training", {}).get("beta_weight", 15.0))
        self.global_qmax = float(config.get("training", {}).get("global_qmax", 60.0))
        # Stage 2 targets from dataloader are log1p(discharge) when use_weighted_mse; model output is log Q
        self.criterion = nn.MSELoss()

    def _write_status(
        self,
        status: str,
        epoch: int,
        train_loss: float,
        val_loss: float,
        val_nse: Optional[float],
        val_kge: Optional[float],
        history: List[Dict[str, Any]],
    ) -> None:
        if self.status_path is None:
            return
        obj = {
            "status": status,
            "stage": "stage2",
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_nse": val_nse,
            "val_kge": val_kge,
            "history": history,
        }
        try:
            with self.status_path.open("w", encoding="utf-8") as f:
                json.dump(obj, f, indent=2)
        except Exception:
            pass

    def run(self) -> Path:
        checkpoint_dir = Path(self.checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        best_val_loss = float("inf")
        best_val_nse: Optional[float] = None
        best_val_kge: Optional[float] = None
        best_state: Optional[Dict[str, Any]] = None
        patience_counter = 0
        history: List[Dict[str, Any]] = []
        self._write_status("running", 0, 0.0, 0.0, None, None, history)

        for epoch in range(1, self.epochs + 1):
            self.model.train()
            train_losses: List[float] = []
            for batch in self.train_loader:
                X = batch[0].to(self.device)
                y_log = batch[1].to(self.device).float()  # already log1p(discharge)
                self.optimizer.zero_grad()
                out = self.model(X)  # [B, 1] log Q
                log_pred = out.squeeze(-1)
                if self.use_weighted_mse:
                    loss = weighted_mse_loss(
                        log_pred, y_log, beta=self.beta_weight, global_qmax=self.global_qmax
                    )
                else:
                    loss = self.criterion(out, y_log.unsqueeze(1))
                loss.backward()
                self.optimizer.step()
                train_losses.append(loss.item())

            train_loss = np.mean(train_losses)
            self.model.eval()
            val_losses: List[float] = []
            all_obs: List[float] = []
            all_sim: List[float] = []
            with torch.no_grad():
                for batch in self.val_loader:
                    X = batch[0].to(self.device)
                    y_log = batch[1].to(self.device).float()
                    out = self.model(X)
                    log_pred = out.squeeze(-1)
                    if self.use_weighted_mse:
                        loss = weighted_mse_loss(
                            log_pred, y_log, beta=self.beta_weight, global_qmax=self.global_qmax
                        )
                    else:
                        loss = self.criterion(out, y_log.unsqueeze(1))
                    val_losses.append(loss.item())
                    # NSE/KGE in linear space (notebook: q_true = expm1(y_log))
                    all_obs.extend(np.expm1(y_log.cpu().numpy()).ravel().tolist())
                    all_sim.extend(np.expm1(out.cpu().numpy()).ravel().tolist())

            val_loss = np.mean(val_losses)
            val_nse = nse(np.array(all_obs), np.array(all_sim))
            val_kge = kge(np.array(all_obs), np.array(all_sim))
            if np.isnan(val_nse):
                val_nse = None
            if np.isnan(val_kge):
                val_kge = None
            self.scheduler.step(val_loss)

            history.append({
                "stage": "stage2",
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_nse": val_nse,
                "val_kge": val_kge,
            })
            self._write_status("running", epoch, train_loss, val_loss, val_nse, val_kge, history)

            if self.ui_callback is not None:
                self.ui_callback(epoch, train_loss, val_loss, val_nse, val_kge)
            if self.experiment_logger is not None:
                m: Dict[str, Any] = {
                    "stage": "stage2",
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                }
                if val_nse is not None:
                    m["val_nse"] = val_nse
                if val_kge is not None:
                    m["val_kge"] = val_kge
                self.experiment_logger.log_metrics(epoch, m)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_val_nse = val_nse
                best_val_kge = val_kge
                best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1
            if patience_counter >= self.patience:
                if best_state is not None:
                    self.model.load_state_dict(best_state)
                break

        self.model.eval()
        if self.test_loader is not None:
            test_losses: List[float] = []
            all_obs_te: List[float] = []
            all_sim_te: List[float] = []
            with torch.no_grad():
                for batch in self.test_loader:
                    X = batch[0].to(self.device)
                    y_log = batch[1].to(self.device).float()
                    out = self.model(X)
                    log_pred = out.squeeze(-1)
                    if self.use_weighted_mse:
                        loss = weighted_mse_loss(
                            log_pred, y_log, beta=self.beta_weight, global_qmax=self.global_qmax
                        )
                    else:
                        loss = self.criterion(out, y_log.unsqueeze(1))
                    test_losses.append(loss.item())
                    all_obs_te.extend(np.expm1(y_log.cpu().numpy()).ravel().tolist())
                    all_sim_te.extend(np.expm1(out.cpu().numpy()).ravel().tolist())
            self.test_loss = float(np.mean(test_losses))
            self.test_nse = nse(np.array(all_obs_te), np.array(all_sim_te))
            self.test_kge = kge(np.array(all_obs_te), np.array(all_sim_te))
            if np.isnan(self.test_nse):
                self.test_nse = None
            if np.isnan(self.test_kge):
                self.test_kge = None
            if self.experiment_logger is not None:
                m = {"stage": "stage2", "test_loss": self.test_loss}
                if self.test_nse is not None:
                    m["test_nse"] = self.test_nse
                if self.test_kge is not None:
                    m["test_kge"] = self.test_kge
                self.experiment_logger.log_metrics(0, m)
        if self.status_path is not None:
            self._write_status("done", epoch, train_loss, val_loss, val_nse, val_kge, history)

        ckpt_path = checkpoint_dir / "improved_lstm_model.pt"
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "scaler": self.scaler,
                "feature_cols": self.feature_cols,
                "rain_indices": self.feature_groups.get("rain", []),
                "discharge_indices": self.feature_groups.get("discharge", []),
                "terrain_indices": self.feature_groups.get("terrain", []),
                "coord_indices": self.feature_groups.get("coord", []),
                "time_indices": self.feature_groups.get("time", []),
                "model_params": {
                    "hidden_size": self.hidden_size,
                    "num_layers": self.num_layers,
                    "dropout": self.dropout,
                    "embed_dim": self.embed_dim,
                    "seq_len": self.seq_len,
                },
            },
            ckpt_path,
        )
        record = ExperimentRecord(
            run_id=self.run_id,
            basin=self.basin_name,
            stage="stage2",
            config_path=self.config_path,
            seq_len=self.seq_len,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=self.dropout,
            batch_size=self.batch_size,
            learning_rate=self.lr,
            epochs=epoch,
            features_json=self.features_json,
            val_loss=best_val_loss,
            val_nse=best_val_nse,
            val_kge=best_val_kge,
            checkpoint_path=str(ckpt_path),
        )
        log_experiment(record, db_path=self.db_path)
        return ckpt_path
