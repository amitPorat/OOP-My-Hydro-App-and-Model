"""
Experiment tracking for Model Playground. Every training run is saved under
basin_folder/output/experiments/{timestamp}_{experiment_name}/ with config,
metrics, and (later) model weights.
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


def _sanitize_run_name(name: str) -> str:
    """Safe directory name: alphanumeric, underscores, hyphens only."""
    return re.sub(r"[^\w\-]", "_", name or "unnamed").strip("_") or "unnamed"


class ExperimentLogger:
    """
    Creates and manages a single experiment run directory. Use for strict
    experiment tracking: config, per-epoch metrics, and (later) checkpoints.
    """

    def __init__(self, basin_folder: Path, experiment_name: str) -> None:
        """
        Create a new run directory under basin_folder/output/experiments/.

        Parameters
        ----------
        basin_folder : Path
            Basin root (e.g. /path/to/Darga_28_for_test_only).
        experiment_name : str
            User-facing name; will be sanitized and prefixed with timestamp.
        """
        basin_folder = Path(basin_folder).expanduser()
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        safe_name = _sanitize_run_name(experiment_name)
        self.run_dir = (
            basin_folder / "output" / "experiments" / f"{ts}_{safe_name}"
        )
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self._metrics_log: List[Dict[str, Any]] = []

    def save_config(self, config: Dict[str, Any]) -> Path:
        """
        Save full run configuration to config.json in the run directory.

        Parameters
        ----------
        config : dict
            All hyperparameters, dataset splits, model choice, notes, etc.

        Returns
        -------
        Path
            Path to the written config.json.
        """
        path = self.run_dir / "config.json"
        with path.open("w", encoding="utf-8") as f:
            json.dump(config, f, indent=2, sort_keys=False)
        return path

    def log_metrics(self, epoch: int, metrics: Dict[str, Any]) -> None:
        """
        Append one epoch's metrics (e.g. train_loss, val_loss). Stored in
        memory and written to metrics.json on each call so leaderboard can
        read final validation loss.

        Parameters
        ----------
        epoch : int
            Epoch index (1-based or 0-based, your choice).
        metrics : dict
            e.g. {"train_loss": 0.5, "val_loss": 0.6, "val_nse": 0.7}.
        """
        record = {"epoch": epoch, **metrics}
        self._metrics_log.append(record)
        path = self.run_dir / "metrics.json"
        with path.open("w", encoding="utf-8") as f:
            json.dump(self._metrics_log, f, indent=2, sort_keys=False)

    @property
    def run_id(self) -> str:
        """Unique run identifier (directory name)."""
        return self.run_dir.name


def list_experiments(basin_folder: Path) -> List[Path]:
    """
    List all experiment run directories (each contains config.json and
    optionally metrics.json), sorted by creation time descending.

    Returns
    -------
    list of Path
        Paths to run directories.
    """
    exp_root = Path(basin_folder).expanduser() / "output" / "experiments"
    if not exp_root.is_dir():
        return []
    dirs = [p for p in exp_root.iterdir() if p.is_dir()]
    dirs.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return dirs


def load_run_config(run_dir: Path) -> Optional[Dict[str, Any]]:
    """Load config.json from a run directory. Returns None if missing or invalid."""
    path = run_dir / "config.json"
    if not path.is_file():
        return None
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def load_run_metrics(run_dir: Path) -> List[Dict[str, Any]]:
    """Load metrics.json from a run directory. Returns list of per-epoch records."""
    path = run_dir / "metrics.json"
    if not path.is_file():
        return []
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, list) else []
    except Exception:
        return []


def get_final_val_loss(run_dir: Path) -> Optional[float]:
    """Last epoch's val_loss from metrics.json, or None."""
    metrics = load_run_metrics(run_dir)
    if not metrics:
        return None
    last = metrics[-1]
    return last.get("val_loss") if isinstance(last.get("val_loss"), (int, float)) else None
