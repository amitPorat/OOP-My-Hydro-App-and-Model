from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional


DEFAULT_DB_PATH = Path("experiments") / "experiments.db"


@dataclass
class ExperimentRecord:
    run_id: str
    basin: str
    stage: str
    config_path: str
    seq_len: int
    hidden_size: int
    num_layers: int
    dropout: float
    batch_size: int
    learning_rate: float
    epochs: int
    features_json: str
    val_loss: float
    val_nse: Optional[float]
    val_kge: Optional[float]
    checkpoint_path: str


def _get_conn(db_path: Path) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS experiments (
            run_id TEXT PRIMARY KEY,
            basin TEXT,
            stage TEXT,
            config_path TEXT,
            seq_len INTEGER,
            hidden_size INTEGER,
            num_layers INTEGER,
            dropout REAL,
            batch_size INTEGER,
            learning_rate REAL,
            epochs INTEGER,
            features_json TEXT,
            val_loss REAL,
            val_nse REAL,
            val_kge REAL,
            checkpoint_path TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """
    )
    return conn


def log_experiment(record: ExperimentRecord, db_path: Path = DEFAULT_DB_PATH) -> None:
    """
    Insert a completed experiment record into the tracking database.
    """
    conn = _get_conn(db_path)
    with conn:
        conn.execute(
            """
            INSERT OR REPLACE INTO experiments (
                run_id, basin, stage, config_path,
                seq_len, hidden_size, num_layers, dropout,
                batch_size, learning_rate, epochs, features_json,
                val_loss, val_nse, val_kge, checkpoint_path
            )
            VALUES (
                :run_id, :basin, :stage, :config_path,
                :seq_len, :hidden_size, :num_layers, :dropout,
                :batch_size, :learning_rate, :epochs, :features_json,
                :val_loss, :val_nse, :val_kge, :checkpoint_path
            )
            """,
            asdict(record),
        )
    conn.close()


def get_experiments_for_basin(
    basin: str, db_path: Path = DEFAULT_DB_PATH
) -> List[Dict[str, Any]]:
    """
    Fetch all experiments for a specific basin, newest first.
    """
    conn = _get_conn(db_path)
    cur = conn.cursor()
    cur.execute(
        """
        SELECT run_id, basin, stage, config_path,
               seq_len, hidden_size, num_layers, dropout,
               batch_size, learning_rate, epochs, features_json,
               val_loss, val_nse, val_kge, checkpoint_path, created_at
        FROM experiments
        WHERE basin = ?
        ORDER BY created_at DESC
        """,
        (basin,),
    )
    rows = cur.fetchall()
    conn.close()
    cols = [
        "run_id",
        "basin",
        "stage",
        "config_path",
        "seq_len",
        "hidden_size",
        "num_layers",
        "dropout",
        "batch_size",
        "learning_rate",
        "epochs",
        "features_json",
        "val_loss",
        "val_nse",
        "val_kge",
        "checkpoint_path",
        "created_at",
    ]
    return [dict(zip(cols, r)) for r in rows]


def get_best_experiment_for_basin(
    basin: str, db_path: Path = DEFAULT_DB_PATH
) -> Optional[Dict[str, Any]]:
    """
    Return the experiment with the highest NSE for a given basin.
    """
    conn = _get_conn(db_path)
    cur = conn.cursor()
    cur.execute(
        """
        SELECT run_id, basin, stage, config_path,
               seq_len, hidden_size, num_layers, dropout,
               batch_size, learning_rate, epochs, features_json,
               val_loss, val_nse, val_kge, checkpoint_path, created_at
        FROM experiments
        WHERE basin = ? AND val_nse IS NOT NULL
        ORDER BY val_nse DESC
        LIMIT 1
        """,
        (basin,),
    )
    row = cur.fetchone()
    conn.close()
    if row is None:
        return None
    cols = [
        "run_id",
        "basin",
        "stage",
        "config_path",
        "seq_len",
        "hidden_size",
        "num_layers",
        "dropout",
        "batch_size",
        "learning_rate",
        "epochs",
        "features_json",
        "val_loss",
        "val_nse",
        "val_kge",
        "checkpoint_path",
        "created_at",
    ]
    return dict(zip(cols, row))

