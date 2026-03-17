from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Tuple

import numpy as np
import pandas as pd
import torch
from torch import Tensor

from .data_loader import WeatherDataLoader
from .features import UgridBuilder, add_engineered_features
from .model import build_stage_models_from_checkpoints


@dataclass
class InferenceConfig:
    """Typed subset of configuration parameters relevant for inference."""

    discharge_threshold: float
    rain_context_hours: int
    min_event_duration: int
    binary_threshold: float
    use_probabilistic_combination: bool
    seq_len: int


class FloodPredictor:
    """
    End-to-end two-stage LSTM flood prediction pipeline.

    This class replicates the operational logic from the notebooks:

    - Load ICON ensemble forecasts and UGRID static features.
    - Reconstruct Stage 1 (binary) and Stage 2 (regression) LSTM models
      from checkpoints (ensuring identical architecture).
    - Build rolling sequences of length ``SEQ_LEN`` for each station
      cell and timestep.
    - Apply the iterative prediction scheme where predicted discharge
      is fed back as lagged input for subsequent steps.
    """

    def __init__(self, config: Mapping[str, Any], config_path: Path | None = None) -> None:
        """
        Initialize the flood predictor.

        Parameters
        ----------
        config : Mapping[str, Any]
            Parsed configuration dictionary from ``system_config.yaml``.
        config_path : Path, optional
            Path to the configuration file (for logging / bookkeeping
            only).
        """
        self._config = config
        self._config_path = config_path

        model_cfg = config.get("model", {})
        if not isinstance(model_cfg, dict):
            raise ValueError("Configuration key 'model' must be a mapping.")

        infer_cfg = config.get("inference", {})
        if not isinstance(infer_cfg, dict):
            raise ValueError("Configuration key 'inference' must be a mapping.")

        self._inf_cfg = InferenceConfig(
            discharge_threshold=float(infer_cfg.get("discharge_threshold_m3s", 0.01)),
            rain_context_hours=int(infer_cfg.get("rain_context_hours", 12)),
            min_event_duration=int(infer_cfg.get("min_event_duration_timesteps", 6)),
            binary_threshold=float(infer_cfg.get("binary_probability_threshold", 0.5)),
            use_probabilistic_combination=bool(
                infer_cfg.get("use_probabilistic_combination", True)
            ),
            seq_len=int(model_cfg.get("seq_len", 24)),
        )

        data_paths = config.get("data_paths", {})
        if not isinstance(data_paths, dict):
            raise ValueError("Configuration key 'data_paths' must be a mapping.")

        checkpoints = model_cfg.get("checkpoints", {})
        if not isinstance(checkpoints, dict):
            raise ValueError("Configuration key 'model.checkpoints' must be a mapping.")

        self._stage1_path = Path(str(checkpoints.get("stage1_path", ""))).expanduser()
        self._stage2_path = Path(str(checkpoints.get("stage2_path", ""))).expanduser()

        terrain_features = config.get("features", {}).get("terrain_features", [])
        self._terrain_features: List[str] = list(terrain_features)

        self._loader = WeatherDataLoader(config)
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load UGRID once and initialise the feature builder
        ugrid_df = self._loader.load_ugrid()
        self._ugrid_builder = UgridBuilder(ugrid_df, terrain_features=self._terrain_features)

        # Load checkpoints and reconstruct models + scalers
        checkpoint_s1 = torch.load(self._stage1_path, map_location=self._device)
        checkpoint_s2 = torch.load(self._stage2_path, map_location=self._device)
        (
            self._model_s1,
            self._model_s2,
            artefacts,
        ) = build_stage_models_from_checkpoints(checkpoint_s1, checkpoint_s2, self._device)

        self._scaler_s1 = artefacts["scaler_s1"]
        self._scaler_s2 = artefacts["scaler_s2"]
        self._feature_cols_s1: List[str] = artefacts["feature_cols_s1"]  # type: ignore[assignment]
        self._feature_cols_s2: List[str] = artefacts["feature_cols_s2"]  # type: ignore[assignment]

        stations_cfg = config.get("stations", {})
        if not isinstance(stations_cfg, dict):
            raise ValueError("Configuration key 'stations' must be a mapping.")
        self._station_cells: List[int] = list(stations_cfg.get("station_cells", []))

    @property
    def device(self) -> torch.device:
        """Return the torch device used for inference."""
        return self._device

    def _prepare_member_dataframe(self, member_df: pd.DataFrame) -> pd.DataFrame:
        """
        Attach UGRID features and initialise discharge column.

        Parameters
        ----------
        member_df : pandas.DataFrame
            ICON member data frame with columns including ``['ID', 'time', 'rainrate']``.

        Returns
        -------
        pandas.DataFrame
            Data frame with static terrain attributes and an empty
            ``'discharge'`` column ready for feature engineering.
        """
        data_df = self._ugrid_builder.attach_static_features(member_df)
        if "discharge" not in data_df.columns:
            data_df["discharge"] = np.nan

        # Mirror the notebook feature engineering applied before
        # sequence creation. We include lagged discharge features so
        # that Stage 2's feature space matches the training checkpoint.
        data_df = add_engineered_features(
            data_df, include_lagged_discharge=True
        )
        return data_df

    def _create_sequences_iterative(
        self,
        df: pd.DataFrame,
        feature_cols: List[str],
    ) -> Tuple[np.ndarray, List[pd.Timestamp], List[int]]:
        """
        Create rolling sequences for iterative prediction, mirroring the notebook.

        Parameters
        ----------
        df : pandas.DataFrame
            Data frame containing at least ``'ID'`` and ``'time'``
            columns plus the specified feature columns.
        feature_cols : list of str
            Ordered list of features expected by the scaler and models.

        Returns
        -------
        numpy.ndarray
            Array of shape ``[n_sequences, seq_len, n_features]``.
        List[pandas.Timestamp]
            List of timestamps corresponding to the prediction target
            of each sequence.
        List[int]
            List of cell IDs corresponding to each sequence.
        """
        seq_len = self._inf_cfg.seq_len

        sequences: List[np.ndarray] = []
        times: List[pd.Timestamp] = []
        ids: List[int] = []

        df = df.sort_values(["ID", "time"]).reset_index(drop=True)

        for cell_id in self._station_cells:
            cell_data = df[df["ID"] == cell_id].copy()
            cell_data = cell_data.sort_values("time").reset_index(drop=True)

            if len(cell_data) < seq_len:
                continue

            for t_idx in range(seq_len - 1, len(cell_data)):
                seq_slice = cell_data.iloc[t_idx - seq_len + 1 : t_idx + 1]
                feat_values = seq_slice[feature_cols].to_numpy(dtype=float)
                sequences.append(feat_values)
                times.append(cell_data.loc[t_idx, "time"])
                ids.append(int(cell_id))

        if not sequences:
            return np.empty((0, seq_len, len(feature_cols))), [], []

        seq_array = np.stack(sequences, axis=0)
        return seq_array, times, ids

    def run_for_member(self, member_name: str, member_df: pd.DataFrame) -> pd.DataFrame:
        """
        Run the full two-stage prediction for a single ICON ensemble member.

        Parameters
        ----------
        member_name : str
            Human-readable member identifier (e.g. ``'2026011200 mem01'``).
        member_df : pandas.DataFrame
            ICON rain data for the member.

        Returns
        -------
        pandas.DataFrame
            Data frame with timestamps, cell IDs, probabilities,
            discharge predictions, and combined final discharge.
        """
        data_df = self._prepare_member_dataframe(member_df)

        # Stage 1 + Stage 2 feature matrices (scaled using training scalers)
        seq_s1, times_s1, ids_s1 = self._create_sequences_iterative(
            data_df, self._feature_cols_s1
        )
        seq_s2, times_s2, ids_s2 = self._create_sequences_iterative(
            data_df, self._feature_cols_s2
        )

        if seq_s1.shape[0] == 0 or seq_s2.shape[0] == 0:
            return pd.DataFrame(
                columns=[
                    "member",
                    "ID",
                    "time",
                    "prob_event",
                    "q_stage2",
                    "q_final",
                ]
            )

        # Scale features as in training
        n_features_s1 = seq_s1.shape[-1]
        n_features_s2 = seq_s2.shape[-1]
        scaled_s1 = self._scaler_s1.transform(
            seq_s1.reshape(-1, n_features_s1)
        ).reshape(seq_s1.shape[0], self._inf_cfg.seq_len, n_features_s1)
        scaled_s2 = self._scaler_s2.transform(
            seq_s2.reshape(-1, n_features_s2)
        ).reshape(seq_s2.shape[0], self._inf_cfg.seq_len, n_features_s2)

        x_s1 = torch.from_numpy(scaled_s1).float().to(self._device)
        x_s2 = torch.from_numpy(scaled_s2).float().to(self._device)

        with torch.no_grad():
            logits = self._model_s1(x_s1)
            prob_event = torch.sigmoid(logits).cpu().numpy().reshape(-1)
            q_stage2 = self._model_s2(x_s2).cpu().numpy().reshape(-1)

        if self._inf_cfg.use_probabilistic_combination:
            q_final = prob_event * q_stage2
        else:
            mask = prob_event >= self._inf_cfg.binary_threshold
            q_final = np.where(mask, q_stage2, 0.0)

        result = pd.DataFrame(
            {
                "member": member_name,
                "ID": ids_s1,
                "time": times_s1,
                "prob_event": prob_event,
                "q_stage2": q_stage2,
                "q_final": q_final,
            }
        )
        return result

    def run_all_members(self) -> pd.DataFrame:
        """
        Run the prediction pipeline for all available ICON ensemble members.

        Returns
        -------
        pandas.DataFrame
            Concatenated predictions for all members, sorted by member,
            station cell and time.
        """
        members = self._loader.load_all_icon_members()
        outputs: List[pd.DataFrame] = []

        for name, df in members.items():
            outputs.append(self.run_for_member(name, df))

        if not outputs:
            return pd.DataFrame()

        result = pd.concat(outputs, ignore_index=True)
        result = result.sort_values(["member", "ID", "time"]).reset_index(drop=True)
        return result

