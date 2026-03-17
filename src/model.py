from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import torch
from torch import Tensor, nn

# Notebook-exact: weighted MSE in log-space (log_q_pred, log_q_true).
BETA_WEIGHT = 15.0
GLOBAL_QMAX = 60.0


def weighted_mse_loss(
    log_q_pred: Tensor,
    log_q_true: Tensor,
    beta: float = BETA_WEIGHT,
    global_qmax: float = GLOBAL_QMAX,
) -> Tensor:
    """
    Weighted MSE in log-space. Notebook-exact from MY_LAST_JUPYTER.ipynb:
    - Clamp log values to [-10, 10]; Q_true = expm1(log_q_true) clamped to [0, 1000].
    - weight = 1 + beta * (Q_true / global_qmax), clamped to [1, 100].
    - loss = mean(weight * (log_q_pred - log_q_true)^2).
    """
    log_q_pred = torch.clamp(log_q_pred, min=-10.0, max=10.0)
    log_q_true = torch.clamp(log_q_true, min=-10.0, max=10.0)
    q_true = torch.expm1(log_q_true)
    q_true = torch.clamp(q_true, min=0.0, max=1000.0)
    w = 1.0 + beta * (q_true / (global_qmax + 1e-6))
    w = torch.clamp(w, min=1.0, max=100.0)
    diff = (log_q_pred - log_q_true) ** 2
    loss = torch.mean(w * diff)
    if torch.isnan(loss):
        return torch.tensor(0.0, device=log_q_pred.device, requires_grad=True)
    return loss


@dataclass
class StageModelConfig:
    """
    Configuration for a single LSTM stage.

    This mirrors the hyperparameters used in the notebooks to ensure
    that checkpoints can be loaded without shape mismatches.
    """

    hidden_size: int
    num_layers: int
    dropout: float
    embed_dim: int


class IdentityScaler:
    """
    Simple identity scaler used for notebook-compatible checkpoints.

    The original notebooks stored a scikit-learn scaler object alongside
    the model weights. For the current modular implementation we do not
    apply any scaling during training, but FloodPredictor still expects
    a ``scaler`` with a ``transform`` method. This lightweight class
    fulfils that contract and keeps data unchanged.
    """

    def fit(self, X):  # type: ignore[no-untyped-def]
        return self

    def transform(self, X):  # type: ignore[no-untyped-def]
        return X

    def inverse_transform(self, X):  # type: ignore[no-untyped-def]
        return X


class BinaryClassifierLSTM(nn.Module):
    """
    Binary classifier LSTM with embeddings (Stage 1).

    This implementation is a direct modularization of the notebook
    architecture used for the ``TWO_STAGE_STAGE1_HYDROGRAPH`` model.
    """

    def __init__(
        self,
        rain_indices: Sequence[int],
        terrain_indices: Sequence[int],
        coord_indices: Sequence[int],
        time_indices: Sequence[int],
        feature_cols: Sequence[str],
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.2,
        embed_dim: int = 32,
    ) -> None:
        super().__init__()
        self.rain_indices = list(rain_indices)
        self.terrain_indices = list(terrain_indices)
        self.coord_indices = list(coord_indices)
        self.time_indices = list(time_indices)

        if len(self.rain_indices) > 0:
            self.rain_embedding = nn.Sequential(
                nn.Linear(len(self.rain_indices), embed_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(embed_dim, embed_dim),
            )
        else:
            self.rain_embedding = None

        if len(self.terrain_indices) > 0:
            self.terrain_embedding = nn.Sequential(
                nn.Linear(len(self.terrain_indices), embed_dim * 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(embed_dim * 2, embed_dim),
            )
        else:
            self.terrain_embedding = None

        if len(self.coord_indices) > 0:
            self.coord_embedding = nn.Sequential(
                nn.Linear(len(self.coord_indices), embed_dim // 4),
                nn.ReLU(),
                nn.Linear(embed_dim // 4, embed_dim // 4),
            )
        else:
            self.coord_embedding = None

        if len(self.time_indices) > 0:
            self.time_embedding = nn.Sequential(
                nn.Linear(len(self.time_indices), embed_dim // 4),
                nn.ReLU(),
                nn.Linear(embed_dim // 4, embed_dim // 4),
            )
        else:
            self.time_embedding = None

        total_embed_dim = 0
        if self.rain_embedding is not None:
            total_embed_dim += embed_dim
        if self.terrain_embedding is not None:
            total_embed_dim += embed_dim
        if self.coord_embedding is not None:
            total_embed_dim += embed_dim // 4
        if self.time_embedding is not None:
            total_embed_dim += embed_dim // 4

        if total_embed_dim == 0:
            total_embed_dim = len(feature_cols)

        self.lstm = nn.LSTM(
            input_size=total_embed_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
        )
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass.

        Parameters
        ----------
        x : Tensor
            Input batch of shape ``[batch, seq_len, features]``.

        Returns
        -------
        Tensor
            Logits of shape ``[batch, 1]``.
        """
        embeddings: List[Tensor] = []
        if self.rain_embedding is not None and self.rain_indices:
            embeddings.append(self.rain_embedding(x[:, :, self.rain_indices]))
        if self.terrain_embedding is not None and self.terrain_indices:
            embeddings.append(self.terrain_embedding(x[:, :, self.terrain_indices]))
        if self.coord_embedding is not None and self.coord_indices:
            embeddings.append(self.coord_embedding(x[:, :, self.coord_indices]))
        if self.time_embedding is not None and self.time_indices:
            embeddings.append(self.time_embedding(x[:, :, self.time_indices]))

        x_embedded = torch.cat(embeddings, dim=-1) if embeddings else x
        lstm_out, _ = self.lstm(x_embedded)
        return self.fc(lstm_out[:, -1, :])


class ImprovedLSTMWithEmbeddings(nn.Module):
    """
    Improved LSTM with embeddings (Stage 2).

    This architecture matches the ``LSTM_IMPROVED`` model used in the
    notebooks. In particular, the discharge embedding dimension is
    reduced to ``embed_dim // 4`` to lessen dependence on lagged
    discharge while keeping the rest of the structure intact.
    """

    def __init__(
        self,
        rain_indices: Sequence[int],
        discharge_indices: Sequence[int],
        terrain_indices: Sequence[int],
        coord_indices: Sequence[int],
        time_indices: Sequence[int],
        feature_cols: Sequence[str],
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.2,
        embed_dim: int = 32,
    ) -> None:
        super().__init__()
        self.rain_indices = list(rain_indices)
        self.discharge_indices = list(discharge_indices)
        self.terrain_indices = list(terrain_indices)
        self.coord_indices = list(coord_indices)
        self.time_indices = list(time_indices)

        if len(self.rain_indices) > 0:
            self.rain_embedding = nn.Sequential(
                nn.Linear(len(self.rain_indices), embed_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(embed_dim, embed_dim),
            )
        else:
            self.rain_embedding = None

        if len(self.discharge_indices) > 0:
            self.discharge_embedding = nn.Sequential(
                nn.Linear(len(self.discharge_indices), embed_dim // 4),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(embed_dim // 4, embed_dim // 4),
            )
        else:
            self.discharge_embedding = None

        if len(self.terrain_indices) > 0:
            self.terrain_embedding = nn.Sequential(
                nn.Linear(len(self.terrain_indices), embed_dim * 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(embed_dim * 2, embed_dim),
            )
        else:
            self.terrain_embedding = None

        if len(self.coord_indices) > 0:
            self.coord_embedding = nn.Sequential(
                nn.Linear(len(self.coord_indices), embed_dim // 4),
                nn.ReLU(),
                nn.Linear(embed_dim // 4, embed_dim // 4),
            )
        else:
            self.coord_embedding = None

        if len(self.time_indices) > 0:
            self.time_embedding = nn.Sequential(
                nn.Linear(len(self.time_indices), embed_dim // 4),
                nn.ReLU(),
                nn.Linear(embed_dim // 4, embed_dim // 4),
            )
        else:
            self.time_embedding = None

        total_embed_dim = 0
        if self.rain_embedding is not None:
            total_embed_dim += embed_dim
        if self.discharge_embedding is not None:
            total_embed_dim += embed_dim // 4
        if self.terrain_embedding is not None:
            total_embed_dim += embed_dim
        if self.coord_embedding is not None:
            total_embed_dim += embed_dim // 4
        if self.time_embedding is not None:
            total_embed_dim += embed_dim // 4

        if total_embed_dim == 0:
            total_embed_dim = len(feature_cols)

        self.lstm = nn.LSTM(
            input_size=total_embed_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
        )
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass.

        Parameters
        ----------
        x : Tensor
            Input batch of shape ``[batch, seq_len, features]``.

        Returns
        -------
        Tensor
            Predicted discharge of shape ``[batch, 1]``.
        """
        embeddings: List[Tensor] = []
        if self.rain_embedding is not None and self.rain_indices:
            embeddings.append(self.rain_embedding(x[:, :, self.rain_indices]))
        if self.discharge_embedding is not None and self.discharge_indices:
            embeddings.append(self.discharge_embedding(x[:, :, self.discharge_indices]))
        if self.terrain_embedding is not None and self.terrain_indices:
            embeddings.append(self.terrain_embedding(x[:, :, self.terrain_indices]))
        if self.coord_embedding is not None and self.coord_indices:
            embeddings.append(self.coord_embedding(x[:, :, self.coord_indices]))
        if self.time_embedding is not None and self.time_indices:
            embeddings.append(self.time_embedding(x[:, :, self.time_indices]))

        x_embedded = torch.cat(embeddings, dim=-1) if embeddings else x
        lstm_out, _ = self.lstm(x_embedded)
        return self.fc(lstm_out[:, -1, :])


def build_stage_models_from_checkpoints(
    checkpoint_s1: Dict[str, object],
    checkpoint_s2: Dict[str, object],
    device: torch.device,
) -> Tuple[BinaryClassifierLSTM, ImprovedLSTMWithEmbeddings, Dict[str, object]]:
    """
    Reconstruct Stage 1 and Stage 2 models from saved checkpoints.

    Parameters
    ----------
    checkpoint_s1 : Dict[str, object]
        Stage 1 checkpoint dictionary as stored in the training
        notebook (must contain ``'model_state_dict'``,
        ``'feature_cols'`` and ``'model_params'``).
    checkpoint_s2 : Dict[str, object]
        Stage 2 checkpoint dictionary with analogous structure.
    device : torch.device
        Device on which to place the reconstructed models.

    Returns
    -------
    BinaryClassifierLSTM
        Reconstructed Stage 1 model in evaluation mode.
    ImprovedLSTMWithEmbeddings
        Reconstructed Stage 2 model in evaluation mode.
    Dict[str, object]
        Dictionary with auxiliary artefacts such as scalers and
        feature column lists.
    """
    params_s1 = checkpoint_s1.get("model_params", {})
    params_s2 = checkpoint_s2.get("model_params", {})

    feature_cols_s1: List[str] = list(checkpoint_s1["feature_cols"])
    feature_cols_s2: List[str] = list(checkpoint_s2["feature_cols"])

    # Indices are stored at the top level of the checkpoint dictionaries
    # in the training notebooks, not inside ``model_params``.
    rain_indices_s1 = checkpoint_s1.get("rain_indices", params_s1.get("rain_indices", []))
    terrain_indices_s1 = checkpoint_s1.get("terrain_indices", params_s1.get("terrain_indices", []))
    coord_indices_s1 = checkpoint_s1.get("coord_indices", params_s1.get("coord_indices", []))
    time_indices_s1 = checkpoint_s1.get("time_indices", params_s1.get("time_indices", []))

    rain_indices_s2 = checkpoint_s2.get("rain_indices", params_s2.get("rain_indices", []))
    discharge_indices_s2 = checkpoint_s2.get(
        "discharge_indices", params_s2.get("discharge_indices", [])
    )
    terrain_indices_s2 = checkpoint_s2.get("terrain_indices", params_s2.get("terrain_indices", []))
    coord_indices_s2 = checkpoint_s2.get("coord_indices", params_s2.get("coord_indices", []))
    time_indices_s2 = checkpoint_s2.get("time_indices", params_s2.get("time_indices", []))

    model_s1 = BinaryClassifierLSTM(
        rain_indices=rain_indices_s1,
        terrain_indices=terrain_indices_s1,
        coord_indices=coord_indices_s1,
        time_indices=time_indices_s1,
        feature_cols=feature_cols_s1,
        hidden_size=int(params_s1.get("hidden_size", 128)),
        num_layers=int(params_s1.get("num_layers", 2)),
        dropout=float(params_s1.get("dropout", 0.2)),
        embed_dim=int(params_s1.get("embed_dim", 32)),
    ).to(device)

    model_s2 = ImprovedLSTMWithEmbeddings(
        rain_indices=rain_indices_s2,
        discharge_indices=discharge_indices_s2,
        terrain_indices=terrain_indices_s2,
        coord_indices=coord_indices_s2,
        time_indices=time_indices_s2,
        feature_cols=feature_cols_s2,
        hidden_size=int(params_s2.get("hidden_size", 128)),
        num_layers=int(params_s2.get("num_layers", 2)),
        dropout=float(params_s2.get("dropout", 0.2)),
        embed_dim=int(params_s2.get("embed_dim", 32)),
    ).to(device)

    model_s1.load_state_dict(checkpoint_s1["model_state_dict"])  # type: ignore[arg-type]
    model_s2.load_state_dict(checkpoint_s2["model_state_dict"])  # type: ignore[arg-type]

    model_s1.eval()
    model_s2.eval()

    artefacts: Dict[str, object] = {
        "scaler_s1": checkpoint_s1.get("scaler"),
        "scaler_s2": checkpoint_s2.get("scaler"),
        "feature_cols_s1": feature_cols_s1,
        "feature_cols_s2": feature_cols_s2,
    }
    return model_s1, model_s2, artefacts

