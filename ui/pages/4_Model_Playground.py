"""
Model Playground: command center for training ML models (LSTM, PINN, Curriculum)
on merged Rain–Discharge data with strict experiment tracking.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import streamlit as st
import yaml

from src.experiment_tracker import (
    ExperimentLogger,
    get_final_val_loss,
    list_experiments,
    load_run_config,
    load_run_metrics,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


BASINS_DIR = PROJECT_ROOT / "configs" / "basins"
SYSTEM_CONFIG = PROJECT_ROOT / "configs" / "system_config.yaml"


def _list_basin_configs() -> List[Path]:
    if not BASINS_DIR.is_dir():
        if SYSTEM_CONFIG.is_file():
            return [SYSTEM_CONFIG]
        return []
    return sorted(BASINS_DIR.glob("*.yaml"))


def _load_config(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _get_basin_folder(cfg: Dict[str, Any]) -> Optional[Path]:
    raw = cfg.get("data_paths", {}).get("basin_folder")
    if not raw:
        return None
    return Path(str(raw)).expanduser()


def _build_playground_config(
    experiment_name: str,
    experiment_notes: str,
    model_architecture: str,
    epochs: int,
    learning_rate: float,
    batch_size: int,
    seq_len: int,
    hidden_size_s1: int,
    hidden_size_s2: int,
    num_layers: int,
    dropout: float,
    embed_dim: int,
    beta_weight: float,
    global_qmax: float,
    early_stop_patience: int,
    lr_patience: int,
    lr_factor: float,
    lr_min: float,
    train_years: List[int],
    val_years: List[int],
    test_years: List[int],
    apply_forecast_augmentation: bool = False,
) -> Dict[str, Any]:
    """Build full config dict for ExperimentLogger and backend."""
    return {
        "experiment_name": experiment_name or "unnamed",
        "experiment_notes": experiment_notes,
        "model_architecture": model_architecture,
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
        "test_years": test_years,
        "apply_forecast_augmentation": apply_forecast_augmentation,
    }


def _get_available_merge_years(basin_folder: Path) -> List[int]:
    merge_dir = basin_folder / "output" / "rain_with_discharge"
    if not merge_dir.is_dir():
        return []
    years = []
    for p in merge_dir.glob("rain_with_discharge_*.parquet"):
        try:
            suffix = p.stem.replace("rain_with_discharge_", "")
            years.append(int(suffix))
        except ValueError:
            continue
    return sorted(years)


def _render_tab_config_and_training(basin_folder: Path) -> None:
    st.subheader("Configuration & Training")

    experiment_name = st.text_input(
        "Experiment Name",
        value="",
        placeholder="e.g. LSTM_baseline_epoch50",
    )
    experiment_notes = st.text_area(
        "Experiment Notes (Explain this run)",
        value="",
        placeholder="Describe this run: goal, assumptions, data split choice.",
        height=100,
    )

    model_architecture = st.selectbox(
        "Model Architecture",
        options=[
            "LSTM (Baseline)",
            "PINN (Future)",
            "Curriculum Learning (Future)",
        ],
    )

    st.markdown("**Training**")
    col_t1, col_t2, col_t3 = st.columns(3)
    with col_t1:
        epochs = st.number_input(
            "Epochs",
            min_value=1,
            max_value=500,
            value=50,
            step=1,
            help="Max training epochs (early stopping may stop earlier).",
        )
        batch_size = st.number_input(
            "Batch Size",
            min_value=1,
            max_value=512,
            value=32,
            step=1,
        )
    with col_t2:
        learning_rate = st.number_input(
            "Learning Rate",
            min_value=1e-6,
            max_value=1e-1,
            value=1e-4,
            step=1e-5,
            format="%.6f",
        )
        early_stop_patience = st.number_input(
            "Early stopping patience",
            min_value=1,
            max_value=50,
            value=7,
            step=1,
            help="Stop if no val loss improvement for this many epochs.",
        )
    with col_t3:
        lr_patience = st.number_input(
            "LR scheduler patience",
            min_value=1,
            max_value=20,
            value=5,
            step=1,
            help="Reduce LR after this many epochs without improvement.",
        )
        lr_factor = st.number_input(
            "LR decay factor",
            min_value=0.1,
            max_value=1.0,
            value=0.5,
            step=0.1,
            format="%.1f",
        )
        lr_min = st.number_input(
            "Min learning rate",
            min_value=1e-8,
            max_value=1e-3,
            value=1e-6,
            step=1e-7,
            format="%.0e",
        )

    st.markdown("**Architecture**")
    col_a1, col_a2, col_a3 = st.columns(3)
    with col_a1:
        seq_len = st.number_input(
            "Sequence length (SEQ_LEN)",
            min_value=8,
            max_value=256,
            value=24,
            step=4,
            help="Input sequence length (timesteps).",
        )
        hidden_size_s1 = st.number_input(
            "Hidden dim Stage 1 (binary)",
            min_value=16,
            max_value=1024,
            value=256,
            step=16,
            help="LSTM hidden size for Stage 1 classifier.",
        )
        hidden_size_s2 = st.number_input(
            "Hidden dim Stage 2 (regression)",
            min_value=16,
            max_value=1024,
            value=128,
            step=16,
            help="LSTM hidden size for Stage 2 regression.",
        )
    with col_a2:
        num_layers = st.number_input(
            "Number of LSTM layers",
            min_value=1,
            max_value=6,
            value=2,
            step=1,
        )
        dropout = st.slider(
            "Dropout rate",
            min_value=0.0,
            max_value=0.5,
            value=0.2,
            step=0.05,
        )
    with col_a3:
        embed_dim = st.number_input(
            "Embedding dimension",
            min_value=8,
            max_value=128,
            value=32,
            step=4,
            help="Embedding size for rain/terrain/coord/time groups.",
        )

    st.markdown("**Custom loss (Stage 2)**")
    col_l1, col_l2, _ = st.columns(3)
    with col_l1:
        beta_weight = st.number_input(
            "BETA_WEIGHT",
            min_value=0.0,
            max_value=50.0,
            value=15.0,
            step=0.5,
            format="%.1f",
            help="Weight scale for high-flow samples in weighted MSE.",
        )
    with col_l2:
        global_qmax = st.number_input(
            "GLOBAL_QMAX (m³/s)",
            min_value=1.0,
            max_value=500.0,
            value=60.0,
            step=5.0,
            help="Reference discharge for loss weighting.",
        )

    st.markdown("**Data augmentation (distribution shift)**")
    apply_forecast_augmentation = st.checkbox(
        "Simulate ICON forecast (data augmentation)",
        value=False,
        help="Smooth rain features after 9h to simulate RMCOMP → ICON transition in production.",
    )
    st.caption(
        "When enabled, the first 9 hours of each training sequence stay as radar (RMCOMP) data; "
        "the rest is smoothed with a rolling mean to mimic ICON’s spatial smoothing. "
        "Applied only to training data, not validation."
    )

    available_years = _get_available_merge_years(basin_folder)
    if not available_years:
        st.warning(
            "No merged parquet files found. Run **Rain–Discharge Merge** to produce "
            "`rain_with_discharge_{year}.parquet` in this basin's output folder."
        )
        train_years: List[int] = []
        val_years: List[int] = []
        test_years: List[int] = []
    else:
        st.markdown("**Train / Validation / Test split (by year)**")
        train_years = st.multiselect(
            "Training set years",
            options=available_years,
            default=available_years[:-2] if len(available_years) > 2 else (available_years[:-1] if len(available_years) > 1 else available_years),
            help="Select years used for training.",
        )
        val_years = st.multiselect(
            "Validation set years",
            options=available_years,
            default=[y for y in available_years if y not in train_years][:1] or (available_years[-1:] if available_years else []),
            help="Select years for validation (early stopping, LR scheduling).",
        )
        test_years = st.multiselect(
            "Hold-out years for Test set",
            options=available_years,
            default=[y for y in available_years if y not in train_years and y not in val_years],
            help="Unseen years for final evaluation. Metrics reported at end of training.",
        )
    st.caption("Ensure Train, Validation, and Test years do not overlap for a valid hold-out.")

    start_training = st.button("Start Training", type="primary")

    if start_training:
        if not experiment_name.strip():
            st.error("Please enter an experiment name.")
        elif available_years and (not train_years or not val_years):
            st.error("Select at least one year for training and one for validation.")
        elif model_architecture != "LSTM (Baseline)":
            logger = ExperimentLogger(basin_folder, experiment_name or "unnamed")
            config = _build_playground_config(
                experiment_name, experiment_notes, model_architecture,
                epochs, learning_rate, batch_size, seq_len,
                hidden_size_s1, hidden_size_s2, num_layers, dropout, embed_dim,
                beta_weight, global_qmax, early_stop_patience,
                lr_patience, lr_factor, lr_min, train_years, val_years, test_years,
                apply_forecast_augmentation,
            )
            logger.save_config(config)
            st.success(
                f"Experiment **{logger.run_id}** created. Config saved. "
                f"Training for **{model_architecture}** is not implemented yet."
            )
            st.info(f"Run directory: `{logger.run_dir}`")
        else:
            # LSTM (Baseline): run two-stage training in a thread; poll status for live UI
            from train import run_training_playground
            import threading
            import time
            import json

            logger = ExperimentLogger(basin_folder, experiment_name or "unnamed")
            config = _build_playground_config(
                experiment_name, experiment_notes, model_architecture,
                epochs, learning_rate, batch_size, seq_len,
                hidden_size_s1, hidden_size_s2, num_layers, dropout, embed_dim,
                beta_weight, global_qmax, early_stop_patience,
                lr_patience, lr_factor, lr_min, train_years, val_years, test_years,
                apply_forecast_augmentation,
            )
            logger.save_config(config)
            status_path = logger.run_dir / "training_status.json"

            progress_placeholder = st.empty()
            status_placeholder = st.empty()
            chart_placeholder = st.empty()
            result_placeholder = st.empty()

            training_result: List[Optional[Dict[str, Any]]] = [None]
            training_error: List[Optional[Exception]] = [None]

            def run_in_thread() -> None:
                try:
                    result = run_training_playground(
                        basin_folder=basin_folder,
                        experiment_name=experiment_name or "unnamed",
                        train_years=train_years,
                        val_years=val_years,
                        test_years=test_years,
                        epochs=epochs,
                        learning_rate=learning_rate,
                        batch_size=batch_size,
                        seq_len=seq_len,
                        hidden_size_s1=hidden_size_s1,
                        hidden_size_s2=hidden_size_s2,
                        num_layers=num_layers,
                        dropout=dropout,
                        embed_dim=embed_dim,
                        beta_weight=beta_weight,
                        global_qmax=global_qmax,
                        early_stop_patience=early_stop_patience,
                        lr_patience=lr_patience,
                        lr_factor=lr_factor,
                        lr_min=lr_min,
                        experiment_notes=experiment_notes,
                        apply_forecast_augmentation=apply_forecast_augmentation,
                        ui_callback=None,
                        experiment_logger=logger,
                        status_path=status_path,
                    )
                    training_result[0] = result
                except Exception as e:
                    training_error[0] = e

            thread = threading.Thread(target=run_in_thread, daemon=True)
            thread.start()

            max_total_epochs = epochs * 2
            while thread.is_alive():
                if status_path.is_file():
                    try:
                        with status_path.open("r", encoding="utf-8") as f:
                            status = json.load(f)
                    except Exception:
                        status = {}
                    s = status.get("status", "")
                    stage = status.get("stage", "")
                    epoch = status.get("epoch", 0)
                    train_loss = status.get("train_loss")
                    val_loss = status.get("val_loss")
                    val_nse = status.get("val_nse")
                    val_kge = status.get("val_kge")
                    history = status.get("history", [])

                    # Progress
                    total_so_far = len(history)
                    prog = min(1.0, total_so_far / max_total_epochs) if max_total_epochs else 0
                    txt = f"{stage} — Epoch {epoch}"
                    if isinstance(train_loss, (int, float)):
                        txt += f" — train_loss={train_loss:.4f}"
                    progress_placeholder.progress(prog, text=txt)
                    # Status text
                    line = f"**{stage}** Epoch {epoch}"
                    if isinstance(train_loss, (int, float)):
                        line += f" · train_loss={train_loss:.4f}"
                    if isinstance(val_loss, (int, float)):
                        line += f" · val_loss={val_loss:.4f}"
                    if val_nse is not None:
                        line += f" · NSE={val_nse:.3f}"
                    if val_kge is not None:
                        line += f" · KGE={val_kge:.3f}"
                    status_placeholder.markdown(line)

                    # Line chart: loss over epochs
                    if history:
                        try:
                            import pandas as pd
                            df = pd.DataFrame(history)
                            if "train_loss" in df.columns and "val_loss" in df.columns:
                                chart_placeholder.line_chart(
                                    df[["train_loss", "val_loss"]],
                                    x=None,
                                    y=["train_loss", "val_loss"],
                                    height=280,
                                )
                        except Exception:
                            pass
                time.sleep(1)

            # Final update
            if status_path.is_file():
                try:
                    with status_path.open("r", encoding="utf-8") as f:
                        status = json.load(f)
                    if status.get("status") == "done":
                        progress_placeholder.progress(1.0, text="Done.")
                    status_placeholder.empty()
                except Exception:
                    pass

            if training_error[0] is not None:
                result_placeholder.error(f"Training failed: {training_error[0]}")
                raise training_error[0]
            if training_result[0]:
                r = training_result[0]
                result_placeholder.success(
                    f"Training complete: **{r['run_id']}**. "
                    f"Stage 1: `{r['stage1_path']}`; Stage 2: `{r['stage2_path']}`."
                )
                result_placeholder.info(f"Run directory: `{r['run_dir']}`")
                if r.get("test_loss_s1") is not None or r.get("test_nse") is not None:
                    test_line = "Test set — "
                    if r.get("test_loss_s1") is not None:
                        test_line += f"Stage 1 loss={r['test_loss_s1']:.4f}; "
                    if r.get("test_nse") is not None:
                        test_line += f"Stage 2 NSE={r['test_nse']:.3f}"
                    if r.get("test_kge") is not None:
                        test_line += f", KGE={r['test_kge']:.3f}"
                    result_placeholder.caption(test_line)


def _render_tab_live_monitor() -> None:
    st.subheader("Live Training Monitor")
    st.info(
        "Live metrics (loss, validation scores) will appear here when a training run is in progress."
    )


def _render_tab_experiment_history(basin_folder: Path) -> None:
    st.subheader("Experiment History (Leaderboard)")

    run_dirs = list_experiments(basin_folder)
    if not run_dirs:
        st.info("No experiments found. Run a training from the Configuration tab.")
        return

    rows: List[Dict[str, Any]] = []
    for run_dir in run_dirs:
        config = load_run_config(run_dir)
        metrics = load_run_metrics(run_dir)
        final_val_loss = get_final_val_loss(run_dir)

        run_id = run_dir.name
        notes = (config or {}).get("experiment_notes", "") or ""
        arch = (config or {}).get("model_architecture", "")
        epochs_cfg = (config or {}).get("epochs", "")
        lr = (config or {}).get("learning_rate", "")
        batch = (config or {}).get("batch_size", "")
        h1 = (config or {}).get("hidden_size_s1", (config or {}).get("hidden_size", ""))
        h2 = (config or {}).get("hidden_size_s2", "")
        hidden_str = f"S1={h1}, S2={h2}" if h2 != "" else str(h1)
        config_summary = (
            f"arch={arch}, epochs={epochs_cfg}, lr={lr}, batch={batch}, hidden={hidden_str}"
        )

        rows.append({
            "run_id": run_id,
            "notes": (notes[:80] + "…") if len(notes) > 80 else notes,
            "config": config_summary,
            "final_val_loss": final_val_loss if final_val_loss is not None else "",
            "num_epochs_logged": len(metrics),
        })

    df = pd.DataFrame(rows)
    st.dataframe(df, use_container_width=True, hide_index=True)


def main() -> None:
    st.set_page_config(
        page_title="Model Playground",
        page_icon="🧪",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.title("Model Playground")
    st.markdown(
        "Configure and launch ML training (LSTM, and later PINN / Curriculum Learning) "
        "on merged Rain–Discharge datasets. Every run is tracked with metadata and metrics."
    )

    basin_files = _list_basin_configs()
    if not basin_files:
        st.sidebar.error(
            "No basin config found. Add `configs/system_config.yaml` or configs in `configs/basins/`."
        )
        return

    basin_names = [p.stem for p in basin_files]
    basin_idx = st.sidebar.selectbox(
        "Basin",
        options=list(range(len(basin_files))),
        format_func=lambda i: basin_names[i],
    )
    basin_config_path = basin_files[basin_idx]
    cfg = _load_config(basin_config_path)
    basin_folder = _get_basin_folder(cfg)

    if not basin_folder or not basin_folder.is_dir():
        st.sidebar.warning(
            "Basin folder not set or not found in config. Set `data_paths.basin_folder`."
        )
        st.info("Select a basin with a valid `data_paths.basin_folder` to use the Playground.")
        return

    st.sidebar.caption(f"Basin folder: `{basin_folder}`")

    tab1, tab2, tab3 = st.tabs([
        "⚙️ Configuration & Training",
        "📈 Live Training Monitor",
        "🗄️ Experiment History (Leaderboard)",
    ])

    with tab1:
        _render_tab_config_and_training(basin_folder)

    with tab2:
        _render_tab_live_monitor()

    with tab3:
        _render_tab_experiment_history(basin_folder)


if __name__ == "__main__":
    main()
