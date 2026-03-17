from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import yaml

from prepare_data import main as run_prepare_data
from src.inference import FloodPredictor


def load_config(config_path: Path) -> Dict[str, Any]:
    """
    Load the system configuration from a YAML file.

    Parameters
    ----------
    config_path : Path
        Path to the YAML configuration file.

    Returns
    -------
    Dict[str, Any]
        Parsed configuration dictionary.
    """
    with config_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main() -> None:
    """
    Run an end-to-end test of the full pipeline on local data.

    Steps
    -----
    1. Execute the heavy preprocessing pipeline (UGRID + ICON).
    2. Instantiate the FloodPredictor and run ensemble inference.
    3. Print basic statistics about the resulting predictions.
    """
    project_root = Path(__file__).resolve().parent
    config_path = project_root / "configs" / "system_config.yaml"

    print("=== STEP 1: Running data preparation pipeline ===")
    run_prepare_data()

    print("\n=== STEP 2: Running ensemble inference ===")
    config = load_config(config_path)
    predictor = FloodPredictor(config, config_path=config_path)
    predictions = predictor.run_all_members()

    if predictions.empty:
        raise RuntimeError("End-to-end test failed: predictions DataFrame is empty.")

    print("\nEnd-to-end inference completed successfully.")
    print(f"Total prediction rows: {len(predictions):,}")
    print(f"Unique members: {predictions['member'].nunique()}")
    print(f"Unique station cells: {predictions['ID'].nunique()}")
    print(f"Time range: {predictions['time'].min()} → {predictions['time'].max()}")
    print("Sample head:")
    print(predictions.head())


if __name__ == "__main__":
    main()

