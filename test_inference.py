from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
import yaml

from src.inference import FloodPredictor


def load_config() -> Dict[str, Any]:
    """
    Load the system configuration from the default YAML file.

    Returns
    -------
    Dict[str, Any]
        Parsed configuration dictionary.
    """
    config_path = Path(__file__).resolve().parent / "configs" / "system_config.yaml"
    with config_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main() -> None:
    """Run a minimal end-to-end inference sanity check on a single ICON member."""
    config = load_config()
    predictor = FloodPredictor(config, config_path=None)

    # Load exactly one ICON member to keep the test fast
    members = predictor._loader.load_all_icon_members()  # type: ignore[attr-defined]
    if not members:
        raise RuntimeError("No ICON members available for testing.")

    member_name = sorted(members.keys())[0]
    member_df = members[member_name]

    # Restrict to configured station cells and a short time window
    station_cells: List[int] = list(config.get("stations", {}).get("station_cells", []))  # type: ignore[assignment]
    if station_cells:
        member_df = member_df[member_df["ID"].isin(station_cells)].copy()

    member_df = member_df.sort_values("time")
    if "time" in member_df.columns and not member_df.empty:
        t0 = member_df["time"].min()
        t_end = t0 + pd.Timedelta(hours=6)
        member_df = member_df[member_df["time"] <= t_end].copy()

    print(f"Running inference for member: {member_name}")
    print(f"Subset rows for test: {len(member_df)}")

    result = predictor.run_for_member(member_name, member_df)

    print("\nResulting DataFrame shape:", result.shape)
    print("\nHead:")
    print(result.head())


if __name__ == "__main__":
    main()

