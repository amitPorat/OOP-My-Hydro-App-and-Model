from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import yaml

from src.preprocess import IconPreprocessor, UgridPreprocessor


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
    Execute the data preparation pipeline.

    This script can be run as a batch job to prepare all heavy
    preprocessing artefacts needed by the real-time inference
    and UI system:

    - Generate the UGRID mesh and terrain attributes.
    - Prepare ICON ensemble forecasts (kriged rain on UGRID cells).
    """
    config_path = Path(__file__).resolve().parent / "configs" / "system_config.yaml"
    config = load_config(config_path)

    print("=== UGRID preprocessing ===")
    ugrid_pre = UgridPreprocessor(config)
    ugrid_parquet = ugrid_pre.run()
    print(f"UGRID with terrain saved at: {ugrid_parquet}")

    print("\n=== ICON preprocessing ===")
    icon_pre = IconPreprocessor(config)
    try:
        icon_pre.run()
    except NotImplementedError as exc:
        print(
            "ICON preprocessing is not yet fully implemented in "
            "`IconPreprocessor.run()`. Please port the kriging logic "
            "from the notebooks into this method."
        )
        print(f"Details: {exc}")

    print("\nData preparation pipeline finished.")


if __name__ == "__main__":
    main()

