from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def run_streamlit() -> int:
    """
    Launch the Streamlit multi-page UI (Home page).

    Returns
    -------
    int
        Exit code from the Streamlit process.
    """
    app_path = Path(__file__).resolve().parent / "ui" / "Home.py"
    cmd = [sys.executable, "-m", "streamlit", "run", str(app_path)]
    return subprocess.call(cmd)


def run_headless_pipeline() -> int:
    """
    Run the end-to-end prediction pipeline in headless mode.

    Notes
    -----
    The actual implementation will be added once the data loading,
    feature engineering, and model inference modules are in place.
    """
    # Placeholder for future FloodPredictor integration
    print("Headless pipeline execution is not yet implemented.")
    return 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """
    Parse command-line arguments for the entry point.

    Parameters
    ----------
    argv : list[str] or None, optional
        Command-line arguments (excluding the program name). If None,
        they are taken from ``sys.argv[1:]``.

    Returns
    -------
    argparse.Namespace
        Parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description="Israel National Flood Forecasting Center - operational system entry point",
    )
    parser.add_argument(
        "--mode",
        choices=["ui", "headless"],
        default="ui",
        help="Execution mode: 'ui' to launch Streamlit, 'headless' to run the pipeline.",
    )
    return parser.parse_args(argv)


def main() -> int:
    """
    Main entry point.

    Returns
    -------
    int
        Process exit code.
    """
    args = parse_args()
    if args.mode == "ui":
        return run_streamlit()
    return run_headless_pipeline()


if __name__ == "__main__":
    raise SystemExit(main())

