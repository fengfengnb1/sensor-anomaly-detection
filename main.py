"""
main.py
-------
Command-line entry point for the sensor anomaly detection pipeline.

Usage examples
--------------
# Run with synthetic data (no download needed)
python main.py --mode synthetic

# Run two of the three methods
python main.py --mode synthetic --method all

# Run on your own CSV file
python main.py --mode file --input data/raw/my_sensor_data.csv

# Run a specific method only
python main.py --mode synthetic --method isolation_forest

# Show plots interactively instead of saving
python main.py --mode synthetic --no-save
"""

import argparse
from src.pipeline import run, SENSOR_COLS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sensor time-series anomaly detection pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--mode", choices=["synthetic", "file"], default="synthetic",
        help="Data source: 'synthetic' generates sensor data, 'file' reads a CSV.",
    )
    parser.add_argument(
        "--input", type=str, default=None,
        help="Path to input CSV file (required when --mode file).",
    )
    parser.add_argument(
        "--channels", nargs="+", default=SENSOR_COLS,
        help="Sensor channel column names (default: temperature humidity).",
    )
    parser.add_argument(
        "--method", choices=["zscore", "isolation_forest", "lstm", "all"], default="all",
        help="Anomaly detection method to run (default: all).",
    )
    parser.add_argument(
        "--output", type=str, default="reports/figures",
        help="Directory for output figures (default: reports/figures).",
    )
    parser.add_argument(
        "--no-save", action="store_true",
        help="Display plots interactively instead of saving to disk.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(
        mode=args.mode,
        input_path=args.input,
        sensor_cols=args.channels,
        method=args.method,
        output_dir=args.output,
        save_figures=not args.no_save,
    )
