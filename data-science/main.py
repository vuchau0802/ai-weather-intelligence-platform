from __future__ import annotations
import argparse
import os
import time
import warnings
from datetime import datetime
from pathlib import Path
import pandas as pd

warnings.filterwarnings("ignore")

OUTPUT_DIR = Path("outputs")
FIGURE_DIR = OUTPUT_DIR / "figures"
MODEL_DIR = OUTPUT_DIR / "models"
PROCESSED_PATH = OUTPUT_DIR / "processed_globalweather.csv"

def ascii_text(value: object) -> str:
    replacements = {
        "σ": "std",
        "²": "2",
        "°": "deg",
        "×": "x",
        "–": "-",
        "—": "-",
        "→": "to",
        "\n": " ",
    }
    text = str(value)
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text.encode("ascii", errors="replace").decode("ascii")

def ensure_output_dirs() -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

def save_metrics(metrics_df: pd.DataFrame) -> Path:
    metrics_path = OUTPUT_DIR / "model_metrics.csv"
    try:
        metrics_df.to_csv(metrics_path)
        return metrics_path
    except PermissionError:
        fallback = OUTPUT_DIR / f"model_metrics_{datetime.now():%Y%m%d_%H%M%S}.csv"
        metrics_df.to_csv(fallback)
        print(f"Could not overwrite {metrics_path}; saved metrics to {fallback}")
        return fallback

def maybe_sample(df: pd.DataFrame, quick: bool) -> pd.DataFrame:
    if not quick:
        return df

    sample_size = min(20_000, len(df))
    sampled = df.sample(sample_size, random_state=42).sort_values("last_updated")
    print(f"Quick mode: sampled {sample_size:,} rows for faster iteration.")
    return sampled

def run_pipeline(data_path: str, quick: bool = False) -> tuple[pd.DataFrame, dict, pd.DataFrame]:
    from src.preprocess import full_pipeline
    from src.explorer import (
        dataset_overview,
        detect_anomalies,
        plot_correlation,
        plot_precipitation,
        plot_temperature,
        plot_time_trends,
    )
    from src.forecasting import run_all_models

    start = time.time()
    ensure_output_dirs()

    print("Weather Trend Forecasting Pipeline")

    print("\n1. Cleaning and preprocessing")
    cleaned_df = full_pipeline(data_path)
    cleaned_df.to_csv(PROCESSED_PATH, index=False)
    print(f"Processed data saved: {PROCESSED_PATH}")
    df = maybe_sample(cleaned_df, quick)

    print("\n2. Dataset overview")
    overview = dataset_overview(df)
    print(f"Rows/columns: {overview['shape']}")
    print(f"Countries: {overview['n_countries']}")
    print(f"Cities: {overview['n_cities']}")
    print(f"Date range: {overview.get('date_range')}")

    print("\n3. Basic EDA")
    plot_temperature(df)
    plot_precipitation(df)
    plot_correlation(df)
    plot_time_trends(df)

    print("\n4. Anomaly detection")
    df_flagged, _, anomaly_summary = detect_anomalies(df, contamination=0.05)
    print("Anomaly summary:", ascii_text(anomaly_summary))

    print("\n5. Forecasting models")
    results, metrics_df, _ = run_all_models(df, target="temperature_celsius", steps=30)
    metrics_path = save_metrics(metrics_df)
    print(f"Metrics saved: {metrics_path}")

    elapsed = time.time() - start

    print(f"\nPipeline complete in {elapsed:.1f} seconds.")
    print(f"Figures: {FIGURE_DIR}")
    print(f"Models: {MODEL_DIR}")

    return df_flagged, results, metrics_df

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Weather trend forecasting analysis")
    parser.add_argument(
        "--data",
        default="data/GlobalWeatherRepository.csv",
        help="Path to GlobalWeatherRepository.csv",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Use a smaller sample for faster local checks",
    )
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    if not os.path.exists(args.data):
        raise FileNotFoundError(
            f"Dataset not found at {args.data}. Download it from Kaggle and place it in data/."
        )

    run_pipeline(args.data, quick=args.quick)
