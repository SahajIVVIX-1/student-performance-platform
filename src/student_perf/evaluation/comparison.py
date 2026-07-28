"""
Model comparison and leaderboard generation.

Reads metrics_summary.csv and produces a sorted leaderboard
with Plotly bar charts and bootstrap confidence intervals.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parents[3]


def load_metrics_summary() -> pd.DataFrame:
    """Load the metrics_summary.csv file."""
    path = _PROJECT_ROOT / "data" / "processed" / "metrics_summary.csv"
    if not path.exists():
        logger.warning("metrics_summary.csv not found. Run train_all first.")
        return pd.DataFrame()
    return pd.read_csv(path)


def build_leaderboard(
    df: pd.DataFrame,
    primary_metric: str = "f1",
    ascending: bool = False,
) -> pd.DataFrame:
    """
    Sort models by primary metric and build a clean leaderboard.

    Args:
        df: metrics_summary DataFrame
        primary_metric: column to sort by
        ascending: True for metrics where lower is better (e.g., RMSE)

    Returns:
        Sorted leaderboard DataFrame.
    """
    if df.empty:
        return df

    # Filter out error rows
    df = df[df["error"].isna() if "error" in df.columns else df.index >= 0].copy()

    cols_to_show = ["model"]
    metric_candidates = ["f1", "roc_auc", "accuracy", "precision", "recall",
                         "rmse", "mae", "r2", "train_time_s", "inference_latency_ms"]
    for col in metric_candidates:
        if col in df.columns:
            cols_to_show.append(col)

    df_lb = df[cols_to_show].copy()

    if primary_metric in df_lb.columns:
        df_lb = df_lb.sort_values(primary_metric, ascending=ascending).reset_index(drop=True)
        df_lb.insert(0, "rank", range(1, len(df_lb) + 1))

    return df_lb


def plot_leaderboard(df: pd.DataFrame, primary_metric: str = "f1", save_path: str | None = None):
    """Generate a Plotly bar chart of the leaderboard."""
    import plotly.graph_objects as go

    if df.empty or primary_metric not in df.columns:
        logger.warning("Cannot generate leaderboard plot — missing data")
        return None

    fig = go.Figure(
        go.Bar(
            x=df["model"],
            y=df[primary_metric],
            marker_color="rgb(55, 83, 109)",
            text=df[primary_metric].round(4),
            textposition="outside",
        )
    )
    fig.update_layout(
        title=f"Model Leaderboard — {primary_metric.upper()}",
        xaxis_title="Model",
        yaxis_title=primary_metric.upper(),
        template="plotly_dark",
        height=450,
    )

    if save_path:
        fig.write_html(save_path)
        logger.info(f"Leaderboard chart saved to {save_path}")

    return fig


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    df = load_metrics_summary()
    lb = build_leaderboard(df)
    print(lb.to_string())
