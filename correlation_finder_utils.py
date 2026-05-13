from typing import Iterable, List, Literal, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas.plotting import scatter_matrix


def numeric_columns(df: pd.DataFrame) -> List[str]:
    return df.select_dtypes(include=[np.number]).columns.tolist()


CorrelationMethod = Literal["pearson", "kendall", "spearman"]


def correlations_matrix(
    df: pd.DataFrame, method: CorrelationMethod = "pearson"
) -> pd.DataFrame:
    cols = numeric_columns(df)
    return df[cols].corr(method=method)


def top_target_correlations(
    df: pd.DataFrame, target: str, n: int = 10, method: CorrelationMethod = "pearson"
) -> pd.Series:
    if target not in df.columns:
        raise KeyError(f"Target column '{target}' not found in DataFrame")
    corr = correlations_matrix(df, method=method)[target].drop(
        labels=[target], errors="ignore"
    )
    return corr.reindex(corr.abs().sort_values(ascending=False).index)


def top_pairwise_correlations(
    df: pd.DataFrame, n: int = 10, method: CorrelationMethod = "pearson"
) -> List[Tuple[str, str, float]]:
    corr = correlations_matrix(df, method=method).abs()
    # mask diagonal and duplicates
    corr.values[np.tril_indices_from(corr.values)] = 0
    flat = (
        corr.stack()
        .reset_index()
        .rename(columns={0: "abs_corr"})
        .sort_values("abs_corr", ascending=False)
    )
    results: List[Tuple[str, str, float]] = []
    for _, row in flat.head(n).iterrows():
        results.append((row["level_0"], row["level_1"], float(row["abs_corr"])))
    return results


def plot_scatter_matrix(
    df: pd.DataFrame,
    columns: Iterable[str],
    figsize: Tuple[int, int] = (10, 10),
    save_path: Optional[str] = None,
):
    cols = [c for c in columns if c in df.columns]
    if len(cols) == 0:
        raise ValueError("No valid columns provided for scatter matrix")
    scatter_matrix(df[cols], figsize=figsize, diagonal="kde")
    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
    plt.close()


def plot_correlation_heatmap(
    df: pd.DataFrame,
    method: CorrelationMethod = "pearson",
    save_path: Optional[str] = None,
):
    corr = correlations_matrix(df, method=method)
    fig, ax = plt.subplots(figsize=(10, 8))
    cax = ax.matshow(corr, cmap="RdBu", vmin=-1, vmax=1)
    fig.colorbar(cax)
    ticks = np.arange(0, len(corr.columns), 1)
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xticklabels(corr.columns, rotation=90)
    ax.set_yticklabels(corr.columns)
    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)


__all__ = [
    "numeric_columns",
    "correlations_matrix",
    "top_target_correlations",
    "top_pairwise_correlations",
    "plot_scatter_matrix",
    "plot_correlation_heatmap",
]
