from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.ticker import PercentFormatter


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
TABLES = ROOT / "results" / "tables"
FIGURES = ROOT / "results" / "figures"


def read_csv(name: str, required: set[str]) -> pd.DataFrame:
    path = DATA / name
    frame = pd.read_csv(path)
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"{name} is missing columns: {sorted(missing)}")
    return frame


def save_figure(name: str) -> None:
    plt.tight_layout()
    plt.savefig(FIGURES / name, dpi=300, bbox_inches="tight")
    plt.close()


def save_table(frame: pd.DataFrame, name: str) -> None:
    frame.to_csv(TABLES / name, index=False, encoding="utf-8-sig")


def smooth_curve(
    x: pd.Series, y: pd.Series, fraction: float = 2 / 3, points: int = 180
) -> tuple[np.ndarray, np.ndarray]:
    x_values = x.to_numpy(dtype=float)
    y_values = y.to_numpy(dtype=float)
    x_grid = np.linspace(x_values.min(), x_values.max(), points)
    neighbor_count = max(3, int(np.ceil(fraction * len(x_values))))
    y_grid = np.empty_like(x_grid)
    for index, center in enumerate(x_grid):
        distance = np.abs(x_values - center)
        bandwidth = np.partition(distance, neighbor_count - 1)[neighbor_count - 1]
        if bandwidth == 0:
            y_grid[index] = y_values[distance == 0].mean()
            continue
        scaled = distance / bandwidth
        weights = np.where(scaled < 1, (1 - scaled**3) ** 3, 0)
        design = np.column_stack([np.ones(len(x_values)), x_values - center])
        weighted_design = design * np.sqrt(weights)[:, None]
        weighted_y = y_values * np.sqrt(weights)
        coefficients, *_ = np.linalg.lstsq(
            weighted_design, weighted_y, rcond=None
        )
        y_grid[index] = coefficients[0]
    return x_grid, y_grid


def annual_series(
    annual: pd.DataFrame, figure: int, panel: str = ""
) -> pd.DataFrame:
    mask = annual["figure"].eq(figure)
    if panel:
        mask &= annual["panel"].eq(panel)
    return annual.loc[mask].sort_values(["series", "year"])


def plot_annual_figures(annual: pd.DataFrame) -> None:
    figure_1 = annual_series(annual, 1)
    plt.figure(figsize=(7.2, 4.5))
    plt.plot(figure_1["year"], figure_1["value"], color="black")
    plt.xlabel("Year")
    plt.ylabel("PwC Publications")
    save_figure("figure_1_pwc_publications_by_year.png")

    figure_2 = annual_series(annual, 2)
    plt.figure(figsize=(7.2, 4.5))
    colors = {"Citation": "#E8756B", "Topic": "#18B6BE"}
    for series, frame in figure_2.groupby("series"):
        plt.plot(
            frame["year"],
            frame["value"],
            linewidth=2.2,
            label=series,
            color=colors[series],
        )
    plt.xlabel("Year")
    plt.ylabel("Share of LIS publications identified")
    plt.gca().yaxis.set_major_formatter(PercentFormatter())
    plt.legend(title="Query Method", frameon=False)
    save_figure("figure_2_method_comparison.png")

    fig, axes = plt.subplots(2, 1, figsize=(7.2, 8.5))
    colors_3 = ["#E8756B", "#18B6BE"]
    for axis, panel in zip(axes, ["A", "B"]):
        frame = annual_series(annual, 3, panel)
        for color, (series, values) in zip(colors_3, frame.groupby("series")):
            axis.plot(
                values["year"],
                values["value"],
                linewidth=2,
                label=series,
                color=color,
            )
        axis.text(-0.13, 1.02, panel, transform=axis.transAxes, weight="bold", size=18)
        axis.set_xlabel("Year")
        axis.legend(frameon=False)
    axes[0].set_ylabel("Count")
    axes[1].set_ylabel("Share")
    axes[1].yaxis.set_major_formatter(PercentFormatter())
    save_figure("figure_3_ml_citation_growth.png")

    fig, axes = plt.subplots(2, 1, figsize=(7.2, 8.5))
    colors_4 = ["#E8756B", "#18B6BE"]
    for axis, panel in zip(axes, ["A", "B"]):
        frame = annual_series(annual, 4, panel)
        for color, (series, values) in zip(colors_4, frame.groupby("series")):
            axis.plot(
                values["year"],
                values["value"],
                linewidth=2,
                label=series,
                color=color,
            )
        axis.text(-0.13, 1.02, panel, transform=axis.transAxes, weight="bold", size=18)
        axis.set_xlabel("Year")
        axis.set_ylabel("Share")
        axis.set_ylim(0, 105)
        axis.yaxis.set_major_formatter(PercentFormatter())
        axis.legend(frameon=False)
    save_figure("figure_4_task_method_coverage.png")


def plot_lag_outputs(lag: pd.DataFrame, paper: pd.DataFrame) -> None:
    table_1 = lag.loc[lag["section"].eq("table_1"), ["label", "mean", "median", "std", "n"]]
    save_table(table_1, "table_1_citation_year_lags.csv")

    table_2 = lag.loc[
        lag["section"].eq("table_2"), ["year", "mean", "median", "n"]
    ].copy()
    table_2["year"] = table_2["year"].astype(int)
    save_table(table_2, "table_2_ml_lags_by_year.csv")

    field = (
        lag.loc[lag["section"].eq("figure_5"), ["label", "mean", "median", "n"]]
        .sort_values("mean")
        .reset_index(drop=True)
    )
    save_table(field, "figure_5_field_lag_values.csv")
    y = np.arange(len(field))
    plt.figure(figsize=(8.2, 7.2))
    plt.barh(y, field["mean"], color="#B8B8B8", label="Mean")
    plt.barh(y, field["median"], color="#D87575", label="Median")
    plt.yticks(y, field["label"])
    plt.xlabel("Citation Age (Years)")
    plt.ylabel("Field Being Cited in LIS Literature")
    plt.legend(frameon=False)
    save_figure("figure_5_field_citation_lags.png")

    save_table(paper, "figure_6_paper_values.csv")
    grid = sns.JointGrid(
        data=paper,
        x="ml_reference_share",
        y="non_ml_mean_lag",
        height=7.5,
        ratio=5,
    )
    grid.plot_joint(sns.scatterplot, s=28, alpha=0.85)
    grid.plot_marginals(sns.histplot, kde=True, bins=70)
    smooth_x, smooth_y = smooth_curve(
        paper["ml_reference_share"],
        paper["non_ml_mean_lag"],
    )
    grid.ax_joint.plot(smooth_x, smooth_y, color="red", linewidth=2)
    grid.ax_joint.set_xlabel("Share of ML in LIS citations")
    grid.ax_joint.set_ylabel(
        "Mean time lag between LIS and its references without ML"
    )
    grid.ax_joint.xaxis.set_major_formatter(PercentFormatter(1))
    grid.figure.savefig(
        FIGURES / "figure_6_ml_share_vs_non_ml_lag.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(grid.figure)


def plot_task_outputs(task: pd.DataFrame) -> None:
    table_3 = task.loc[
        task["section"].eq("table_3"),
        ["rank", "task", "lis_share_pct", "pwc_share_pct"],
    ].copy()
    table_3["rank"] = table_3["rank"].astype(int)
    save_table(table_3, "table_3_first_level_task_shares.csv")

    points = task.loc[task["section"].eq("figure_7")].copy()
    colors = {
        "Over-represented": "#2D8DBD",
        "Near proportional": "#BFC4C7",
        "Under-represented": "#D36C4E",
    }
    plt.figure(figsize=(9.2, 6.4))
    axis = plt.gca()
    axis.axhline(0, color="#555555", linestyle="--", linewidth=1)
    maximum = points["lis_share_pct"].max()
    for direction, frame in points.groupby("direction"):
        sizes = 18 + np.sqrt(frame["lis_share_pct"] / maximum) * 700
        axis.scatter(
            frame["pwc_share_pct"],
            frame["selectivity_log2"],
            s=sizes,
            color=colors[direction],
            alpha=0.82,
            edgecolor="white",
            linewidth=0.5,
            label=direction,
        )
    labels = {
        "Natural Language Processing",
        "Computer Vision",
        "Medical",
        "Graphs",
        "Reasoning",
        "Translation",
        "Time Series",
        "Speech",
        "Computer Code",
        "Adversarial",
        "Sentiment Classification",
    }
    for record in points.loc[points["task"].isin(labels)].itertuples(index=False):
        axis.annotate(
            record.task,
            (record.pwc_share_pct, record.selectivity_log2),
            xytext=(6, 5),
            textcoords="offset points",
            fontsize=8,
        )
    axis.set_xscale("log")
    axis.set_xlabel("PwC publication share by task category, log scale (%)")
    axis.set_ylabel(
        "Selectivity index, log2(LIS citation share / PwC publication share)"
    )
    axis.legend(frameon=False)
    save_figure("figure_7_task_selectivity.png")


def appendix_outputs(boundary: pd.DataFrame) -> None:
    names = {
        "summary": "appendix_b3_boundary_summary.csv",
        "annual": "appendix_b4_boundary_annual.csv",
        "top_tasks": "appendix_boundary_top_tasks.csv",
        "correlations": "appendix_b5_boundary_correlations.csv",
    }
    for section, name in names.items():
        frame = boundary.loc[boundary["section"].eq(section)].drop(
            columns="section"
        )
        save_table(frame.dropna(axis=1, how="all"), name)

    direct_copies = {
        "journal_classification.csv": "appendix_b2_journal_classification.csv",
        "arxiv_coverage.csv": "appendix_b1_arxiv_coverage.csv",
        "topic_search_terms.csv": "appendix_c_topic_search_terms.csv",
    }
    for source, output in direct_copies.items():
        save_table(pd.read_csv(DATA / source), output)


def main() -> None:
    TABLES.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", context="paper")

    annual = read_csv(
        "annual_metrics.csv", {"figure", "panel", "series", "year", "value"}
    )
    lag = read_csv(
        "citation_lag_metrics.csv",
        {"section", "label", "year", "mean", "median", "std", "n"},
    )
    paper = read_csv(
        "paper_ml_share.csv", {"ml_reference_share", "non_ml_mean_lag"}
    )
    task = read_csv(
        "task_metrics.csv",
        {
            "section",
            "rank",
            "task",
            "lis_share_pct",
            "pwc_share_pct",
            "selectivity_log2",
            "direction",
        },
    )
    boundary = read_csv("boundary_checks.csv", {"section", "model"})

    plot_annual_figures(annual)
    plot_lag_outputs(lag, paper)
    plot_task_outputs(task)
    appendix_outputs(boundary)
    print(f"Analysis complete: {ROOT / 'results'}")


if __name__ == "__main__":
    main()
