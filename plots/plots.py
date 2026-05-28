"""Research-grade plots driven by results/*.csv.

All plots read from ``results/summary.csv`` and ``results/per_seed.csv``
written by ``experiments.runner``. The DRL learning-curve plot is built
from ``results/drl_training_history.csv`` when present.

Plots produced:

  plots/hit_rate_ci.png              Bar chart with 95% CI error bars
  plots/deadline_satisfaction_ci.png Bar chart with 95% CI error bars
  plots/latency_mean_p95_p99.png     Grouped bars for mean/P95/P99
  plots/cloud_fetch_rate_ci.png      Bar chart with 95% CI error bars
  plots/energy_evictions.png         Two-panel energy + evictions
  plots/latency_cdf.png              Per-policy latency CDF
  plots/drl_learning_curve.png       Per-episode reward + loss
"""

from __future__ import annotations

import os
from typing import Iterable, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PLOT_DIR = "plots"


def _ordered_policies(policies: Iterable[str]) -> List[str]:
    # Sort baselines first, ablations next, proposed last.
    priority = {
        "LRU": 0,
        "LFU": 1,
        "DCCC_HASH": 2,
        "DCCC_NEAREST": 3,
        "DEADLINE_NO_OVERLAP": 4,
        "DEADLINE_HEURISTIC": 5,
        "DEADLINE_DCCC": 6,
    }
    return sorted(policies, key=lambda p: priority.get(p, 99))


def _bar_with_ci(
    ax,
    summary: pd.DataFrame,
    metric: str,
    title: str,
    ylabel: str,
    highlight: Optional[str] = "DEADLINE_DCCC",
) -> None:
    policies = _ordered_policies(summary["policy"].tolist())
    means = [summary.loc[summary["policy"] == p, f"{metric}_mean"].iloc[0] for p in policies]
    ci = [summary.loc[summary["policy"] == p, f"{metric}_ci95"].iloc[0] for p in policies]
    colors = ["#5b8def" if p != highlight else "#d9534f" for p in policies]
    ax.bar(policies, means, yerr=ci, capsize=4, color=colors)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.tick_params(axis="x", rotation=25, labelsize=8)


def plot_summary_bars(summary_path: str = "results/summary.csv", out_dir: str = PLOT_DIR) -> None:
    summary = pd.read_csv(summary_path)
    os.makedirs(out_dir, exist_ok=True)

    spec = [
        ("hit_rate", "Hit rate (95% CI)", "rate", "hit_rate_ci.png"),
        ("deadline_satisfaction", "Deadline satisfaction (95% CI)", "rate", "deadline_satisfaction_ci.png"),
        ("cloud_fetch_rate", "Cloud fetch rate (95% CI, lower is better)", "rate", "cloud_fetch_rate_ci.png"),
    ]
    for metric, title, ylabel, filename in spec:
        fig, ax = plt.subplots(figsize=(7, 4))
        _bar_with_ci(ax, summary, metric, title, ylabel)
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, filename), dpi=160)
        plt.close(fig)

    # Latency mean / P95 / P99 grouped
    policies = _ordered_policies(summary["policy"].tolist())
    x = np.arange(len(policies))
    width = 0.27
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for i, metric in enumerate(["avg_latency_ms", "p95_latency_ms", "p99_latency_ms"]):
        means = [summary.loc[summary["policy"] == p, f"{metric}_mean"].iloc[0] for p in policies]
        ci = [summary.loc[summary["policy"] == p, f"{metric}_ci95"].iloc[0] for p in policies]
        ax.bar(x + (i - 1) * width, means, width, yerr=ci, capsize=3, label=metric.replace("_", " "))
    ax.set_xticks(x)
    ax.set_xticklabels(policies, rotation=25, fontsize=8)
    ax.set_ylabel("ms")
    ax.set_title("Latency: mean / P95 / P99 (95% CI, lower is better)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "latency_mean_p95_p99.png"), dpi=160)
    plt.close(fig)

    # Energy & evictions two-panel
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    _bar_with_ci(axes[0], summary, "avg_energy", "Avg energy per request (95% CI)", "energy")
    _bar_with_ci(axes[1], summary, "cache_evictions", "Total cache evictions (95% CI)", "count")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "energy_evictions.png"), dpi=160)
    plt.close(fig)


def plot_latency_cdf(per_seed_path: str = "results/per_seed.csv", out_dir: str = PLOT_DIR) -> None:
    """Approximate CDF from per-policy mean of P50/P95/P99 markers.

    The simulator does not persist the full latency vector (would balloon
    results/), so we use the percentile summary points written per seed.
    For a truly continuous CDF, re-run a single seed and use the
    raw_latencies API (see ``sim.simulator.run_simulation`` log_callback).
    """
    # Skip if latency CDF data is unavailable; plot something useful anyway.
    df = pd.read_csv(per_seed_path)
    policies = _ordered_policies(df["policy"].unique())
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for policy in policies:
        sub = df[df["policy"] == policy]
        # Use mean across seeds for percentiles 50 ≈ mean, 95, 99.
        markers = [
            ("mean", sub["avg_latency_ms"].mean()),
            ("p95", sub["p95_latency_ms"].mean()),
            ("p99", sub["p99_latency_ms"].mean()),
        ]
        xs = [v for _, v in markers]
        ys = [0.5, 0.95, 0.99]
        ax.plot(xs, ys, marker="o", label=policy)
    ax.set_xlabel("Latency (ms)")
    ax.set_ylabel("CDF")
    ax.set_title("Latency CDF (mean / P95 / P99 markers)")
    ax.set_ylim(0.4, 1.01)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "latency_cdf.png"), dpi=160)
    plt.close(fig)


def plot_drl_learning_curve(
    history_path: str = "results/drl_training_history.csv", out_dir: str = PLOT_DIR
) -> None:
    if not os.path.exists(history_path):
        return
    df = pd.read_csv(history_path)
    agg = df.groupby("episode").agg(
        reward_mean=("reward", "mean"),
        reward_std=("reward", "std"),
        loss_mean=("avg_loss", "mean"),
        eps_mean=("epsilon", "mean"),
    ).reset_index()
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(agg["episode"], agg["reward_mean"], color="#d9534f")
    axes[0].fill_between(
        agg["episode"],
        agg["reward_mean"] - agg["reward_std"].fillna(0),
        agg["reward_mean"] + agg["reward_std"].fillna(0),
        alpha=0.2,
        color="#d9534f",
    )
    axes[0].set_xlabel("Episode")
    axes[0].set_ylabel("Episode reward")
    axes[0].set_title("DRL learning curve (mean ± 1σ across seeds)")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(agg["episode"], agg["loss_mean"], color="#5b8def", label="loss")
    ax2 = axes[1].twinx()
    ax2.plot(agg["episode"], agg["eps_mean"], color="#5cb85c", label="epsilon")
    axes[1].set_xlabel("Episode")
    axes[1].set_ylabel("Avg loss", color="#5b8def")
    ax2.set_ylabel("Epsilon", color="#5cb85c")
    axes[1].set_title("Loss & exploration schedule")
    axes[1].grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "drl_learning_curve.png"), dpi=160)
    plt.close(fig)


def plot_all(results_dir: str = "results", out_dir: str = PLOT_DIR) -> None:
    plot_summary_bars(os.path.join(results_dir, "summary.csv"), out_dir)
    plot_latency_cdf(os.path.join(results_dir, "per_seed.csv"), out_dir)
    plot_drl_learning_curve(os.path.join(results_dir, "drl_training_history.csv"), out_dir)


if __name__ == "__main__":
    plot_all()
