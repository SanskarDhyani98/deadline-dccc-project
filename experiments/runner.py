"""Multi-seed experiment runner.

For every (policy, seed) pair this runs ``sim.simulator.run_simulation``
and aggregates the resulting per-replication rows into:

  results/per_seed.csv      — every (policy, seed) row
  results/summary.csv       — per-policy mean ± 95% CI across seeds
  results/significance.csv  — paired t / Wilcoxon vs the best non-proposed
                              baseline, per metric

For ``DEADLINE_DCCC`` the DRL agents are trained per-seed (so each seed
yields an independently trained policy — closer to true variance).
Heuristic ablations (``DEADLINE_HEURISTIC`` etc.) reuse the same
simulator without a DRL hook.
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict
from typing import Dict, List, Optional

import pandas as pd

from experiments.stats import mean_ci95, paired_t_pvalue, wilcoxon_signed_rank_pvalue
from sim.simulator import build_world, run_simulation, DRL_POLICY


METRIC_COLUMNS = [
    "hit_rate",
    "edge_hit_rate",
    "coop_hit_rate",
    "cloud_fetch_rate",
    "deadline_satisfaction",
    "avg_latency_ms",
    "p95_latency_ms",
    "p99_latency_ms",
    "avg_energy",
    "cache_evictions",
    "load_imbalance",
]

# Higher-is-better vs. lower-is-better. Used to pick the best baseline
# and to decide the direction of "improvement %".
HIGHER_IS_BETTER = {
    "hit_rate": True,
    "edge_hit_rate": True,
    "coop_hit_rate": True,
    "cloud_fetch_rate": False,
    "deadline_satisfaction": True,
    "avg_latency_ms": False,
    "p95_latency_ms": False,
    "p99_latency_ms": False,
    "avg_energy": False,
    "cache_evictions": False,
    "load_imbalance": False,
}


def _run_for_policy(policy: str, seed: int, config: dict, train_drl: bool, train_log: List[dict]) -> Dict:
    world = build_world(seed=seed, config=config)
    if policy == DRL_POLICY:
        if not train_drl:
            # Fall back to heuristic (DRL hook = None) for fast smoke runs.
            metrics = run_simulation(policy=policy, seed=seed, config=config, world=world)
        else:
            from drl.trainer import make_eval_hook, train_drl_agents

            agents, history = train_drl_agents(seed=seed, config=config, verbose=False)
            train_log.append({"seed": seed, "history": history})
            # Re-build a fresh world so caches don't carry over training state.
            world = build_world(seed=seed, config=config)
            hook = make_eval_hook(agents)
            metrics = run_simulation(
                policy=policy, seed=seed, config=config, world=world, drl_hook=hook
            )
    else:
        metrics = run_simulation(policy=policy, seed=seed, config=config, world=world)
    return metrics.to_row()


def run_multi_seed(
    config: dict,
    policies: Optional[List[str]] = None,
    seeds: Optional[List[int]] = None,
    train_drl: bool = True,
    output_dir: str = "results",
) -> Dict[str, pd.DataFrame]:
    """Run the full experiment matrix and write per-seed + summary CSVs."""
    policies = policies or config["policies"]
    seeds = seeds or config.get("seeds", [42])
    os.makedirs(output_dir, exist_ok=True)

    rows: List[Dict] = []
    train_log: List[Dict] = []
    for policy in policies:
        for seed in seeds:
            print(f"[run] policy={policy:<22} seed={seed}")
            rows.append(_run_for_policy(policy, seed, config, train_drl, train_log))

    per_seed = pd.DataFrame(rows)
    per_seed.to_csv(os.path.join(output_dir, "per_seed.csv"), index=False)

    summary = _summarize(per_seed)
    summary.to_csv(os.path.join(output_dir, "summary.csv"), index=False)

    significance = _significance(per_seed, policies)
    significance.to_csv(os.path.join(output_dir, "significance.csv"), index=False)

    if train_log:
        _save_training_history(train_log, output_dir)

    return {"per_seed": per_seed, "summary": summary, "significance": significance}


def _summarize(per_seed: pd.DataFrame) -> pd.DataFrame:
    summary_rows: List[Dict] = []
    for policy, group in per_seed.groupby("policy"):
        row: Dict[str, float] = {"policy": policy, "n_seeds": len(group)}
        for metric in METRIC_COLUMNS:
            mu, half = mean_ci95(group[metric].tolist())
            row[f"{metric}_mean"] = mu
            row[f"{metric}_ci95"] = half
        summary_rows.append(row)
    return pd.DataFrame(summary_rows)


def _significance(per_seed: pd.DataFrame, policies: List[str]) -> pd.DataFrame:
    """Paired tests of DEADLINE_DCCC vs. the strongest non-deadline baseline."""
    if DRL_POLICY not in policies:
        return pd.DataFrame()
    proposed = per_seed[per_seed["policy"] == DRL_POLICY].sort_values("seed")
    baselines = [p for p in policies if not p.startswith("DEADLINE")]
    if not baselines:
        return pd.DataFrame()
    out_rows: List[Dict] = []
    for metric in METRIC_COLUMNS:
        higher_better = HIGHER_IS_BETTER[metric]
        # Pick best baseline by mean on this metric.
        best_baseline = None
        best_mean = -float("inf") if higher_better else float("inf")
        for b in baselines:
            mu = per_seed[per_seed["policy"] == b][metric].mean()
            if higher_better and mu > best_mean:
                best_mean, best_baseline = mu, b
            elif not higher_better and mu < best_mean:
                best_mean, best_baseline = mu, b
        baseline_rows = per_seed[per_seed["policy"] == best_baseline].sort_values("seed")
        x = proposed[metric].tolist()
        y = baseline_rows[metric].tolist()
        prop_mean = sum(x) / len(x)
        rel_change_pct = (
            100.0 * (prop_mean - best_mean) / abs(best_mean) if best_mean != 0 else 0.0
        )
        out_rows.append(
            {
                "metric": metric,
                "best_baseline": best_baseline,
                "proposed_mean": prop_mean,
                "baseline_mean": best_mean,
                "relative_change_pct": rel_change_pct,
                "higher_is_better": higher_better,
                "paired_t_p": paired_t_pvalue(x, y),
                "wilcoxon_p": wilcoxon_signed_rank_pvalue(x, y),
            }
        )
    return pd.DataFrame(out_rows)


def _save_training_history(train_log: List[Dict], output_dir: str) -> None:
    history_rows: List[Dict] = []
    for entry in train_log:
        seed = entry["seed"]
        history = entry["history"]
        for episode, (r, l, e) in enumerate(
            zip(history.episode_reward, history.episode_loss, history.episode_epsilon)
        ):
            history_rows.append(
                {"seed": seed, "episode": episode + 1, "reward": r, "avg_loss": l, "epsilon": e}
            )
    pd.DataFrame(history_rows).to_csv(os.path.join(output_dir, "drl_training_history.csv"), index=False)


def load_config(path: str = "configs/simulation_config.json") -> dict:
    with open(path, "r") as f:
        return json.load(f)


def main() -> None:
    parser = argparse.ArgumentParser(description="Deadline-aware DCCC multi-seed experiments.")
    parser.add_argument("--config", default="configs/simulation_config.json")
    parser.add_argument("--seeds", type=int, nargs="*", default=None, help="Override seed list.")
    parser.add_argument("--policies", nargs="*", default=None, help="Override policy list.")
    parser.add_argument("--no-drl", action="store_true", help="Skip DRL training (use heuristic core for DEADLINE_DCCC).")
    parser.add_argument("--output-dir", default="results")
    parser.add_argument("--episodes", type=int, default=None, help="Override num_episodes for DRL training.")
    args = parser.parse_args()

    config = load_config(args.config)
    if args.episodes is not None:
        config["num_episodes"] = args.episodes
    out = run_multi_seed(
        config=config,
        policies=args.policies,
        seeds=args.seeds,
        train_drl=not args.no_drl,
        output_dir=args.output_dir,
    )
    print("\n=== Summary (mean across seeds) ===")
    print(out["summary"].to_string(index=False))
    print("\n=== Significance (DEADLINE_DCCC vs best baseline) ===")
    print(out["significance"].to_string(index=False))


if __name__ == "__main__":
    main()
