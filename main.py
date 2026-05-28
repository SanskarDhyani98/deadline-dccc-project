"""CLI entrypoint for the deadline-aware DCCC research simulator.

Examples:

  # Full research run: 5 seeds, all policies, DRL training enabled.
  python main.py

  # Quick smoke run (no DRL training, 1 seed, baselines only).
  python main.py --no-drl --seeds 42 --policies LRU LFU DCCC_HASH

  # DRL ablation only.
  python main.py --policies DEADLINE_HEURISTIC DEADLINE_DCCC --episodes 20

After the run completes, ``results/`` and ``plots/`` are populated:
  results/per_seed.csv, summary.csv, significance.csv,
  drl_training_history.csv (when DRL is trained)
  plots/*.png
"""

from __future__ import annotations

import argparse
import sys

from experiments.runner import load_config, run_multi_seed
from plots.plots import plot_all


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Deadline-aware DCCC research simulator.")
    parser.add_argument("--config", default="configs/simulation_config.json")
    parser.add_argument("--seeds", type=int, nargs="*", default=None, help="Override seed list.")
    parser.add_argument("--policies", nargs="*", default=None, help="Override policy list.")
    parser.add_argument(
        "--no-drl",
        action="store_true",
        help="Disable DRL training; DEADLINE_DCCC then runs the heuristic core.",
    )
    parser.add_argument("--episodes", type=int, default=None, help="Override num_episodes for DRL.")
    parser.add_argument("--output-dir", default="results")
    parser.add_argument("--skip-plots", action="store_true")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
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
    if not out["significance"].empty:
        print("\n=== Significance (DEADLINE_DCCC vs best baseline) ===")
        print(out["significance"].to_string(index=False))

    if not args.skip_plots:
        plot_all(results_dir=args.output_dir)
        print("\nWrote plots to plots/*.png")
    return 0


if __name__ == "__main__":
    sys.exit(main())
