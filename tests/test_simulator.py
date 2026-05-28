"""Smoke tests for the simulator and policies.

These are not full property tests — they pin behaviour we rely on for
the research claims so that future refactors don't silently change the
metric pipeline.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments.runner import load_config
from sim.simulator import build_world, run_simulation, DRL_POLICY


def _small_config() -> dict:
    cfg = load_config(os.path.join(os.path.dirname(__file__), "..", "configs", "simulation_config.json"))
    cfg["num_requests"] = 400
    cfg["num_contents"] = 40
    cfg["seeds"] = [42, 43]
    return cfg


def test_simulator_runs_for_every_policy():
    cfg = _small_config()
    for policy in cfg["policies"]:
        if policy == DRL_POLICY:
            # DRL hook left at None → falls back to heuristic core.
            metrics = run_simulation(policy=policy, seed=42, config=cfg)
        else:
            metrics = run_simulation(policy=policy, seed=42, config=cfg)
        row = metrics.to_row()
        assert row["requests"] == cfg["num_requests"]
        assert 0.0 <= row["hit_rate"] <= 1.0
        assert 0.0 <= row["deadline_satisfaction"] <= 1.0


def test_proposed_beats_dccc_hash_on_deadline():
    cfg = _small_config()
    seeds = [42, 43, 44]
    proposed = []
    hash_baseline = []
    for seed in seeds:
        world_a = build_world(seed=seed, config=cfg)
        world_b = build_world(seed=seed, config=cfg)
        proposed.append(
            run_simulation(policy="DEADLINE_HEURISTIC", seed=seed, config=cfg, world=world_a).to_row()["deadline_satisfaction"]
        )
        hash_baseline.append(
            run_simulation(policy="DCCC_HASH", seed=seed, config=cfg, world=world_b).to_row()["deadline_satisfaction"]
        )
    # We don't require it on every single seed but on the mean.
    assert sum(proposed) / len(proposed) >= sum(hash_baseline) / len(hash_baseline)


def test_world_is_deterministic_per_seed():
    cfg = _small_config()
    a = build_world(seed=42, config=cfg)
    b = build_world(seed=42, config=cfg)
    assert [c.content_id for c in a.contents] == [c.content_id for c in b.contents]
    assert [r.content_id for r in a.requests[:50]] == [r.content_id for r in b.requests[:50]]


if __name__ == "__main__":
    test_simulator_runs_for_every_policy()
    test_proposed_beats_dccc_hash_on_deadline()
    test_world_is_deterministic_per_seed()
    print("all smoke tests passed")
