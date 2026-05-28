"""Deadline-aware DCCC scoring primitives.

This module is the analytical core of the proposed routing policy. The
simulator and the DRL agent both consume the same primitives so the
heuristic ablation and the learned policy are directly comparable.

Score weights are calibrated for the heterogeneous-capacity, region-skewed
workload defined in configs/simulation_config.json. They are not tuned per
seed; the ablation study in the README characterises sensitivity.
"""

from __future__ import annotations

import math
from typing import Iterable, Protocol


class _NodeLike(Protocol):
    node_id: int
    load: float
    cache: dict

    def has_content(self, content_id: int) -> bool: ...


class _ContentLike(Protocol):
    content_id: int
    size_mb: int
    base_popularity: float


def newton_popularity(
    content: _ContentLike,
    time_step: int,
    request_count: int,
    cooling_lambda: float = 0.0004,
) -> float:
    """Newton-cooling-inspired popularity.

    Combines a content's intrinsic Zipf popularity with a running request
    count and a time-decay term so that historically popular but currently
    cold content cools off and is eventually evicted by the ICE-style
    reward.
    """
    request_impact = request_count / 20.0
    return (content.base_popularity + request_impact) * math.exp(-cooling_lambda * time_step)


def common_overlap(node: _NodeLike, candidate_nodes: Iterable[_NodeLike]) -> float:
    """Fraction of a node's cache that is also held by some other candidate.

    A value of 1.0 means every item the node holds is duplicated elsewhere
    in the cluster — the node contributes no unique capacity to the
    cooperative cache. The score penalises this so the cluster naturally
    soft-partitions content.
    """
    if not node.cache:
        return 0.0
    other_contents: set = set()
    for other in candidate_nodes:
        if other.node_id != node.node_id:
            other_contents.update(other.cache.keys())
    if not other_contents:
        return 0.0
    held = set(node.cache.keys())
    return len(held & other_contents) / max(1, len(held))


def deadline_aware_score(
    node: _NodeLike,
    content: _ContentLike,
    user_region: int,
    popularity: float,
    latency_ms: float,
    d1_ms: float,
    common_overlap_ratio: float,
    drop_overlap: bool = False,
) -> float:
    """Deadline-aware DCCC node-selection score (higher is better).

    Terms:
      + 2.0 popularity           reward serving popular content from edge
      + 6.0 hit_chance           strongly prefer nodes that already cache it
      - 0.5 common_overlap       discourage duplicated caching
      - 0.08 latency_ms          prefer faster paths
      - 1.2 load**2              non-linear: penalise saturation, not utilisation
      - 0.4 deadline_risk        penalise paths likely to miss d1

    ``drop_overlap`` is used by the DEADLINE_NO_OVERLAP ablation only.
    """
    hit_chance = 1.0 if node.has_content(content.content_id) else min(0.5, popularity)
    deadline_risk = max(0.0, latency_ms - d1_ms)
    overlap_term = 0.0 if drop_overlap else 0.5 * common_overlap_ratio

    return (
        2.0 * popularity
        + 6.0 * hit_chance
        - overlap_term
        - 0.08 * latency_ms
        - 1.2 * (node.load ** 2)
        - 0.4 * deadline_risk
    )
