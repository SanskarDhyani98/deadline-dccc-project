"""Deterministic single-seed simulator.

All four families of policies (LRU, LFU, DCCC_HASH, deadline-aware) share
this simulator. Differences between policies are confined to two hooks:

  1. ``_select_node``       — which node to route the request to
  2. ``_insert_after_serve`` — how the serving node updates its cache

The simulator computes a research-grade metric panel: hit rate
(edge / cooperative / total), cloud-fetch rate, deadline-satisfaction,
mean / P95 / P99 latency, average energy, total cache evictions, and
load imbalance.

The DRL hook is intentionally injected from outside (``drl_act``) so the
simulator stays usable in isolation for ablation and baseline runs.
"""

from __future__ import annotations

import math
import random
import statistics
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

from models.content import Content
from models.edge_node import EdgeNode
from models.request import Request
from policies.deadline_dccc_policy import (
    common_overlap,
    deadline_aware_score,
    newton_popularity,
)


PolicyName = str

DRL_POLICY = "DEADLINE_DCCC"
HEURISTIC_POLICY = "DEADLINE_HEURISTIC"
NO_OVERLAP_POLICY = "DEADLINE_NO_OVERLAP"
DCCC_HASH_POLICY = "DCCC_HASH"
DCCC_NEAREST_POLICY = "DCCC_NEAREST"
LRU_POLICY = "LRU"
LFU_POLICY = "LFU"

_DEADLINE_FAMILY = {DRL_POLICY, HEURISTIC_POLICY, NO_OVERLAP_POLICY}
_DCCC_FAMILY = {DCCC_HASH_POLICY, DCCC_NEAREST_POLICY} | _DEADLINE_FAMILY


@dataclass
class World:
    contents: List[Content]
    nodes: List[EdgeNode]
    requests: List[Request]
    content_map: Dict[int, Content]


@dataclass
class SimMetrics:
    policy: PolicyName
    seed: int
    requests: int
    edge_hits: int
    coop_hits: int
    cloud_fetches: int
    deadline_satisfied: int
    cache_evictions: int
    latencies: List[float] = field(default_factory=list)
    energy_costs: List[float] = field(default_factory=list)
    serves_per_node: List[int] = field(default_factory=list)

    def to_row(self) -> Dict[str, float]:
        n = max(1, self.requests)
        sorted_latencies = sorted(self.latencies)

        def pct(p: float) -> float:
            if not sorted_latencies:
                return 0.0
            idx = min(len(sorted_latencies) - 1, int(p * len(sorted_latencies)))
            return sorted_latencies[idx]

        serves = self.serves_per_node
        mean_serves = (sum(serves) / len(serves)) if serves else 0.0
        load_imbalance = (max(serves) / (mean_serves + 1e-9)) if serves else 0.0
        return {
            "policy": self.policy,
            "seed": self.seed,
            "requests": self.requests,
            "hit_rate": (self.edge_hits + self.coop_hits) / n,
            "edge_hit_rate": self.edge_hits / n,
            "coop_hit_rate": self.coop_hits / n,
            "cloud_fetch_rate": self.cloud_fetches / n,
            "deadline_satisfaction": self.deadline_satisfied / n,
            "avg_latency_ms": statistics.mean(self.latencies) if self.latencies else 0.0,
            "p95_latency_ms": pct(0.95),
            "p99_latency_ms": pct(0.99),
            "avg_energy": statistics.mean(self.energy_costs) if self.energy_costs else 0.0,
            "cache_evictions": self.cache_evictions,
            "load_imbalance": load_imbalance,
        }


def _weighted_choice(rng: random.Random, weights: List[float]) -> int:
    total = sum(weights)
    marker = rng.random() * total
    cumulative = 0.0
    for index, weight in enumerate(weights):
        cumulative += weight
        if cumulative >= marker:
            return index
    return len(weights) - 1


def _generate_contents(rng: random.Random, config: dict) -> List[Content]:
    contents: List[Content] = []
    size_min = config.get("content_size_min_mb", 50)
    size_max = config.get("content_size_max_mb", 2500)
    zipf_exp = config.get("zipf_exponent", 0.85)
    bias_min = config.get("region_bias_min", 0.1)
    bias_max = config.get("region_bias_max", 1.0)
    for cid in range(config["num_contents"]):
        size = rng.randint(size_min, size_max)
        base_popularity = 1.0 / ((cid + 1) ** zipf_exp)
        region_bias = [round(rng.uniform(bias_min, bias_max), 2) for _ in range(config["num_regions"])]
        content_type = rng.choice(config["content_types"])
        contents.append(
            Content(
                content_id=cid,
                size_mb=size,
                base_popularity=base_popularity,
                region_bias=region_bias,
                content_type=content_type,
            )
        )
    return contents


def _generate_nodes(config: dict) -> List[EdgeNode]:
    n = config["num_edge_nodes"]
    profile = config.get("capacity_profile") or [config["cache_capacity_mb"]] * n
    if len(profile) < n:
        profile = list(profile) + [config["cache_capacity_mb"]] * (n - len(profile))
    return [
        EdgeNode(node_id=i, region=i % config["num_regions"], cache_capacity_mb=int(profile[i]))
        for i in range(n)
    ]


def _seed_initial_distribution(
    contents: List[Content], nodes: List[EdgeNode], config: dict | None = None
) -> None:
    """Original DCCC-hash seed distribution.

    Used as the starting point for *every* policy so all start with the
    same cache state. Subsequent evictions follow the policy under test.
    """
    threshold = (config or {}).get("secondary_cache_threshold", 0.25)
    for content in contents:
        primary = nodes[content.content_id % len(nodes)]
        primary.add_content(content.content_id, content.size_mb)
        if content.base_popularity > threshold:
            secondary = nodes[(content.content_id + 1) % len(nodes)]
            secondary.add_content(content.content_id, content.size_mb)


def _generate_requests(rng: random.Random, contents: List[Content], config: dict) -> List[Request]:
    region_weights = config.get("region_request_weights") or [1.0] * config["num_regions"]
    region_weight_mult = config.get("region_weight_multiplier", 2.5)
    deadline = config["strict_deadline_ms"]
    requests: List[Request] = []
    for rid in range(config["num_requests"]):
        user_region = _weighted_choice(rng, region_weights)
        weights = [
            content.base_popularity * (1 + region_weight_mult * content.region_bias[user_region])
            for content in contents
        ]
        selected = contents[_weighted_choice(rng, weights)]
        requests.append(
            Request(
                request_id=rid,
                user_region=user_region,
                content_id=selected.content_id,
                deadline_ms=deadline,
            )
        )
    return requests


def build_world(seed: int, config: dict) -> World:
    """Construct a reproducible world (same seed → identical world).

    All policies for a given seed see the *same* request stream and the
    *same* initial cache distribution. Cross-policy variance therefore
    reflects routing/caching decisions, not workload sampling.
    """
    rng = random.Random(seed)
    contents = _generate_contents(rng, config)
    nodes = _generate_nodes(config)
    _seed_initial_distribution(contents, nodes, config)
    requests = _generate_requests(rng, contents, config)
    content_map = {c.content_id: c for c in contents}
    return World(contents=contents, nodes=nodes, requests=requests, content_map=content_map)


def edge_latency_ms(node: EdgeNode, content: Content, user_region: int, config: dict) -> float:
    transfer = config["transfer_ms_per_mb"] * content.size_mb
    load_delay = config.get("load_delay_ms", 10.0) * node.load
    region_delay = abs(node.region - user_region) * config.get("region_delay_ms_per_hop", 4.0)
    return config["base_edge_latency_ms"] + transfer + load_delay + region_delay


def _select_node(
    policy: PolicyName,
    candidate_nodes: List[EdgeNode],
    all_nodes: List[EdgeNode],
    content: Content,
    user_region: int,
    popularity: float,
    d1_ms: float,
    config: dict,
) -> Tuple[EdgeNode, float]:
    """Return (selected_node, best_score). Score is 0 for non-scoring policies."""
    if policy == DCCC_HASH_POLICY:
        return all_nodes[content.content_id % len(all_nodes)], 0.0

    if policy == DCCC_NEAREST_POLICY:
        pool = candidate_nodes or all_nodes
        return min(pool, key=lambda n: edge_latency_ms(n, content, user_region, config)), 0.0

    if policy in _DEADLINE_FAMILY:
        drop_overlap = policy == NO_OVERLAP_POLICY
        best_node: Optional[EdgeNode] = None
        best_score = float("-inf")
        for node in all_nodes:
            latency = edge_latency_ms(node, content, user_region, config)
            overlap = common_overlap(node, all_nodes)
            score = deadline_aware_score(
                node=node,
                content=content,
                user_region=user_region,
                popularity=popularity,
                latency_ms=latency,
                d1_ms=d1_ms,
                common_overlap_ratio=overlap,
                drop_overlap=drop_overlap,
                config=config,
            )
            if score > best_score:
                best_score = score
                best_node = node
        assert best_node is not None
        return best_node, best_score

    # LRU / LFU baselines: choose nearest reachable node, decide cache replacement later.
    pool = candidate_nodes or all_nodes
    return min(pool, key=lambda n: edge_latency_ms(n, content, user_region, config)), 0.0


def _cooperative_lookup(
    nodes: List[EdgeNode], content: Content, user_region: int, config: dict
) -> Tuple[Optional[EdgeNode], float]:
    holders = [n for n in nodes if n.has_content(content.content_id)]
    if not holders:
        return None, float("inf")
    best = min(holders, key=lambda n: edge_latency_ms(n, content, user_region, config))
    coop_latency = config["coop_path_latency_ms"] + edge_latency_ms(best, content, user_region, config) / 2.0
    return best, coop_latency


def _insert_after_serve(
    policy: PolicyName,
    cache_target: EdgeNode,
    content: Content,
    popularity: float,
    time_step: int,
    hit_type: str,
    config: dict | None = None,
) -> int:
    if policy == LRU_POLICY:
        return cache_target.add_lru(content, time_step)
    if policy == LFU_POLICY:
        return cache_target.add_lfu(content, time_step)
    # All DCCC/deadline policies use ICE-style insertion, gated to avoid
    # blindly replicating already-hot edge hits.
    ice_gate = (config or {}).get("ice_cache_gate", 0.6)
    should_cache = hit_type != "edge" or popularity * math.log1p(content.size_mb) > ice_gate
    if not should_cache:
        return 0
    return cache_target.add_ice_style(content, time_step, popularity)


def run_simulation(
    policy: PolicyName,
    seed: int,
    config: dict,
    world: Optional[World] = None,
    drl_hook: Optional[Callable] = None,
    log_callback: Optional[Callable[[dict], None]] = None,
) -> SimMetrics:
    """Run one (policy, seed) replication.

    ``world`` lets the caller reuse a built world across policies for a
    fixed seed to make comparisons paired. ``drl_hook`` is called for the
    DEADLINE_DCCC policy on every request with the state dict; it must
    return one of {0,1,2}: SERVE_LOCAL / FORWARD_DCCC / DEFAULT_POLICY.
    """
    if world is None:
        world = build_world(seed, config)
    # Use a separate RNG for the run so simulator stochastics (load
    # noise, partial accessibility) are reproducible *given* the world.
    rng = random.Random(seed * 1_000_003 + 17)

    nodes = world.nodes
    content_map = world.content_map
    num_regions = config["num_regions"]
    d1 = config["strict_deadline_ms"]
    d2 = config["cooperative_deadline_ms"]
    load_inc = config["load_increment_per_serve"]
    load_decay = config["load_decay"]
    energy_local = config["energy_local_factor"]
    energy_cloud = config["energy_cloud_factor"]

    metrics = SimMetrics(
        policy=policy,
        seed=seed,
        requests=0,
        edge_hits=0,
        coop_hits=0,
        cloud_fetches=0,
        deadline_satisfied=0,
        cache_evictions=0,
        serves_per_node=[0] * len(nodes),
    )

    request_counts: Dict[int, int] = {c.content_id: 0 for c in world.contents}

    for time_step, request in enumerate(world.requests):
        content = content_map[request.content_id]
        request_counts[content.content_id] += 1
        popularity = newton_popularity(content, time_step, request_counts[content.content_id], config=config)

        # Decay loads. Add tiny stochastic jitter so two policies with
        # identical routing don't become bit-identical (research figures
        # need a touch of variance to plot error bars over).
        load_noise_amplitude = config.get("load_noise_amplitude", 0.005)
        for node in nodes:
            node.load = max(0.0, node.load * load_decay + rng.random() * load_noise_amplitude)

        cross_region_prob = config.get("cross_region_prob", 0.35)
        candidate_nodes = [
            n for n in nodes if n.region == request.user_region or rng.random() < cross_region_prob
        ]
        if not candidate_nodes:
            candidate_nodes = [min(nodes, key=lambda n: abs(n.region - request.user_region))]

        selected_node, _ = _select_node(
            policy=policy,
            candidate_nodes=candidate_nodes,
            all_nodes=nodes,
            content=content,
            user_region=request.user_region,
            popularity=popularity,
            d1_ms=d1,
            config=config,
        )

        selected_latency = edge_latency_ms(selected_node, content, request.user_region, config)
        serving_node = selected_node
        hit_type = "cloud"
        latency = 0.0

        # DRL hook overrides routing for DEADLINE_DCCC only.
        drl_action = None
        if policy == DRL_POLICY and drl_hook is not None:
            drl_action = drl_hook(
                node=selected_node,
                content=content,
                latency=selected_latency,
                deadline_ms=request.deadline_ms,
                candidate_nodes=candidate_nodes,
                all_nodes=nodes,
            )

        # Routing decision tree
        if drl_action == 1:
            # FORWARD_DCCC: skip local, go straight to cooperative
            coop_node, coop_latency = _cooperative_lookup(nodes, content, request.user_region, config)
            if coop_node is not None and coop_latency <= d2:
                latency, hit_type, serving_node = coop_latency, "cooperative", coop_node
            else:
                latency = config["cloud_latency_ms"] + config["cloud_transfer_ms_per_mb"] * content.size_mb
                hit_type = "cloud"
        else:
            if selected_node.has_content(content.content_id) and selected_latency <= d1:
                latency, hit_type = selected_latency, "edge"
            else:
                coop_node, coop_latency = _cooperative_lookup(nodes, content, request.user_region, config)
                if policy in _DCCC_FAMILY and coop_node is not None and coop_latency <= d2:
                    latency, hit_type, serving_node = coop_latency, "cooperative", coop_node
                else:
                    latency = config["cloud_latency_ms"] + config["cloud_transfer_ms_per_mb"] * content.size_mb
                    hit_type = "cloud"

        # Insertion: cache on whichever node actually served the request.
        cache_target = serving_node if hit_type == "cooperative" else selected_node
        metrics.cache_evictions += _insert_after_serve(
            policy=policy,
            cache_target=cache_target,
            content=content,
            popularity=popularity,
            time_step=time_step,
            hit_type=hit_type,
            config=config,
        )

        max_node_load = config.get("max_node_load", 1.5)
        if hit_type != "cloud":
            serving_node.load = min(max_node_load, serving_node.load + load_inc)
            metrics.serves_per_node[serving_node.node_id] += 1

        if hit_type == "edge":
            metrics.edge_hits += 1
        elif hit_type == "cooperative":
            metrics.coop_hits += 1
        else:
            metrics.cloud_fetches += 1

        deadline = d1 if hit_type == "edge" else d2
        if latency <= deadline:
            metrics.deadline_satisfied += 1

        energy_factor = energy_local if hit_type in ("edge", "cooperative") else energy_cloud

        metrics.latencies.append(latency)
        metrics.energy_costs.append(energy_factor * content.size_mb)
        metrics.requests += 1

        if log_callback is not None:
            log_callback(
                {
                    "request_id": request.request_id,
                    "policy": policy,
                    "hit_type": hit_type,
                    "selected_node": selected_node.node_id,
                    "serving_node": serving_node.node_id if hit_type != "cloud" else "cloud",
                    "latency_ms": latency,
                    "deadline_met": latency <= deadline,
                    "drl_action": drl_action,
                }
            )

    return metrics
