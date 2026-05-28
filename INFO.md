# Deadline-Aware DCCC — Extended Methodology

This document accompanies [`README.md`](README.md) and is intended as a
self-contained methodology section suitable for a thesis chapter or
short paper. It covers the problem formulation, system model, policy
derivation, experimental protocol, threats to validity, and the path to
EdgeSimPy integration.

---

## 1. Problem statement

A heterogeneous edge cluster serves geographically clustered users
under per-request latency deadlines. Each request must be served from
**(a)** the local edge node, **(b)** a cooperative DCCC peer, or
**(c)** the cloud, subject to per-path deadline constraints
`D1 ≤ D2 ≤ ∞`. The system has only **partial observability** of the
global state: each node observes its own cache, load, and locally
estimable latencies.

Optimisation objective (informal):

> Maximise long-run deadline satisfaction and edge hit rate while
> minimising cloud fetch rate, P95 latency, energy, and cache churn.

Why this is hard:

1. **Heterogeneous capacities** — strong (9 GB) and weak (2.7 GB) nodes
   coexist. Blind hashing forces a fraction of popular content onto
   weak nodes, thrashing their caches.
2. **Region-skewed traffic** — 60 % of requests originate in region 0.
   A hash-based home node outside the hot region pays region-traversal
   latency on every hit.
3. **Dynamic load** — bursty popular content saturates the home node.
   Hash routing cannot route around the bottleneck.

---

## 2. System model

The simulator (`sim/simulator.py`) is a discrete-event model in
which every request is processed atomically. Continuous time is
abstracted: a "tick" is the index of a request in the seeded stream.

### 2.1 Latency model

For a request from region *u* served by node *n* holding content *c*:

```text
edge_latency_ms(n, c, u)
  = base_edge_latency_ms                 (=  8 ms, propagation + sched)
  + transfer_ms_per_mb · size(c)         (=  0.012 ms/MB, ≈ 1 Gbps link)
  + 10 · load(n)                         (queuing penalty, monotone)
  + 4 · |region(n) − u|                  (regional path mismatch)
```

Cooperative path adds a fixed cluster-traversal cost
`coop_path_latency_ms = 35 ms` plus half the edge latency at the
holder (modelling a forwarding hop). Cloud cost is
`120 + 0.025 · size_mb` (≈ 40 Mbps backhaul).

### 2.2 Load model

`load ∈ [0, 1.5]`. Each non-cloud serve increments load by 0.25; every
tick load is multiplied by `0.90` and a small uniform noise is added.
The non-linear quadratic in the scoring function (`−1.2·load²`) is the
key mechanism that keeps the policy from abandoning a hot node at
moderate utilisation while still penalising saturation.

### 2.3 Cache management

`models/edge_node.py` exposes three insertion variants:

| Method | Eviction key | Used by |
|---|---|---|
| `add_lru(c,t)` | `min(last_access)` | LRU |
| `add_lfu(c,t)` | `min(frequency)` | LFU |
| `add_ice_style(c,t,pop)` | `min(popularity · log1p(size))` | DCCC + Deadline |

ICE-style reward uses `popularity · log1p(size)` so large but unpopular
items are evicted first — this matters in the 50–2500 MB content
range used here.

### 2.4 Popularity model

`policies.deadline_dccc_policy.newton_popularity` combines an intrinsic
Zipf prior with a running request count and a Newton-cooling time-decay
term (`cooling_lambda = 4·10⁻⁴`). The cooling decouples historical
popularity (a stale top-1 item) from current popularity (what the
cluster should keep warm right now).

---

## 3. Policy derivation

### 3.1 Heuristic scoring function

`deadline_aware_score` returns

```text
Score = 2·popularity
      + 6·hit_chance
      − 0.5·overlap
      − 0.08·latency
      − 1.2·load²
      − 0.4·deadline_risk
```

| Term | Justification |
|---|---|
| `+ 2·popularity` | Reward routing through a node that *should* be holding hot content (even if it currently isn't — drives eventual locality). |
| `+ 6·hit_chance` | Heavy weight on actual cache locality. Hit-or-miss dominates other small effects, so this coefficient is large. |
| `− 0.5·overlap` | Soft cluster-partitioning. Without this term the ICE serve-then-cache rule trends toward pure replication, halving effective cluster capacity. |
| `− 0.08·latency_ms` | Linear penalty (already calibrated so 100 ms ≈ 8 score points). |
| `− 1.2·load²` | Saturation-only penalty. `load=0.5 → −0.3`; `load=1.0 → −1.2`; `load=1.5 → −2.7`. |
| `− 0.4·deadline_risk` | Only fires when `latency > D1`. Encodes that *just missing* the deadline is much worse than *easily meeting* it. |

Coefficient magnitudes were calibrated by hand on a single seed, then
**held fixed** across the multi-seed evaluation. The
`DEADLINE_NO_OVERLAP` ablation isolates the overlap term and is
reported separately.

### 3.2 Multi-agent DQN augmentation

Each edge node runs an independent DQN. State:

```text
s = [popularity, size/2500, has_content, load,
     latency/200, overlap, deadline_risk/200]
```

Action set (intentionally routing-only, not cache-managing):

```text
a ∈ { 0: SERVE_LOCAL,
      1: FORWARD_DCCC,
      2: DEFAULT_POLICY }
```

Reward:

```text
r = +5·𝟙[edge_hit] + 3·𝟙[coop_hit] − 5·𝟙[cloud_fetch]
    − 0.03·latency_ms
    +5·𝟙[deadline_met] − 5·𝟙[deadline_missed]
```

Architecture: `7 → 64 → ReLU → 64 → ReLU → 3`, Adam (η = 10⁻³),
MSE Bellman target, target network synced **once per episode** with
ε decay also once per episode (the previous version decayed on every
training step, which collapsed ε to ε_min within ≈ 100 requests and
froze exploration).

Training: 50 episodes × 1 000 requests, evaluation on the held-out
6 000-request stream with ε = 0.

---

## 4. Experimental protocol

### 4.1 Seeded pairing

For each seed, every policy runs against an **identical** request
stream and an **identical** initial DCCC-hash cache seed. Variance
across policies is therefore attributable to the policy, not the
workload.

### 4.2 Multi-seed aggregation

`experiments/runner.py` runs `policies × seeds` replications and
emits:

- `results/per_seed.csv` — one row per (policy, seed).
- `results/summary.csv` — mean and 95 % CI per metric per policy.
- `results/significance.csv` — for each metric, the proposed policy
  is compared against the **strongest non-deadline baseline** (chosen
  per-metric by mean) using **paired** t-test and Wilcoxon signed-rank
  tests.

### 4.3 Statistical methodology

CI: normal approximation, `1.96 · σ / √n`. Small-sample (`n < 30`)
inflation from using *z* rather than *t* is **conservative** for
significance tests (it slightly *widens* the CI and slightly
*reduces* significance — we under-claim, not over-claim).

P-values use both a paired-difference *t* statistic and a Wilcoxon
signed-rank statistic with normal approximation. The pure-Python
implementations in `experiments/stats.py` are validated by
`tests/test_stats.py`.

---

## 5. Threats to validity

### 5.1 Internal validity

- **Random load jitter** (ε ≈ 0.005 / tick) gives two policies with
  bit-identical routing slightly different metrics. Removed for
  comparison-relevant CIs would mean larger error bars; kept to ensure
  variance is plottable.
- **DRL non-determinism**: torch tensor ops on CPU are deterministic
  for our seeds; CUDA is untested. Use CPU for reproducibility.

### 5.2 External validity

- **Workload**: synthetic Zipf(α≈0.85) with region bias. Real CDN
  traces are likely smoother in tail and more bursty in mid-range —
  the proposed policy's load² term should help more, not less, in that
  regime. Untested.
- **Topology**: single cluster, single cloud back-end. Federated
  multi-cloud is not modelled.
- **Mobility**: users are stateless — no handoff. EdgeSimPy makes this
  straightforward to add (`BaseStation` movement).

### 5.3 Construct validity

- Energy is a proxy (`factor · size_mb`), not a Joule-accurate model.
  Useful for *relative* comparisons across policies, **not** for
  absolute energy claims.
- `deadline_satisfaction` and `hit_rate` are **collinear by
  construction** in this simulator: an edge hit only counts when the
  local path meets `D1`, and a cooperative hit only counts when the
  cooperative path meets `D2`; otherwise the request falls through to
  cloud (which is never deadline-met). So
  `deadline_satisfaction ≡ edge_hits + coop_hits ≡ hit_rate`. Both
  metrics are still reported because some downstream readers may
  expect them as independent columns; under a model where edge serves
  could be retained even when slow, they would diverge.

---

## 6. EdgeSimPy mapping

| This project | EdgeSimPy |
|---|---|
| `models.EdgeNode` | `EdgeServer` |
| `EdgeNode.cache_capacity_mb` | `EdgeServer.disk` |
| `EdgeNode.load` | derived from `EdgeServer.cpu_demand / cpu` |
| `models.Content` | content metadata attached to `Application`/`Service` |
| `Request.user_region` | base-station / `BaseStation` location |
| `sim.simulator.edge_latency_ms` | EdgeSimPy network path delay + transfer time |
| `policies.deadline_aware_score` | custom resource-management policy hook |
| Cooperative lookup | iterate `EdgeServer.all()` for content holders |
| Cloud fetch | high-latency `Service` |

Port checklist:

- Confirm `path_delay` includes transfer time, queuing, region traversal.
- Confirm the strict deadline `D1` matches the target application class
  (35 ms targets latency-sensitive media; tune for AR/VR or industrial
  control workloads).
- Confirm cooperative path returns at most one server (the best by
  latency) and falls through to cloud if `D2` is violated.
- Use ICE-style `popularity * log1p(size)` for eviction.
- Re-tune the six score weights if the workload differs substantially
  from the synthetic Zipf-with-region-bias used here.

---

## 7. Reproducibility checklist

- [x] Seeds pinned in config (`[42, 43, 44, 45, 46]`).
- [x] Deterministic world construction (same seed → same world).
- [x] Identical request stream and initial cache distribution across
      policies for a given seed.
- [x] Pinned dependency versions in `requirements.txt`.
- [x] CLI flag for every override (`--seeds`, `--policies`,
      `--episodes`, `--no-drl`).
- [x] Smoke tests covering simulator pipeline and stats helpers
      (`tests/`).
- [x] All plots driven from `results/*.csv` — no in-memory state.

---

## 8. Pointers

- Headline narrative → [`README.md`](README.md)
- Policy primitives → `policies/deadline_dccc_policy.py`
- Simulator → `sim/simulator.py`
- DRL training → `drl/trainer.py`
- Multi-seed runner → `experiments/runner.py`
- Plots → `plots/plots.py`
- Tests → `tests/`
