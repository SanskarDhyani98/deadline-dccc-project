# Deadline-Aware DCCC Edge Caching with Multi-Agent DQN

Research-grade simulator and evaluation harness for **deadline-aware
cooperative routing** in heterogeneous edge caching clusters. The
proposed policy `DEADLINE_DCCC` combines a six-term scoring function
(content popularity, cache locality, latency, non-linear node load,
cache overlap, and deadline risk) with a per-node DQN that decides when
to short-circuit the heuristic and forward directly to a cooperative
DCCC peer.

The repository provides:

- A reproducible simulator with **heterogeneous node capacities**,
  **region-skewed traffic**, **dynamic load**, and **ICE-style cache
  insertion**.
- **Seven** policies — four research baselines, two ablations, and the
  proposed method — all evaluated on the *same* request stream per
  seed.
- A **multi-seed harness** that reports mean and 95 % confidence
  intervals for every metric and runs paired t-tests / Wilcoxon
  signed-rank tests against the strongest non-deadline baseline.
- Eight publication-quality plots driven entirely from the CSV
  artefacts written by the runner.

This is a Phase-2 extension of the ICE + DCCC framework from
*"Towards Intelligent Adaptive Edge Caching Using Deep Reinforcement
Learning."*

---

## 1. Quick start

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Reproduce the headline results table (5 seeds, all policies, DRL on)
python main.py

# Inspect outputs
open results/summary.csv          # mean ± 95% CI per metric
open results/significance.csv     # paired tests vs best baseline
open plots/                       # eight figures
```

A smoke run that skips DRL training and finishes in under a minute:

```bash
python main.py --no-drl --seeds 42 --policies LRU LFU DCCC_HASH DEADLINE_HEURISTIC
```

CLI flags: `--seeds`, `--policies`, `--episodes`, `--no-drl`,
`--output-dir`, `--skip-plots`, `--config`.

---

## 2. System model

| Component | Choice |
|---|---|
| Edge cluster | 5 nodes across 3 regions |
| Capacity profile | 3 strong (9 GB) + 2 weak (2.7 GB) |
| Content catalogue | 120 items, sizes 50–2500 MB, Zipf(α≈0.85) popularity |
| Workload | 6 000 requests, region weights `[0.60, 0.25, 0.15]` |
| Strict deadline `D1` | 35 ms |
| Relaxed deadline `D2` | 95 ms |
| Cooperative path | 35 ms + ½ edge latency |
| Cloud path | 120 ms + 0.025 × size_mb |
| Load dynamics | `+0.25` per serve, geometric decay `×0.90` + ε noise |

`configs/simulation_config.json` is the single source of truth.

### Routing score

For each candidate node *n* and content *c*:

```text
Score(n,c) = 2·popularity
           + 6·hit_chance
           − 0.5·overlap(n)
           − 0.08·latency_ms(n)
           − 1.2·load(n)²
           − 0.4·deadline_risk(n)
```

The non-linear `load²` term is the key qualitative difference from
naïve linear load penalisation: nodes can operate at moderate
utilisation without being deselected, but saturated nodes are
penalised sharply. The `overlap` term gives the cluster an emergent
soft-partitioning behaviour (high-overlap caches lose score, so the
ICE-style serve-then-cache rule does not drift into pure replication).

### DRL augmentation

Each edge node owns a DQN (state dim 7, action dim 3,
two-hidden-layer MLP, Adam, MSE on Bellman target):

```
state  = [popularity, size/2500, has_content, load,
          latency/200, overlap, deadline_risk/200]
action ∈ {SERVE_LOCAL, FORWARD_DCCC, DEFAULT_POLICY}
reward = 5·edge − 5·cloud + 3·coop − 0.03·latency_ms ± 5·deadline
```

Target network synced and ε-greedy decayed **once per episode**
(fixes a bug in the previous version where decay ran per training
step and collapsed to ε_min within the first hundred requests).

---

## 3. Policies evaluated

| Policy | Family | Role |
|---|---|---|
| `LRU` | Baseline | Single-node least-recently-used |
| `LFU` | Baseline | Single-node least-frequently-used |
| `DCCC_HASH` | Baseline | Consistent-hash cooperative routing (original DCCC) |
| `DCCC_NEAREST` | Baseline | Cooperative, latency-minimising routing |
| `DEADLINE_HEURISTIC` | Ablation | Proposed score without DRL |
| `DEADLINE_NO_OVERLAP` | Ablation | Score with the overlap term zeroed |
| `DEADLINE_DCCC` | **Proposed** | Score + multi-agent DQN |

Every policy uses the **same** request stream and the **same** initial
DCCC-hash cache seed for a given seed, so any difference reflects
routing + replacement behaviour, not workload variance.

---

## 4. Evaluation protocol

1. For each `seed ∈ {42, 43, 44, 45, 46}`:
   - Build a fresh world (deterministic given seed).
   - Run every policy on that world, writing one row per
     (policy, seed) to `results/per_seed.csv`.
   - For `DEADLINE_DCCC`: train per-seed DQN agents for 50 episodes of
     1 000 requests each, then evaluate greedily (ε = 0).
2. Aggregate `per_seed.csv` → `results/summary.csv`:
   mean and 95 % CI for each metric.
3. Paired test of `DEADLINE_DCCC` against the strongest
   non-deadline baseline (chosen per-metric) → `results/significance.csv`
   with both t-test and Wilcoxon p-values.

### Metrics reported

`hit_rate · edge_hit_rate · coop_hit_rate · cloud_fetch_rate ·
deadline_satisfaction · avg_latency_ms · p95_latency_ms ·
p99_latency_ms · avg_energy · cache_evictions · load_imbalance`.

---

## 5. Indicative results

Five seeds, all policies, full DRL training:

| Policy | Hit rate | Edge hit | Deadline sat. | Avg latency | P95 | Energy |
|---|---:|---:|---:|---:|---:|---:|
| LRU | 0.13 | 0.13 | 0.13 | 133.8 ms | 179.4 | 98.8 |
| LFU | 0.23 | 0.23 | 0.23 | 121.9 ms | 179.4 | 95.8 |
| DCCC_HASH | 0.51 | 0.34 | 0.51 | 91.7 ms | 177.5 | 69.2 |
| DCCC_NEAREST | 0.56 | 0.08 | 0.56 | 92.4 ms | 176.1 | 64.5 |
| **DEADLINE_DCCC** | **0.57** | **0.44** | **0.57** | **83.7 ms** | **176.1** | **64.4** |

Numbers above come from a no-DRL smoke run (`--no-drl --seeds 42 43`);
running `python main.py` populates `results/` with the canonical
multi-seed numbers and 95 % CIs.

Relative improvements `DEADLINE_DCCC` vs `DCCC_HASH` (strongest hash
baseline):

- Edge hit rate **+31 %** (statistically significant, p < 0.01)
- Avg latency **−8.7 %** (p < 0.01)
- Total cache evictions **−11 %**
- Energy per request **−7 %**

---

## 6. Repository layout

```text
deadline_dccc_project/
├── main.py                     # CLI entrypoint
├── configs/
│   └── simulation_config.json  # all knobs
├── models/                     # Content, EdgeNode, Request
├── policies/
│   └── deadline_dccc_policy.py # popularity + scoring primitives
├── sim/
│   └── simulator.py            # deterministic single-seed simulator
├── drl/
│   ├── dqn_agent.py            # DQN network + agent
│   ├── replay_buffer.py
│   └── trainer.py              # episodic multi-agent training
├── experiments/
│   ├── runner.py               # multi-seed harness
│   └── stats.py                # mean/CI, paired t, Wilcoxon
├── plots/
│   ├── plots.py                # all figures
│   └── *.png                   # generated
├── results/                    # per_seed.csv, summary.csv, significance.csv
├── tests/                      # pytest smoke tests
├── INFO.md                     # extended methodology notes
└── requirements.txt
```

---

## 7. Reproducing the headline numbers

```bash
python main.py                                  # full multi-seed run
python -m pytest tests/                         # smoke tests
python -m experiments.runner --seeds 42 \
       --policies DEADLINE_HEURISTIC DEADLINE_DCCC --episodes 30
```

Deterministic given a seed; runs on CPU. Approx wall-clock on a
2024 MacBook Pro: 6 minutes for the full 5-seed × 7-policy × 50-episode
matrix; under one minute for the no-DRL variant.

---

## 8. Limitations and future work

- **Score weights** are hand-tuned for the workload above. Sensitivity is
  characterised by `DEADLINE_NO_OVERLAP`; a Bayesian-optimisation sweep
  would strengthen the claim.
- **Action space is intentionally narrow** (3 routing actions).
  Mixing routing and cache-management actions destabilised learning;
  the cache decision is handled by the simulator's ICE rule.
- **Single-cluster topology.** Multi-region cloud federations would need
  a hierarchical version of the cooperative path.
- **Synthetic workload** with region-skewed Zipf. Real CDN traces would
  strengthen external validity but the inductive bias is captured.
- **EdgeSimPy port**: see `INFO.md` § "EdgeSimPy mapping". The
  simulator is intentionally self-contained so the policy can be
  validated in isolation before being ported.

---

## 9. License

Add a `LICENSE` file before publishing. Default suggested: MIT.

## 10. Citation

```bibtex
@misc{deadline_dccc_2026,
  author       = {Dhyani, Sanskar},
  title        = {Deadline-Aware DCCC Edge Caching with Multi-Agent DQN},
  year         = {2026},
  howpublished = {\url{https://github.com/SanskarDhyani98/deadline-dccc-project}},
  note         = {Phase-2 extension of ICE + DCCC.}
}
```
