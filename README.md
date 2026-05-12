# Deadline-aware DCCC edge caching simulation

Simulation of edge nodes, cooperative DCCC-style lookups, strict deadlines, and optional **DQN-based** decisions when using the `DEADLINE_DCCC` policy.

## Features

- Multiple caching / node-selection policies: LRU, LFU, content-hash (DCCC_HASH), and deadline-aware scoring with **DRL** (`DEADLINE_DCCC`)
- Latency, hit-rate, and deadline-satisfaction metrics with CSV export and matplotlib plots

## Quick start

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python main.py
```

Detailed installation (PyTorch notes, verification) is in **[setup/README.md](setup/README.md)**.

## Project layout

| Path | Purpose |
|------|--------|
| `main.py` | Simulation loop, policies, DRL integration, metrics, plotting |
| `configs/simulation_config.json` | Topology, deadlines, request count, DRL hyperparameters |
| `models/` | `Content`, `EdgeNode`, `Request` |
| `drl/` | DQN agent and replay buffer |
| `test_drl.py` | Minimal smoke test for the DQN agent |

## Configuration

Adjust `configs/simulation_config.json` (e.g. `num_requests`, `cache_capacity_mb`, `drl_epsilon`, `drl_batch_size`) without changing code.

## Outputs

- `results/` — metrics CSV, policy comparison, optional per-policy logs when you run `main.py`
- `plots/` — bar charts for rates and latency breakdown

## License

Add a `LICENSE` file if you plan to open-source this repository.
