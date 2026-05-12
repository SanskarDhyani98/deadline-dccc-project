# Deadline-Aware DCCC Edge Caching using Multi-Agent DRL

This project implements a Phase-2 extension of the research paper **"Towards Intelligent Adaptive Edge Caching Using Deep Reinforcement Learning."**

The base paper proposes:

- **ICE**: Intelligent Caching at the Edge using DQN.
- **DCCC**: Distributed Cooperative Caching Cluster for multi-node cooperative edge caching.

This project extends the original ICE+DCCC framework by adding:

- Deadline-aware request routing
- Decentralized cooperative node selection
- POMDP-based multi-node formulation
- Multi-agent Deep Reinforcement Learning
- Edge/cooperative/cloud service decision logic
- Baseline comparison against LRU, LFU, and DCCC hashing

---

## Project Objective

The goal is to simulate an intelligent edge caching system where users request content from distributed edge nodes.

Each edge node has:

- limited cache capacity
- different cached contents
- dynamic load
- region-based proximity
- partially overlapping content with other nodes

For each user request, the system decides whether the request should be served by:

1. the selected local edge node,
2. a cooperative DCCC node,
3. or the cloud.

The main objective is to improve:

- deadline satisfaction
- average latency
- cache hit rate
- cooperative cache usage
- cloud fetch reduction

---

## Motivation

Traditional cloud-based systems suffer from high latency, network congestion, and high energy consumption. Edge caching reduces delay by storing popular content closer to users.

However, edge nodes have limited storage and dynamic workloads. The original DCCC framework distributes content across cooperative edge nodes, but routing is mainly based on hashing and load balancing.

This project adds a deadline-aware decision layer so that the system chooses the node most likely to satisfy a request within a given deadline.

---

## System Architecture

```text
Users
  ↓
Candidate Edge Nodes
  ↓
Deadline-Aware DCCC Decision Layer
  ↓
Local Edge Node / Cooperative DCCC Node / Cloud
```

Each edge node stores a subset of the global content catalogue.

Some contents overlap between nodes, creating a common cooperative cache space.

---

## Key Concepts

### Edge Caching

Edge caching stores frequently requested content closer to users to reduce latency and cloud dependency.

### ICE

ICE is the base intelligent caching method from the paper. It uses DQN to decide whether to:

- add content to cache
- delete content from cache
- keep cache unchanged

### DCCC

DCCC stands for Distributed Cooperative Caching Cluster. It allows multiple edge nodes to cooperate and serve content collectively.

### Deadline-Aware Routing

Each request has a strict deadline:

```text
d1 = strict local edge deadline
```

If the local edge cannot satisfy the request, the request may be forwarded to cooperative DCCC nodes under:

```text
d2 = relaxed cooperative deadline
```

### POMDP

The system is modeled as a POMDP because each node does not know the complete global state of all other nodes.

A node only observes local information such as:

- local cache status
- content popularity
- content size
- node load
- latency
- deadline risk
- overlap with nearby nodes

### Multi-Agent DRL

Each edge node acts as an independent learning agent. The DQN agent learns whether to:

- serve locally
- forward to cooperative DCCC
- use default deadline-aware fallback policy

---

## Implemented Policies

The simulator compares four policies:

| Policy | Description |
|---|---|
| LRU | Least Recently Used baseline |
| LFU | Least Frequently Used baseline |
| DCCC_HASH | Original DCCC-style hash routing |
| DEADLINE_DCCC | Proposed deadline-aware DRL-based policy |

---

## Deadline-Aware Node Score

For the heuristic deadline-aware node selection, each candidate node is scored using:

```text
Score_i(c) =
2.0 * HitProbability
+ 1.5 * Popularity
- 0.18 * Latency
- 4.0 * NodeLoad
- 2.5 * DeadlineRisk
- 0.8 * Overlap
```

Where:

- `HitProbability` indicates whether the node likely has the content.
- `Popularity` represents demand for the content.
- `Latency` is the response time from user to node.
- `NodeLoad` represents node congestion.
- `DeadlineRisk` penalizes nodes that may miss the deadline.
- `Overlap` prevents excessive duplicated caching.

---

## DRL State Space

The DQN state contains 7 features:

```text
[
  popularity,
  content_size_normalized,
  cache_hit,
  node_load,
  latency_normalized,
  overlap,
  deadline_risk_normalized
]
```

This represents the local observation of an edge node.

---

## DRL Action Space

The final DRL action space contains 3 routing actions:

| Action ID | Action | Meaning |
|---:|---|---|
| 0 | SERVE_LOCAL | Try selected edge node first |
| 1 | FORWARD_DCCC | Forward directly to cooperative DCCC |
| 2 | DEFAULT_POLICY | Use normal deadline-aware fallback |

Earlier versions used 5 actions including cache, evict, and keep, but that mixed routing and cache-management decisions. The final design uses routing-only actions for cleaner learning.

---

## Reward Function

The reward function rewards successful and fast serving:

```text
+5 for edge hit
+3 for cooperative hit
-5 for cloud fetch
-0.03 * latency
+5 if deadline met
-5 if deadline missed
```

This encourages:

- cache hits
- cooperative hits
- low latency
- deadline satisfaction
- reduced cloud fetches

---

## Simulation Workflow

For each request:

1. Generate a user request from a region.
2. Select candidate edge nodes.
3. Calculate latency, load, deadline risk, and cache status.
4. Select the best node using policy logic.
5. For `DEADLINE_DCCC`, the DRL agent chooses an action.
6. Serve from local edge, cooperative DCCC, or cloud.
7. Record metrics.
8. Repeat for all requests.

---

## Training Workflow

For the proposed DRL policy:

1. Create one DQN agent per edge node.
2. Train agents over multiple episodes.
3. Store experiences in replay buffers.
4. Use epsilon-greedy exploration.
5. Decay epsilon once per episode.
6. Evaluate using mostly greedy learned behavior.

Example training output:

```text
Episode 1/50  | Reward=528.12  | Avg Loss=91.66 | Epsilon=0.9700
Episode 10/50 | Reward=1532.45 | Avg Loss=27.29 | Epsilon=0.7374
Episode 30/50 | Reward=1191.32 | Avg Loss=13.87 | Epsilon=0.4010
Episode 50/50 | Reward=1061.90 | Avg Loss=12.41 | Epsilon=0.2181
```

This shows that loss decreases and the model learns more stable actions over time.

---

## Final Results

Final policy comparison:

| Policy | Hit Rate | Edge Hit Rate | Cloud Fetch Rate | Deadline Satisfaction | Average Latency |
|---|---:|---:|---:|---:|---:|
| LRU | 0.7088 | 0.2180 | 0.2912 | 0.6448 | 69.83 ms |
| LFU | 0.7070 | 0.2383 | 0.2930 | 0.5745 | 71.40 ms |
| DCCC_HASH | 0.6705 | 0.2705 | 0.3295 | 0.5772 | 73.65 ms |
| DEADLINE_DCCC | 0.7082 | 0.3188 | 0.2918 | 0.7035 | 67.35 ms |

---

## Result Interpretation

The proposed `DEADLINE_DCCC` policy achieved:

- highest deadline satisfaction
- lowest average latency
- highest edge hit rate
- competitive overall hit rate

This validates the main project hypothesis:

```text
Deadline-aware cooperative routing improves real-time QoS in edge caching systems.
```

Although the hit rate is similar to LRU and LFU, the proposed policy performs better for latency-sensitive scenarios because it prioritizes deadline satisfaction and fast edge/cooperative serving.

---

## Project Structure

```text
deadline-dccc-project/
│
├── main.py
│
├── configs/
│   └── simulation_config.json
│
├── models/
│   ├── content.py
│   ├── edge_node.py
│   └── request.py
│
├── drl/
│   ├── dqn_agent.py
│   └── replay_buffer.py
│
├── results/
│   ├── metrics.csv
│   ├── policy_comparison.csv
│   └── *_log.txt
│
├── plots/
│   ├── performance_rates.png
│   └── latency_breakdown.png
│
├── requirements.txt
├── .gitignore
└── README.md
```

---

## How to Run

### 1. Clone the repository

```bash
git clone https://github.com/SanskarDhyani98/deadline-dccc-project.git
cd deadline-dccc-project
```

### 2. Create virtual environment

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

If `requirements.txt` is not available, install manually:

```bash
pip install numpy pandas matplotlib torch networkx tqdm
```

### 4. Run simulation

```bash
python3 main.py
```

---

## Output Files

After running, the project generates:

```text
results/metrics.csv
results/policy_comparison.csv
plots/performance_rates.png
plots/latency_breakdown.png
results/DEADLINE_DCCC_log.txt
```

---

## Configuration

Main parameters are stored in:

```text
configs/simulation_config.json
```

Example:

```json
{
  "num_edge_nodes": 5,
  "num_regions": 3,
  "num_contents": 120,
  "cache_capacity_mb": 9000,
  "strict_deadline_ms": 30,
  "cooperative_deadline_ms": 95,
  "base_edge_latency_ms": 8,
  "cloud_latency_ms": 120,
  "num_requests": 6000,
  "num_episodes": 50,
  "train_requests_per_episode": 1000,
  "eval_requests": 6000,
  "drl_state_size": 7,
  "drl_action_size": 3,
  "drl_learning_rate": 0.001,
  "drl_gamma": 0.95,
  "drl_epsilon": 1.0,
  "drl_epsilon_min": 0.05,
  "drl_epsilon_decay": 0.97,
  "drl_batch_size": 64,
  "content_types": [
    "video",
    "image",
    "iot",
    "audio"
  ],
  "policies": [
    "LRU",
    "LFU",
    "DCCC_HASH",
    "DEADLINE_DCCC"
  ]
}
```

---

## Important Design Decisions

### 1. Separating Hit Rate and Deadline Satisfaction

A cache hit may still miss the deadline. Therefore, hit rate and deadline satisfaction were treated as separate metrics.

### 2. Using POMDP

Each node only observes local information, not the entire network state. Therefore, the system is naturally partially observable.

### 3. Moving from 5 DRL Actions to 3 DRL Actions

Initial action space included cache, evict, and keep. This confused routing and cache-management decisions. The final DRL action space focuses only on routing decisions.

### 4. Episodic Training

Single-run DRL was unstable. Episodic training allowed replay memory accumulation, epsilon decay, and more stable policy learning.

### 5. Epsilon Decay Once Per Episode

Earlier, epsilon decayed too quickly when updated inside every training step. It was corrected to decay once per episode.

---

## Limitations

- The current simulator is a custom EdgeSimPy-style simulation and is not yet fully integrated with native EdgeSimPy APIs.
- Cache eviction is simplified.
- DRL state transition currently uses simplified next-state modeling.
- Network latency is simulated using a formula rather than a real topology.
- Results may vary due to random initialization.

---

## Future Work

- Full EdgeSimPy topology integration
- Advanced cache replacement with learned eviction
- Separate DRL agents for routing and caching
- Actor-Critic or Multi-Agent DQN
- More realistic mobility model
- Energy-aware optimization
- Larger node topology
- Real workload traces

---

## Conclusion

This project successfully extends ICE+DCCC with a deadline-aware cooperative routing mechanism using multi-agent DRL.

The proposed `DEADLINE_DCCC` method achieves the best deadline satisfaction and lowest latency among tested policies, while maintaining competitive hit rate.

This demonstrates that deadline-aware DRL-based cooperative routing can improve real-time QoS in decentralized edge caching systems.
