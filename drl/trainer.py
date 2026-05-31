"""Episodic multi-agent DQN training for the DEADLINE_DCCC policy.

One agent per edge node observes its own local state for every request
routed to it, regardless of which node ends up actually serving. This
gives each agent enough on-policy data without sharing weights.

The trainer is decoupled from the simulator: it calls
``sim.simulator.run_simulation`` for the *training* episodes with the
agents wired in as ``drl_hook``. It records reward and loss per episode
so the learning curve can be plotted.
"""

from __future__ import annotations

import math
import os
import random
from dataclasses import dataclass, field
from typing import Dict, List

from drl.dqn_agent import DQNAgent
from policies.deadline_dccc_policy import common_overlap


@dataclass
class TrainingHistory:
    episode_reward: List[float] = field(default_factory=list)
    episode_loss: List[float] = field(default_factory=list)
    episode_epsilon: List[float] = field(default_factory=list)


def build_drl_state(
    node,
    content,
    candidate_nodes,
    latency_ms: float,
    deadline_ms: float,
) -> List[float]:
    """7-dim local observation used by every agent."""
    return [
        content.base_popularity,
        content.size_mb / 2500.0,
        1.0 if node.has_content(content.content_id) else 0.0,
        node.load,
        latency_ms / 200.0,
        common_overlap(node, candidate_nodes),
        max(0.0, latency_ms - deadline_ms) / 200.0,
    ]


def make_agents(node_ids: List[int], config: dict) -> Dict[int, DQNAgent]:
    return {
        nid: DQNAgent(
            state_size=config["drl_state_size"],
            action_size=config["drl_action_size"],
            learning_rate=config["drl_learning_rate"],
            gamma=config["drl_gamma"],
            epsilon=config["drl_epsilon"],
            epsilon_min=config["drl_epsilon_min"],
            epsilon_decay=config["drl_epsilon_decay"],
            batch_size=config["drl_batch_size"],
            hidden_size=config.get("drl_hidden_size", 64),
        )
        for nid in node_ids
    }


def reward_for(hit_type: str, latency: float, deadline_met: bool, config=None) -> float:
    cfg = config or {}
    r_edge = cfg.get("reward_edge_hit", 5.0)
    r_coop = cfg.get("reward_coop_hit", 3.0)
    r_cloud = cfg.get("reward_cloud_penalty", 5.0)
    r_lat_coef = cfg.get("reward_latency_coef", 0.03)
    r_deadline = cfg.get("reward_deadline_bonus", 5.0)

    reward = 0.0
    if hit_type == "edge":
        reward += r_edge
    elif hit_type == "cooperative":
        reward += r_coop
    elif hit_type == "cloud":
        reward -= r_cloud
    reward -= r_lat_coef * latency
    reward += r_deadline if deadline_met else -r_deadline
    return reward


def train_drl_agents(seed: int, config: dict, verbose: bool = True) -> tuple[Dict[int, DQNAgent], TrainingHistory]:
    """Train one DQN per node by simulating shortened episodes.

    Returns the trained agents and the per-episode history (for plots).
    """
    # Lazy import to keep simulator independent of torch when not needed.
    from sim.simulator import build_world, run_simulation, edge_latency_ms, DRL_POLICY

    world = build_world(seed=seed, config=config)
    agents = make_agents([n.node_id for n in world.nodes], config)
    history = TrainingHistory()

    # Truncate the request stream to a training horizon per episode.
    train_horizon = int(config["train_requests_per_episode"])
    full_requests = world.requests

    for episode in range(config["num_episodes"]):
        # Shuffle the training slice each episode for more iid samples.
        rng = random.Random(seed * 7919 + episode)
        sliced = list(full_requests[:train_horizon])
        rng.shuffle(sliced)
        episode_world = world  # share contents/nodes, swap requests
        episode_world.requests = sliced

        # Hook that picks an action and trains after the simulator
        # tells us what hit_type / latency happened.
        last_state: Dict[int, list] = {}
        last_action: Dict[int, int] = {}
        rewards = 0.0
        losses: List[float] = []

        def drl_hook(node, content, latency, deadline_ms, candidate_nodes, all_nodes):
            state = build_drl_state(node, content, candidate_nodes, latency, deadline_ms)
            agent = agents[node.node_id]
            action = agent.act(state)
            last_state[node.node_id] = state
            last_action[node.node_id] = action
            return action

        def log_callback(record: dict) -> None:
            nonlocal rewards
            served_id = record["selected_node"]
            if served_id not in last_state:
                return
            state = last_state.pop(served_id)
            action = last_action.pop(served_id)
            r = reward_for(record["hit_type"], record["latency_ms"], record["deadline_met"], config=config)
            rewards += r
            # next_state ≈ state (we don't model true next state)
            agents[served_id].remember(state, action, r, state, False)
            loss = agents[served_id].train()
            if loss is not None:
                losses.append(loss)

        run_simulation(
            policy=DRL_POLICY,
            seed=seed,
            config=config,
            world=episode_world,
            drl_hook=drl_hook,
            log_callback=log_callback,
        )

        # Target network sync + epsilon decay once per episode.
        for agent in agents.values():
            agent.update_target_network()
            agent.decay_epsilon()

        avg_loss = sum(losses) / len(losses) if losses else 0.0
        history.episode_reward.append(rewards)
        history.episode_loss.append(avg_loss)
        history.episode_epsilon.append(next(iter(agents.values())).epsilon)

        if verbose:
            print(
                f"  Episode {episode + 1:>3}/{config['num_episodes']} | "
                f"reward={rewards:>9.2f} | avg_loss={avg_loss:>7.4f} | "
                f"eps={history.episode_epsilon[-1]:.3f}"
            )

    # Restore full request stream so the caller can run evaluation on it.
    world.requests = full_requests
    return agents, history


def save_agents(agents: Dict[int, DQNAgent], directory: str) -> None:
    os.makedirs(directory, exist_ok=True)
    for node_id, agent in agents.items():
        agent.save(os.path.join(directory, f"agent_node_{node_id}.pt"))


def make_eval_hook(agents: Dict[int, DQNAgent]):
    """Hook for evaluation runs: greedy, no learning."""
    for agent in agents.values():
        agent.epsilon = 0.0

    def _hook(node, content, latency, deadline_ms, candidate_nodes, all_nodes):
        state = build_drl_state(node, content, candidate_nodes, latency, deadline_ms)
        return agents[node.node_id].act(state)

    return _hook
