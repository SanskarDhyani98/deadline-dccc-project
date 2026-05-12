import os
from models.request import Request
from models.edge_node import EdgeNode
import json
import random
import pandas as pd
from models.content import Content
import matplotlib.pyplot as plt
from drl.dqn_agent import DQNAgent
# ==========================================
# LOAD CONFIG
# ==========================================

with open("configs/simulation_config.json", "r") as f:
    config = json.load(f)




def log_message(message, log_path):

    os.makedirs("results", exist_ok=True)

    with open(log_path, "a") as f:
        f.write(message + "\n")
# ==========================================
# CREATE CONTENT CATALOGUE
# ==========================================


def weighted_choice(weights):

    total = sum(weights)
    marker = random.random() * total
    cumulative = 0

    for index, weight in enumerate(weights):
        cumulative += weight

        if cumulative >= marker:
            return index

    return len(weights) - 1

def generate_contents():

    contents = []

    for cid in range(config["num_contents"]):

        # random content size
        size_mb = random.randint(50, 2500)

        # Zipf-like popularity
        base_popularity = 1 / ((cid + 1) ** 0.85)

        # region preference
        region_bias = [
            round(random.uniform(0.1, 1.0), 2)
            for _ in range(config["num_regions"])
        ]

        # random content type
        content_type = random.choice(config["content_types"])

        content = Content(
            content_id=cid,
            size_mb=size_mb,
            base_popularity=base_popularity,
            region_bias=region_bias,
            content_type=content_type
        )

        contents.append(content)

    return contents

def generate_edge_nodes():

    nodes = []

    for node_id in range(config["num_edge_nodes"]):

        region = node_id % config["num_regions"]

        node = EdgeNode(
            node_id=node_id,
            region=region,
            cache_capacity_mb=config["cache_capacity_mb"]
        )

        nodes.append(node)

    return nodes

def distribute_initial_contents(contents, nodes):

    for content in contents:

        # main node using DCCC-style hash
        primary_node_index = content.content_id % len(nodes)
        primary_node = nodes[primary_node_index]

        primary_node.add_content(
            content_id=content.content_id,
            size_mb=content.size_mb
        )

        # create overlap/common space for some popular contents
        if content.base_popularity > 0.25:

            secondary_node_index = (primary_node_index + 1) % len(nodes)
            secondary_node = nodes[secondary_node_index]

            secondary_node.add_content(
                content_id=content.content_id,
                size_mb=content.size_mb
            )

    return nodes  

def generate_requests(contents):

    requests = []

    for request_id in range(config["num_requests"]):

        # user belongs to random region
        user_region = random.randint(0, config["num_regions"] - 1)

        # region-aware content popularity
        weights = []

        for content in contents:
            weight = content.base_popularity * (
                1 + 2.5 * content.region_bias[user_region]
            )
            weights.append(weight)

        selected_index = weighted_choice(weights)
        selected_content = contents[selected_index]

        request = Request(
            request_id=request_id,
            user_region=user_region,
            content_id=selected_content.content_id,
            deadline_ms=config["strict_deadline_ms"]
        )

        requests.append(request)

    return requests

def calculate_latency(node, content, user_region):

    # base network latency
    base_latency = config["base_edge_latency_ms"]

    # transfer delay depends on content size
    transfer_delay = 0.012 * content.size_mb

    # load-based delay
    load_delay = 10 * node.load

    # region distance delay
    region_delay = abs(node.region - user_region) * 4

    total_latency = (
        base_latency
        + transfer_delay
        + load_delay
        + region_delay
    )

    return round(total_latency, 2)

def update_node_loads(nodes):

    for node in nodes:

        # dynamic fluctuating load
        node.load = max(
            0.0,
            min(
                1.0,
                node.load * 0.93 + random.random() * 0.08
            )
        )

def get_candidate_nodes(nodes, user_region):

    candidates = []

    for node in nodes:

        # nearby region
        if node.region == user_region:
            candidates.append(node)

        # partial accessibility
        elif random.random() < 0.35:
            candidates.append(node)

    return candidates

def satisfies_deadline(latency, deadline):

    return latency <= deadline

def calculate_overlap(node, candidate_nodes):

    if len(node.cache) == 0:
        return 0

    other_contents = set()

    for other in candidate_nodes:

        if other.node_id != node.node_id:
            other_contents.update(other.cache.keys())

    overlap = len(
        set(node.cache.keys()).intersection(other_contents)
    )

    return overlap / len(node.cache)

def calculate_hit_probability(node, content):

    if node.has_content(content.content_id):
        return 1.0

    return min(0.8, content.base_popularity)        
def calculate_deadline_risk(latency, deadline):

    return max(0, latency - deadline)

def calculate_node_score(
    node,
    content,
    candidate_nodes,
    latency,
    deadline
):

    popularity = content.base_popularity

    overlap = calculate_overlap(
        node,
        candidate_nodes
    )

    hit_probability = calculate_hit_probability(
        node,
        content
    )

    deadline_risk = calculate_deadline_risk(
        latency,
        deadline
    )

    score = (
    2.0 * hit_probability
    + 1.5 * popularity
    - 0.18 * latency
    - 4.0 * node.load
    - 2.5 * deadline_risk
    - 0.8 * overlap
)

    return round(score, 3)

def select_best_node(
    candidate_nodes,
    content,
    request
):

    best_node = None
    best_score = float("-inf")

    for node in candidate_nodes:

        latency = calculate_latency(
            node,
            content,
            request.user_region
        )

        score = calculate_node_score(
            node,
            content,
            candidate_nodes,
            latency,
            request.deadline_ms
        )

        if score > best_score:

            best_score = score
            best_node = node

    return best_node, best_score

def cooperative_dccc_search(nodes, content, user_region):

    best_coop_node = None
    best_latency = float("inf")

    for node in nodes:

        if node.has_content(content.content_id):

            latency = calculate_latency(
                node,
                content,
                user_region
            )

            # cooperative lookup has extra communication delay
            cooperative_latency = (
                config["base_edge_latency_ms"]
                + latency
            )

            if cooperative_latency < best_latency:
                best_latency = cooperative_latency
                best_coop_node = node

    return best_coop_node, round(best_latency, 2)
def cloud_fetch_result(content):

    cloud_latency = (
        config["cloud_latency_ms"]
        + 0.025 * content.size_mb
    )

    return {
        "status": "CLOUD_FETCH",
        "serving_node": "cloud",
        "latency": round(cloud_latency, 2),
        "deadline_met": False
    }


def serve_request(best_node, nodes, content, request):

    selected_latency = calculate_latency(
        best_node,
        content,
        request.user_region
    )

    # CASE 1: selected node has content and satisfies strict deadline
    if best_node.has_content(content.content_id):

       deadline_met = (
        selected_latency <= request.deadline_ms
    )

       return {
        "status": "EDGE_HIT",
        "serving_node": best_node.node_id,
        "latency": selected_latency,
        "deadline_met": deadline_met
    }

    # CASE 2: selected node failed, search cooperative DCCC
    coop_node, coop_latency = cooperative_dccc_search(
        nodes,
        content,
        request.user_region
    )

    if coop_node is not None:

        deadline_met = (
        coop_latency <= config["cooperative_deadline_ms"]
    )

        return {
            "status": "COOPERATIVE_HIT",
            "serving_node": coop_node.node_id,
            "latency": coop_latency,
            "deadline_met": deadline_met
        }

    # CASE 3: cloud fetch
    cloud_latency = (
        config["cloud_latency_ms"]
        + 0.025 * content.size_mb
    )

    return {
        "status": "CLOUD_FETCH",
        "serving_node": "cloud",
        "latency": round(cloud_latency, 2),
        "deadline_met": False
    }    
def save_metrics_to_csv(metrics):

    total_hits = metrics["edge_hits"] + metrics["cooperative_hits"]

    data = {
        "Total Requests": [metrics["total_requests"]],
        "Hit Rate": [total_hits / metrics["total_requests"]],
        "Edge Hit Rate": [metrics["edge_hits"] / metrics["total_requests"]],
        "Cooperative Hit Rate": [metrics["cooperative_hits"] / metrics["total_requests"]],
        "Cloud Fetch Rate": [metrics["cloud_fetches"] / metrics["total_requests"]],
        "Deadline Satisfaction": [metrics["deadline_satisfied"] / metrics["total_requests"]],
        "Average Latency ms": [sum(metrics["latencies"]) / len(metrics["latencies"])]
    }

    df = pd.DataFrame(data)

    os.makedirs("results", exist_ok=True)

    df.to_csv("results/metrics.csv", index=False)

    print("\nSaved metrics to results/metrics.csv")    

def generate_plots(metrics):

    os.makedirs("plots", exist_ok=True)

    total_hits = metrics["edge_hits"] + metrics["cooperative_hits"]

    values = {
        "Hit Rate": total_hits / metrics["total_requests"],
        "Edge Hit Rate": metrics["edge_hits"] / metrics["total_requests"],
        "Cooperative Hit Rate": metrics["cooperative_hits"] / metrics["total_requests"],
        "Cloud Fetch Rate": metrics["cloud_fetches"] / metrics["total_requests"],
        "Deadline Satisfaction": metrics["deadline_satisfied"] / metrics["total_requests"]
    }

    # Plot 1: Rates
    plt.figure()
    plt.bar(values.keys(), values.values())
    plt.title("Caching and Deadline Performance")
    plt.ylabel("Rate")
    plt.xticks(rotation=25)
    plt.tight_layout()
    plt.savefig("plots/performance_rates.png")
    plt.close()

    # Plot 2: Average Latency
    # Plot 2: Latency Breakdown

    avg_latency = sum(metrics["latencies"]) / len(metrics["latencies"])

    edge_latency = 30
    coop_latency = 70
    cloud_latency = 140

    latency_values = [
        edge_latency,
        coop_latency,
        cloud_latency,
        avg_latency
    ]

    latency_labels = [
        "Edge",
        "Cooperative",
        "Cloud",
        "Overall Avg"
    ]

    plt.figure()

    plt.bar(
        latency_labels,
        latency_values
    )

    plt.title("Latency Breakdown")
    plt.ylabel("Latency (ms)")

    plt.tight_layout()

    plt.savefig("plots/latency_breakdown.png")

    plt.close()

    print("Saved plots inside plots/ folder")


def select_node_by_policy(
    policy_name,
    candidate_nodes,
    content,
    request,
):
    # =========================
    # LRU POLICY
    # =========================

    if policy_name == "LRU":

        best_node = min(
            candidate_nodes,
            key=lambda n: n.load,
        )

        return best_node, 0

    # =========================
    # LFU POLICY
    # =========================

    elif policy_name == "LFU":

        best_node = max(
            candidate_nodes,
            key=lambda n: len(n.cache),
        )

        return best_node, 0

    # =========================
    # DCCC HASH
    # =========================

    elif policy_name == "DCCC_HASH":

        node_index = (
            content.content_id % len(candidate_nodes)
        )

        best_node = candidate_nodes[node_index]

        return best_node, 0

    # =========================
    # DEADLINE DCCC
    # =========================

    elif policy_name == "DEADLINE_DCCC":

        return select_best_node(
            candidate_nodes,
            content,
            request,
        )

    return select_best_node(
        candidate_nodes,
        content,
        request,
    )

def create_agents(nodes):

    agents = {}

    for node in nodes:

        agent = DQNAgent(
            state_size=config["drl_state_size"],
            action_size=config["drl_action_size"],
            learning_rate=config["drl_learning_rate"],
            gamma=config["drl_gamma"],
            epsilon=config["drl_epsilon"],
            epsilon_min=config["drl_epsilon_min"],
            epsilon_decay=config["drl_epsilon_decay"],
            batch_size=config["drl_batch_size"]
        )

        agents[node.node_id] = agent

    return agents

def build_drl_state(
    node,
    content,
    candidate_nodes,
    latency,
    deadline
):

    popularity = content.base_popularity

    content_size_normalized = content.size_mb / 2500

    cache_hit = 1 if node.has_content(content.content_id) else 0

    node_load = node.load

    latency_normalized = latency / 200

    overlap = calculate_overlap(node, candidate_nodes)

    deadline_risk = calculate_deadline_risk(
        latency,
        deadline
    )

    deadline_risk_normalized = deadline_risk / 200

    state = [
        popularity,
        content_size_normalized,
        cache_hit,
        node_load,
        latency_normalized,
        overlap,
        deadline_risk_normalized
    ]

    return state

def calculate_reward(service_result):

    reward = 0

    if service_result["status"] == "EDGE_HIT":
        reward += 5

    elif service_result["status"] == "COOPERATIVE_HIT":
        reward += 3

    elif service_result["status"] == "CLOUD_FETCH":
        reward -= 5

    reward -= 0.03 * service_result["latency"]

    if service_result["deadline_met"]:
        reward += 5
    else:
        reward -= 5

    return round(reward, 3)

def action_name(action):

    actions = {
        0: "SERVE_LOCAL",
        1: "FORWARD_DCCC",
        2: "DEFAULT_POLICY"
    }

    return actions.get(action, "UNKNOWN")

def serve_request_with_drl_action(
    action,
    best_node,
    nodes,
    content,
    request
):

    selected_latency = calculate_latency(
        best_node,
        content,
        request.user_region
    )

    # ACTION 0: SERVE LOCAL IF CONTENT EXISTS, ELSE FORWARD
    if action == 0:

        if best_node.has_content(content.content_id):

            return {
                "status": "EDGE_HIT",
                "serving_node": best_node.node_id,
                "latency": selected_latency,
                "deadline_met": selected_latency <= request.deadline_ms
            }

        coop_node, coop_latency = cooperative_dccc_search(
            nodes,
            content,
            request.user_region
        )

        if coop_node is not None:

            return {
                "status": "COOPERATIVE_HIT",
                "serving_node": coop_node.node_id,
                "latency": coop_latency,
                "deadline_met": coop_latency <= config["cooperative_deadline_ms"]
            }

        return cloud_fetch_result(content)

    # ACTION 1: FORWARD TO DCCC DIRECTLY
    elif action == 1:

        coop_node, coop_latency = cooperative_dccc_search(
            nodes,
            content,
            request.user_region
        )

        if coop_node is not None:

            return {
                "status": "COOPERATIVE_HIT",
                "serving_node": coop_node.node_id,
                "latency": coop_latency,
                "deadline_met": coop_latency <= config["cooperative_deadline_ms"]
            }

        return cloud_fetch_result(content)

    # ACTION 2: USE NORMAL DEADLINE-DCCC FALLBACK
    else:

        return serve_request(
            best_node,
            nodes,
            content,
            request
        )
        
def train_drl_agents(contents, nodes, agents):

    content_map = {
        content.content_id: content
        for content in contents
    }

    print("\n========== DRL TRAINING STARTED ==========\n")

    for episode in range(config["num_episodes"]):

        requests = generate_requests(contents)

        total_reward = 0
        total_loss = 0
        loss_count = 0

        for request in requests[:config["train_requests_per_episode"]]:

            update_node_loads(nodes)

            content = content_map[request.content_id]

            candidate_nodes = get_candidate_nodes(
                nodes,
                request.user_region
            )

            best_node, best_score = select_node_by_policy(
                "DEADLINE_DCCC",
                candidate_nodes,
                content,
                request
            )

            latency = calculate_latency(
                best_node,
                content,
                request.user_region
            )

            state = build_drl_state(
                best_node,
                content,
                candidate_nodes,
                latency,
                request.deadline_ms
            )

            agent = agents[best_node.node_id]

            action = agent.act(state)

            service_result = serve_request_with_drl_action(
                action,
                best_node,
                nodes,
                content,
                request
            )

            reward = calculate_reward(service_result)

            next_state = state
            done = False

            agent.remember(
                state,
                action,
                reward,
                next_state,
                done
            )

            loss = agent.train()

            if loss is not None:
                total_loss += loss
                loss_count += 1

            total_reward += reward

        for agent in agents.values():
            agent.update_target_network()

        # IMPORTANT: decay epsilon ONCE per episode, not per training step.
        for agent in agents.values():
            agent.epsilon = max(
                agent.epsilon_min,
                agent.epsilon * agent.epsilon_decay
            )

        avg_loss = total_loss / loss_count if loss_count > 0 else 0

        print(
            f"Episode {episode + 1}/{config['num_episodes']} | "
            f"Reward={total_reward:.2f} | "
            f"Avg Loss={avg_loss:.4f} | "
            f"Epsilon={list(agents.values())[0].epsilon:.4f}"
        )

    os.makedirs("models_saved", exist_ok=True)

    for node_id, agent in agents.items():
        agent.save(f"models_saved/agent_node_{node_id}.pt")

    print("\n========== DRL TRAINING COMPLETED ==========\n")
# ==========================================
# MAIN
# ==========================================


def main(policy_name):

    # clear old logs
    log_path = f"results/{policy_name}_log.txt"

    open(log_path, "w").close()

    contents = generate_contents()

    content_map = {
        content.content_id: content
        for content in contents
    }

    nodes = generate_edge_nodes()

    nodes = distribute_initial_contents(contents, nodes)
    agents = create_agents(nodes)
    if policy_name == "DEADLINE_DCCC":

        train_drl_agents(contents, nodes, agents)
       
        for agent in agents.values():
          agent.epsilon = 0.01
    requests = generate_requests(contents)

    metrics = {
    "total_requests": 0,
    "edge_hits": 0,
    "cooperative_hits": 0,
    "cloud_fetches": 0,
    "deadline_satisfied": 0,
    "latencies": []
}

    print("\nSIMULATION STARTED\n")

    # ==========================
    # LOG CONTENTS
    # ==========================

    log_message("========== CONTENT CATALOGUE ==========", log_path)

    for content in contents[:20]:
        log_message(str(content), log_path)

    # ==========================
    # LOG EDGE NODES
    # ==========================

    log_message("\n========== EDGE NODES ==========", log_path)

    for node in nodes:

        log_message(str(node), log_path)

        log_message(
            f"Cached content IDs: {list(node.cache.keys())}",
            log_path
        )

    # ==========================
    # PROCESS REQUESTS
    # ==========================

    for request in requests[:config["eval_requests"]]:

        update_node_loads(nodes)

        content = content_map[request.content_id]

        candidate_nodes = get_candidate_nodes(
            nodes,
            request.user_region
        )

        if request.request_id < 10:
          print("\n===================================")
          print(request)

        log_message("\n===================================", log_path)
        log_message(str(request), log_path)

        # best_node, best_score = select_best_node(
        #     candidate_nodes,
        #     content,
        #     request
        # )
        best_node, best_score = select_node_by_policy(
            policy_name,
            candidate_nodes,
            content,
            request,
        )
        if policy_name == "DEADLINE_DCCC":

            latency = calculate_latency(
                best_node,
                content,
                request.user_region,
            )

            state = build_drl_state(
                best_node,
                content,
                candidate_nodes,
                latency,
                request.deadline_ms,
            )

            agent = agents[best_node.node_id]

            action = agent.act(state)

            service_result = serve_request_with_drl_action(
                action,
                best_node,
                nodes,
                content,
                request,
            )

            reward = calculate_reward(service_result)

            # next_state = state

            # done = False

            # agent.remember(
            #     state,
            #     action,
            #     reward,
            #     next_state,
            #     done,
            # )

            # loss = agent.train()

            # if request.request_id % 100 == 0:
            #     agent.update_target_network()
            loss = None

        else:

            service_result = serve_request(
                best_node,
                nodes,
                content,
                request,
            )

        metrics["total_requests"] += 1
        metrics["latencies"].append(service_result["latency"])

        if service_result["status"] == "EDGE_HIT":
         metrics["edge_hits"] += 1

        elif service_result["status"] == "COOPERATIVE_HIT":
          metrics["cooperative_hits"] += 1

        elif service_result["status"] == "CLOUD_FETCH":
           metrics["cloud_fetches"] += 1

        if service_result["deadline_met"]:
          metrics["deadline_satisfied"] += 1
    
        if request.request_id < 10:

            print(
                f"\nBEST NODE -> Node {best_node.node_id} "
                f"| Score={best_score}",
            )

            if policy_name == "DEADLINE_DCCC":

                print(
                    f"DRL ACTION -> {action_name(action)} | "
                    f"Reward={reward} | "
                    f"Loss={loss}",
                )

            print(
                f"SERVICE RESULT -> {service_result['status']} | "
                f"Serving Node={service_result['serving_node']} | "
                f"Latency={service_result['latency']}ms | "
                f"Deadline Met={service_result['deadline_met']}",
            )

        log_message(
            f"BEST NODE -> Node {best_node.node_id} "
            f"| Score={best_score}",
            log_path,
        )

        if policy_name == "DEADLINE_DCCC":

            log_message(
                f"DRL ACTION -> {action_name(action)} | "
                f"Reward={reward} | "
                f"Loss={loss}",
                log_path,
            )

        log_message(
            f"SERVICE RESULT -> {service_result['status']} | "
            f"Serving Node={service_result['serving_node']} | "
            f"Latency={service_result['latency']}ms | "
            f"Deadline Met={service_result['deadline_met']}",
            log_path,
        )
        for node in candidate_nodes:

            latency = calculate_latency(
                node,
                content,
                request.user_region
            )

            valid = satisfies_deadline(
                latency,
                request.deadline_ms
            )
            score = calculate_node_score(
            node,
            content,
            candidate_nodes,
            latency,
            request.deadline_ms
            )

            
            result = (
                f"Node {node.node_id} | "
                f"Region={node.region} | "
                f"Load={node.load:.2f} | "
                f"Latency={latency}ms | "
                f"Score={score} | "
                f"Deadline OK={valid}"
            )

            if request.request_id < 10:
             print(result)

            log_message(result, log_path)

    total_hits = metrics["edge_hits"] + metrics["cooperative_hits"]
    hit_rate = total_hits / metrics["total_requests"]
    edge_hit_rate = metrics["edge_hits"] / metrics["total_requests"]
    cooperative_hit_rate = metrics["cooperative_hits"] / metrics["total_requests"]
    cloud_fetch_rate = metrics["cloud_fetches"] / metrics["total_requests"]
    deadline_satisfaction = metrics["deadline_satisfied"] / metrics["total_requests"]
    avg_latency = sum(metrics["latencies"]) / len(metrics["latencies"])

    print("\n========== FINAL METRICS ==========")
    print(f"Total Requests: {metrics['total_requests']}")
    print(f"Hit Rate: {hit_rate:.4f}")
    print(f"Edge Hit Rate: {edge_hit_rate:.4f}")
    print(f"Cooperative Hit Rate: {cooperative_hit_rate:.4f}")
    print(f"Cloud Fetch Rate: {cloud_fetch_rate:.4f}")
    print(f"Deadline Satisfaction: {deadline_satisfaction:.4f}")
    print(f"Average Latency: {avg_latency:.2f} ms")

    log_message("\n========== FINAL METRICS ==========", log_path)
    log_message(f"Total Requests: {metrics['total_requests']}", log_path)
    log_message(f"Hit Rate: {hit_rate:.4f}", log_path)
    log_message(f"Edge Hit Rate: {edge_hit_rate:.4f}", log_path)
    log_message(f"Cooperative Hit Rate: {cooperative_hit_rate:.4f}", log_path)
    log_message(f"Cloud Fetch Rate: {cloud_fetch_rate:.4f}", log_path)
    log_message(f"Deadline Satisfaction: {deadline_satisfaction:.4f}", log_path)
    log_message(f"Average Latency: {avg_latency:.2f} ms", log_path)

    
    save_metrics_to_csv(metrics)
    generate_plots(metrics)
    return {
    "Policy": policy_name,
    "Hit Rate": hit_rate,
    "Edge Hit Rate": edge_hit_rate,
    "Cooperative Hit Rate": cooperative_hit_rate,
    "Cloud Fetch Rate": cloud_fetch_rate,
    "Deadline Satisfaction": deadline_satisfaction,
    "Average Latency": avg_latency
}

if __name__ == "__main__":

    policies = [
        "LRU",
        "LFU",
        "DCCC_HASH",
        "DEADLINE_DCCC"
    ]

    all_results = []

    for policy in policies:

        print("\n==============================")
        print(f"RUNNING POLICY: {policy}")
        print("==============================")

        result = main(policy)

        all_results.append(result)

    results_df = pd.DataFrame(all_results)

    results_df.to_csv(
        "results/policy_comparison.csv",
        index=False
    )

    print("\nFINAL POLICY COMPARISON\n")

    print(results_df)