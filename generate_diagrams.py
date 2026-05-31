"""Generate the 6 architecture/diagram figures needed by the LaTeX report."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import matplotlib.patheffects as pe
import numpy as np

OUT = "plots"
DPI = 180

BLUE   = "#2c3e50"
LBLUE  = "#3498db"
ORANGE = "#e67e22"
GREEN  = "#27ae60"
RED    = "#c0392b"
GREY   = "#95a5a6"
WHITE  = "#ffffff"
LGREY  = "#ecf0f1"

def savefig(name):
    plt.savefig(f"{OUT}/{name}", dpi=DPI, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    plt.close()
    print(f"  saved {OUT}/{name}")


# ── Figure 1: System Architecture ─────────────────────────────────────────────
def fig_system_architecture():
    fig, ax = plt.subplots(figsize=(10, 6.5))
    ax.set_xlim(0, 10); ax.set_ylim(0, 7)
    ax.axis("off")

    def box(x, y, w, h, color, text, fontsize=9, textcolor="white", radius=0.25):
        bp = FancyBboxPatch((x - w/2, y - h/2), w, h,
                            boxstyle=f"round,pad=0.05,rounding_size={radius}",
                            fc=color, ec="white", lw=1.5, zorder=3)
        ax.add_patch(bp)
        ax.text(x, y, text, ha="center", va="center", fontsize=fontsize,
                color=textcolor, fontweight="bold", zorder=4,
                wrap=True, multialignment="center")

    def arrow(x1, y1, x2, y2):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="-|>", color=BLUE, lw=1.5),
                    zorder=2)

    # ── Users (3 regions) ──────────────────────────────────────
    ax.text(5, 6.6, "Users (3 Geographic Regions)", ha="center", fontsize=11,
            fontweight="bold", color=BLUE)
    colors_u = [GREEN, "#16a085", "#1abc9c"]
    labels_u = ["Region 0\n60% traffic", "Region 1\n25% traffic", "Region 2\n15% traffic"]
    for i, (ux, lbl, uc) in enumerate(zip([2, 5, 8], labels_u, colors_u)):
        box(ux, 5.9, 1.7, 0.7, uc, lbl, fontsize=8)

    # ── Decision layer ────────────────────────────────────────
    for ux in [2, 5, 8]:
        arrow(ux, 5.55, ux, 5.05)
    box(5, 4.75, 7.5, 0.7, BLUE,
        "Deadline-Aware DCCC Decision Layer\n"
        "(deadline_aware_score + DRL routing hook)",
        fontsize=9)

    # ── Edge nodes ────────────────────────────────────────────
    ax.text(5, 4.05, "Edge Cluster (5 Nodes)", ha="center", fontsize=10,
            fontweight="bold", color=BLUE)
    node_x   = [1.0, 2.8, 4.6, 6.4, 8.2]
    node_cap = [9,   9,   2.7, 9,   2.7]
    for i, (nx, cap) in enumerate(zip(node_x, node_cap)):
        c = LBLUE if cap == 9 else ORANGE
        box(nx, 3.35, 1.55, 0.85, c,
            f"Node {i}\n{cap} GB", fontsize=8)
        arrow(5, 4.4, nx, 3.75)

    # coop arrows between adjacent nodes
    for i in range(len(node_x) - 1):
        ax.annotate("", xy=(node_x[i+1] - 0.78, 3.35),
                    xytext=(node_x[i] + 0.78, 3.35),
                    arrowprops=dict(arrowstyle="<->", color=GREY, lw=1.0, linestyle="dashed"),
                    zorder=2)
    ax.text(4.6, 2.9, "Cooperative DCCC Lookup  ←—→", ha="center",
            fontsize=7.5, color=GREY, style="italic")

    # ── Cloud ─────────────────────────────────────────────────
    for nx in node_x:
        arrow(nx, 2.93, 5, 2.2)
    box(5, 1.85, 4, 0.65, RED, "Cloud  (120 ms base latency, fallback only)", fontsize=9)

    # legend
    leg = [mpatches.Patch(fc=LBLUE, label="Strong node (9 GB)"),
           mpatches.Patch(fc=ORANGE, label="Weak node (2.7 GB)"),
           mpatches.Patch(fc=GREEN,  label="User regions"),
           mpatches.Patch(fc=RED,    label="Cloud fallback")]
    ax.legend(handles=leg, loc="lower right", fontsize=8, framealpha=0.9)

    ax.set_title("System Architecture — Heterogeneous Edge Cluster with DCCC",
                 fontsize=12, fontweight="bold", pad=8, color=BLUE)
    savefig("system_architecture.png")


# ── Figure 2: Capacity Profile ─────────────────────────────────────────────
def fig_capacity_profile():
    fig, ax = plt.subplots(figsize=(7, 4))
    nodes = ["Node 0", "Node 1", "Node 2\n(weak)", "Node 3", "Node 4\n(weak)"]
    caps  = [9, 9, 2.7, 9, 2.7]
    cols  = [LBLUE, LBLUE, ORANGE, LBLUE, ORANGE]
    bars  = ax.bar(nodes, caps, color=cols, edgecolor="white", linewidth=1.2,
                   width=0.55, zorder=3)
    ax.axhline(9,   color=LBLUE,  linestyle="--", lw=1.2, alpha=0.5, label="Strong cap = 9 GB")
    ax.axhline(2.7, color=ORANGE, linestyle="--", lw=1.2, alpha=0.5, label="Weak cap = 2.7 GB")
    ax.set_ylabel("Cache Capacity (GB)", fontsize=11)
    ax.set_title("Heterogeneous Edge Node Capacity Profile\n(3 strong × 9 GB, 2 weak × 2.7 GB)",
                 fontsize=11, fontweight="bold", color=BLUE)
    ax.set_ylim(0, 11)
    ax.grid(axis="y", alpha=0.35, zorder=0)
    for bar, cap in zip(bars, caps):
        ax.text(bar.get_x() + bar.get_width()/2, cap + 0.2, f"{cap} GB",
                ha="center", va="bottom", fontsize=10, fontweight="bold")
    leg = [mpatches.Patch(fc=LBLUE,  label="Strong node (9 GB)"),
           mpatches.Patch(fc=ORANGE, label="Weak node (2.7 GB)")]
    ax.legend(handles=leg, fontsize=9)
    ax.spines[["top","right"]].set_visible(False)
    plt.tight_layout()
    savefig("capacity_profile.png")


# ── Figure 3: Routing Decision Tree ───────────────────────────────────────────
def fig_routing_decision():
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.set_xlim(0, 10); ax.set_ylim(0, 8)
    ax.axis("off")

    def box(x, y, w, h, color, text, fs=9, tc="white"):
        bp = FancyBboxPatch((x - w/2, y - h/2), w, h,
                            boxstyle="round,pad=0.05,rounding_size=0.2",
                            fc=color, ec="white", lw=1.5, zorder=3)
        ax.add_patch(bp)
        ax.text(x, y, text, ha="center", va="center", fontsize=fs,
                color=tc, fontweight="bold", zorder=4, multialignment="center")

    def arr(x1, y1, x2, y2, label="", lside="center"):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="-|>", color=BLUE, lw=1.4), zorder=2)
        if label:
            mx, my = (x1+x2)/2, (y1+y2)/2
            ax.text(mx + (0.15 if lside == "right" else -0.15 if lside == "left" else 0),
                    my, label, ha=lside, va="center", fontsize=8, color=BLUE)

    # START
    box(5, 7.5, 2.2, 0.65, BLUE, "Request arrives\n(content, region, deadline)", fs=8.5)
    arr(5, 7.17, 5, 6.5)

    box(5, 6.15, 3.2, 0.65, BLUE, "Compute deadline_aware_score()\nfor all 5 edge nodes", fs=8.5)
    arr(5, 5.82, 5, 5.15)

    box(5, 4.82, 2.8, 0.55, "#7f8c8d", "Select best-scoring node", fs=8.5)
    arr(5, 4.55, 5, 3.85)

    # DRL decision diamond
    box(5, 3.55, 3.0, 0.55, "#8e44ad",
        "[DEADLINE_DCCC] DQN action?", fs=8)

    # SERVE_LOCAL branch
    arr(3.5, 3.55, 2.0, 3.55, "action=0\nlocal-first", "left")
    box(2.0, 3.1, 2.0, 0.55, "#7f8c8d", "has_content\nAND latency ≤ D₁?", fs=7.5)
    arr(2.0, 2.82, 0.9, 2.3, "YES", "left")
    arr(2.0, 2.82, 3.1, 2.3, "NO",  "right")
    box(0.9, 1.95, 1.4, 0.55, GREEN, "EDGE HIT\n✓ fast path", fs=7.5)

    # FORWARD_DCCC branch
    arr(5, 3.27, 5, 2.5, "action=1\nor miss")
    box(5, 2.2, 2.8, 0.55, LBLUE, "Cooperative DCCC lookup\n(all 5 nodes)", fs=8)
    arr(5, 1.92, 5, 1.25)
    box(5, 0.95, 2.5, 0.55, "#7f8c8d", "coop_latency ≤ D₂?", fs=8)
    arr(3.75, 0.95, 2.5, 0.95, "YES", "left")
    arr(6.25, 0.95, 7.5, 0.95, "NO", "right")
    box(2.5, 0.6, 1.6, 0.55, "#16a085", "COOP HIT\n✓ within D₂", fs=7.5)
    box(7.5, 0.6, 1.6, 0.55, RED, "CLOUD\nFETCH ✗", fs=7.5)

    # Action=2 branch
    arr(6.5, 3.55, 8.2, 3.55, "action=2\nheuristic", "right")
    box(8.2, 3.1, 1.8, 0.55, GREEN, "Apply\nDEADLINE\nHEURISTIC", fs=7.5)

    # from NO → coop
    arr(3.1, 2.3, 5, 2.3)

    ax.set_title("Deadline-Aware Routing Decision Tree", fontsize=12,
                 fontweight="bold", color=BLUE, pad=6)
    savefig("routing_decision.png")


# ── Figure 4: Score Term Contributions ────────────────────────────────────────
def fig_score_terms():
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    # left: bar chart of coefficient magnitudes / direction
    terms  = ["popularity\n(×2.0)", "hit_chance\n(×6.0)", "overlap\n(×-0.5)",
              "latency\n(×-0.08)", "load²\n(×-1.2)", "deadline_risk\n(×-0.4)"]
    values = [2.0, 6.0, -0.5, -0.08*50, -1.2, -0.4*20]  # example at lat=50ms, risk=20ms
    colors = [GREEN if v > 0 else RED for v in values]
    ax = axes[0]
    bars = ax.barh(terms, values, color=colors, edgecolor="white", height=0.55)
    ax.axvline(0, color=BLUE, lw=1.5)
    ax.set_xlabel("Contribution to Score (example request)", fontsize=10)
    ax.set_title("Score Term Contributions\n(example: latency=50ms, risk=20ms)",
                 fontsize=10, fontweight="bold", color=BLUE)
    for bar, val in zip(bars, values):
        xpos = val + (0.1 if val >= 0 else -0.1)
        ax.text(xpos, bar.get_y() + bar.get_height()/2,
                f"{val:+.1f}", va="center",
                ha="left" if val >= 0 else "right", fontsize=8.5, fontweight="bold")
    ax.spines[["top","right"]].set_visible(False)
    ax.grid(axis="x", alpha=0.3)

    # right: load² penalty curve
    ax2 = axes[1]
    load = np.linspace(0, 1.5, 300)
    penalty = -1.2 * load**2
    ax2.plot(load, penalty, color=RED, lw=2.5, label=r"$-1.2 \times \mathrm{load}^2$")
    ax2.axvline(0.5, color=GREY, lw=1, linestyle="--")
    ax2.axvline(1.0, color=ORANGE, lw=1, linestyle="--")
    ax2.axvline(1.5, color=RED, lw=1, linestyle="--")
    ax2.text(0.5, -0.4, "load=0.5\n−0.30", ha="center", fontsize=8, color=GREY)
    ax2.text(1.0, -1.3, "load=1.0\n−1.20", ha="center", fontsize=8, color=ORANGE)
    ax2.text(1.5, -2.9, "load=1.5\n−2.70", ha="center", fontsize=8, color=RED)
    ax2.set_xlabel("Node Load", fontsize=10)
    ax2.set_ylabel("Load² Score Penalty", fontsize=10)
    ax2.set_title("Non-Linear Load Penalty\n(saturation-only penalisation)",
                  fontsize=10, fontweight="bold", color=BLUE)
    ax2.grid(alpha=0.3)
    ax2.spines[["top","right"]].set_visible(False)
    ax2.legend(fontsize=9)

    plt.suptitle("Scoring Function — Term Contributions and Load Penalty",
                 fontsize=11, fontweight="bold", color=BLUE, y=1.01)
    plt.tight_layout()
    savefig("score_terms.png")


# ── Figure 5: DQN Architecture ────────────────────────────────────────────────
def fig_dqn_architecture():
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.set_xlim(-0.5, 10.5); ax.set_ylim(-0.5, 6.5)
    ax.axis("off")

    layer_specs = [
        (1.0,  7, "Input Layer\n(State Space)",  LBLUE,  ["popularity", "size/2500",
         "has_content", "load", "latency/200", "overlap", "risk/200"]),
        (3.8,  4, "Hidden Layer 1\n(64 units + ReLU)", GREEN, None),
        (6.5,  4, "Hidden Layer 2\n(64 units + ReLU)", GREEN, None),
        (9.3,  3, "Output Layer\n(Q-values)",     ORANGE, ["SERVE_LOCAL", "FORWARD_DCCC",
                                                            "DEFAULT_POLICY"]),
    ]

    node_radius = 0.22
    layer_nodes = {}  # store y-coords per layer

    for lx, n_nodes, label, color, node_labels in layer_specs:
        ys = np.linspace(0.5, 5.5, n_nodes) if n_nodes <= 7 else np.linspace(0.3, 5.7, n_nodes)
        layer_nodes[lx] = ys
        for i, y in enumerate(ys):
            circle = plt.Circle((lx, y), node_radius, fc=color, ec="white", lw=1.5, zorder=3)
            ax.add_patch(circle)
            if node_labels and i < len(node_labels):
                side = -0.38 if lx < 5 else 0.38
                ha   = "right" if lx < 5 else "left"
                ax.text(lx + side, y, node_labels[i], ha=ha, va="center",
                        fontsize=7.5, color=BLUE)
        ax.text(lx, -0.3, label, ha="center", va="top", fontsize=8.5,
                fontweight="bold", color=color, multialignment="center")

    # Connections between adjacent layers
    layer_xs = [spec[0] for spec in layer_specs]
    for i in range(len(layer_xs) - 1):
        x1, x2 = layer_xs[i], layer_xs[i+1]
        y1s, y2s = layer_nodes[x1], layer_nodes[x2]
        # draw all connections (thin)
        for y1 in y1s:
            for y2 in y2s:
                ax.plot([x1 + node_radius, x2 - node_radius], [y1, y2],
                        color=GREY, alpha=0.25, lw=0.7, zorder=1)

    # Layer size annotations
    sizes = ["7", "64", "64", "3"]
    for lx, sz in zip(layer_xs, sizes):
        ax.text(lx, 6.2, sz, ha="center", va="center", fontsize=12,
                fontweight="bold", color=BLUE)
    ax.text(5.15, 6.2, "→", ha="center", va="center", fontsize=14, color=BLUE)
    ax.text(7.9,  6.2, "→", ha="center", va="center", fontsize=14, color=BLUE)

    ax.set_title("DQN Architecture per Agent — 7-dim State → 64 → 64 → 3 Q-values",
                 fontsize=11, fontweight="bold", color=BLUE, pad=6)
    plt.tight_layout()
    savefig("dqn_architecture.png")


# ── Figure 6: DRL Training Workflow ───────────────────────────────────────────
def fig_training_workflow():
    fig, ax = plt.subplots(figsize=(9, 8))
    ax.set_xlim(0, 9); ax.set_ylim(0, 9.5)
    ax.axis("off")

    def box(x, y, w, h, color, text, fs=9, tc="white", shape="rect"):
        if shape == "diamond":
            dx, dy = w/2, h/2
            diamond = plt.Polygon([[x, y+dy], [x+dx, y], [x, y-dy], [x-dx, y]],
                                  fc=color, ec="white", lw=1.5, zorder=3)
            ax.add_patch(diamond)
        else:
            bp = FancyBboxPatch((x-w/2, y-h/2), w, h,
                                boxstyle="round,pad=0.05,rounding_size=0.2",
                                fc=color, ec="white", lw=1.5, zorder=3)
            ax.add_patch(bp)
        ax.text(x, y, text, ha="center", va="center", fontsize=fs,
                color=tc, fontweight="bold", zorder=4, multialignment="center")

    def arr(x1, y1, x2, y2, label="", lside="center"):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="-|>", color=BLUE, lw=1.4), zorder=2)
        if label:
            mx, my = (x1+x2)/2, (y1+y2)/2
            ax.text(mx + (0.15 if lside=="right" else -0.15 if lside=="left" else 0),
                    my, label, ha=lside, va="center", fontsize=8, color=BLUE)

    box(4.5, 9.1, 3.5, 0.65, BLUE, "Initialise 5 DQN Agents\n(ε=1.0, random weights)", fs=8.5)
    arr(4.5, 8.77, 4.5, 8.15)

    box(4.5, 7.85, 3.0, 0.55, "#7f8c8d", "Episode loop: 50 episodes", fs=8.5)
    arr(4.5, 7.57, 4.5, 6.97)

    box(4.5, 6.67, 3.5, 0.55, "#7f8c8d", "Build world (seed) — 1000 requests", fs=8.5)
    arr(4.5, 6.40, 4.5, 5.80)

    box(4.5, 5.50, 3.8, 0.55, LBLUE,
        "For each request:\nobserve state s, pick ε-greedy action a", fs=8)
    arr(4.5, 5.22, 4.5, 4.62)

    box(4.5, 4.32, 3.8, 0.55, LBLUE,
        "Execute action → get reward r, next state s'\nStore (s, a, r, s') in replay buffer", fs=8)
    arr(4.5, 4.05, 4.5, 3.45)

    box(4.5, 3.15, 3.5, 0.55, GREEN,
        "Sample batch (64) → compute Bellman target\nUpdate policy net via Adam (MSE loss)", fs=8)
    arr(4.5, 2.87, 4.5, 2.27)

    box(4.5, 1.97, 3.8, 0.55, "#9b59b6",
        "End of episode:\nDecay ε (×0.97) · Sync target network", fs=8)
    arr(4.5, 1.70, 4.5, 1.10)

    box(4.5, 0.80, 3.2, 0.55, "#7f8c8d", "episode < 50?", fs=8.5, shape="diamond")

    # YES loop back
    ax.annotate("", xy=(4.5, 7.57), xytext=(6.2, 0.80),
                arrowprops=dict(arrowstyle="-|>", color=GREEN, lw=1.4,
                                connectionstyle="arc3,rad=-0.25"), zorder=2)
    ax.text(7.1, 4.2, "YES\n(next episode)", ha="center", fontsize=8, color=GREEN)

    # NO → evaluation
    arr(2.9, 0.80, 1.5, 0.80, "NO", "left")
    box(0.85, 0.80, 1.4, 0.55, ORANGE, "Evaluate\nε = 0\n(greedy)", fs=7.5)

    ax.set_title("Episodic DRL Training Workflow (50 episodes × 1000 requests)",
                 fontsize=11, fontweight="bold", color=BLUE, pad=6)
    savefig("training_workflow.png")


if __name__ == "__main__":
    print("Generating architecture figures...")
    fig_system_architecture()
    fig_capacity_profile()
    fig_routing_decision()
    fig_score_terms()
    fig_dqn_architecture()
    fig_training_workflow()
    print("Done — 6 figures written to plots/")
