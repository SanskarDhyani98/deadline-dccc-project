"""Generate visual class diagram and sequence diagram PNGs."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import matplotlib.patheffects as pe
import numpy as np

OUT   = "plots"
DPI   = 180
BLUE  = "#2c3e50"
LBLUE = "#3498db"
TEAL  = "#16a085"
ORANGE= "#e67e22"
GREEN = "#27ae60"
PURPLE= "#8e44ad"
RED   = "#c0392b"
LGREY = "#ecf0f1"
GREY  = "#7f8c8d"
WHITE = "#ffffff"


# ── helpers ──────────────────────────────────────────────────────────────────
def savefig(name):
    plt.savefig(f"{OUT}/{name}", dpi=DPI, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    plt.close()
    print(f"  saved {OUT}/{name}")


def uml_class_box(ax, x, y, width, header, attrs, methods, color):
    """Draw a UML class box at (x,y) top-left corner."""
    row_h   = 0.28
    pad     = 0.12
    n_attr  = len(attrs)
    n_meth  = len(methods)
    h_head  = 0.42
    h_attr  = n_attr  * row_h + 2 * pad
    h_meth  = n_meth  * row_h + 2 * pad
    total_h = h_head + h_attr + h_meth

    # header
    head_box = FancyBboxPatch((x, y - h_head), width, h_head,
                              boxstyle="round,pad=0,rounding_size=0",
                              fc=color, ec=BLUE, lw=1.6, zorder=3)
    ax.add_patch(head_box)
    ax.text(x + width/2, y - h_head/2, header,
            ha="center", va="center", fontsize=10, fontweight="bold",
            color=WHITE, zorder=4)

    # attributes section
    attr_box = FancyBboxPatch((x, y - h_head - h_attr), width, h_attr,
                              boxstyle="round,pad=0,rounding_size=0",
                              fc=LGREY, ec=BLUE, lw=1.6, zorder=3)
    ax.add_patch(attr_box)
    for i, attr in enumerate(attrs):
        ty = y - h_head - pad - (i + 0.5) * row_h
        ax.text(x + 0.1, ty, attr, ha="left", va="center",
                fontsize=7.5, color=BLUE, family="monospace", zorder=4)

    # methods section
    meth_box = FancyBboxPatch((x, y - total_h), width, h_meth,
                              boxstyle="round,pad=0,rounding_size=0",
                              fc=WHITE, ec=BLUE, lw=1.6, zorder=3)
    ax.add_patch(meth_box)
    for i, meth in enumerate(methods):
        ty = y - h_head - h_attr - pad - (i + 0.5) * row_h
        ax.text(x + 0.1, ty, meth, ha="left", va="center",
                fontsize=7.5, color=TEAL, family="monospace", zorder=4)

    return total_h   # return height so caller can draw connectors


# ══════════════════════════════════════════════════════════════════════════════
#  CLASS DIAGRAM
# ══════════════════════════════════════════════════════════════════════════════
def fig_class_diagram():
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.set_xlim(0, 14); ax.set_ylim(-8.5, 1)
    ax.axis("off")
    ax.set_facecolor(WHITE)

    # ── four class boxes ─────────────────────────────────────
    classes = {
        "Content": dict(
            x=0.4, y=0.5, w=3.0, color=LBLUE,
            attrs=[
                "+ content_id : int",
                "+ size_mb    : int",
                "+ base_popularity : float",
                "+ region_bias : List[float]",
                "+ content_type : str",
            ],
            methods=[
                "+ __repr__() : str",
            ],
        ),
        "EdgeNode": dict(
            x=4.2, y=0.5, w=3.6, color=TEAL,
            attrs=[
                "+ node_id     : int",
                "+ region      : int",
                "+ cache_capacity_mb : int",
                "+ cache       : Dict[int,Content]",
                "+ used_mb     : int",
                "+ load        : float",
                "+ frequency   : Dict[int,int]",
                "+ last_access : Dict[int,int]",
                "+ reward_score: Dict[int,float]",
                "+ eviction_count : int",
            ],
            methods=[
                "+ has_content(cid) : bool",
                "+ add_content(c)   : None",
                "+ add_lru(c, t)    : None",
                "+ add_lfu(c, t)    : None",
                "+ add_ice_style(c,t,pop) : None",
                "- _drop(cid)       : None",
            ],
        ),
        "Request": dict(
            x=0.4, y=-5.2, w=3.0, color=ORANGE,
            attrs=[
                "+ request_id   : int",
                "+ content_id   : int",
                "+ user_region  : int",
                "+ deadline_ms  : int",
                "+ time_step    : int",
            ],
            methods=[
                "+ __repr__() : str",
            ],
        ),
        "DQNAgent": dict(
            x=8.6, y=0.5, w=3.8, color=PURPLE,
            attrs=[
                "+ node_id      : int",
                "+ state_size   : int  (=7)",
                "+ action_size  : int  (=3)",
                "+ epsilon      : float",
                "+ epsilon_min  : float",
                "+ epsilon_decay: float",
                "+ gamma        : float",
                "+ memory       : ReplayBuffer",
                "+ policy_net   : DQNNetwork",
                "+ target_net   : DQNNetwork",
            ],
            methods=[
                "+ act(state) : int",
                "+ remember(s,a,r,s',done) : None",
                "+ train()    : float",
                "+ decay_epsilon()       : None",
                "+ update_target_network(): None",
            ],
        ),
    }

    boxes = {}
    for name, cfg in classes.items():
        h = uml_class_box(ax, cfg["x"], cfg["y"], cfg["w"],
                          name, cfg["attrs"], cfg["methods"], cfg["color"])
        boxes[name] = cfg | {"h": h}

    # ── relationships ─────────────────────────────────────────
    def arrow(x1, y1, x2, y2, label="", style="->", color=GREY):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(
                        arrowstyle=style, color=color, lw=1.5,
                        connectionstyle="arc3,rad=0.0"),
                    zorder=5)
        if label:
            mx, my = (x1+x2)/2, (y1+y2)/2
            ax.text(mx, my + 0.12, label, ha="center", fontsize=8,
                    color=color, style="italic", zorder=6)

    # Content ──(1..*)── EdgeNode  (aggregation)
    arrow(3.4, -1.5,  4.2, -1.5,  "1..* caches", "->", LBLUE)
    # Request ──── Content  (dependency)
    arrow(1.9, -5.2,  1.9, -4.2,  "uses", "->", ORANGE)
    # DQNAgent ──── EdgeNode  (association, one per node)
    arrow(8.6, -2.2,  7.8, -2.2,  "1 per node", "-|>", PURPLE)
    # EdgeNode ──── Request  (processes)
    arrow(5.0, -7.2,  3.4, -6.2,  "processes", "->", TEAL)

    # ── legend ────────────────────────────────────────────────
    legend_items = [
        mpatches.Patch(fc=LBLUE,  label="Content (data entity)"),
        mpatches.Patch(fc=TEAL,   label="EdgeNode (infrastructure)"),
        mpatches.Patch(fc=ORANGE, label="Request (workload)"),
        mpatches.Patch(fc=PURPLE, label="DQNAgent (DRL)"),
    ]
    ax.legend(handles=legend_items, loc="lower right", fontsize=9,
              framealpha=0.95, ncol=2)

    ax.set_title("Class Diagram — Core Domain Model",
                 fontsize=13, fontweight="bold", color=BLUE,
                 y=1.01, pad=4)
    fig.tight_layout()
    savefig("class_diagram_visual.png")


# ══════════════════════════════════════════════════════════════════════════════
#  SEQUENCE DIAGRAM
# ══════════════════════════════════════════════════════════════════════════════
def fig_sequence_diagram():
    fig, ax = plt.subplots(figsize=(14, 10))
    ax.set_xlim(-0.5, 13.5); ax.set_ylim(-11.5, 1.2)
    ax.axis("off")
    ax.set_facecolor(WHITE)

    # lifeline positions (x-coordinate)
    actors = {
        "User":       0.5,
        "Simulator":  2.3,
        "Policy":     4.1,
        "AllNodes":   5.9,
        "BestNode":   7.7,
        "DQNAgent":   9.5,
        "CoopNode/\nCloud": 11.3,
    }
    colors = {
        "User":       GREEN,
        "Simulator":  BLUE,
        "Policy":     TEAL,
        "AllNodes":   GREY,
        "BestNode":   LBLUE,
        "DQNAgent":   PURPLE,
        "CoopNode/\nCloud": RED,
    }

    TOP = 0.8
    BOT = -11.0

    # ── actor header boxes ────────────────────────────────────
    for name, x in actors.items():
        bp = FancyBboxPatch((x - 0.55, TOP - 0.35), 1.1, 0.55,
                            boxstyle="round,pad=0.05,rounding_size=0.12",
                            fc=colors[name], ec=WHITE, lw=1.5, zorder=4)
        ax.add_patch(bp)
        ax.text(x, TOP - 0.075, name, ha="center", va="center",
                fontsize=8, fontweight="bold", color=WHITE, zorder=5,
                multialignment="center")
        # dashed lifeline
        ax.plot([x, x], [TOP - 0.35, BOT], color=colors[name],
                lw=1.0, linestyle="--", alpha=0.45, zorder=1)

    # ── message arrows ────────────────────────────────────────
    def msg(x1, x2, y, label, color=BLUE, ret=False, note=None):
        """Draw a horizontal message arrow at height y."""
        style = "<-" if ret else "-|>"
        ax.annotate("", xy=(x2, y), xytext=(x1, y),
                    arrowprops=dict(arrowstyle=style, color=color, lw=1.5),
                    zorder=3)
        # label above arrow
        mx = (x1 + x2) / 2
        ax.text(mx, y + 0.10, label, ha="center", va="bottom",
                fontsize=7.5, color=color, zorder=4)
        if note:
            ax.text(max(x1, x2) + 0.1, y, note, ha="left", va="center",
                    fontsize=7, color=GREY, style="italic", zorder=4)

    def divider(y, label, color=LGREY):
        ax.axhline(y, xmin=0.01, xmax=0.99, color=color,
                   lw=0.8, linestyle=":", zorder=1)
        ax.text(-0.3, y, label, ha="right", va="center",
                fontsize=7, color=GREY, style="italic")

    # activation boxes
    def act(x, y_top, y_bot, color):
        bp = FancyBboxPatch((x - 0.08, y_bot), 0.16, y_top - y_bot,
                            boxstyle="round,pad=0,rounding_size=0",
                            fc=color, ec=WHITE, lw=1.0, alpha=0.55, zorder=2)
        ax.add_patch(bp)

    # ── messages ─────────────────────────────────────────────
    ax.text(6.4, 0.1, "Request Processing Flow — DEADLINE_DCCC Policy",
            ha="center", fontsize=11, fontweight="bold", color=BLUE, zorder=6)

    y = -0.5
    msg(actors["User"], actors["Simulator"], y,
        "Request(content_id, region, deadline_ms)", GREEN)
    act(actors["Simulator"], y, -10.5, LBLUE)

    y -= 0.7
    msg(actors["Simulator"], actors["Policy"], y,
        "compute_popularity(content, t)", BLUE)

    y -= 0.5
    msg(actors["Policy"], actors["Simulator"], y,
        "popularity : float", TEAL, ret=True)

    y -= 0.6
    msg(actors["Simulator"], actors["AllNodes"], y,
        "score all 5 nodes →  deadline_aware_score()", BLUE)

    y -= 0.5
    msg(actors["AllNodes"], actors["Simulator"], y,
        "scores[ ] : List[float]", GREY, ret=True)

    y -= 0.6
    divider(y + 0.3, "select best")
    msg(actors["Simulator"], actors["BestNode"], y,
        "selected node (highest score)", BLUE)
    act(actors["BestNode"], y, -7.8, TEAL)

    y -= 0.7
    msg(actors["BestNode"], actors["DQNAgent"], y,
        "build_drl_state(node, content, latency, deadline)", LBLUE)
    act(actors["DQNAgent"], y, y - 0.5, PURPLE)

    y -= 0.5
    msg(actors["DQNAgent"], actors["BestNode"], y,
        "action ∈ {0:SERVE_LOCAL, 1:FORWARD_DCCC, 2:DEFAULT}", PURPLE, ret=True)

    # alt block
    y -= 0.3
    alt_top = y
    rect = FancyBboxPatch(
        (actors["BestNode"] - 0.55, y - 2.4),
        actors["CoopNode/\nCloud"] - actors["BestNode"] + 1.1, 2.4,
        boxstyle="round,pad=0.05,rounding_size=0.1",
        fc="#f0f4f8", ec=BLUE, lw=1.2, zorder=1)
    ax.add_patch(rect)
    ax.text(actors["BestNode"] - 0.5, y - 0.05,
            "alt", fontsize=8, fontweight="bold", color=BLUE, zorder=5)

    y -= 0.55
    ax.text(actors["BestNode"] - 0.45, y + 0.08,
            "[action=0 AND has_content AND latency ≤ D₁]",
            fontsize=7, color=TEAL, style="italic", zorder=5)
    msg(actors["BestNode"], actors["Simulator"], y,
        "EDGE_HIT  (latency_ms)", TEAL, ret=True)

    y -= 0.5
    ax.plot([actors["BestNode"] - 0.55,
             actors["CoopNode/\nCloud"] + 0.55], [y, y],
            color=BLUE, lw=0.8, linestyle="--", zorder=2)
    ax.text(actors["BestNode"] - 0.45, y + 0.08,
            "[action=1 OR local miss OR latency > D₁]",
            fontsize=7, color=ORANGE, style="italic", zorder=5)

    y -= 0.5
    msg(actors["BestNode"], actors["CoopNode/\nCloud"], y,
        "cooperative_lookup(content_id)", ORANGE)

    y -= 0.5
    msg(actors["CoopNode/\nCloud"], actors["Simulator"], y,
        "COOP_HIT / CLOUD_FETCH  (latency_ms)", RED, ret=True)

    y -= 0.5
    divider(y + 0.2, "cache + metrics")
    msg(actors["Simulator"], actors["BestNode"], y,
        "add_ice_style(content, t, popularity)", BLUE)

    y -= 0.5
    msg(actors["Simulator"], actors["Simulator"], y - 0.15,
        "record metrics (hit_type, latency, deadline_met, energy)", BLUE)
    ax.annotate("", xy=(actors["Simulator"] + 0.25, y - 0.15),
                xytext=(actors["Simulator"] + 0.7, y - 0.15),
                arrowprops=dict(arrowstyle="-|>", color=BLUE, lw=1.2,
                                connectionstyle="arc3,rad=-0.5"))

    y -= 0.6
    msg(actors["Simulator"], actors["User"], y,
        "response  (served)", GREEN, ret=True)

    ax.set_title("Sequence Diagram — Request Processing Flow (DEADLINE_DCCC)",
                 fontsize=12, fontweight="bold", color=BLUE, pad=8, y=1.01)
    fig.tight_layout()
    savefig("sequence_diagram_visual.png")


if __name__ == "__main__":
    print("Generating visual UML diagrams...")
    fig_class_diagram()
    fig_sequence_diagram()
    print("Done.")
