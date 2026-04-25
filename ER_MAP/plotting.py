"""
ER_MAP/plotting.py
==================
Per-phase + cross-phase visualization for ER-MAP GRPO training.

Reads the ``training_metrics.json`` file produced by
``ER_MAP/training/train_grpo.py`` and produces one comprehensive
multi-panel PNG per curriculum phase plus a cross-phase comparison
chart.

Output layout under ``output_dir``::

    plots/
    ├── phase1_dashboard.png         # 6-panel dashboard for Phase 1
    ├── phase2_dashboard.png         # 6-panel dashboard for Phase 2
    ├── phase3_dashboard.png         # 6-panel dashboard for Phase 3
    ├── all_phases_overview.png      # cross-phase overview (single plot)
    └── all_phases_comparison.png    # phase summary bar charts

Each phase dashboard packs:

    1. Reward growth — raw + rolling mean
    2. Win-rate evolution
    3. Outcome distribution (stacked bars over episode bins)
    4. Reward components (mean per component within the phase)
    5. GRPO loss + KL divergence (per group update)
    6. Episode length distribution

Usage (from anywhere):

    from ER_MAP.plotting import plot_per_phase_dashboards
    plot_per_phase_dashboards(
        "er_map_grpo_checkpoints/training_metrics.json",
        "er_map_grpo_checkpoints/plots",
    )

Or from the CLI::

    python -m ER_MAP.plotting \\
        --metrics er_map_grpo_checkpoints/training_metrics.json \\
        --out     er_map_grpo_checkpoints/plots
"""
from __future__ import annotations

import json
import os
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

# Matplotlib import is deferred to plotting time so the module can be
# imported in environments without matplotlib (e.g. for type checking).


# ---------------------------------------------------------------------------
# Color scheme — consistent across every chart for visual continuity
# ---------------------------------------------------------------------------

OUTCOME_COLORS = {
    # Trainer-style outcome names (used by ER_MAP/training/train_grpo.py)
    "WIN":         "#10b981",  # emerald
    "PARTIAL":     "#3b82f6",  # blue
    "INCORRECT":   "#f59e0b",  # amber
    "AMA_LOSS":    "#a855f7",  # violet
    "FATAL_LOSS":  "#ef4444",  # red
    # Baseline-eval outcome names (used by ER_MAP/evaluate_baseline.py)
    "AMA":         "#a855f7",  # violet — same as AMA_LOSS
    "WRONG":       "#f59e0b",  # amber  — same as INCORRECT
    "FATAL":       "#ef4444",  # red    — same as FATAL_LOSS
    # Raw env-event names (used when the script falls through to the
    # default branch — e.g. `terminal_partial` for partial-credit wins).
    "terminal_win":       "#10b981",  # emerald
    "terminal_partial":   "#3b82f6",  # blue (partial credit)
    "terminal_incorrect": "#f59e0b",  # amber
    "terminal_ama":       "#a855f7",  # violet
    "terminal_fatal":     "#ef4444",  # red
    "TRUNCATED":   "#9ca3af",  # gray
    "MAX_STEPS":   "#6b7280",  # darker gray
    "ERROR":       "#1f2937",  # near-black
    "done":        "#9ca3af",  # gray
    "unknown":     "#6b7280",  # gray
}
OUTCOME_ORDER = [
    "WIN", "terminal_win",
    "PARTIAL", "terminal_partial",
    "INCORRECT", "WRONG", "terminal_incorrect",
    "AMA_LOSS", "AMA", "terminal_ama",
    "FATAL_LOSS", "FATAL", "terminal_fatal",
    "TRUNCATED", "MAX_STEPS", "done", "ERROR", "unknown",
]

# Each reward component gets a distinctive color so the per-phase chart
# matches the colors used in the dashboard's "Live Rewards" panel.
COMPONENT_COLORS = {
    "process":       "#22d3ee",  # cyan
    "diagnosis":     "#3b82f6",  # blue
    "plan":          "#8b5cf6",  # violet
    "labs":          "#06b6d4",  # teal
    "treatment":     "#10b981",  # emerald
    "empathy":       "#ec4899",  # pink
    "milestones":    "#f59e0b",  # amber
    "consent":       "#84cc16",  # lime
    "documentation": "#a855f7",  # purple
    "emergency_id":  "#ef4444",  # red
    "penalties":     "#64748b",  # slate
}

PHASE_COLOR = {
    1: "#06b6d4",  # teal
    2: "#3b82f6",  # blue
    3: "#a855f7",  # purple
}

PHASE_NAME = {
    1: "Phase 1: Tool Mastery",
    2: "Phase 2: Clinical Reasoning",
    3: "Phase 3: Empathetic Negotiation",
}


# ---------------------------------------------------------------------------
# Loading + utilities
# ---------------------------------------------------------------------------

def load_metrics(path: str) -> List[Dict[str, Any]]:
    """Load the per-episode metrics list dumped by ``train_grpo.train``."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Metrics file not found: {p}")
    with p.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Expected a JSON list at {p}; got {type(data).__name__}")
    return data


def split_by_phase(metrics: List[Dict[str, Any]]) -> Dict[int, List[Dict[str, Any]]]:
    """Bucket episode records by their ``phase`` field (1, 2, or 3)."""
    buckets: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    for m in metrics:
        buckets[int(m.get("phase", 1))].append(m)
    return dict(sorted(buckets.items()))


def rolling_mean(xs: Sequence[float], window: int = 10) -> List[float]:
    """Simple right-aligned rolling mean (matches what the env scheduler uses)."""
    out: List[float] = []
    buf: List[float] = []
    for x in xs:
        buf.append(x)
        if len(buf) > window:
            buf.pop(0)
        out.append(sum(buf) / len(buf))
    return out


# ---------------------------------------------------------------------------
# Per-phase dashboard
# ---------------------------------------------------------------------------

def _plot_phase_dashboard(phase_id: int, episodes: List[Dict[str, Any]],
                          out_path: str) -> bool:
    """
    Render a single 6-panel dashboard PNG for one curriculum phase.

    Returns True on success, False if the phase has no episodes.
    """
    import matplotlib.pyplot as plt
    import numpy as np

    if not episodes:
        return False

    ep_idx       = [m["episode"] for m in episodes]
    raw          = [m.get("raw_reward", 0.0) for m in episodes]
    verified     = [m.get("verified_reward", 0.0) for m in episodes]
    win_rates    = [m.get("rolling_win_rate", 0.0) for m in episodes]
    avg_rewards  = [m.get("rolling_avg_reward", 0.0) for m in episodes]
    outcomes     = [m.get("outcome", "unknown") for m in episodes]
    steps        = [m.get("steps", 0) for m in episodes]
    components   = [m.get("reward_components", {}) for m in episodes]

    fig = plt.figure(figsize=(16, 10), constrained_layout=True)
    gs = fig.add_gridspec(3, 3)
    fig.patch.set_facecolor("white")

    phase_color = PHASE_COLOR.get(phase_id, "#3b82f6")
    fig.suptitle(
        f"{PHASE_NAME.get(phase_id, f'Phase {phase_id}')}  "
        f"\u2014  {len(episodes)} episodes  "
        f"\u2014  win rate {sum(1 for o in outcomes if o == 'WIN') / len(episodes):.0%}",
        fontsize=15, fontweight="bold", color=phase_color, y=1.02,
    )

    # ----- Panel 1: Reward growth ------------------------------------------------
    ax1 = fig.add_subplot(gs[0, :2])
    ax1.scatter(ep_idx, raw, alpha=0.35, s=18, color=phase_color, label="raw episode reward")
    ax1.plot(ep_idx, rolling_mean(raw, window=10), linewidth=2.5, color="#111827",
             label="rolling mean (w=10)")
    ax1.plot(ep_idx, rolling_mean(verified, window=10), linewidth=2,
             color="#10b981", linestyle="--", label="verified rolling mean")
    ax1.axhline(0, color="gray", linestyle=":", alpha=0.5)
    ax1.set_title("Reward growth", fontweight="bold")
    ax1.set_xlabel("Episode")
    ax1.set_ylabel("Reward")
    ax1.legend(loc="upper left", fontsize=9)
    ax1.grid(alpha=0.25)

    # ----- Panel 2: Win-rate evolution ------------------------------------------
    ax2 = fig.add_subplot(gs[0, 2])
    ax2.plot(ep_idx, win_rates, linewidth=2.5, color="#10b981")
    ax2.fill_between(ep_idx, 0, win_rates, color="#10b981", alpha=0.2)
    ax2.set_ylim(0, 1.0)
    ax2.set_title("Rolling win rate (w=20)", fontweight="bold")
    ax2.set_xlabel("Episode")
    ax2.set_ylabel("Win rate")
    ax2.grid(alpha=0.25)

    # ----- Panel 3: Outcome distribution over episode bins ----------------------
    ax3 = fig.add_subplot(gs[1, :2])
    bin_size = max(1, len(episodes) // 12)  # ~12 bins per phase
    bins = []
    bin_labels = []
    for start in range(0, len(episodes), bin_size):
        chunk = outcomes[start:start + bin_size]
        bins.append(Counter(chunk))
        ep_start = episodes[start]["episode"]
        ep_end = episodes[min(start + bin_size, len(episodes)) - 1]["episode"]
        bin_labels.append(f"{ep_start}-{ep_end}")

    bottom = np.zeros(len(bins))
    x = np.arange(len(bins))
    for outcome in OUTCOME_ORDER:
        heights = np.array([b.get(outcome, 0) for b in bins])
        if heights.sum() == 0:
            continue
        ax3.bar(x, heights, bottom=bottom, color=OUTCOME_COLORS[outcome],
                label=outcome, edgecolor="white", linewidth=0.5)
        bottom += heights
    ax3.set_xticks(x)
    ax3.set_xticklabels(bin_labels, rotation=40, ha="right", fontsize=8)
    ax3.set_title("Outcome distribution over time", fontweight="bold")
    ax3.set_ylabel("Episodes per bin")
    ax3.legend(loc="upper right", fontsize=8, ncol=2)
    ax3.grid(alpha=0.2, axis="y")

    # ----- Panel 4: Reward components (mean within phase) -----------------------
    ax4 = fig.add_subplot(gs[1, 2])
    component_means: Dict[str, float] = defaultdict(float)
    counts: Dict[str, int] = defaultdict(int)
    for c in components:
        for k, v in c.items():
            component_means[k] += float(v)
            counts[k] += 1
    if component_means:
        ordered = sorted(
            component_means.keys(),
            key=lambda k: component_means[k] / max(counts[k], 1),
            reverse=True,
        )
        means = [component_means[k] / max(counts[k], 1) for k in ordered]
        colors = [COMPONENT_COLORS.get(k, "#94a3b8") for k in ordered]
        bars = ax4.barh(range(len(ordered)), means, color=colors,
                        edgecolor="white", linewidth=0.5)
        ax4.set_yticks(range(len(ordered)))
        ax4.set_yticklabels(ordered, fontsize=8)
        ax4.axvline(0, color="gray", linestyle=":", alpha=0.5)
        ax4.set_title("Reward components (mean / episode)", fontweight="bold")
        ax4.set_xlabel("Reward")
        ax4.grid(alpha=0.2, axis="x")
        for bar, value in zip(bars, means):
            ax4.text(value, bar.get_y() + bar.get_height() / 2,
                     f" {value:+.2f}", va="center",
                     ha="left" if value >= 0 else "right",
                     fontsize=7, color="#374151")
    else:
        ax4.text(0.5, 0.5, "No reward-component data",
                 transform=ax4.transAxes, ha="center", va="center", color="gray")
        ax4.set_title("Reward components (mean / episode)", fontweight="bold")

    # ----- Panel 5: GRPO loss + KL divergence per update ------------------------
    ax5 = fig.add_subplot(gs[2, :2])
    update_eps = [m["episode"] for m in episodes if m.get("grpo_update")]
    losses     = [m["grpo_update"]["loss"] for m in episodes if m.get("grpo_update")]
    kls        = [m["grpo_update"]["kl"]   for m in episodes if m.get("grpo_update")]
    if update_eps:
        l1 = ax5.plot(update_eps, losses, marker="o", color="#3b82f6",
                      linewidth=1.8, label="loss")
        ax5b = ax5.twinx()
        l2 = ax5b.plot(update_eps, kls, marker="s", color="#ef4444",
                       linewidth=1.5, label="KL", alpha=0.85)
        ax5.set_title("GRPO update statistics", fontweight="bold")
        ax5.set_xlabel("Episode (last episode of group)")
        ax5.set_ylabel("Loss", color="#3b82f6")
        ax5b.set_ylabel("KL", color="#ef4444")
        ax5.tick_params(axis="y", labelcolor="#3b82f6")
        ax5b.tick_params(axis="y", labelcolor="#ef4444")
        ax5.axhline(0, color="gray", linestyle=":", alpha=0.5)
        ax5.grid(alpha=0.2)
        lines = l1 + l2
        ax5.legend(lines, [l.get_label() for l in lines], loc="upper right", fontsize=8)
    else:
        ax5.text(0.5, 0.5, "No GRPO update stats logged\n(set --no-dry-run or upgrade train_grpo.py)",
                 transform=ax5.transAxes, ha="center", va="center", color="gray")
        ax5.set_title("GRPO update statistics", fontweight="bold")

    # ----- Panel 6: Episode-length distribution ---------------------------------
    ax6 = fig.add_subplot(gs[2, 2])
    if steps:
        ax6.hist(steps, bins=range(0, max(steps) + 2), color=phase_color,
                 alpha=0.85, edgecolor="white")
        ax6.axvline(sum(steps) / len(steps), color="#111827", linestyle="--",
                    linewidth=1.5, label=f"mean={sum(steps) / len(steps):.1f}")
        ax6.set_title("Episode length distribution", fontweight="bold")
        ax6.set_xlabel("Steps")
        ax6.set_ylabel("Episodes")
        ax6.legend(fontsize=8)
        ax6.grid(alpha=0.2, axis="y")
    else:
        ax6.set_visible(False)

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return True


# ---------------------------------------------------------------------------
# Cross-phase overview (single chart, all 3 phases on the same axes)
# ---------------------------------------------------------------------------

def _plot_all_phases_overview(metrics: List[Dict[str, Any]], out_path: str) -> None:
    import matplotlib.pyplot as plt

    if not metrics:
        return

    eps    = [m["episode"] for m in metrics]
    raw    = [m.get("raw_reward", 0.0) for m in metrics]
    rolling = [m.get("rolling_avg_reward", 0.0) for m in metrics]
    wr     = [m.get("rolling_win_rate", 0.0) for m in metrics]
    phase  = [int(m.get("phase", 1)) for m in metrics]

    fig, axes = plt.subplots(2, 1, figsize=(13, 7), sharex=True,
                             constrained_layout=True)
    fig.patch.set_facecolor("white")
    fig.suptitle("ER-MAP GRPO training — full curriculum overview",
                 fontsize=14, fontweight="bold", y=1.02)

    # Reward axis
    axes[0].scatter(eps, raw, alpha=0.30, s=14, color="#3b82f6",
                    label="raw episode reward")
    axes[0].plot(eps, rolling, linewidth=2.5, color="#111827",
                 label="rolling avg reward (w=20)")
    axes[0].axhline(0, color="gray", linestyle=":", alpha=0.5)
    axes[0].set_ylabel("Reward")
    axes[0].legend(loc="upper left", fontsize=9)
    axes[0].grid(alpha=0.25)

    # Win-rate axis
    axes[1].plot(eps, wr, linewidth=2.5, color="#10b981")
    axes[1].fill_between(eps, 0, wr, color="#10b981", alpha=0.18)
    axes[1].set_ylim(0, 1.0)
    axes[1].set_ylabel("Rolling win rate")
    axes[1].set_xlabel("Episode")
    axes[1].grid(alpha=0.25)

    # Phase boundary lines + colored bands
    for i in range(1, len(phase)):
        if phase[i] != phase[i - 1]:
            for ax in axes:
                ax.axvline(eps[i], color="red", linestyle=":", alpha=0.6, linewidth=1.5)
            axes[0].text(eps[i], axes[0].get_ylim()[1] * 0.92,
                         f" \u2192 Phase {phase[i]}", color="red",
                         fontsize=9, fontweight="bold")

    # Faint background colour bands per phase
    cur_phase = phase[0]
    band_start = eps[0]
    for i in range(1, len(phase) + 1):
        if i == len(phase) or phase[i] != cur_phase:
            band_end = eps[i - 1]
            for ax in axes:
                ax.axvspan(band_start, band_end,
                           color=PHASE_COLOR.get(cur_phase, "#3b82f6"),
                           alpha=0.06, zorder=0)
            if i < len(phase):
                cur_phase = phase[i]
                band_start = eps[i]

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Cross-phase summary bars
# ---------------------------------------------------------------------------

def _plot_phase_comparison(buckets: Dict[int, List[Dict[str, Any]]],
                           out_path: str) -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    if not buckets:
        return

    phases = sorted(buckets.keys())
    avg_reward    = [sum(m.get("raw_reward", 0.0) for m in buckets[p]) / max(len(buckets[p]), 1) for p in phases]
    win_rate      = [sum(1 for m in buckets[p] if m.get("outcome") == "WIN") / max(len(buckets[p]), 1) for p in phases]
    avg_steps     = [sum(m.get("steps", 0) for m in buckets[p]) / max(len(buckets[p]), 1) for p in phases]
    fatal_rate    = [sum(1 for m in buckets[p] if m.get("outcome") == "FATAL_LOSS") / max(len(buckets[p]), 1) for p in phases]

    fig, axes = plt.subplots(1, 4, figsize=(16, 4.2), constrained_layout=True)
    fig.patch.set_facecolor("white")
    fig.suptitle("Phase-by-phase comparison", fontsize=14, fontweight="bold", y=1.05)

    labels = [f"Phase {p}" for p in phases]
    colors = [PHASE_COLOR.get(p, "#3b82f6") for p in phases]
    metrics = [
        ("Avg raw reward",  avg_reward, "Reward",  "{:+.2f}"),
        ("Win rate",        win_rate,   "Rate",    "{:.0%}"),
        ("Avg steps",       avg_steps,  "Steps",   "{:.1f}"),
        ("Fatal-loss rate", fatal_rate, "Rate",    "{:.0%}"),
    ]
    for ax, (title, values, ylabel, fmt) in zip(axes, metrics):
        bars = ax.bar(labels, values, color=colors,
                      edgecolor="white", linewidth=0.6)
        ax.set_title(title, fontweight="bold")
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.2, axis="y")
        if title in ("Win rate", "Fatal-loss rate"):
            ax.set_ylim(0, max(1.0, max(values) * 1.15))
        for bar, v in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height(),
                    " " + fmt.format(v), ha="center", va="bottom",
                    fontsize=9, color="#111827")

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def plot_per_phase_dashboards(
    metrics_path: str,
    output_dir: str,
    *,
    phases: Optional[Sequence[int]] = None,
) -> Dict[str, str]:
    """
    Read a ``training_metrics.json`` file and produce:
        - one 6-panel dashboard PNG per phase that has episodes
        - one cross-phase overview PNG (all phases on shared axes)
        - one phase-comparison bar chart PNG

    Returns a dict mapping ``{logical_name: written_path}`` for every
    PNG that was actually written.
    """
    metrics = load_metrics(metrics_path)
    buckets = split_by_phase(metrics)

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    written: Dict[str, str] = {}

    # Per-phase dashboards
    target_phases = list(phases) if phases else sorted(buckets.keys())
    for p in target_phases:
        eps = buckets.get(int(p), [])
        if not eps:
            continue
        path = out_dir / f"phase{p}_dashboard.png"
        ok = _plot_phase_dashboard(int(p), eps, str(path))
        if ok:
            written[f"phase{p}_dashboard"] = str(path)

    # Cross-phase overview + comparison
    overview = out_dir / "all_phases_overview.png"
    _plot_all_phases_overview(metrics, str(overview))
    written["all_phases_overview"] = str(overview)

    comparison = out_dir / "all_phases_comparison.png"
    _plot_phase_comparison(buckets, str(comparison))
    written["all_phases_comparison"] = str(comparison)

    return written


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Baseline-eval plots (clean single-panel episode-vs-reward histograms)
# ---------------------------------------------------------------------------

def plot_baseline_phase_histogram(
    results: List[Dict[str, Any]],
    phase_id: int,
    out_path: str,
    *,
    title_suffix: str = "Baseline (no RL)",
) -> str:
    """
    Render a single clean episode-vs-reward bar chart for one phase of a
    baseline evaluation run. One bar per episode, colored by outcome.

    Returns the absolute path of the PNG that was written.
    """
    import matplotlib.pyplot as plt

    if not results:
        raise ValueError(f"No results to plot for phase {phase_id}")

    episodes = [r.get("episode", i + 1) for i, r in enumerate(results)]
    rewards  = [float(r.get("total_reward", 0.0)) for r in results]
    outcomes = [r.get("outcome", "unknown") for r in results]
    colors   = [OUTCOME_COLORS.get(o, OUTCOME_COLORS["unknown"]) for o in outcomes]

    fig, ax = plt.subplots(figsize=(12, 5.5), constrained_layout=True)
    fig.patch.set_facecolor("white")

    ax.bar(episodes, rewards, color=colors, edgecolor="white", linewidth=0.6,
           width=0.85)
    ax.axhline(0, color="#374151", linewidth=1, linestyle="-", alpha=0.6)

    avg = sum(rewards) / len(rewards)
    win_rate = sum(1 for o in outcomes if o == "WIN") / len(outcomes)
    mean_line = ax.axhline(avg, color="#111827", linewidth=1.5, linestyle="--",
                            alpha=0.85, label=f"mean = {avg:+.2f}")

    ax.set_xticks(episodes)
    ax.set_xticklabels([str(e) for e in episodes], fontsize=9)
    ax.set_xlabel("Episode", fontsize=12, fontweight="bold")
    ax.set_ylabel("Total reward", fontsize=12, fontweight="bold")
    ax.set_title(
        f"{title_suffix} — {PHASE_NAME.get(phase_id, f'Phase {phase_id}')}  "
        f"(n={len(results)}, win rate {win_rate:.0%})",
        fontsize=13, fontweight="bold",
        color=PHASE_COLOR.get(phase_id, "#3b82f6"),
        pad=12,
    )
    ax.grid(alpha=0.25, axis="y")

    # Single, unified legend: outcome swatches + the mean line, all in
    # one legend in the upper-right so the chart stays clean.
    used_outcomes = [o for o in OUTCOME_ORDER if o in outcomes]
    handles = [
        plt.Rectangle((0, 0), 1, 1, color=OUTCOME_COLORS[o], label=o)
        for o in used_outcomes
    ]
    handles.append(mean_line)
    ax.legend(handles=handles, loc="upper right", fontsize=9,
              framealpha=0.92, ncol=1, title="Outcome / mean",
              title_fontsize=9)

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return str(Path(out_path).resolve())


def plot_baseline_phase_comparison(
    results_by_phase: Dict[int, List[Dict[str, Any]]],
    out_path: str,
) -> str:
    """
    Cross-phase summary for a baseline run: 3 mini bar charts of
    win-rate / avg-reward / fatal-rate per phase. Useful as a single
    quick sanity glance before/after we plug in the trained model.
    """
    import matplotlib.pyplot as plt

    if not results_by_phase:
        raise ValueError("No baseline results to compare")

    phases = sorted(results_by_phase.keys())
    win_rate    = [sum(1 for r in results_by_phase[p] if r.get("outcome") == "WIN") /
                   max(len(results_by_phase[p]), 1) for p in phases]
    avg_reward  = [sum(float(r.get("total_reward", 0.0)) for r in results_by_phase[p]) /
                   max(len(results_by_phase[p]), 1) for p in phases]
    fatal_rate  = [sum(1 for r in results_by_phase[p]
                       if r.get("outcome") in ("FATAL", "FATAL_LOSS")) /
                   max(len(results_by_phase[p]), 1) for p in phases]

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.0), constrained_layout=True)
    fig.patch.set_facecolor("white")
    fig.suptitle("Baseline (no RL) — per-phase summary",
                 fontsize=14, fontweight="bold", y=1.06)

    labels = [f"Phase {p}" for p in phases]
    colors = [PHASE_COLOR.get(p, "#3b82f6") for p in phases]

    for ax, (title, vals, ylabel, fmt) in zip(axes, [
        ("Win rate",        win_rate,   "Rate",   "{:.0%}"),
        ("Avg reward",      avg_reward, "Reward", "{:+.2f}"),
        ("Fatal/Wrong rate", fatal_rate, "Rate",  "{:.0%}"),
    ]):
        bars = ax.bar(labels, vals, color=colors, edgecolor="white", linewidth=0.6)
        ax.set_title(title, fontweight="bold")
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.2, axis="y")
        if "rate" in title.lower():
            ax.set_ylim(0, max(1.0, (max(vals) if vals else 0) * 1.15))
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                    " " + fmt.format(v), ha="center", va="bottom",
                    fontsize=10, color="#111827")

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return str(Path(out_path).resolve())


def _cli() -> None:
    import argparse
    parser = argparse.ArgumentParser(
        description="Render per-phase + cross-phase ER-MAP training plots."
    )
    parser.add_argument(
        "--metrics", default="er_map_grpo_checkpoints/training_metrics.json",
        help="Path to training_metrics.json from train_grpo",
    )
    parser.add_argument(
        "--out", default="er_map_grpo_checkpoints/plots",
        help="Output directory for the rendered PNG files",
    )
    parser.add_argument(
        "--phases", default="", help="Optional comma-separated subset of phases (e.g. 1,2)"
    )
    args = parser.parse_args()

    phase_list = None
    if args.phases.strip():
        phase_list = [int(p.strip()) for p in args.phases.split(",") if p.strip()]

    written = plot_per_phase_dashboards(args.metrics, args.out, phases=phase_list)
    print("=" * 60)
    print(f"  Plotted {len(written)} chart(s):")
    for name, path in written.items():
        print(f"    {name:<28s} -> {path}")
    print("=" * 60)


if __name__ == "__main__":
    _cli()
