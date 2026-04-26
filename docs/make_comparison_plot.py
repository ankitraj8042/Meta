"""
docs/make_comparison_plot.py
============================
Build the trained-vs-baseline comparison plot for the README + blog + video.

Reads:
  - baseline_eval/baseline_results.json
        per-episode rollouts of the untrained Doctor (one record per
        episode, fields: phase, total_reward, outcome, steps, ...)
  - er_map_grpo_checkpoints/training_metrics.json
        per-update GRPO trace from train_grpo.py / Kaggle notebook
        (fields used: phase, group_avg_reward, ...)
        *Optional fallback*: er_map_grpo_checkpoints/eval_results.json
        if you ran a clean post-training eval pass with --episodes-per-phase.

Writes:
  docs/plots/trained_vs_baseline.png          (2x2 panel for README/blog)
  docs/plots/trained_vs_baseline_summary.json (numeric summary used by blog)

Usage:
    python docs/make_comparison_plot.py
    python docs/make_comparison_plot.py --metrics path/to/training_metrics.json
    python docs/make_comparison_plot.py --window 20  # rolling avg window

Designed to gracefully degrade: if the trained metrics aren't downloaded yet
the script still builds a baseline-only plot so you have something to ship
while training is running.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_BASELINE = REPO_ROOT / "baseline_eval" / "baseline_results.json"
DEFAULT_TRAINED = REPO_ROOT / "er_map_grpo_checkpoints" / "training_metrics.json"
DEFAULT_OUT_PNG = REPO_ROOT / "docs" / "plots" / "trained_vs_baseline.png"
DEFAULT_OUT_JSON = REPO_ROOT / "docs" / "plots" / "trained_vs_baseline_summary.json"


def _load_baseline(path: Path) -> dict[int, list[dict]]:
    """Group baseline records by phase. Empty dict if path missing."""
    if not path.is_file():
        return {}
    raw = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(raw, dict) and "results" in raw:
        records = raw["results"]
    else:
        records = raw
    by_phase: dict[int, list[dict]] = defaultdict(list)
    for rec in records:
        ph = int(rec.get("phase", 0))
        by_phase[ph].append(rec)
    return by_phase


def _load_trained(path: Path) -> list[dict]:
    """Read the GRPO training trace. Returns [] if file missing."""
    if not path.is_file():
        return []
    raw = json.loads(path.read_text(encoding="utf-8"))
    # Accept either a flat list of updates, or {"updates":[...]} form
    if isinstance(raw, dict) and "updates" in raw:
        return raw["updates"]
    if isinstance(raw, list):
        return raw
    return []


def _rolling(xs: list[float], window: int) -> list[float]:
    out = []
    for i in range(len(xs)):
        lo = max(0, i - window + 1)
        chunk = xs[lo : i + 1]
        out.append(sum(chunk) / len(chunk))
    return out


def _summary(by_phase: dict[int, list[dict]]) -> dict[int, dict]:
    s: dict[int, dict] = {}
    for ph, recs in by_phase.items():
        if not recs:
            continue
        n = len(recs)
        wins = sum(1 for r in recs if r.get("outcome") == "WIN")
        ama = sum(1 for r in recs if r.get("outcome") == "AMA")
        wrong = sum(1 for r in recs if r.get("outcome") in ("WRONG", "FATAL"))
        avg = mean(r["total_reward"] for r in recs)
        s[ph] = {
            "n": n,
            "win_rate": wins / n,
            "ama_rate": ama / n,
            "wrong_rate": wrong / n,
            "avg_reward": avg,
        }
    return s


def _plot(
    baseline_by_phase: dict[int, list[dict]],
    trained_updates: list[dict],
    window: int,
    out_png: Path,
) -> dict:
    out_png.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(13, 8.5), constrained_layout=True)

    # ----- TL: per-phase avg reward bar comparison -----
    ax = axes[0, 0]
    phases = [1, 2, 3]
    summary_b = _summary(baseline_by_phase)
    avgs_b = [summary_b.get(p, {}).get("avg_reward", 0.0) for p in phases]
    ax.bar([p - 0.18 for p in phases], avgs_b, width=0.36,
           label="Baseline (no RL)", color="#999999")

    # If we have trained metrics, derive per-phase post-training avg
    summary_t: dict[int, dict] = {}
    if trained_updates:
        groups_by_phase: dict[int, list[float]] = defaultdict(list)
        for upd in trained_updates:
            ph = int(upd.get("phase", upd.get("current_phase", 0)) or 0)
            r = upd.get("group_avg_reward",
                        upd.get("rolling_avg_reward",
                                upd.get("avg_reward")))
            if r is None:
                continue
            groups_by_phase[ph].append(float(r))
        # Use *last 20%* of each phase as "trained" estimate (post-curriculum)
        for ph, vals in groups_by_phase.items():
            if not vals:
                continue
            tail = vals[max(1, int(0.8 * len(vals))):]
            summary_t[ph] = {"n": len(tail), "avg_reward": mean(tail)}
        avgs_t = [summary_t.get(p, {}).get("avg_reward", 0.0) for p in phases]
        ax.bar([p + 0.18 for p in phases], avgs_t, width=0.36,
               label="Trained (GRPO)", color="#3b7dd8")

    ax.set_xticks(phases)
    ax.set_xticklabels(["Phase 1\nTool Mastery",
                        "Phase 2\nReasoning",
                        "Phase 3\nEmpathy"])
    ax.set_ylabel("Average episode reward")
    ax.set_title("Reward improvement: baseline vs trained")
    ax.axhline(0, color="black", linewidth=0.5)
    ax.legend()
    ax.grid(True, axis="y", alpha=0.25)

    # ----- TR: baseline outcome breakdown -----
    ax = axes[0, 1]
    bars_data = {"WIN": [], "AMA": [], "WRONG/FATAL": [], "OTHER": []}
    for ph in phases:
        recs = baseline_by_phase.get(ph, [])
        n = max(1, len(recs))
        wins = sum(1 for r in recs if r.get("outcome") == "WIN")
        ama = sum(1 for r in recs if r.get("outcome") == "AMA")
        wrong = sum(1 for r in recs if r.get("outcome") in ("WRONG", "FATAL"))
        other = n - wins - ama - wrong
        bars_data["WIN"].append(wins / n)
        bars_data["AMA"].append(ama / n)
        bars_data["WRONG/FATAL"].append(wrong / n)
        bars_data["OTHER"].append(other / n)

    bottom = [0.0] * 3
    colors = {"WIN": "#3eb16f", "AMA": "#e08a3a",
              "WRONG/FATAL": "#c84d4d", "OTHER": "#aaaaaa"}
    for label, vals in bars_data.items():
        ax.bar(phases, vals, bottom=bottom, label=label, color=colors[label])
        bottom = [b + v for b, v in zip(bottom, vals)]
    ax.set_xticks(phases)
    ax.set_xticklabels(["Phase 1", "Phase 2", "Phase 3"])
    ax.set_ylabel("Fraction of episodes")
    ax.set_title("Baseline outcomes by curriculum phase")
    ax.set_ylim(0, 1.0)
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, axis="y", alpha=0.25)

    # ----- BL: GRPO learning curve -----
    ax = axes[1, 0]
    if trained_updates:
        rewards = [
            float(u.get("group_avg_reward",
                        u.get("rolling_avg_reward",
                              u.get("avg_reward", 0.0))))
            for u in trained_updates
        ]
        rolling = _rolling(rewards, window)
        x = list(range(1, len(rewards) + 1))
        ax.plot(x, rewards, color="#bbbbbb", linewidth=0.9, label="per group")
        ax.plot(x, rolling, color="#3b7dd8", linewidth=2.0,
                label=f"rolling avg (w={window})")

        # Phase-boundary markers
        last_phase = None
        for i, u in enumerate(trained_updates, start=1):
            ph = u.get("phase", u.get("current_phase"))
            if ph is not None and ph != last_phase:
                if last_phase is not None:
                    ax.axvline(i, color="black", linewidth=0.6,
                               linestyle="--", alpha=0.6)
                ax.text(i, ax.get_ylim()[1] * 0.98 if i > 1 else 0.02,
                        f" P{ph}", fontsize=9, color="black",
                        verticalalignment="top")
                last_phase = ph

        ax.set_xlabel("GRPO update step")
        ax.set_ylabel("Group avg reward")
        ax.set_title("Training curve (GRPO + curriculum)")
        ax.legend(loc="lower right", fontsize=9)
        ax.grid(True, alpha=0.25)
    else:
        ax.text(0.5, 0.5,
                "Training metrics not found yet.\n"
                f"Expected at:\n{DEFAULT_TRAINED.relative_to(REPO_ROOT)}\n\n"
                "Run after Kaggle training completes.",
                ha="center", va="center", fontsize=10, color="#666",
                transform=ax.transAxes)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title("Training curve (GRPO + curriculum)")

    # ----- BR: baseline reward histogram per phase -----
    ax = axes[1, 1]
    for ph, color in zip(phases, ["#3b7dd8", "#e08a3a", "#c84d4d"]):
        recs = baseline_by_phase.get(ph, [])
        if not recs:
            continue
        rewards = [r["total_reward"] for r in recs]
        ax.hist(rewards, bins=12, alpha=0.55, label=f"Phase {ph} (n={len(rewards)})",
                color=color)
    ax.set_xlabel("Episode total reward (baseline)")
    ax.set_ylabel("Episodes")
    ax.set_title("Baseline reward distribution by phase")
    ax.axvline(0, color="black", linewidth=0.5)
    ax.legend(fontsize=9)
    ax.grid(True, axis="y", alpha=0.25)

    fig.suptitle(
        "ER-MAP — Trained vs Baseline (Doctor agent on TriageEnv)",
        fontsize=14, fontweight="bold",
    )
    fig.savefig(out_png, dpi=140)
    plt.close(fig)

    return {
        "baseline": _summary(baseline_by_phase),
        "trained_phase_means": summary_t,
        "rolling_window": window,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", default=str(DEFAULT_BASELINE))
    ap.add_argument("--metrics", default=str(DEFAULT_TRAINED))
    ap.add_argument("--out-png", default=str(DEFAULT_OUT_PNG))
    ap.add_argument("--out-json", default=str(DEFAULT_OUT_JSON))
    ap.add_argument("--window", type=int, default=10,
                    help="Rolling avg window for the GRPO learning curve")
    args = ap.parse_args()

    baseline_by_phase = _load_baseline(Path(args.baseline))
    trained_updates = _load_trained(Path(args.metrics))

    if not baseline_by_phase:
        print(f"WARN: no baseline data at {args.baseline}; plot will be empty.")
    else:
        n = sum(len(v) for v in baseline_by_phase.values())
        print(f"Baseline: {n} episodes across {len(baseline_by_phase)} phase(s).")

    if trained_updates:
        print(f"Trained: {len(trained_updates)} GRPO update records.")
    else:
        print(f"Trained metrics not found at {args.metrics} — plotting baseline only.")

    summary = _plot(
        baseline_by_phase=baseline_by_phase,
        trained_updates=trained_updates,
        window=args.window,
        out_png=Path(args.out_png),
    )
    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_json).write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"\nWrote {args.out_png}")
    print(f"Wrote {args.out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
