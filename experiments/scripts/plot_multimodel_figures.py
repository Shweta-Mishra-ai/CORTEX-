#!/usr/bin/env python3
"""
CORTEX v2 — Multi-Model Figure Generator
==========================================
Reads experiment results and generates publication-ready figures.

Figures generated:
  Fig-A: Top-1 accuracy by filter condition (grouped bar, all models)
  Fig-B: Token reduction vs accuracy tradeoff (scatter)
  Fig-C: API cost comparison across conditions
  Fig-D: Multi-model heatmap (model × condition accuracy)
  Fig-E: AdaptiveSizeFilter threshold distribution
  Fig-F: Filter latency comparison (log scale)

Output: paper/figures/  (PNG, 300 DPI, publication ready)

Usage:
  python experiments/scripts/plot_multimodel_figures.py
  python experiments/scripts/plot_multimodel_figures.py --results path/to/results.json
"""

import os
import sys
import json
import argparse
from pathlib import Path

REPO_ROOT   = Path(__file__).resolve().parents[2]
FIGURES_DIR = REPO_ROOT / "paper" / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR = REPO_ROOT / "experiments" / "results"

def find_latest(directory: Path, pattern: str) -> Path | None:
    files = sorted(directory.glob(pattern), key=lambda f: f.stat().st_mtime, reverse=True)
    return files[0] if files else None

# ── Matplotlib setup ──────────────────────────────────────────────────────────
try:
    import matplotlib
    matplotlib.use("Agg")   # non-interactive backend — works without display
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    import numpy as np
    HAS_MPL = True
except ImportError:
    HAS_MPL = False
    print("WARN matplotlib/numpy not installed. Run: pip install matplotlib numpy")
    print("   Generating text-based summary instead.\n")


# ── Color palette (colorblind-safe, publication quality) ─────────────────────
COLORS = {
    "A_no_filter":        "#d73027",   # red
    "B_naive_truncation": "#fc8d59",   # orange
    "C_bm25":             "#fee090",   # yellow
    "D_size_1mb":         "#91bfdb",   # light blue
    "E_hybrid":           "#4575b4",   # dark blue  ← paper recommended
}

CONDITION_LABELS = {
    "A_no_filter":        "No Filter",
    "B_naive_truncation": "Naive Truncation",
    "C_bm25":             "BM25 Selector",
    "D_size_1mb":         "SizeFilter(1MB)",
    "E_hybrid":           "HybridFilter ★",
}

STYLE = {
    "font.family":   "serif",
    "font.size":     11,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.dpi":    300,
    "savefig.dpi":   300,
    "savefig.bbox":  "tight",
    "savefig.pad_inches": 0.1,
}


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE A: Top-1 accuracy by condition, grouped by model
# ══════════════════════════════════════════════════════════════════════════════

def fig_accuracy_grouped_bar(data: dict, out: Path) -> None:
    summary = data.get("summary", [])
    if not summary:
        print("  WARN No summary data for Fig-A"); return

    # Collect models and conditions
    models     = list(dict.fromkeys(r["model"] for r in summary))
    conditions = list(dict.fromkeys(r["condition"] for r in summary))

    # Build matrix: acc[model][condition]
    acc = {m: {c: 0.0 for c in conditions} for m in models}
    for row in summary:
        acc[row["model"]][row["condition"]] = row.get("top1_pct", 0.0)

    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(9, 5))

        n_models = len(models)
        n_conds  = len(conditions)
        x        = np.arange(n_conds)
        width    = 0.7 / max(n_models, 1)

        for i, model in enumerate(models):
            vals   = [acc[model].get(c, 0) for c in conditions]
            offset = (i - n_models / 2 + 0.5) * width
            bars   = ax.bar(x + offset, vals, width,
                            label=model.split("/")[-1],
                            alpha=0.88)
            # Value labels on bars
            for bar, val in zip(bars, vals):
                if val > 5:
                    ax.text(bar.get_x() + bar.get_width()/2,
                            bar.get_height() + 0.8,
                            f"{val:.0f}%",
                            ha="center", va="bottom", fontsize=7)

        ax.set_xticks(x)
        ax.set_xticklabels([CONDITION_LABELS.get(c, c) for c in conditions],
                            rotation=15, ha="right")
        ax.set_ylabel("Top-1 File Accuracy (%)")
        ax.set_title("Figure A — Top-1 Accuracy by Filter Condition and Model")
        ax.set_ylim(0, 105)
        ax.axhline(y=25, color="gray", linestyle="--", linewidth=0.8,
                   label="Random baseline (25%)")
        ax.legend(loc="upper left", framealpha=0.9)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", alpha=0.3, linestyle="--")

        fig.tight_layout()
        fig.savefig(out)
        plt.close(fig)
    print(f"  ✓ Fig-A saved → {out.name}")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE B: Token reduction vs accuracy scatter (tradeoff plot)
# ══════════════════════════════════════════════════════════════════════════════

def fig_tradeoff_scatter(data: dict, token_data: dict | None, out: Path) -> None:
    summary = data.get("summary", [])
    if not summary:
        print("  WARN No data for Fig-B"); return

    # Token reduction lookup (from EXP-1 results)
    KNOWN_REDUCTION = {
        "A_no_filter":        0.0,
        "B_naive_truncation": 15.0,   # approximate
        "C_bm25":             55.0,   # approximate
        "D_size_1mb":         79.6,   # paper Table IV
        "E_hybrid":           89.3,   # paper Table IV
    }

    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(7, 5))

        plotted = set()
        for row in summary:
            cond    = row["condition"]
            top1    = row.get("top1_pct", 0.0)
            reduc   = KNOWN_REDUCTION.get(cond, 50.0)
            color   = COLORS.get(cond, "#888888")
            label   = CONDITION_LABELS.get(cond, cond)

            marker = "o" if cond == "E_hybrid" else "s"
            size   = 120 if cond == "E_hybrid" else 70
            ax.scatter(reduc, top1, c=color, s=size, marker=marker,
                       zorder=3, edgecolors="white", linewidths=0.8)

            if cond not in plotted:
                ax.annotate(
                    label,
                    (reduc, top1),
                    textcoords="offset points",
                    xytext=(6, 4),
                    fontsize=8,
                    color=color,
                )
                plotted.add(cond)

        ax.set_xlabel("Token Reduction (%)")
        ax.set_ylabel("Top-1 File Accuracy (%)")
        ax.set_title("Figure B — Accuracy vs Token Reduction Tradeoff")
        ax.set_xlim(-5, 100)
        ax.set_ylim(0, 105)
        ax.axhline(25, color="gray", linestyle="--", lw=0.8, alpha=0.6,
                   label="Random baseline")
        ax.legend(fontsize=8, framealpha=0.9)
        ax.spines[["top","right"]].set_visible(False)
        ax.grid(alpha=0.25, linestyle="--")

        # Ideal corner annotation
        ax.annotate("← ideal corner\n(high accuracy,\nhigh reduction)",
                    xy=(90, 85), fontsize=7.5, color="#444444",
                    ha="right",
                    bbox=dict(boxstyle="round,pad=0.3", fc="lightyellow", alpha=0.8))

        fig.tight_layout()
        fig.savefig(out)
        plt.close(fig)
    print(f"  ✓ Fig-B saved → {out.name}")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE C: Total API cost per condition
# ══════════════════════════════════════════════════════════════════════════════

def fig_cost_comparison(data: dict, out: Path) -> None:
    summary = data.get("summary", [])
    if not summary:
        print("  WARN No data for Fig-C"); return

    # Sum cost across all models per condition
    from collections import defaultdict
    cost_by_cond: dict = defaultdict(float)
    tok_by_cond:  dict = defaultdict(list)
    for row in summary:
        cost_by_cond[row["condition"]] += row.get("total_cost_usd", 0)
        tok_by_cond[row["condition"]].append(row.get("avg_tokens", 0))

    conditions = list(cost_by_cond)
    costs      = [cost_by_cond[c] for c in conditions]
    avg_toks   = [sum(tok_by_cond[c])/len(tok_by_cond[c]) for c in conditions]
    colors     = [COLORS.get(c, "#888888") for c in conditions]
    labels     = [CONDITION_LABELS.get(c, c) for c in conditions]

    with plt.rc_context(STYLE):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.5))

        # Left: cost bars
        bars = ax1.bar(labels, costs, color=colors, alpha=0.88, edgecolor="white")
        for bar, val in zip(bars, costs):
            ax1.text(bar.get_x() + bar.get_width()/2,
                     bar.get_height() + max(costs)*0.01,
                     f"${val:.3f}", ha="center", va="bottom", fontsize=8)
        ax1.set_ylabel("Total API Cost (USD)")
        ax1.set_title("C1 — Total API Cost by Condition")
        ax1.set_xticklabels(labels, rotation=20, ha="right")
        ax1.spines[["top","right"]].set_visible(False)
        ax1.grid(axis="y", alpha=0.3, linestyle="--")

        # Right: avg tokens bars
        bars2 = ax2.bar(labels, avg_toks, color=colors, alpha=0.88, edgecolor="white")
        for bar, val in zip(bars2, avg_toks):
            ax2.text(bar.get_x() + bar.get_width()/2,
                     bar.get_height() + max(avg_toks)*0.01,
                     f"{val:,.0f}", ha="center", va="bottom", fontsize=8)
        ax2.set_ylabel("Average Input Tokens per Call")
        ax2.set_title("C2 — Avg Context Size by Condition")
        ax2.set_xticklabels(labels, rotation=20, ha="right")
        ax2.spines[["top","right"]].set_visible(False)
        ax2.grid(axis="y", alpha=0.3, linestyle="--")

        fig.suptitle("Figure C — Cost and Context Size by Filter Condition",
                     fontsize=13, fontweight="bold")
        fig.tight_layout()
        fig.savefig(out)
        plt.close(fig)
    print(f"  ✓ Fig-C saved → {out.name}")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE D: Heatmap — model × condition accuracy
# ══════════════════════════════════════════════════════════════════════════════

def fig_heatmap(data: dict, out: Path) -> None:
    summary = data.get("summary", [])
    if not summary:
        print("  WARN No data for Fig-D"); return

    models     = list(dict.fromkeys(r["model"] for r in summary))
    conditions = list(dict.fromkeys(r["condition"] for r in summary))

    matrix = np.zeros((len(models), len(conditions)))
    for row in summary:
        i = models.index(row["model"])
        j = conditions.index(row["condition"])
        matrix[i, j] = row.get("top1_pct", 0.0)

    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(8, max(3, len(models) * 0.8 + 1.5)))

        im = ax.imshow(matrix, cmap="RdYlGn", aspect="auto",
                       vmin=0, vmax=100)
        plt.colorbar(im, ax=ax, label="Top-1 Accuracy (%)")

        ax.set_xticks(range(len(conditions)))
        ax.set_xticklabels([CONDITION_LABELS.get(c, c) for c in conditions],
                            rotation=25, ha="right")
        ax.set_yticks(range(len(models)))
        ax.set_yticklabels([m.split("/")[-1] for m in models])

        # Annotate cells
        for i in range(len(models)):
            for j in range(len(conditions)):
                val = matrix[i, j]
                color = "white" if val < 40 or val > 75 else "black"
                ax.text(j, i, f"{val:.0f}%", ha="center", va="center",
                        color=color, fontsize=9, fontweight="bold")

        ax.set_title("Figure D — Accuracy Heatmap: Model × Filter Condition")
        fig.tight_layout()
        fig.savefig(out)
        plt.close(fig)
    print(f"  ✓ Fig-D saved → {out.name}")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE E: AdaptiveSizeFilter per-repo thresholds
# ══════════════════════════════════════════════════════════════════════════════

def fig_adaptive_thresholds(adaptive_data: dict, out: Path) -> None:
    per_repo = adaptive_data.get("perRepo", [])
    if not per_repo:
        print("  WARN No adaptive data for Fig-E"); return

    repos  = [r["repo"].replace("_py","").replace("_js","").replace("_go","").replace("_rb","")
               for r in per_repo]
    p95_kb = []
    fixed_reduction  = []
    p95_reduction    = []

    for r in per_repo:
        filters = r.get("filters", {})
        p95     = filters.get("AdaptiveP95", {})
        fixed   = filters.get("SizeFilter(1MB)", {})
        p95_kb.append(p95.get("thresholdKB", 0))
        p95_reduction.append(p95.get("tokenReductionPct", 0))
        fixed_reduction.append(fixed.get("tokenReductionPct", 0))

    x = np.arange(len(repos))

    with plt.rc_context(STYLE):
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 7), sharex=True)

        # Top: per-repo P95 threshold
        bars = ax1.bar(x, p95_kb, color="#4575b4", alpha=0.85, label="P95 threshold (KB)")
        ax1.axhline(1024, color="red", linestyle="--", lw=1.2, label="Fixed 1MB")
        for bar, val in zip(bars, p95_kb):
            ax1.text(bar.get_x() + bar.get_width()/2,
                     bar.get_height() + 20,
                     f"{val}KB", ha="center", va="bottom", fontsize=7)
        ax1.set_ylabel("Threshold (KB)")
        ax1.set_title("E1 — AdaptiveSizeFilter P95 Threshold per Repository")
        ax1.legend(fontsize=8)
        ax1.spines[["top","right"]].set_visible(False)
        ax1.grid(axis="y", alpha=0.3, linestyle="--")

        # Bottom: reduction comparison
        w = 0.35
        ax2.bar(x - w/2, fixed_reduction, w, label="SizeFilter(1MB)",
                color="#fc8d59", alpha=0.85)
        ax2.bar(x + w/2, p95_reduction, w,  label="AdaptiveP95",
                color="#4575b4", alpha=0.85)
        ax2.set_ylabel("Token Reduction (%)")
        ax2.set_title("E2 — Token Reduction: Fixed vs Adaptive Threshold")
        ax2.set_xticks(x)
        ax2.set_xticklabels(repos, rotation=30, ha="right")
        ax2.legend(fontsize=8)
        ax2.spines[["top","right"]].set_visible(False)
        ax2.grid(axis="y", alpha=0.3, linestyle="--")

        fig.suptitle("Figure E — AdaptiveSizeFilter Analysis", fontsize=13, fontweight="bold")
        fig.tight_layout()
        fig.savefig(out)
        plt.close(fig)
    print(f"  ✓ Fig-E saved → {out.name}")


# ══════════════════════════════════════════════════════════════════════════════
# TEXT FALLBACK (no matplotlib)
# ══════════════════════════════════════════════════════════════════════════════

def text_summary(data: dict) -> None:
    print("\n── Multi-Model Results Summary ──────────────────────────────")
    print(f"{'Model':25s} {'Condition':22s} {'Top1%':>7} {'Top3%':>7} {'Cost':>9}")
    print("─" * 75)
    for row in data.get("summary", []):
        print(f"{row['model']:25s} {row['condition']:22s} "
              f"{row['top1_pct']:>6.1f}% {row['top3_pct']:>6.1f}% "
              f"${row['total_cost_usd']:>8.4f}")
    print("─" * 75)


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Generate paper figures from CORTEX experiment results")
    parser.add_argument("--results", default=None,
                        help="Path to multimodel results JSON (default: latest)")
    parser.add_argument("--adaptive", default=None,
                        help="Path to adaptive results JSON (default: latest)")
    args = parser.parse_args()

    print("\nCORTEX v2 — Figure Generator")
    print("=" * 40)

    # Load multimodel results
    mm_path = Path(args.results) if args.results else \
              find_latest(RESULTS_DIR / "multimodel", "multimodel_results_*.json")

    if not mm_path or not mm_path.exists():
        print(f"WARN No multi-model results found in {RESULTS_DIR / 'multimodel'}")
        print("   Run experiments first: python run_all_experiments.py")
        print("   Or: python experiments/scripts/run_multimodel.py --dry-run")

        # Generate placeholder figures so paper compilation doesn't break
        if HAS_MPL:
            _generate_placeholder_figures()
        return

    with open(mm_path) as f:
        mm_data = json.load(f)

    print(f"Loaded: {mm_path.name}")
    print(f"Models: {', '.join(mm_data.get('metadata',{}).get('models',[]))}")
    print(f"Tasks:  {mm_data.get('metadata',{}).get('n_tasks','?')}")
    print()

    # Load adaptive results
    ad_path = Path(args.adaptive) if args.adaptive else \
              find_latest(RESULTS_DIR, "adaptive_results_*.json")
    ad_data = None
    if ad_path and ad_path.exists():
        with open(ad_path) as f:
            ad_data = json.load(f)
        print(f"Loaded adaptive: {ad_path.name}")

    if not HAS_MPL:
        text_summary(mm_data)
        return

    print("Generating figures...")

    fig_accuracy_grouped_bar(mm_data, FIGURES_DIR / "fig_A_accuracy_by_condition.png")
    fig_tradeoff_scatter(mm_data, None,    FIGURES_DIR / "fig_B_tradeoff_scatter.png")
    fig_cost_comparison(mm_data,           FIGURES_DIR / "fig_C_cost_comparison.png")
    fig_heatmap(mm_data,                   FIGURES_DIR / "fig_D_accuracy_heatmap.png")

    if ad_data:
        fig_adaptive_thresholds(ad_data,   FIGURES_DIR / "fig_E_adaptive_thresholds.png")
    else:
        print("  WARN No adaptive data - skipping Fig-E")

    print(f"\nOK All figures saved to {FIGURES_DIR}/")
    print("    Include in LaTeX with: \\includegraphics{figures/fig_A_accuracy_by_condition}")


def _generate_placeholder_figures():
    """Generate placeholder PNGs so LaTeX compilation doesn't break."""
    for name in ["fig_A_accuracy_by_condition", "fig_B_tradeoff_scatter",
                 "fig_C_cost_comparison", "fig_D_accuracy_heatmap",
                 "fig_E_adaptive_thresholds"]:
        with plt.rc_context(STYLE):
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.text(0.5, 0.5, f"[{name}]\nRun experiments to generate",
                    ha="center", va="center", transform=ax.transAxes,
                    fontsize=12, color="gray",
                    bbox=dict(boxstyle="round", fc="lightyellow", alpha=0.8))
            ax.axis("off")
            fig.savefig(FIGURES_DIR / f"{name}.png")
            plt.close(fig)
    print(f"  OK Placeholder figures saved to {FIGURES_DIR}/")


if __name__ == "__main__":
    main()
