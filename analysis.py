"""
Option Pricing Tuning — Post-Run Analysis
==========================================
Run after an autoresearch session to visualize experiment progress.

    python analysis.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd


def load_results(path="results.tsv"):
    """Load the experiment log."""
    p = Path(path)
    if not p.exists():
        print(f"No results file found at {p}")
        sys.exit(1)
    df = pd.read_csv(p, sep="\t")
    df.index.name = "experiment"
    return df


def analyze(df):
    """Print summary statistics and experiment history."""
    n = len(df)
    n_keep = (df["status"] == "keep").sum()
    n_discard = (df["status"] == "discard").sum()
    n_crash = (df["status"] == "crash").sum()

    print("=" * 65)
    print("OPTION PRICING TUNING — EXPERIMENT ANALYSIS")
    print("=" * 65)
    print(f"\nTotal experiments:  {n}")
    print(f"  Kept:             {n_keep} ({100*n_keep/max(n,1):.0f}%)")
    print(f"  Discarded:        {n_discard} ({100*n_discard/max(n,1):.0f}%)")
    print(f"  Crashed:          {n_crash} ({100*n_crash/max(n,1):.0f}%)")

    if n == 0:
        return

    # Baseline vs best
    baseline = df.iloc[0]
    best_idx = df["combined_score"].idxmax()
    best = df.iloc[best_idx]

    print(f"\n{'Metric':<20} {'Baseline':>12} {'Best':>12} {'Δ':>12}")
    print("-" * 58)
    for col in ["combined_score", "sharpe", "mape"]:
        if col in df.columns:
            b = baseline[col]
            be = best[col]
            delta = be - b
            print(f"{col:<20} {b:>12.6f} {be:>12.6f} {delta:>+12.6f}")

    # All kept experiments
    kept = df[df["status"] == "keep"].copy()
    if len(kept) > 0:
        print(f"\n{'─' * 65}")
        print("KEPT EXPERIMENTS (chronological):")
        print(f"{'─' * 65}")
        for i, (_, row) in enumerate(kept.iterrows()):
            print(f"  {i+1:3d}. [{row['commit']:.7s}] "
                  f"score={row['combined_score']:.4f}  "
                  f"sharpe={row['sharpe']:.3f}  "
                  f"mape={row['mape']:.4f}  "
                  f"— {row['description']}")

    # Top improvements (by delta from previous best)
    if len(kept) > 1:
        kept["delta"] = kept["combined_score"].diff()
        top = kept.nlargest(5, "delta")
        print(f"\n{'─' * 65}")
        print("TOP 5 IMPROVEMENTS:")
        print(f"{'─' * 65}")
        for _, row in top.iterrows():
            print(f"  Δ={row['delta']:+.4f}  [{row['commit']:.7s}] {row['description']}")

    # Try to plot if matplotlib is available
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle("Option Pricing Tuning — Experiment Progress", fontsize=14)

        # 1. Combined score over time
        ax = axes[0, 0]
        colors = {"keep": "#2ecc71", "discard": "#e74c3c", "crash": "#95a5a6"}
        for status, color in colors.items():
            mask = df["status"] == status
            if mask.any():
                ax.scatter(df.index[mask], df.loc[mask, "combined_score"],
                          c=color, s=30, alpha=0.7, label=status)
        # Running best
        running_best = df["combined_score"].cummax()
        ax.plot(running_best, "k-", linewidth=1.5, alpha=0.5, label="running best")
        ax.set_xlabel("Experiment #")
        ax.set_ylabel("Combined Score")
        ax.set_title("Score Progress")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        # 2. Sharpe ratio over time
        ax = axes[0, 1]
        if "sharpe" in df.columns:
            for status, color in colors.items():
                mask = df["status"] == status
                if mask.any():
                    ax.scatter(df.index[mask], df.loc[mask, "sharpe"],
                              c=color, s=30, alpha=0.7)
            ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
            ax.set_xlabel("Experiment #")
            ax.set_ylabel("Sharpe Ratio")
            ax.set_title("Signal Quality (Sharpe)")
            ax.grid(True, alpha=0.3)

        # 3. Pricing MAPE over time
        ax = axes[1, 0]
        if "mape" in df.columns:
            for status, color in colors.items():
                mask = df["status"] == status
                if mask.any():
                    ax.scatter(df.index[mask], df.loc[mask, "mape"],
                              c=color, s=30, alpha=0.7)
            ax.set_xlabel("Experiment #")
            ax.set_ylabel("MAPE")
            ax.set_title("Pricing Accuracy (lower = better)")
            ax.grid(True, alpha=0.3)

        # 4. Pareto front: MAPE vs Sharpe
        ax = axes[1, 1]
        if "sharpe" in df.columns and "mape" in df.columns:
            for status, color in colors.items():
                mask = df["status"] == status
                if mask.any():
                    ax.scatter(df.loc[mask, "mape"], df.loc[mask, "sharpe"],
                              c=color, s=30, alpha=0.7, label=status)
            ax.set_xlabel("MAPE (lower = better)")
            ax.set_ylabel("Sharpe (higher = better)")
            ax.set_title("Pareto Front")
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig("progress.png", dpi=150)
        print(f"\nChart saved to progress.png")

    except ImportError:
        print("\nInstall matplotlib to generate charts: pip install matplotlib")


def main():
    df = load_results()
    analyze(df)


if __name__ == "__main__":
    main()
