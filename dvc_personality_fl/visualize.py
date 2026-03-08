"""
visualize.py — Matplotlib plotting utilities for FL experiments.

Generates and saves:
  1. Accuracy vs. rounds
  2. Loss vs. rounds
  3. Per-client personality scores
  4. Side-by-side comparison of Basic FL vs. Personality FL
"""

import os
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import matplotlib
import numpy as np

from . import config

matplotlib.use("Agg")  # Non-interactive backend for saving PNGs


def _ensure_output_dir():
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)


# ── Individual plots ──────────────────────────────────────────────────

def plot_accuracy_vs_rounds(
    rounds: List[int],
    accuracies: List[float],
    mode: str,
    save: bool = True,
) -> None:
    """Plot accuracy over FL rounds for a single mode."""
    _ensure_output_dir()
    plt.figure(figsize=(8, 5))
    plt.plot(rounds, accuracies, marker="o", linewidth=2, color="#2196F3")
    plt.title(f"Accuracy vs. Rounds  ({mode.upper()} FL)", fontsize=14)
    plt.xlabel("Round")
    plt.ylabel("Accuracy")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    if save:
        path = os.path.join(config.OUTPUT_DIR, f"accuracy_{mode}.png")
        plt.savefig(path, dpi=150)
        print(f"  📊  Saved → {path}")
    plt.close()


def plot_loss_vs_rounds(
    rounds: List[int],
    losses: List[float],
    mode: str,
    save: bool = True,
) -> None:
    """Plot loss over FL rounds for a single mode."""
    _ensure_output_dir()
    plt.figure(figsize=(8, 5))
    plt.plot(rounds, losses, marker="s", linewidth=2, color="#FF5722")
    plt.title(f"Loss vs. Rounds  ({mode.upper()} FL)", fontsize=14)
    plt.xlabel("Round")
    plt.ylabel("Loss")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    if save:
        path = os.path.join(config.OUTPUT_DIR, f"loss_{mode}.png")
        plt.savefig(path, dpi=150)
        print(f"  📊  Saved → {path}")
    plt.close()


def plot_personality_scores(
    client_ids: List[int],
    scores: List[float],
    metrics_list: Optional[List[Dict[str, float]]] = None,
    save: bool = True,
) -> None:
    """
    Bar chart of personality scores per client.

    If *metrics_list* is provided, a stacked breakdown of individual
    components is shown instead.
    """
    _ensure_output_dir()
    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(client_ids))
    width = 0.5

    if metrics_list is not None:
        w = config.PERSONALITY_WEIGHTS
        bottom = np.zeros(len(client_ids))
        colors = ["#4CAF50", "#2196F3", "#FF9800", "#9C27B0"]
        labels = ["Stability", "Reliability", "Compute Power", "Data Diversity"]
        keys = ["stability", "reliability", "compute_power", "data_diversity"]
        weight_keys = keys  # same order

        for i, (key, label, color) in enumerate(zip(keys, labels, colors)):
            vals = np.array([m[key] * w[key] for m in metrics_list])
            ax.bar(x, vals, width, bottom=bottom, label=label, color=color, alpha=0.85)
            bottom += vals
    else:
        ax.bar(x, scores, width, color="#7C4DFF", alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels([f"Client {cid}" for cid in client_ids])
    ax.set_ylabel("Personality Score")
    ax.set_title("Per-Client Personality Scores", fontsize=14)
    if metrics_list is not None:
        ax.legend(loc="upper right")
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()

    if save:
        path = os.path.join(config.OUTPUT_DIR, "personality_scores.png")
        plt.savefig(path, dpi=150)
        print(f"  📊  Saved → {path}")
    plt.close()


def plot_comparison(
    basic_results: Dict[str, List[float]],
    personality_results: Dict[str, List[float]],
    save: bool = True,
) -> None:
    """
    Side-by-side comparison of Basic FL vs. Personality FL.

    Parameters
    ----------
    basic_results / personality_results : dict
        Must contain keys: "rounds", "accuracies", "losses"
    """
    _ensure_output_dir()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Accuracy comparison
    ax1.plot(
        basic_results["rounds"], basic_results["accuracies"],
        marker="o", label="Basic FL", color="#2196F3", linewidth=2,
    )
    ax1.plot(
        personality_results["rounds"], personality_results["accuracies"],
        marker="^", label="Personality FL", color="#4CAF50", linewidth=2,
    )
    ax1.set_title("Accuracy Comparison", fontsize=14)
    ax1.set_xlabel("Round")
    ax1.set_ylabel("Accuracy")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Loss comparison
    ax2.plot(
        basic_results["rounds"], basic_results["losses"],
        marker="s", label="Basic FL", color="#FF5722", linewidth=2,
    )
    ax2.plot(
        personality_results["rounds"], personality_results["losses"],
        marker="D", label="Personality FL", color="#FF9800", linewidth=2,
    )
    ax2.set_title("Loss Comparison", fontsize=14)
    ax2.set_xlabel("Round")
    ax2.set_ylabel("Loss")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    if save:
        path = os.path.join(config.OUTPUT_DIR, "comparison_basic_vs_personality.png")
        plt.savefig(path, dpi=150)
        print(f"  📊  Saved → {path}")
    plt.close()
