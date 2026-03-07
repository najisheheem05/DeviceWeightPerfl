"""
main.py — CLI entry point for the Federated Learning experiment.

Usage
─────
    # Standard FedAvg
    python -m fed_personality_fl.main --mode basic

    # Personality-weighted FL
    python -m fed_personality_fl.main --mode personality

    # Custom settings
    python -m fed_personality_fl.main --mode personality --rounds 15 --clients 5 --partition noniid

    # Run BOTH modes and generate comparison plot
    python -m fed_personality_fl.main --mode compare
"""

import argparse
import json
import os
import sys
from typing import Dict, List

import numpy as np

from . import config
from .utils import set_seed
from .server import run_simulation
from .dataset import get_partitioned_data, get_client_labels
from .personality import (
    generate_personality_metrics,
    compute_data_diversity,
    compute_personality_score,
)
from .visualize import (
    plot_accuracy_vs_rounds,
    plot_loss_vs_rounds,
    plot_personality_scores,
    plot_comparison,
)


def _extract_history(history) -> Dict[str, List[float]]:
    """
    Pull per-round accuracy and loss from a Flower History object.

    Flower stores distributed evaluation metrics as:
        history.metrics_distributed["accuracy"]  → [(round, value), ...]
        history.losses_distributed               → [(round, value), ...]
    """
    # Accuracies
    acc_tuples = history.metrics_distributed.get("accuracy", [])
    rounds_acc = [r for r, _ in acc_tuples]
    accuracies = [a for _, a in acc_tuples]

    # Losses
    loss_tuples = history.losses_distributed
    rounds_loss = [r for r, _ in loss_tuples]
    losses = [l for _, l in loss_tuples]

    # Use union of round numbers (they should match)
    rounds = rounds_acc if rounds_acc else rounds_loss

    return {
        "rounds": rounds,
        "accuracies": accuracies,
        "losses": losses,
    }


def _print_personality_table(
    num_clients: int,
    partition: str,
) -> tuple:
    """Print personality metrics for all clients and return data for plotting."""
    set_seed()
    client_indices, train_ds, _ = get_partitioned_data(partition, num_clients)

    print("\n╔══════════════════════════════════════════════════════════════════╗")
    print("║               DEVICE PERSONALITY METRICS                       ║")
    print("╠══════════╦════════════╦════════════╦═══════════╦═══════════╦════╣")
    print("║ Client   ║ Stability  ║ Reliability║ Compute   ║ Diversity ║ P_k║")
    print("╠══════════╬════════════╬════════════╬═══════════╬═══════════╬════╣")

    client_ids = []
    scores = []
    metrics_list = []

    for cid in range(num_clients):
        labels = get_client_labels(cid, client_indices, train_ds)
        metrics = generate_personality_metrics(cid)
        metrics["data_diversity"] = compute_data_diversity(labels)
        score = compute_personality_score(metrics)

        client_ids.append(cid)
        scores.append(score)
        metrics_list.append(metrics)

        print(
            f"║ Client {cid} ║   {metrics['stability']:.4f}  "
            f"║   {metrics['reliability']:.4f}  "
            f"║  {metrics['compute_power']:.4f}  "
            f"║  {metrics['data_diversity']:.4f}  "
            f"║{score:.4f}║"
        )

    print("╚══════════╩════════════╩════════════╩═══════════╩═══════════╩════╝\n")

    return client_ids, scores, metrics_list


def _run_single_mode(mode: str, args) -> Dict[str, List[float]]:
    """Run one FL experiment and generate individual plots."""
    print(f"\n{'='*60}")
    print(f"  RUNNING  {mode.upper()}  FEDERATED LEARNING")
    print(f"  Rounds: {args.rounds}  |  Clients: {args.clients}  |  Partition: {args.partition}")
    print(f"{'='*60}\n")

    history = run_simulation(
        mode=mode,
        num_rounds=args.rounds,
        num_clients=args.clients,
        partition=args.partition,
    )

    results = _extract_history(history)

    # Print per-round summary
    print(f"\n{'─'*40}")
    print(f"  Results — {mode.upper()} FL")
    print(f"{'─'*40}")
    for i, r in enumerate(results["rounds"]):
        acc = results["accuracies"][i] if i < len(results["accuracies"]) else "N/A"
        loss = results["losses"][i] if i < len(results["losses"]) else "N/A"
        if isinstance(acc, float):
            print(f"  Round {r:>2d}:  accuracy = {acc:.4f}   loss = {loss:.4f}")
        else:
            print(f"  Round {r:>2d}:  accuracy = {acc}   loss = {loss}")

    # Individual plots
    if results["rounds"]:
        plot_accuracy_vs_rounds(results["rounds"], results["accuracies"], mode)
        plot_loss_vs_rounds(results["rounds"], results["losses"], mode)

    # Save results as JSON for later comparison
    json_path = os.path.join(config.OUTPUT_DIR, f"results_{mode}.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  💾  Results saved → {json_path}")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Federated Learning with Device Personality Weighting",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["basic", "personality", "compare"],
        default="basic",
        help="FL mode: 'basic' (FedAvg), 'personality' (weighted), or 'compare' (run both)",
    )
    parser.add_argument(
        "--rounds", type=int, default=config.NUM_ROUNDS,
        help=f"Number of FL rounds (default: {config.NUM_ROUNDS})",
    )
    parser.add_argument(
        "--clients", type=int, default=config.NUM_CLIENTS,
        help=f"Number of simulated clients (default: {config.NUM_CLIENTS})",
    )
    parser.add_argument(
        "--partition", type=str, default=config.DATA_PARTITION,
        choices=["iid", "noniid"],
        help=f"Data partition strategy (default: {config.DATA_PARTITION})",
    )
    args = parser.parse_args()

    set_seed()

    # ── Print personality table (always useful context) ────────────────
    client_ids, scores, metrics_list = _print_personality_table(
        args.clients, args.partition,
    )
    plot_personality_scores(client_ids, scores, metrics_list)

    # ── Run requested mode(s) ──────────────────────────────────────────
    if args.mode == "compare":
        basic_results = _run_single_mode("basic", args)
        personality_results = _run_single_mode("personality", args)
        plot_comparison(basic_results, personality_results)
        print("\n✅  Comparison complete — see outputs/ for all plots.\n")
    else:
        _run_single_mode(args.mode, args)
        print(f"\n✅  {args.mode.upper()} FL experiment complete.\n")


if __name__ == "__main__":
    main()
