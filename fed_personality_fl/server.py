"""
server.py — Federated learning server orchestration.

Uses Flower's simulation API to run FL with 5 local clients.
Depending on the selected mode the server picks:
  • "basic"       → standard FedAvg
  • "personality" → PersonalityWeightedStrategy
"""

from typing import Dict, List, Optional, Tuple

import flwr as fl
from flwr.common import Metrics, Context

from . import config
from .model import MNISTNet
from .strategy import PersonalityWeightedStrategy
from .client import FlowerClient
from .dataset import get_partitioned_data, get_client_loaders, get_client_labels
from .utils import get_parameters_from_model, set_seed


def _weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    """Aggregate evaluation metrics (accuracy) across clients."""
    total = sum(n for n, _ in metrics)
    acc = sum(n * m["accuracy"] for n, m in metrics) / total
    return {"accuracy": acc}


def run_simulation(
    mode: str = config.MODE,
    num_rounds: int = config.NUM_ROUNDS,
    num_clients: int = config.NUM_CLIENTS,
    partition: str = config.DATA_PARTITION,
) -> fl.server.history.History:
    """
    Run the full FL simulation and return the Flower History object.

    Parameters
    ----------
    mode : str
        "basic" or "personality"
    num_rounds : int
        Number of FL communication rounds.
    num_clients : int
        Number of simulated clients.
    partition : str
        "iid" or "noniid"

    Returns
    -------
    history : fl.server.history.History
        Contains per-round distributed loss & accuracy.
    """
    set_seed()

    # ── Data preparation ───────────────────────────────────────────────
    client_indices, train_ds, test_ds = get_partitioned_data(partition, num_clients)

    # ── Client factory for Flower simulation ───────────────────────────
    def client_fn(context: Context) -> fl.client.Client:
        cid_int = int(context.node_config["partition-id"])
        train_loader, test_loader = get_client_loaders(
            cid_int, client_indices, train_ds, test_ds
        )
        labels = get_client_labels(cid_int, client_indices, train_ds)
        return FlowerClient(
            client_id=cid_int,
            train_loader=train_loader,
            test_loader=test_loader,
            client_labels=labels,
            mode=mode,
        ).to_client()

    # ── Strategy selection ─────────────────────────────────────────────
    initial_model = MNISTNet()
    initial_params = get_parameters_from_model(initial_model)

    common_kwargs = dict(
        fraction_fit=1.0,            # use all clients each round
        fraction_evaluate=1.0,
        min_fit_clients=num_clients,
        min_evaluate_clients=num_clients,
        min_available_clients=num_clients,
        evaluate_metrics_aggregation_fn=_weighted_average,
        initial_parameters=fl.common.ndarrays_to_parameters(initial_params),
    )

    if mode == "personality":
        strategy = PersonalityWeightedStrategy(**common_kwargs)
        print("\n🧠  Strategy: PersonalityWeightedStrategy")
    else:
        strategy = fl.server.strategy.FedAvg(**common_kwargs)
        print("\n📦  Strategy: Standard FedAvg")

    # ── Run simulation ─────────────────────────────────────────────────
    history = fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=num_clients,
        config=fl.server.ServerConfig(num_rounds=num_rounds),
        strategy=strategy,
        client_resources={"num_cpus": 1, "num_gpus": 0.0},
    )

    return history
