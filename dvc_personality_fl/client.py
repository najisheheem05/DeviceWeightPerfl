"""
client.py — Flower NumPy client for federated MNIST training.

Each FlowerClient wraps:
  • a local MNISTNet model
  • a training DataLoader  (partition-specific)
  • a test DataLoader      (shared evaluation set)

In "personality" mode the client additionally computes and reports its
personality score inside the fit() metrics dict so that the
PersonalityWeightedStrategy can use it during aggregation.
"""

from typing import Dict, List, Tuple

import numpy as np
import flwr as fl

from . import config
from .model import get_model, train_one_epoch, evaluate
from .personality import (
    generate_personality_metrics,
    compute_data_diversity,
    compute_personality_score,
)
from .utils import get_device, get_parameters_from_model, set_parameters_to_model


class FlowerClient(fl.client.NumPyClient):
    """Flower client that trains a local MNISTNet model."""

    def __init__(
        self,
        client_id: int,
        train_loader,
        test_loader,
        client_labels: np.ndarray,
        mode: str = config.MODE,
        dataset: str = config.DATASET,
    ):
        self.client_id = client_id
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.client_labels = client_labels
        self.mode = mode
        self.device = get_device()

        # Local model (re-created each round from global weights)
        self.model = get_model(dataset).to(self.device)

        # Pre-compute personality (deterministic per client)
        self.personality_metrics = generate_personality_metrics(client_id)
        self.personality_metrics["data_diversity"] = compute_data_diversity(
            client_labels
        )
        self.personality_score = compute_personality_score(self.personality_metrics)

    # ── Flower interface ──────────────────────────────────────────────

    def get_parameters(self, config: Dict) -> List[np.ndarray]:
        """Return current model parameters as NumPy arrays."""
        return get_parameters_from_model(self.model)

    def set_parameters(self, parameters: List[np.ndarray]) -> None:
        """Load global parameters into the local model."""
        set_parameters_to_model(self.model, parameters)

    def fit(
        self, parameters: List[np.ndarray], config_dict: Dict
    ) -> Tuple[List[np.ndarray], int, Dict]:
        """
        Local training round.

        1. Set global weights → local model
        2. Train for LOCAL_EPOCHS epochs
        3. Return updated weights + num_examples + metrics
        """
        self.set_parameters(parameters)

        # Local training
        for _ in range(config.LOCAL_EPOCHS):
            train_one_epoch(self.model, self.train_loader, self.device)

        # Metrics to send back
        metrics: Dict = {"client_id": self.client_id}
        if self.mode == "personality":
            metrics["personality_score"] = float(self.personality_score)

        num_examples = len(self.train_loader.dataset)
        return self.get_parameters({}), num_examples, metrics

    def evaluate(
        self, parameters: List[np.ndarray], config_dict: Dict
    ) -> Tuple[float, int, Dict]:
        """
        Evaluate the global model on the shared test set.

        Returns (loss, num_examples, {"accuracy": …}).
        """
        self.set_parameters(parameters)
        loss, accuracy = evaluate(self.model, self.test_loader, self.device)
        num_examples = len(self.test_loader.dataset)
        return float(loss), num_examples, {"accuracy": float(accuracy)}
