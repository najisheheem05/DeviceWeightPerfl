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
import torch
import flwr as fl

from . import config
from .model import get_model, train_one_epoch, evaluate
from .personality import (
    generate_personality_metrics,
    compute_data_diversity,
    compute_personality_score,
    compute_dynamic_personality_score,
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
        2. Save a copy of the old parameters
        3. Train for LOCAL_EPOCHS epochs, tracking loss
        4. Compute dynamic personality score from real metrics
        5. Return updated weights + num_examples + metrics
        """
        self.set_parameters(parameters)

        use_dynamic = (config.PERSONALITY_MODE == "dynamic")

        # ── Snapshot parameters before training (dynamic mode only) ────
        if self.mode == "personality" and use_dynamic:
            old_params = [p.clone().detach() for p in self.model.parameters()]

        # ── Local training ─────────────────────────────────────────────
        total_loss = 0.0
        for _ in range(config.LOCAL_EPOCHS):
            epoch_loss = train_one_epoch(self.model, self.train_loader, self.device)
            total_loss += epoch_loss
        avg_loss = total_loss / max(config.LOCAL_EPOCHS, 1)

        # ── Build metrics dict ─────────────────────────────────────────
        metrics: Dict = {"client_id": self.client_id}

        if self.mode == "personality":
            if use_dynamic:
                # Compute real per-round metrics
                update_mag = 0.0
                for new_p, old_p in zip(self.model.parameters(), old_params):
                    update_mag += (new_p - old_p).norm(2).item() ** 2
                update_mag = update_mag ** 0.5

                _, val_accuracy = evaluate(self.model, self.test_loader, self.device)

                score = compute_dynamic_personality_score(
                    training_loss=avg_loss,
                    val_accuracy=val_accuracy,
                    update_magnitude=update_mag,
                    data_diversity=self.personality_metrics["data_diversity"],
                )
            else:
                # Static: use pre-computed random personality score
                score = self.personality_score

            metrics["personality_score"] = float(score)

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
