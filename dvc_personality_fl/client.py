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
        2. (Behavioural) Check reliability — possibly skip this round
        3. (Behavioural) Scale local epochs by compute_power
        4. Train for the determined number of epochs, tracking loss
        5. (Behavioural) Add noise based on stability
        6. Compute dynamic personality score from real metrics
        7. Return updated weights + num_examples + metrics
        """
        self.set_parameters(parameters)

        simulate = config.SIMULATE_BEHAVIOUR
        use_dynamic = (config.PERSONALITY_MODE == "dynamic")

        # ── 1. Reliability → Round dropout ─────────────────────────────
        if simulate:
            reliability = self.personality_metrics["reliability"]
            rng = np.random.RandomState(
                config.SEED + self.client_id + hash("reliability") % 10000
                + getattr(self, "_round_counter", 0)
            )
            if rng.random() > reliability:
                # Client "drops out" — return unchanged global params
                self._round_counter = getattr(self, "_round_counter", 0) + 1
                metrics: Dict = {
                    "client_id": self.client_id,
                    "dropped": True,
                }
                if self.mode == "personality":
                    metrics["personality_score"] = 0.0
                return self.get_parameters({}), 0, metrics

        # ── 2. Compute power → Reduced local epochs ───────────────────
        local_epochs = config.LOCAL_EPOCHS
        if simulate:
            compute_power = self.personality_metrics["compute_power"]
            local_epochs = max(1, int(config.LOCAL_EPOCHS * compute_power))

        # ── Snapshot parameters before training (dynamic mode only) ────
        if self.mode == "personality" and use_dynamic:
            old_params = [p.clone().detach() for p in self.model.parameters()]

        # ── Local training ─────────────────────────────────────────────
        total_loss = 0.0
        for _ in range(local_epochs):
            epoch_loss = train_one_epoch(self.model, self.train_loader, self.device)
            total_loss += epoch_loss
        avg_loss = total_loss / max(local_epochs, 1)

        # ── 3. Stability → Gradient noise ──────────────────────────────
        if simulate:
            stability = self.personality_metrics["stability"]
            noise_level = (1.0 - stability) * config.STABILITY_NOISE_SCALE
            if noise_level > 0:
                with torch.no_grad():
                    for p in self.model.parameters():
                        noise = torch.randn_like(p) * noise_level * p.std()
                        p.add_(noise)

        # ── Increment round counter for reproducible dropout ───────────
        self._round_counter = getattr(self, "_round_counter", 0) + 1

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

                # Compute individual component scores for visualization
                loss_score = max(0.0, 1.0 - avg_loss)
                um_score = 1.0 / (1.0 + np.exp(-0.5 * (update_mag - 1.0)))
                dd_score = self.personality_metrics["data_diversity"]

                score = compute_dynamic_personality_score(
                    training_loss=avg_loss,
                    val_accuracy=val_accuracy,
                    update_magnitude=update_mag,
                    data_diversity=dd_score,
                )

                # Report component scores for stacked bar visualization
                metrics["dyn_loss_score"] = float(loss_score)
                metrics["dyn_val_accuracy"] = float(val_accuracy)
                metrics["dyn_um_score"] = float(um_score)
                metrics["dyn_data_diversity"] = float(dd_score)
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
