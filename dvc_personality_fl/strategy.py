"""
strategy.py — Custom Flower aggregation strategy for personality-weighted FL.

Provides:
  • PersonalityWeightedStrategy  — extends FedAvg so that the global model
    update is a weighted average where each client's contribution is scaled
    by its personality score *and* dataset size.

    When SOFTMAX_TEMPERATURE > 0, the combined scores are passed through a
    temperature-controlled softmax to prevent any single client from
    dominating the aggregation.

The personality score P_k is passed from each client via the Flower
metrics dict under the key "personality_score".
"""

from typing import Dict, List, Optional, Tuple, Union
from logging import WARNING

import numpy as np
from flwr.common import (
    FitRes,
    Parameters,
    Scalar,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import FedAvg

from . import config


class PersonalityWeightedStrategy(FedAvg):
    """
    FedAvg variant that weights each client's update by its personality score.

    Clients must include a ``"personality_score"`` key in the metrics dict
    returned from ``fit()``.  If a client does not provide the key, a default
    score of 1.0 (i.e. no extra weighting) is used.
    """

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """
        Personality-weighted aggregation with optional softmax temperature.

        Steps
        -----
        1. Extract per-client weights (ndarrays), sample counts, and
           personality scores from the FitRes objects.
        2. Compute combined weight  c_k = P_k × n_k  for each client.
        3. Apply softmax temperature (if enabled) for smoother weighting.
        4. Aggregate:  W_global[i] = Σ w_k · W_k[i]
        """
        if not results:
            return None, {}

        # ── Collect per-client data ────────────────────────────────────
        client_weights = []   # list of list-of-ndarrays
        combined_scores = []  # P_k * n_k

        for _, fit_res in results:
            ndarrays = parameters_to_ndarrays(fit_res.parameters)
            n_k = fit_res.num_examples

            # Retrieve personality score from client metrics (default 1.0)
            p_k = fit_res.metrics.get("personality_score", 1.0)

            client_weights.append(ndarrays)
            combined_scores.append(p_k * n_k)

        # ── Apply softmax temperature ──────────────────────────────────
        scores = np.array(combined_scores, dtype=np.float64)
        temperature = config.SOFTMAX_TEMPERATURE

        if temperature > 0:
            # Subtract max for numerical stability
            scores_shifted = scores / temperature
            scores_shifted -= scores_shifted.max()
            exp_scores = np.exp(scores_shifted)
            weights = exp_scores / exp_scores.sum()
        else:
            # Raw weighting (no softmax)
            weights = scores / scores.sum()

        # ── Weighted aggregation ───────────────────────────────────────
        # For each parameter tensor, compute the weighted sum
        num_layers = len(client_weights[0])
        aggregated = []
        for layer_idx in range(num_layers):
            weighted_sum = np.zeros_like(client_weights[0][layer_idx])
            for client_idx in range(len(client_weights)):
                weighted_sum += weights[client_idx] * client_weights[client_idx][layer_idx]
            aggregated.append(weighted_sum)

        # ── Aggregate metrics for logging ──────────────────────────────
        metrics_aggregated: Dict[str, Scalar] = {
            "personality_weighted": True,
        }

        return ndarrays_to_parameters(aggregated), metrics_aggregated
