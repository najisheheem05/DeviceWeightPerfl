"""
personality.py — Device personality metrics for federated clients.

Each simulated client is assigned four behavioural metrics:
  • stability      — how consistent the device's connection is (0.7 – 1.0)
  • reliability    — probability of successful round completion  (0.6 – 1.0)
  • compute_power  — relative processing capability              (0.5 – 1.0)
  • data_diversity — normalised entropy of the client's label distribution

These are combined into a single personality score P_k used for
weighted aggregation in the PersonalityWeightedStrategy.

Score formula:
    P_k = 0.30 * stability
        + 0.30 * reliability
        + 0.20 * compute_power
        + 0.20 * data_diversity
"""

import numpy as np
from typing import Dict

from . import config


def generate_personality_metrics(client_id: int, seed: int = config.SEED) -> Dict[str, float]:
    """
    Generate random personality metrics for a given client.

    A deterministic seed derived from (global_seed + client_id) ensures
    the same client always gets the same metrics within an experiment.

    Returns
    -------
    metrics : dict
        Keys: stability, reliability, compute_power
    """
    rng = np.random.RandomState(seed + client_id)
    return {
        "stability":     round(rng.uniform(0.70, 1.00), 4),
        "reliability":   round(rng.uniform(0.60, 1.00), 4),
        "compute_power": round(rng.uniform(0.50, 1.00), 4),
    }


def compute_data_diversity(labels: np.ndarray, num_classes: int = 10) -> float:
    """
    Compute normalised entropy of the client's label distribution.

    A uniform distribution yields 1.0 (maximum diversity);
    a single-class distribution yields 0.0.

    Parameters
    ----------
    labels : np.ndarray
        Array of integer class labels for one client's dataset.
    num_classes : int
        Total number of possible classes.

    Returns
    -------
    diversity : float   in [0, 1]
    """
    if len(labels) == 0:
        return 0.0

    counts = np.bincount(labels, minlength=num_classes).astype(float)
    probs = counts / counts.sum()

    # Avoid log(0) — zero-probability classes contribute 0 entropy
    probs = probs[probs > 0]
    entropy = -np.sum(probs * np.log(probs))
    max_entropy = np.log(num_classes)

    return round(float(entropy / max_entropy), 4)


def compute_personality_score(metrics: Dict[str, float]) -> float:
    """
    Combine individual metrics into a single personality score.

    P_k = w_s * stability + w_r * reliability
        + w_c * compute_power + w_d * data_diversity

    Parameters
    ----------
    metrics : dict
        Must contain keys: stability, reliability, compute_power, data_diversity.

    Returns
    -------
    score : float
    """
    w = config.PERSONALITY_WEIGHTS
    score = (
        w["stability"]      * metrics["stability"]
        + w["reliability"]  * metrics["reliability"]
        + w["compute_power"] * metrics["compute_power"]
        + w["data_diversity"] * metrics["data_diversity"]
    )
    return round(score, 4)
