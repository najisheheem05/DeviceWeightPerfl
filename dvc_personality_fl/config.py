"""
config.py — Central configuration.

All hyperparameters, paths, and experiment settings live here so that
every other module can import them from a single source of truth.
"""

import os

# ── Project paths ──────────────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs")

# Create directories if they don't exist
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Federated learning settings ───────────────────────────────────────
NUM_CLIENTS = 5  # Number of simulated FL clients
NUM_ROUNDS = 10  # Number of FL communication rounds
LOCAL_EPOCHS = 5  # Epochs each client trains per round
BATCH_SIZE = 32  # Mini-batch size for local training
LEARNING_RATE = 1e-3  # Adam optimizer learning rate

# ── Dataset selection ─────────────────────────────────────────────────
# "mnist"     — 28×28 grayscale handwritten digits (10 classes)
# "fsn-mnist" — 28×28 grayscale fashion items      (10 classes)
# "cifar10"   — 32×32 colour natural images         (10 classes)
DATASET = "cifar10"

# ── Data partitioning ─────────────────────────────────────────────────
# "iid"    — each client gets a uniform random split
# "noniid" — Dirichlet-based heterogeneous label distribution
DATA_PARTITION = "noniid"
DIRICHLET_ALPHA = 0.5  # Lower α → more skewed label distributions

# ── Mode toggle ────────────────────────────────────────────────────────
# "basic"       — standard FedAvg aggregation
# "personality" — personality-weighted aggregation
MODE = "basic"

# ── Personality scoring mode ───────────────────────────────────────────
# "static"  — random per-client metrics (stability, reliability, etc.)
#              computed once at client init and fixed for all rounds
# "dynamic" — real per-round metrics (training loss, val accuracy, etc.)
#              recomputed every round from actual training signals
PERSONALITY_MODE = "static"

# ── Personality weight coefficients for static mode───────────────────────────────────
# Used in compute_personality_score():
#   P_k = w_s * stability + w_r * reliability
#       + w_c * compute_power + w_d * data_diversity
PERSONALITY_WEIGHTS = {
    "stability": 0.25,
    "reliability": 0.25,
    "compute_power": 0.25,
    "data_diversity": 0.25,
}

# ── Dynamic personality weight coefficients ────────────────────────────
# Used in compute_dynamic_personality_score() with real per-round metrics:
#   P_k = w_tl * (1 - norm_loss) + w_va * val_accuracy
#       + w_um * norm_update_mag  + w_dd * data_diversity
DYNAMIC_PERSONALITY_WEIGHTS = {
    "training_loss": 0.15,
    "val_accuracy": 0.35,
    "update_magnitude": 0.15,
    "data_diversity": 0.25,
}

# ── Softmax temperature for aggregation weighting ─────────────────────
# Controls how sharply personality scores differentiate client weights.
# Higher → more uniform (closer to FedAvg), Lower → more aggressive.
# Set to 0.0 to disable softmax and use raw P_k * n_k weighting.
SOFTMAX_TEMPERATURE = 0.0

# ── Behavioural simulation ─────────────────────────────────────────────
# When True, personality metrics have real effects on client behaviour:
#   • reliability   → probability of participating (else returns zero update)
#   • compute_power → scales LOCAL_EPOCHS (fewer epochs for weaker devices)
#   • stability     → Gaussian noise added to updates (noisier for unstable)
SIMULATE_BEHAVIOUR = True

# Standard deviation multiplier for stability-based gradient noise.
# Noise σ = (1 - stability) * STABILITY_NOISE_SCALE * param_std
STABILITY_NOISE_SCALE = 0.1


# ── Random seed (for reproducibility) ─────────────────────────────────
SEED = 42
