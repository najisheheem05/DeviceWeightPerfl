"""
config.py — Central configuration for the Federated Learning project.

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
LOCAL_EPOCHS = 2  # Epochs each client trains per round
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
DATA_PARTITION = "iid"
DIRICHLET_ALPHA = 0.5  # Lower α → more skewed label distributions

# ── Mode toggle ────────────────────────────────────────────────────────
# "basic"       — standard FedAvg aggregation
# "personality" — personality-weighted aggregation
MODE = "basic"

# ── Personality weight coefficients ────────────────────────────────────
# Used in compute_personality_score():
#   P_k = w_s * stability + w_r * reliability
#       + w_c * compute_power + w_d * data_diversity
PERSONALITY_WEIGHTS = {
    "stability": 0.30,
    "reliability": 0.30,
    "compute_power": 0.20,
    "data_diversity": 0.20,
}

# ── Random seed (for reproducibility) ─────────────────────────────────
SEED = 42
