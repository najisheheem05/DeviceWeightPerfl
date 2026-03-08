"""
utils.py — Utility helpers shared across the project.

Provides:
  • Reproducible seeding
  • Device selection (CUDA / CPU)
  • Conversions between Flower Parameters ↔ PyTorch state dicts
"""

import random
import numpy as np
import torch
from collections import OrderedDict
from typing import List

from . import config


def set_seed(seed: int = config.SEED) -> None:
    """Set random seeds for Python, NumPy, and PyTorch for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    """Return the best available device (CUDA GPU or CPU)."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ── Flower ↔ PyTorch weight conversions ────────────────────────────────

def get_parameters_from_model(model: torch.nn.Module) -> List[np.ndarray]:
    """Extract model parameters as a list of NumPy arrays (Flower format)."""
    return [val.cpu().numpy() for _, val in model.state_dict().items()]


def set_parameters_to_model(model: torch.nn.Module, parameters: List[np.ndarray]) -> None:
    """Load a list of NumPy arrays into a PyTorch model's state dict."""
    params_dict = zip(model.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
    model.load_state_dict(state_dict, strict=True)
