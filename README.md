# Federated Learning with Device Personality Weighting

A research-style federated learning project that implements **standard FedAvg** and a novel **personality-weighted aggregation** strategy. Built with **PyTorch** and **Flower (flwr)**.

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?logo=pytorch&logoColor=white)
![Flower](https://img.shields.io/badge/Flower-1.5+-4CAF50)

---

## Overview

This project simulates a federated learning system across **5 clients** on the **MNIST** dataset. It supports two aggregation modes:

| Mode            | Strategy            | Aggregation Formula                                                    |
| --------------- | ------------------- | ---------------------------------------------------------------------- |
| **Basic**       | FedAvg              | $W_{global} = \sum \frac{n_k}{N} \cdot W_k$                            |
| **Personality** | PersonalityWeighted | $W_{global} = \frac{\sum P_k \cdot n_k \cdot W_k}{\sum P_k \cdot n_k}$ |

In **personality mode**, each client is assigned behavioural metrics (stability, reliability, compute power, data diversity) that are combined into a personality score $P_k$, giving higher influence to more capable/reliable devices during model aggregation.

---

## Project Structure

```
deviceWeighFed/
├── requirements.txt
├── README.md
├── venv/                          # Python virtual environment
└── fed_personality_fl/
    ├── __init__.py
    ├── config.py                  # Central hyperparameters & settings
    ├── dataset.py                 # MNIST loading, IID & Non-IID partitioning
    ├── model.py                   # CNN architecture, train & evaluate
    ├── personality.py             # Device personality metrics & scoring
    ├── strategy.py                # PersonalityWeightedStrategy (extends FedAvg)
    ├── client.py                  # Flower NumPy client
    ├── server.py                  # Simulation orchestration
    ├── visualize.py               # Matplotlib plot generation
    ├── utils.py                   # Seeding, device selection, weight conversions
    ├── main.py                    # CLI entry point
    └── outputs/                   # Generated plots & result JSONs
```

---

## implements

### 1. Clone & Setup

```bash
cd /path/to/deviceWeighFed

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Run Experiments

```bash
# Standard FedAvg (10 rounds, 5 clients, Non-IID by default)
python -m fed_personality_fl.main --mode basic

# Personality-weighted FL
python -m fed_personality_fl.main --mode personality

# Run BOTH modes & generate comparison plot
python -m fed_personality_fl.main --mode compare
```

### 3. Custom Settings

```bash
python -m fed_personality_fl.main \
    --mode personality \
    --rounds 15 \
    --clients 5 \
    --partition iid
```

| Argument      | Options                           | Default  | Description                       |
| ------------- | --------------------------------- | -------- | --------------------------------- |
| `--mode`      | `basic`, `personality`, `compare` | `basic`  | Aggregation strategy              |
| `--rounds`    | int                               | `10`     | Number of FL communication rounds |
| `--clients`   | int                               | `5`      | Number of simulated clients       |
| `--partition` | `iid`, `noniid`                   | `noniid` | Data partitioning strategy        |

---

## 🧪 Model Architecture

Simple CNN classifier for MNIST:

```
Input (1×28×28)
  → Conv2d(1, 32, 3, padding=1) → ReLU → MaxPool2d(2)
  → Conv2d(32, 64, 3, padding=1) → ReLU → MaxPool2d(2)
  → Flatten
  → Linear(64×7×7, 128) → ReLU
  → Linear(128, 10)
```

- **Loss**: CrossEntropyLoss
- **Optimizer**: Adam (lr = 0.001)

---

## 📊 Device Personality Metrics

Each client is assigned four metrics that model real-world device characteristics:

| Metric             | Range       | Weight | Description                    |
| ------------------ | ----------- | ------ | ------------------------------ |
| **Stability**      | 0.70 – 1.00 | 0.30   | Connection consistency         |
| **Reliability**    | 0.60 – 1.00 | 0.30   | Round completion probability   |
| **Compute Power**  | 0.50 – 1.00 | 0.20   | Relative processing capability |
| **Data Diversity** | 0.00 – 1.00 | 0.20   | Normalized label entropy       |

**Personality Score:**

$$P_k = 0.30 \cdot \text{stability} + 0.30 \cdot \text{reliability} + 0.20 \cdot \text{compute\_power} + 0.20 \cdot \text{data\_diversity}$$

---

## Data Partitioning

### IID

Each client receives a uniform random split of the training data.

### Non-IID (Dirichlet)

Labels are distributed across clients using a Dirichlet distribution with concentration parameter α = 0.5. Lower α produces more skewed (heterogeneous) label distributions across clients, simulating realistic federated scenarios.

---

## Generated Outputs

All results are saved to `fed_personality_fl/outputs/`:

| File                                  | Description                                     |
| ------------------------------------- | ----------------------------------------------- |
| `accuracy_basic.png`                  | Accuracy vs. rounds (Basic FL)                  |
| `accuracy_personality.png`            | Accuracy vs. rounds (Personality FL)            |
| `loss_basic.png`                      | Loss vs. rounds (Basic FL)                      |
| `loss_personality.png`                | Loss vs. rounds (Personality FL)                |
| `personality_scores.png`              | Stacked bar chart of per-client scores          |
| `comparison_basic_vs_personality.png` | Side-by-side comparison (with `--mode compare`) |
| `results_basic.json`                  | Raw per-round metrics (Basic)                   |
| `results_personality.json`            | Raw per-round metrics (Personality)             |

---

## Configuration

All hyperparameters are centralized in [`config.py`](fed_personality_fl/config.py):

```python
NUM_CLIENTS      = 5
NUM_ROUNDS       = 10
LOCAL_EPOCHS     = 2
BATCH_SIZE       = 32
LEARNING_RATE    = 1e-3
DATA_PARTITION   = "noniid"
DIRICHLET_ALPHA  = 0.5
SEED             = 42
```

---

## Dependencies

- Python ≥ 3.10
- PyTorch ≥ 2.0
- torchvision ≥ 0.15
- Flower (flwr) ≥ 1.5 (with simulation extras)
- Matplotlib ≥ 3.7
- NumPy ≥ 1.24

---
