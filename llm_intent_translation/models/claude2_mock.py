"""
Claude2 is a closed model (Anthropic API only).
This is a mocked evaluation interface for reproducibility.
"""

from evaluators.metrics import compute_metrics
import random
import numpy as np

def evaluate_claude2_stub(val_file: str):
    # Simulate true/false predictions from val set
    with open(val_file, 'r') as f:
        lines = f.readlines()
    y_true = [int(eval(line)["label"]) for line in lines]
    y_pred = [random.choice([0, 1]) for _ in y_true]  

    np.random.seed(42)
    metrics = compute_metrics(y_true, y_pred)
    return metrics
