from sklearn.metrics import (
    precision_score, recall_score, f1_score, accuracy_score
)
from typing import List, Dict


def compute_metrics(y_true: List[int], y_pred: List[int]) -> Dict[str, float]:
    """
    Compute performance metrics for classification.
    
    Returns:
        Dictionary with keys: accuracy, precision, recall, f1
    """
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "recall": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "f1": f1_score(y_true, y_pred, average="macro", zero_division=0)
    }
