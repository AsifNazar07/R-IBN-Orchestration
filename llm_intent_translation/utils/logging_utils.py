import os
import csv
from torch.utils.tensorboard import SummaryWriter
from typing import Dict


def init_tensorboard(log_dir: str) -> SummaryWriter:
    """
    Initialize TensorBoard writer.

    Args:
        log_dir: Directory to store logs
    Returns:
        TensorBoard SummaryWriter instance
    """
    return SummaryWriter(log_dir=log_dir)


def log_metrics_to_csv(epoch: int, train_loss: float, val_metrics: Dict[str, float], log_dir: str):
    """
    Log evaluation metrics to CSV after each epoch.

    Args:
        epoch: Current epoch number
        train_loss: Average training loss for the epoch
        val_metrics: Dictionary with F1, precision, recall, accuracy
        log_dir: Output directory to store log file
    """
    csv_file = os.path.join(log_dir, "metrics.csv")
    file_exists = os.path.isfile(csv_file)

    with open(csv_file, mode='a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["epoch", "train_loss", "f1", "precision", "recall", "accuracy"])
        writer.writerow([
            epoch,
            round(train_loss, 6),
            round(val_metrics["f1"], 6),
            round(val_metrics["precision"], 6),
            round(val_metrics["recall"], 6),
            round(val_metrics["accuracy"], 6)
        ])
