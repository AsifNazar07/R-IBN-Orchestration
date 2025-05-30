import argparse
import numpy as np
import json
import os
from models.knn_verifier import KNNContradictionDetector
from utils.config import load_config

def load_embeddings(path: str):
    """
    Loads embeddings from a .npy file. Assumes shape (N, embedding_dim).
    """
    return np.load(path)

def load_labels(path: str):
    """
    Loads labels (0 or 1) from a .txt or .npy file.
    """
    if path.endswith(".txt"):
        return np.loadtxt(path, dtype=int)
    elif path.endswith(".npy"):
        return np.load(path)
    else:
        raise ValueError("Unsupported label file format.")

def main():
    parser = argparse.ArgumentParser(description="KNN-based Contradiction Validation")
    parser.add_argument("--config", required=True, help="Path to knn.yml config file")
    parser.add_argument("--train_embeddings", required=True, help="Path to .npy file of training embeddings")
    parser.add_argument("--train_labels", required=True, help="Path to .txt or .npy file of labels (0=no contradiction, 1=contradiction)")
    parser.add_argument("--test_embedding", required=True, help="Path to .npy file of single test embedding (shape: [embedding_dim])")
    args = parser.parse_args()

    config = load_config(args.config)

    # Load and train
    X_train = load_embeddings(args.train_embeddings)
    y_train = load_labels(args.train_labels)

    knn = KNNContradictionDetector(k=config.k_neighbors)
    knn.fit(X_train, y_train)

    # Predict contradiction
    x_new = np.load(args.test_embedding)
    prediction = knn.predict(x_new)

    print("Contradiction Detected" if prediction == 1 else "No Contradiction")

if __name__ == "__main__":
    main()
