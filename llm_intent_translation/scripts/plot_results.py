import argparse
import pandas as pd
import matplotlib.pyplot as plt

def plot_metrics(csv_path):
    df = pd.read_csv(csv_path)

    metrics = ["train_loss", "f1", "precision", "recall", "accuracy"]
    for metric in metrics:
        if metric not in df.columns:
            continue
        plt.figure()
        plt.plot(df["epoch"], df[metric], marker='o')
        plt.title(f"{metric} over epochs")
        plt.xlabel("Epoch")
        plt.ylabel(metric.capitalize())
        plt.grid(True)
        plt.savefig(csv_path.replace(".csv", f"_{metric}.png"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True, help="Path to metrics.csv file")
    args = parser.parse_args()
    plot_metrics(args.csv)
