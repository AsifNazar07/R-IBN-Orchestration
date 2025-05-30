import os
import argparse
from pathlib import Path

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from typing import List


def find_result_files(base_dir: str) -> List[Path]:
    result_files = []
    for root, dirs, files in os.walk(base_dir):
        for file in files:
            if file == 'results.csv':
                result_files.append(Path(root) / file)
    return result_files


def extract_metadata(path: Path) -> dict:
    try:
        parts = path.parts
        idx = parts.index("logs")
        topology = parts[idx + 1]
        agent = parts[idx + 2]
    except Exception:
        topology = "Unknown"
        agent = "Unknown"
    return {"Topology": topology, "Agent": agent}


def load_and_annotate(files: List[Path]) -> pd.DataFrame:
    frames = []
    for file in files:
        try:
            df = pd.read_csv(file)
            meta = extract_metadata(file)
            for key, val in meta.items():
                df[key] = val
            frames.append(df)
        except Exception as e:
            print(f"Error reading {file}: {e}")
    return pd.concat(frames, ignore_index=True)


def plot_metric(data: pd.DataFrame, metric: str, output_dir: Path, style: str = "whitegrid", formats: List[str] = ["pdf", "png"]):
    sns.set_style(style)
    plt.figure(figsize=(9, 7))
    ax = sns.boxplot(data=data, x="Agent", y=metric, hue="Topology", palette="Set2")
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_title(f"{metric} across Agents and Topologies", fontsize=14)
    plt.tight_layout()

    for ext in formats:
        filename = output_dir / f"{metric.replace(' ', '_')}.{ext}"
        plt.savefig(filename)
        print(f"Saved plot: {filename}")
    plt.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Plot training results from multiple agents and topologies.")
    parser.add_argument('--logdir', type=str, default='./logs', help="Directory containing training logs.")
    parser.add_argument('--output', type=str, default='./results', help="Directory to save plots.")
    parser.add_argument('--metrics', type=str, nargs='+', default=["reward", "accept_rate"],
                        help="Metrics to visualize (columns in results.csv).")
    parser.add_argument('--formats', type=str, nargs='+', default=["pdf", "png"], help="Output file formats.")
    parser.add_argument('--style', type=str, default="whitegrid", help="Seaborn plot style.")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    result_files = find_result_files(args.logdir)
    if not result_files:
        print("No results.csv files found.")
        exit(1)

    results_df = load_and_annotate(result_files)
    for metric in args.metrics:
        if metric not in results_df.columns:
            print(f"Skipping {metric}: column not found.")
            continue
        plot_metric(results_df, metric, Path(args.output), style=args.style, formats=args.formats)
