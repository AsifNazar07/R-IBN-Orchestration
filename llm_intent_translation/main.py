import argparse
import subprocess

def train(model, config):
    print(f"Training model: {model}")
    subprocess.run(f"python scripts/train_model.py --model {model} --config {config}", shell=True)

def evaluate(config, model_path):
    print(f"Evaluating model: {model_path}")
    subprocess.run(f"python scripts/evaluate_model.py --config {config} --model {model_path}", shell=True)

def plot(csv_path):
    print(f"Plotting results from: {csv_path}")
    subprocess.run(f"python scripts/plot_results.py --csv {csv_path}", shell=True)

def validate_knn(config, train_embeddings, train_labels, test_embedding):
    print("Running KNN contradiction validation...")
    cmd = (
        f"python scripts/validate_contradictions.py "
        f"--config {config} "
        f"--train_embeddings {train_embeddings} "
        f"--train_labels {train_labels} "
        f"--test_embedding {test_embedding}"
    )
    subprocess.run(cmd, shell=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Unified CLI for LLM-Orchestration Pipeline")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    train_parser = subparsers.add_parser("train", help="Train a model")
    train_parser.add_argument("--model", required=True)
    train_parser.add_argument("--config", required=True)

    eval_parser = subparsers.add_parser("eval", help="Evaluate a trained model")
    eval_parser.add_argument("--config", required=True)
    eval_parser.add_argument("--model_path", required=True)

    plot_parser = subparsers.add_parser("plot", help="Plot CSV training metrics")
    plot_parser.add_argument("--csv", required=True)

    knn_parser = subparsers.add_parser("validate_knn", help="Run contradiction detection with KNN")
    knn_parser.add_argument("--config", required=True)
    knn_parser.add_argument("--train_embeddings", required=True)
    knn_parser.add_argument("--train_labels", required=True)
    knn_parser.add_argument("--test_embedding", required=True)

    args = parser.parse_args()

    if args.command == "train":
        train(args.model, args.config)
    elif args.command == "eval":
        evaluate(args.config, args.model_path)
    elif args.command == "plot":
        plot(args.csv)
    elif args.command == "validate_knn":
        validate_knn(args.config, args.train_embeddings, args.train_labels, args.test_embedding)
    else:
        parser.print_help()
