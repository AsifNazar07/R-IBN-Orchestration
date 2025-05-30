import argparse
import torch
import os
from models.bilstm_attention import BiLSTMWithAttention
from utils.preprocessing import load_lumi_dataset
from evaluators.metrics import compute_metrics
from utils.config import load_config

def evaluate_model(config_path: str, model_path: str):
    config = load_config(config_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load dataset and vocab
    _, val_dataset, vocab, padding_idx = load_lumi_dataset(
        config.train_file, config.val_file, config.max_len
    )

    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=config.eval_batch_size)
    model = BiLSTMWithAttention(
        vocab_size=len(vocab),
        embedding_dim=config.embedding_dim,
        hidden_dim=config.hidden_dim,
        output_dim=config.num_labels,
        padding_idx=padding_idx,
        dropout=config.dropout
    ).to(device)

    # Load weights
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # Run evaluation
    y_true, y_pred = [], []
    with torch.no_grad():
        for input_ids, attention_mask, labels in val_loader:
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            labels = labels.to(device)

            logits = model(input_ids, attention_mask)
            predictions = torch.argmax(torch.softmax(logits, dim=1), dim=1)

            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predictions.cpu().numpy())

    metrics = compute_metrics(y_true, y_pred)
    print(f"\nEvaluation Results from {model_path}:")
    for key, val in metrics.items():
        print(f"  {key.capitalize()}: {val:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to bilstm.yml")
    parser.add_argument("--model", required=True, help="Path to saved model (e.g. best_model.pt)")
    args = parser.parse_args()

    evaluate_model(args.config, args.model)
