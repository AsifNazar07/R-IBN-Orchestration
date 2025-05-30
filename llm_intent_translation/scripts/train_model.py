import argparse
import yaml
from trainers.huggingface_trainer import HFTrainer
from trainers.bilstm_trainer import BiLSTMTrainer
from utils.config import load_config

def get_trainer_class(model_name):
    if model_name.lower() in ['bert', 'gpt2', 't5', 'roberta']:
        return HFTrainer
    elif model_name.lower() == 'bilstm':
        return BiLSTMTrainer
    else:
        raise ValueError(f"Unsupported model: {model_name}")

def main():
    parser = argparse.ArgumentParser(description="Train LLM for intent translation")
    parser.add_argument('--model', required=True, help="Model name: bert | gpt2 | t5 | roberta | bilstm")
    parser.add_argument('--config', required=True, help="Path to YAML config")
    args = parser.parse_args()

    config = load_config(args.config)
    trainer_cls = get_trainer_class(args.model)
    trainer = trainer_cls(config)
    trainer.train()

if __name__ == "__main__":
    main()
