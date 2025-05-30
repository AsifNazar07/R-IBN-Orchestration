from transformers import (AutoTokenizer, AutoModelForSequenceClassification, Trainer,
                          TrainingArguments, DataCollatorWithPadding)
from datasets import load_dataset
import torch
import os
from evaluators.metrics import compute_metrics

class HFTrainer:
    def __init__(self, config):
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_name)
        self.dataset = self.load_lumi_dataset()
        self.model = AutoModelForSequenceClassification.from_pretrained(
            config.model_name, num_labels=config.num_labels)

    def load_lumi_dataset(self):
        data_files = {"train": self.config.train_file, "validation": self.config.val_file}
        dataset = load_dataset("json", data_files=data_files)
        
        def preprocess(example):
            return self.tokenizer(example["text"], truncation=True, max_length=128)
        
        tokenized = dataset.map(preprocess, batched=True)
        return tokenized

    def train(self):
        output_dir = os.path.join(self.config.output_dir, self.config.model_name.replace('/', '_'))
        args = TrainingArguments(
            output_dir=output_dir,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            learning_rate=self.config.learning_rate,
            per_device_train_batch_size=self.config.batch_size,
            per_device_eval_batch_size=self.config.eval_batch_size,
            num_train_epochs=self.config.epochs,
            weight_decay=self.config.weight_decay,
            logging_dir=f"{output_dir}/logs",
            logging_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="f1",
            report_to=["tensorboard"],
        )
        data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer)
        trainer = Trainer(
            model=self.model,
            args=args,
            train_dataset=self.dataset["train"],
            eval_dataset=self.dataset["validation"],
            tokenizer=self.tokenizer,
            data_collator=data_collator,
            compute_metrics=compute_metrics,
        )
        trainer.train()
        trainer.save_model(output_dir)
