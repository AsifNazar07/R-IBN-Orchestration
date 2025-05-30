import os
import torch
from torch.utils.data import DataLoader
from torch.nn import CrossEntropyLoss
from torch.optim import AdamW
from torch.nn.functional import softmax
from tqdm import tqdm
from sklearn.metrics import precision_recall_fscore_support, accuracy_score

from models.bilstm_attention import BiLSTMWithAttention
from utils.preprocessing import load_lumi_dataset
from evaluators.metrics import compute_metrics
from utils.logging_utils import log_metrics_to_csv, init_tensorboard

class BiLSTMTrainer:
    def __init__(self, config):
        self.cfg = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.train_dataset, self.val_dataset, self.vocab, self.padding_idx = load_lumi_dataset(
            config.train_file, config.val_file, config.max_len
        )

        self.model = BiLSTMWithAttention(
            vocab_size=len(self.vocab),
            embedding_dim=config.embedding_dim,
            hidden_dim=config.hidden_dim,
            output_dim=config.num_labels,
            padding_idx=self.padding_idx,
            dropout=config.dropout
        ).to(self.device)

        self.criterion = CrossEntropyLoss()
        self.optimizer = AdamW(self.model.parameters(), lr=config.learning_rate)
        self.train_loader = DataLoader(self.train_dataset, batch_size=config.batch_size, shuffle=True)
        self.val_loader = DataLoader(self.val_dataset, batch_size=config.eval_batch_size)

        self.log_dir = os.path.join(config.output_dir, 'bilstm')
        os.makedirs(self.log_dir, exist_ok=True)
        self.tb_writer = init_tensorboard(self.log_dir)

    def train(self):
        best_f1 = 0
        for epoch in range(1, self.cfg.epochs + 1):
            self.model.train()
            total_loss = 0
            for batch in tqdm(self.train_loader, desc=f"Epoch {epoch}"):
                input_ids, attention_mask, labels = [x.to(self.device) for x in batch]
                self.optimizer.zero_grad()
                outputs = self.model(input_ids, attention_mask)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()

            val_metrics = self.evaluate()
            val_f1 = val_metrics['f1']
            if val_f1 > best_f1:
                best_f1 = val_f1
                torch.save(self.model.state_dict(), os.path.join(self.log_dir, "best_model.pt"))

            print(f"Epoch {epoch} | Train Loss: {total_loss:.4f} | Val F1: {val_f1:.4f}")
            log_metrics_to_csv(epoch, total_loss, val_metrics, self.log_dir)
            for k, v in val_metrics.items():
                self.tb_writer.add_scalar(f"Val/{k}", v, epoch)

    def evaluate(self):
        self.model.eval()
        y_true, y_pred = [], []
        with torch.no_grad():
            for batch in self.val_loader:
                input_ids, attention_mask, labels = [x.to(self.device) for x in batch]
                outputs = self.model(input_ids, attention_mask)
                probs = softmax(outputs, dim=1)
                preds = torch.argmax(probs, dim=1)
                y_true.extend(labels.cpu().numpy())
                y_pred.extend(preds.cpu().numpy())
        return compute_metrics(y_true, y_pred)
