import torch
import torch.nn as nn
import torch.nn.functional as F


class BiLSTMWithAttention(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, padding_idx, dropout=0.3):
        super(BiLSTMWithAttention, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=padding_idx)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.attention = nn.Linear(hidden_dim * 2, 1)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * 2, output_dim)

    def forward(self, input_ids, attention_mask):
        embeds = self.embedding(input_ids)
        packed_output, _ = self.lstm(embeds)
        attn_weights = F.softmax(self.attention(packed_output).squeeze(-1), dim=1)
        weighted = torch.sum(packed_output * attn_weights.unsqueeze(-1), dim=1)
        out = self.dropout(weighted)
        return self.fc(out)
