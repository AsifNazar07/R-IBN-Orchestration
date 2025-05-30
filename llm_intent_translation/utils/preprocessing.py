# utils/preprocessing.py

import json
import torch
from torch.utils.data import Dataset
from collections import Counter
from sklearn.model_selection import train_test_split
from typing import List, Tuple

SEED = 42

def build_vocab(texts: List[str], min_freq: int = 2) -> dict:
    counter = Counter()
    for line in texts:
        counter.update(line.lower().split())

    vocab = {'<pad>': 0, '<unk>': 1}
    for word, freq in counter.items():
        if freq >= min_freq:
            vocab[word] = len(vocab)
    return vocab

def encode_text(text: str, vocab: dict, max_len: int) -> Tuple[List[int], List[int]]:
    tokens = text.lower().split()
    ids = [vocab.get(tok, vocab['<unk>']) for tok in tokens[:max_len]]
    padding = [vocab['<pad>']] * (max_len - len(ids))
    attention_mask = [1] * len(ids) + [0] * len(padding)
    return ids + padding, attention_mask

class LumiIntentDataset(Dataset):
    def __init__(self, examples: List[dict], vocab: dict, max_len: int):
        self.examples = examples
        self.vocab = vocab
        self.max_len = max_len

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        item = self.examples[idx]
        input_ids, attn_mask = encode_text(item['text'], self.vocab, self.max_len)
        label = int(item['label'])
        return torch.tensor(input_ids), torch.tensor(attn_mask), torch.tensor(label)

def load_lumi_dataset(train_path: str, val_path: str, max_len: int):
    with open(train_path, 'r') as f:
        train_data = [json.loads(l) for l in f]
    with open(val_path, 'r') as f:
        val_data = [json.loads(l) for l in f]

    all_texts = [ex['text'] for ex in train_data + val_data]
    vocab = build_vocab(all_texts)
    padding_idx = vocab['<pad>']

    train_dataset = LumiIntentDataset(train_data, vocab, max_len)
    val_dataset = LumiIntentDataset(val_data, vocab, max_len)

    return train_dataset, val_dataset, vocab, padding_idx
