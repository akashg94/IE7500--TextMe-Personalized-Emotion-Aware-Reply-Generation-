"""
model.py
--------
Bidirectional LSTM Mood Classifier for TextMe.
"""

import torch
import torch.nn as nn
import numpy as np
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import config


def load_glove_matrix(vocab, glove_path, embed_dim):
    """Build embedding matrix from pre-trained GloVe vectors."""
    print(f"Loading GloVe vectors from {glove_path} ...")
    glove = {}
    with open(glove_path, "r", encoding="utf-8") as f:
        for line in f:
            values = line.split()
            word   = values[0]
            vector = np.array(values[1:], dtype="float32")
            glove[word] = vector
    print(f"GloVe loaded: {len(glove)} vectors")

    vocab_size   = len(vocab)
    embed_matrix = np.zeros((vocab_size, embed_dim), dtype="float32")
    found = 0
    not_found = 0
    for word, idx in vocab.items():
        if word in glove:
            embed_matrix[idx] = glove[word]
            found += 1
        else:
            embed_matrix[idx] = np.random.normal(scale=0.1, size=(embed_dim,))
            not_found += 1

    print(f"Words found in GloVe : {found}")
    print(f"Words not in GloVe   : {not_found}")
    return embed_matrix


class MoodClassifier(nn.Module):
    """Bidirectional LSTM classifier for mood detection."""

    def __init__(self, vocab_size, embed_matrix, embed_dim=None,
                 hidden_size=None, num_layers=None, num_classes=None,
                 dropout=None, freeze_embed=True):
        super(MoodClassifier, self).__init__()

        embed_dim   = embed_dim   or config.EMBED_DIM
        hidden_size = hidden_size or config.HIDDEN_SIZE
        num_layers  = num_layers  or config.NUM_LAYERS
        num_classes = num_classes or config.NUM_CLASSES
        dropout     = dropout     or config.DROPOUT

        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.embedding.weight = nn.Parameter(
            torch.tensor(embed_matrix, dtype=torch.float32)
        )
        if freeze_embed:
            self.embedding.weight.requires_grad = False

        self.lstm = nn.LSTM(
            input_size    = embed_dim,
            hidden_size   = hidden_size,
            num_layers    = num_layers,
            batch_first   = True,
            bidirectional = True,
            dropout       = dropout if num_layers > 1 else 0.0
        )

        self.dropout = nn.Dropout(dropout)
        self.fc      = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, x):
        embedded = self.dropout(self.embedding(x))
        lstm_out, (hidden, _) = self.lstm(embedded)
        final_hidden = torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1)
        final_hidden = self.dropout(final_hidden)
        logits = self.fc(final_hidden)
        return logits

    def predict(self, x):
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            probs  = torch.softmax(logits, dim=1)
            preds  = torch.argmax(probs, dim=1)
        return preds, probs


if __name__ == "__main__":
    print("=== MoodClassifier Model Test ===\n")

    vocab_size  = 5000
    embed_dim   = config.EMBED_DIM
    batch_size  = config.BATCH_SIZE
    seq_len     = config.MAX_SEQ_LEN
    num_classes = config.NUM_CLASSES

    print(f"Vocab size  : {vocab_size}")
    print(f"Embed dim   : {embed_dim}")
    print(f"Batch size  : {batch_size}")
    print(f"Seq length  : {seq_len}")
    print(f"Num classes : {num_classes}\n")

    dummy_embed_matrix = np.random.randn(vocab_size, embed_dim).astype("float32")

    model = MoodClassifier(vocab_size=vocab_size, embed_matrix=dummy_embed_matrix)
    print("Model architecture:")
    print(model)

    total_params     = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTotal parameters     : {total_params:,}")
    print(f"Trainable parameters : {trainable_params:,}")
    print(f"Frozen (GloVe)       : {total_params - trainable_params:,}\n")

    dummy_input = torch.randint(0, vocab_size, (batch_size, seq_len))
    print(f"Input shape  : {dummy_input.shape}")

    logits = model(dummy_input)
    print(f"Output shape : {logits.shape}")
    print(f"Expected     : [{batch_size}, {num_classes}]\n")

    preds, probs = model.predict(dummy_input)
    print(f"Predictions shape   : {preds.shape}")
    print(f"Probabilities shape : {probs.shape}")
    print(f"Sample prediction   : {preds[0].item()} -> {config.MOOD_CLASSES[preds[0].item()]}")
    print(f"Sample probs sum    : {probs[0].sum().item():.4f}\n")

    assert logits.shape == (batch_size, num_classes)
    assert abs(probs[0].sum().item() - 1.0) < 1e-5
    print("All assertions passed.")
    print("Model is ready for training.")
