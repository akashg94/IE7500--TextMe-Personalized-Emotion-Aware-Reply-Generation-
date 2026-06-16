"""
dataset.py
----------
DailyDialog DataLoader for TextMe Seq2Seq Reply Generator.

Loads DailyDialog dataset, extracts consecutive (input, response) pairs,
builds a shared vocabulary, pads sequences, and returns PyTorch DataLoaders.

Usage:
    from seq2seq_generator.dataset import create_dataloaders

    train_loader, val_loader, test_loader, vocab = create_dataloaders(
        train_df, val_df, test_df
    )
"""

import os
import sys
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from collections import Counter

import nltk
nltk.download('punkt',     quiet=True)
nltk.download('punkt_tab', quiet=True)
from nltk.tokenize import word_tokenize

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import config


def load_dailydialog(data_dir):
    """
    Load DailyDialog CSV files from data directory.

    DailyDialog is structured with three columns:
        dialog  - full conversation as a string
        act     - communication act labels per turn
        emotion - emotion labels per turn

    Args:
        data_dir (str): Path to data/ folder containing train.csv,
                        validation.csv, test.csv

    Returns:
        tuple: (train_df, val_df, test_df)
    """
    train_df = pd.read_csv(os.path.join(data_dir, "dailydialog/train.csv"))
    val_df   = pd.read_csv(os.path.join(data_dir, "dailydialog/validation.csv"))
    test_df  = pd.read_csv(os.path.join(data_dir, "dailydialog/test.csv"))

    print(f"Train conversations : {len(train_df)}")
    print(f"Val conversations   : {len(val_df)}")
    print(f"Test conversations  : {len(test_df)}")

    return train_df, val_df, test_df


def clean_turns(dialog):
    """
    Split a raw dialog string into individual turns.

    Args:
        dialog (str): Raw dialog string from DailyDialog CSV

    Returns:
        list[str]: List of individual turn strings
    """
    dialog = dialog.replace("\n", " ")
    turns  = dialog.split("  ")
    return [t.strip() for t in turns if t.strip()]


def clean_text(t):
    """
    Clean a single dialog turn string.
    Removes brackets, quotes, and fixes spacing around punctuation.

    Args:
        t (str): Raw turn string

    Returns:
        str: Cleaned turn string
    """
    t = t.strip()
    t = t.lstrip("['").rstrip("']")
    t = t.strip("'").strip('"')
    t = t.replace(" ,", ",").replace(" .", ".")
    t = t.replace(" ?", "?").replace(" !", "!")
    t = " ".join(t.split())
    return t


def extract_pairs(df):
    """
    Extract consecutive (input, response) pairs from all conversations.

    Each consecutive pair of turns in a dialog becomes one training sample:
        Turn i   -> input
        Turn i+1 -> response

    Args:
        df (pd.DataFrame): DailyDialog dataframe with 'dialog' column

    Returns:
        list[dict]: List of {"input": str, "response": str} dicts
    """
    pairs = []

    for dialog_str in df["dialog"]:
        turns = clean_turns(str(dialog_str))

        if len(turns) < 2:
            continue

        for i in range(len(turns) - 1):
            pairs.append({
                "input":    clean_text(turns[i]),
                "response": clean_text(turns[i + 1])
            })

    return pairs


def build_vocab(pairs, min_freq=2):
    """
    Build shared vocabulary from training conversation pairs.
    Only includes words appearing at least min_freq times.
    Special tokens from config are always included.

    Args:
        pairs    (list[dict]): Training pairs with 'input' and 'response' keys
        min_freq (int)       : Minimum word frequency to include in vocab

    Returns:
        dict: Word -> index mapping
    """
    counter = Counter()

    for pair in pairs:
        counter.update(word_tokenize(pair["input"].lower()))
        counter.update(word_tokenize(pair["response"].lower()))

    # Special tokens at fixed indices
    vocab = {
        config.PAD_TOKEN: 0,  # <PAD> padding
        config.SOS_TOKEN: 1,  # <SOS> start of sequence
        config.EOS_TOKEN: 2,  # <EOS> end of sequence
        config.UNK_TOKEN: 3,  # <UNK> unknown word
    }

    for word, freq in counter.items():
        if freq >= min_freq:
            vocab[word] = len(vocab)

    return vocab


def tokenize_and_pad(text, vocab, max_len):
    """
    Tokenize text, convert to indices, add SOS/EOS tokens, and pad.

    Args:
        text    (str) : Raw input text
        vocab   (dict): Word -> index mapping
        max_len (int) : Target sequence length

    Returns:
        list[int]: Padded token index sequence of length max_len
    """
    tokens  = word_tokenize(text.lower())
    indices = [vocab.get(t, vocab[config.UNK_TOKEN]) for t in tokens]

    # Truncate to fit SOS and EOS
    indices = indices[:max_len - 2]

    # Add SOS at start and EOS at end
    indices = [vocab[config.SOS_TOKEN]] + indices + [vocab[config.EOS_TOKEN]]

    # Pad to max_len
    padding = [vocab[config.PAD_TOKEN]] * (max_len - len(indices))
    return indices + padding


class DailyDialogDataset(Dataset):
    """
    PyTorch Dataset for DailyDialog conversation pairs.

    Each item returns a tuple of:
        input_tensor    (torch.Tensor): shape [MAX_SEQ_LEN]
        response_tensor (torch.Tensor): shape [MAX_SEQ_LEN]

    Args:
        pairs   (list[dict]): Conversation pairs with 'input' and 'response'
        vocab   (dict)      : Word -> index mapping
        max_len (int)       : Sequence length to pad/truncate to
    """

    def __init__(self, pairs, vocab, max_len=None):
        self.pairs   = pairs
        self.vocab   = vocab
        self.max_len = max_len or config.MAX_SEQ_LEN

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        pair = self.pairs[idx]

        src = tokenize_and_pad(pair["input"],    self.vocab, self.max_len)
        tgt = tokenize_and_pad(pair["response"], self.vocab, self.max_len)

        return (
            torch.tensor(src, dtype=torch.long),
            torch.tensor(tgt, dtype=torch.long)
        )


def create_dataloaders(train_df, val_df, test_df, batch_size=None):
    """
    Full pipeline: extract pairs, build vocab, return DataLoaders.

    Vocabulary is built from training data only to prevent data leakage.

    Args:
        train_df   (pd.DataFrame): Training conversations
        val_df     (pd.DataFrame): Validation conversations
        test_df    (pd.DataFrame): Test conversations
        batch_size (int)         : Batch size (default: config.BATCH_SIZE)

    Returns:
        tuple: (train_loader, val_loader, test_loader, vocab)
    """
    batch_size = batch_size or config.BATCH_SIZE

    # Extract pairs
    print("Extracting conversation pairs...")
    train_pairs = extract_pairs(train_df)
    val_pairs   = extract_pairs(val_df)
    test_pairs  = extract_pairs(test_df)

    print(f"Train pairs : {len(train_pairs)}")
    print(f"Val pairs   : {len(val_pairs)}")
    print(f"Test pairs  : {len(test_pairs)}")

    # Build vocab from training data only
    print("\nBuilding vocabulary...")
    vocab = build_vocab(train_pairs, min_freq=2)
    print(f"Vocabulary size : {len(vocab)}")
    print(f"Special tokens  : PAD=0, SOS=1, EOS=2, UNK=3")

    # Create datasets
    train_dataset = DailyDialogDataset(train_pairs, vocab)
    val_dataset   = DailyDialogDataset(val_pairs,   vocab)
    test_dataset  = DailyDialogDataset(test_pairs,  vocab)

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False)
    test_loader  = DataLoader(test_dataset,  batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader, vocab


# ── Test when run directly ──
if __name__ == "__main__":
    print("=== DailyDialog DataLoader Test ===\n")

    # Load data
    train_df, val_df, test_df = load_dailydialog(config.DATA_DIR)

    # Create DataLoaders
    train_loader, val_loader, test_loader, vocab = create_dataloaders(
        train_df, val_df, test_df
    )

    # Verify batch shapes
    src_batch, tgt_batch = next(iter(train_loader))
    print(f"\nInput batch shape   : {src_batch.shape}")   # [64, 64]
    print(f"Response batch shape: {tgt_batch.shape}")   # [64, 64]

    # Show sample pair
    sample_src = src_batch[0].tolist()
    idx_to_word = {v: k for k, v in vocab.items()}
    decoded = [idx_to_word.get(i, config.UNK_TOKEN) for i in sample_src
               if i not in (vocab[config.PAD_TOKEN],)]
    print(f"\nSample decoded input: {' '.join(decoded[:10])} ...")

    assert src_batch.shape == (config.BATCH_SIZE, config.MAX_SEQ_LEN), \
        f"Expected [{config.BATCH_SIZE}, {config.MAX_SEQ_LEN}]"
    assert tgt_batch.shape == (config.BATCH_SIZE, config.MAX_SEQ_LEN), \
        f"Expected [{config.BATCH_SIZE}, {config.MAX_SEQ_LEN}]"

    print("\nAll assertions passed.")
    print("DailyDialog DataLoader is ready.")
