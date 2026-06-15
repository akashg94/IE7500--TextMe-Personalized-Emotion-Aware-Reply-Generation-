"""
dataset.py
----------
GoEmotions DataLoader for TextMe Mood Classifier.

Loads GoEmotions dataset, maps 28 emotion labels to 18 TextMe mood classes,
tokenizes text, pads sequences, and returns PyTorch DataLoaders.

Usage:
    from mood_classifier.dataset import create_dataloaders
    train_loader, val_loader, test_loader, vocab = create_dataloaders(
        train_df, val_df, test_df
    )
"""

import os
import sys
import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from collections import Counter

import torch
from torch.utils.data import Dataset, DataLoader

nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import config

# ── GoEmotions 28 label names (index 0-27) ──
GOEMOTION_LABELS = [
    "admiration",       # 0
    "amusement",        # 1
    "anger",            # 2
    "annoyance",        # 3
    "approval",         # 4
    "caring",           # 5
    "confusion",        # 6
    "curiosity",        # 7
    "desire",           # 8
    "disappointment",   # 9
    "disapproval",      # 10
    "disgust",          # 11
    "embarrassment",    # 12
    "excitement",       # 13
    "fear",             # 14
    "gratitude",        # 15
    "grief",            # 16
    "joy",              # 17
    "love",             # 18
    "nervousness",      # 19
    "optimism",         # 20
    "pride",            # 21
    "realization",      # 22
    "relief",           # 23
    "remorse",          # 24
    "sadness",          # 25
    "surprise",         # 26
    "neutral",          # 27
]

# ── TextMe 18 Mood Classes (from config) ──
TEXTME_LABELS = config.MOOD_CLASSES

# ── Mood class to index mapping ──
label_to_idx = {
    label: idx
    for idx, label in enumerate(TEXTME_LABELS)
}

# ── GoEmotions → TextMe mapping (fixed) ──
GOEMOTIONS_TO_TEXTME = {
    "admiration":     "grateful",     # appreciating someone
    "amusement":      "funny",        # something is funny
    "anger":          "angry",        # direct anger
    "annoyance":      "angry",        # mild anger, irritation
    "approval":       "supportive",   # agreeing, encouraging
    "caring":         "supportive",   # showing care
    "confusion":      "curious",      # not understanding — fixed from question
    "curiosity":      "curious",      # wanting to know more
    "desire":         "romantic",     # wanting someone
    "disappointment": "emotional",    # feeling let down
    "disapproval":    "angry",        # strong disagreement
    "disgust":        "angry",        # strong negative reaction
    "embarrassment":  "emotional",    # feeling ashamed
    "excitement":     "excited",      # very hyped up
    "fear":           "anxious",      # scared of something
    "gratitude":      "grateful",     # saying thank you
    "grief":          "emotional",    # deep sadness, loss
    "joy":            "excited",      # happy, positive — fixed from casual
    "love":           "romantic",     # loving feeling
    "nervousness":    "anxious",      # nervous about something
    "optimism":       "excited",      # positive about future — fixed from supportive
    "pride":          "excited",      # proud of achievement — fixed from professional
    "realization":    "curious",      # understanding something new
    "relief":         "casual",       # relaxed, no longer worried
    "remorse":        "apology",      # feeling guilty, sorry
    "sadness":        "emotional",    # feeling sad
    "surprise":       "excited",      # unexpected news
    "neutral":        "casual",       # no strong emotion
}


def build_vocab(texts, min_freq=2):
    """
    Build vocabulary from training texts.
    Only includes words appearing at least min_freq times.
    Includes special tokens from config.

    Args:
        texts    (pd.Series): Training text samples
        min_freq (int)      : Minimum word frequency to include

    Returns:
        dict: Word -> index mapping
    """
    counter = Counter()
    for text in texts:
        tokens = word_tokenize(str(text).lower())
        counter.update(tokens)

    # Special tokens from config
    vocab = {
        config.PAD_TOKEN: 0,  # <PAD>
        config.SOS_TOKEN: 1,  # <SOS>
        config.EOS_TOKEN: 2,  # <EOS>
        config.UNK_TOKEN: 3,  # <UNK>
    }

    for token, freq in counter.items():
        if freq >= min_freq:
            vocab[token] = len(vocab)

    return vocab


def tokenize_and_pad(text, vocab):
    """
    Tokenize a text string, convert to indices, and pad to MAX_SEQ_LEN.

    Args:
        text  (str) : Raw input text
        vocab (dict): Word -> index mapping

    Returns:
        list[int]: Padded sequence of token indices, length = MAX_SEQ_LEN
    """
    tokens = word_tokenize(str(text).lower())

    ids = [
        vocab.get(token, vocab[config.UNK_TOKEN])
        for token in tokens
    ]

    # Truncate to MAX_SEQ_LEN
    ids = ids[:config.MAX_SEQ_LEN]

    # Pad to MAX_SEQ_LEN
    while len(ids) < config.MAX_SEQ_LEN:
        ids.append(vocab[config.PAD_TOKEN])

    return ids


def map_labels(label_string):
    """
    Convert a GoEmotions label string to a TextMe mood class index.
    Handles multi-label annotations using majority vote.

    Args:
        label_string (str): Comma-separated label indices e.g. "2,14"

    Returns:
        int: TextMe mood class index
    """
    try:
        label_indices = list(map(int, str(label_string).split(",")))
    except ValueError:
        return label_to_idx["casual"]

    mapped = []
    for idx in label_indices:
        if idx < len(GOEMOTION_LABELS):
            emotion = GOEMOTION_LABELS[idx]
            if emotion in GOEMOTIONS_TO_TEXTME:
                mapped.append(GOEMOTIONS_TO_TEXTME[emotion])

    if len(mapped) == 0:
        return label_to_idx["casual"]

    # Majority vote for multi-label samples
    final_label = max(set(mapped), key=mapped.count)
    return label_to_idx[final_label]


class GoEmotionsDataset(Dataset):
    """
    PyTorch Dataset for GoEmotions.
    Each item returns (input_ids_tensor, mood_label_tensor).

    Args:
        df    (pd.DataFrame): Dataset with 'text' and 'labels' columns
        vocab (dict)        : Word -> index mapping
    """

    def __init__(self, df, vocab):
        self.df    = df.reset_index(drop=True)
        self.vocab = vocab

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row   = self.df.iloc[idx]
        text  = row["text"]
        label = row["labels"]

        input_ids = tokenize_and_pad(text, self.vocab)
        target    = map_labels(label)

        return (
            torch.tensor(input_ids, dtype=torch.long),
            torch.tensor(target,    dtype=torch.long)
        )


def create_dataloaders(train_df, val_df, test_df, batch_size=None):
    """
    Build vocabulary and create DataLoaders for train, val, and test sets.

    Args:
        train_df   (pd.DataFrame): Training data
        val_df     (pd.DataFrame): Validation data
        test_df    (pd.DataFrame): Test data
        batch_size (int)         : Batch size (defaults to config.BATCH_SIZE)

    Returns:
        tuple: (train_loader, val_loader, test_loader, vocab)
    """
    batch_size = batch_size or config.BATCH_SIZE  # 64

    # Build vocab from training data only
    vocab = build_vocab(train_df["text"])

    train_dataset = GoEmotionsDataset(train_df, vocab)
    val_dataset   = GoEmotionsDataset(val_df,   vocab)
    test_dataset  = GoEmotionsDataset(test_df,  vocab)

    train_loader = DataLoader(
        train_dataset,
        batch_size = batch_size,
        shuffle    = True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size = batch_size,
        shuffle    = False
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size = batch_size,
        shuffle    = False
    )

    return train_loader, val_loader, test_loader, vocab


# ── Test when run directly ──
if __name__ == "__main__":
    print("=== GoEmotions DataLoader Test ===\n")

    # Create small dummy dataframe for testing
    dummy_data = {
        "text": [
            "I am so happy today!",
            "This makes me really angry.",
            "I love you so much.",
            "I am nervous about tomorrow.",
            "Thank you for everything.",
            "That is so funny lol.",
            "I feel so sad right now.",
            "Can you help me with this?",
        ],
        "labels": ["17", "2", "18", "19", "15", "1", "25", "7"]
    }

    dummy_df = pd.DataFrame(dummy_data)

    # Split into train/val/test
    train_df = dummy_df[:5]
    val_df   = dummy_df[5:7]
    test_df  = dummy_df[7:]

    # Create DataLoaders
    train_loader, val_loader, test_loader, vocab = create_dataloaders(
        train_df, val_df, test_df, batch_size=4
    )

    print(f"Vocabulary size : {len(vocab)}")
    print(f"Train batches   : {len(train_loader)}")
    print(f"Val batches     : {len(val_loader)}")
    print(f"Test batches    : {len(test_loader)}")

    # Check one batch
    for input_ids, labels in train_loader:
        print(f"\nBatch input shape : {input_ids.shape}")
        print(f"Batch label shape : {labels.shape}")
        print(f"Sample label idx  : {labels[0].item()} -> {config.MOOD_CLASSES[labels[0].item()]}")
        break

    # Check label mapping
    print("\nLabel mapping checks:")
    tests = [
        ("17", "excited"),    # joy -> excited (fixed)
        ("15", "grateful"),   # gratitude -> grateful
        ("2",  "angry"),      # anger -> angry
        ("18", "romantic"),   # love -> romantic
        ("6",  "curious"),    # confusion -> curious (fixed)
        ("21", "excited"),    # pride -> excited (fixed)
        ("20", "excited"),    # optimism -> excited (fixed)
        ("27", "casual"),     # neutral -> casual
    ]

    all_passed = True
    for label_str, expected in tests:
        result_idx  = map_labels(label_str)
        result_mood = config.MOOD_CLASSES[result_idx]
        status      = "PASS" if result_mood == expected else "FAIL"
        if status == "FAIL":
            all_passed = False
        print(f"  [{status}] label={label_str} -> {result_mood} (expected: {expected})")

    print(f"\nAll tests passed: {all_passed}")
    print("\nDataLoader is ready for training.")
