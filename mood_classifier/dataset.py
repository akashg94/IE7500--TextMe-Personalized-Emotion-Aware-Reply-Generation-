import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from collections import Counter

import torch
from torch.utils.data import Dataset, DataLoader

nltk.download('punkt')

MAX_SEQ_LEN = 64
BATCH_SIZE = 32

GOEMOTION_LABELS = [
    "admiration", "amusement", "anger", "annoyance", "approval",
    "caring", "confusion", "curiosity", "desire", "disappointment",
    "disapproval", "disgust", "embarrassment", "excitement",
    "fear", "gratitude", "grief", "joy", "love", "nervousness",
    "optimism", "pride", "realization", "relief", "remorse",
    "sadness", "surprise", "neutral"
]

TEXTME_LABELS = [
    "casual", "emotional", "excited", "urgent", "romantic",
    "flirty", "angry", "anxious", "grateful", "apology",
    "question", "checking_in", "supportive", "curious",
    "professional", "naughty", "funny", "family",
]

label_to_idx = {
    label: idx
    for idx, label in enumerate(TEXTME_LABELS)
}

GOEMOTIONS_TO_TEXTME = {
    "admiration": "supportive",
    "amusement": "funny",
    "anger": "angry",
    "annoyance": "angry",
    "approval": "supportive",
    "caring": "supportive",
    "confusion": "question",
    "curiosity": "curious",
    "desire": "romantic",
    "disappointment": "emotional",
    "disapproval": "angry",
    "disgust": "angry",
    "embarrassment": "emotional",
    "excitement": "excited",
    "fear": "anxious",
    "gratitude": "grateful",
    "grief": "emotional",
    "joy": "casual",
    "love": "romantic",
    "nervousness": "anxious",
    "optimism": "supportive",
    "pride": "professional",
    "realization": "curious",
    "relief": "casual",
    "remorse": "apology",
    "sadness": "emotional",
    "surprise": "excited",
    "neutral": "casual",
}

def build_vocab(texts, min_freq=2):

    counter = Counter()

    for text in texts:
        tokens = word_tokenize(str(text).lower())
        counter.update(tokens)

    vocab = {
        "<PAD>": 0,
        "<UNK>": 1
    }

    for token, freq in counter.items():
        if freq >= min_freq:
            vocab[token] = len(vocab)

    return vocab


def tokenize_and_pad(text, vocab):

    tokens = word_tokenize(str(text).lower())

    ids = [
        vocab.get(token, vocab["<UNK>"])
        for token in tokens
    ]

    ids = ids[:MAX_SEQ_LEN]

    while len(ids) < MAX_SEQ_LEN:
        ids.append(vocab["<PAD>"])

    return ids

def map_labels(label_string):

    label_indices = list(
        map(int, str(label_string).split(","))
    )

    mapped = []

    for idx in label_indices:
        emotion = GOEMOTION_LABELS[idx]

        if emotion in GOEMOTIONS_TO_TEXTME:
            mapped.append(
                GOEMOTIONS_TO_TEXTME[emotion]
            )

    if len(mapped) == 0:
        return label_to_idx["casual"]

    final_label = max(
        set(mapped),
        key=mapped.count
    )

    return label_to_idx[final_label]

class GoEmotionsDataset(Dataset):

    def __init__(self, df, vocab):

        self.df = df
        self.vocab = vocab

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):

        row = self.df.iloc[idx]

        text = row["text"]
        label = row["labels"]

        input_ids = tokenize_and_pad(
            text,
            self.vocab
        )

        target = map_labels(label)

        return (
            torch.tensor(input_ids,
                         dtype=torch.long),
            torch.tensor(target,
                         dtype=torch.long)
        )
    
def create_dataloaders(
    train_df,
    val_df,
    test_df,
    batch_size=BATCH_SIZE
):

    vocab = build_vocab(
        train_df["text"]
    )

    train_dataset = GoEmotionsDataset(
        train_df,
        vocab
    )

    val_dataset = GoEmotionsDataset(
        val_df,
        vocab
    )

    test_dataset = GoEmotionsDataset(
        test_df,
        vocab
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False
    )

    return (
        train_loader,
        val_loader,
        test_loader,
        vocab
    )
