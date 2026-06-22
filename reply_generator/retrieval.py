"""
retrieval.py
------------
Retrieval-based candidate response generator for TextMe.

Uses:
- DailyDialog conversation pairs
- GloVe sentence embeddings
- Cosine similarity search

Returns top-k candidate replies.
"""

import os
import pickle
import numpy as np

import nltk
nltk.download("punkt", quiet=True)

from nltk.tokenize import word_tokenize
from sklearn.metrics.pairwise import cosine_similarity

from seq2seq_generator.dataset import (
    load_dailydialog,
    extract_pairs
)

import config


# =====================================================
# Configuration
# =====================================================

GLOVE_PATH = os.path.join(
    config.DATA_DIR,
    "glove",
    "glove.6B.100d.txt"
)

CACHE_FILE = os.path.join(
    os.path.dirname(__file__),
    "dialog_embeddings.pkl"
)

EMBED_DIM = 100


# =====================================================
# Load GloVe
# =====================================================

def load_glove():

    glove = {}

    with open(GLOVE_PATH, "r", encoding="utf8") as f:

        for line in f:

            values = line.split()

            word = values[0]

            vector = np.asarray(
                values[1:],
                dtype=np.float32
            )

            glove[word] = vector

    print(f"Loaded {len(glove)} GloVe vectors")

    return glove


GLOVE = load_glove()


# =====================================================
# Sentence Embedding
# =====================================================

def sentence_embedding(text):

    tokens = word_tokenize(str(text).lower())

    vectors = [
        GLOVE[token]
        for token in tokens
        if token in GLOVE
    ]

    if len(vectors) == 0:
        return np.zeros(EMBED_DIM)

    return np.mean(vectors, axis=0)


# =====================================================
# Build / Load Embedding Cache
# =====================================================

def build_embedding_cache():

    print("Loading DailyDialog...")

    train_df, val_df, test_df = load_dailydialog(
        config.DATA_DIR
    )

    pairs = (
        extract_pairs(train_df)
        + extract_pairs(val_df)
        + extract_pairs(test_df)
    )

    cache = []

    for pair in pairs:

        embedding = sentence_embedding(
            pair["input"]
        )

        cache.append(
            {
                "input": pair["input"],
                "response": pair["response"],
                "embedding": embedding
            }
        )

    with open(CACHE_FILE, "wb") as f:
        pickle.dump(cache, f)

    print(
        f"Saved {len(cache)} embeddings "
        f"to {CACHE_FILE}"
    )

    return cache


def load_embedding_cache():

    if os.path.exists(CACHE_FILE):

        with open(CACHE_FILE, "rb") as f:
            return pickle.load(f)

    return build_embedding_cache()


# =====================================================
# Mood Filter
# =====================================================

MOOD_KEYWORDS = {

    "grateful": [
        "thank",
        "thanks"
    ],

    "apology": [
        "sorry"
    ],

    "supportive": [
        "hope",
        "good luck",
        "take care"
    ],

    "romantic": [
        "love",
        "miss you"
    ],

    "question": [
        "?"
    ]
}


def mood_match(response, mood):

    if mood not in MOOD_KEYWORDS:
        return True

    response = response.lower()

    return any(
        keyword in response
        for keyword in MOOD_KEYWORDS[mood]
    )


# =====================================================
# Retrieval Function
# =====================================================

def retrieve_candidates(
    message,
    mood,
    top_k=3
):

    cache = load_embedding_cache()

    query_embedding = sentence_embedding(
        message
    )

    candidates = []

    for item in cache:

        if not mood_match(
            item["response"],
            mood
        ):
            continue

        score = cosine_similarity(
            query_embedding.reshape(1, -1),
            item["embedding"].reshape(1, -1)
        )[0][0]

        candidates.append(
            (
                item["response"],
                score
            )
        )

    candidates.sort(
        key=lambda x: x[1],
        reverse=True
    )

    return candidates[:top_k]


# =====================================================
# Test
# =====================================================

if __name__ == "__main__":

    results = retrieve_candidates(
        message="hey are you free this weekend?",
        mood="casual",
        top_k=3
    )

    print("\nTop Candidates:\n")

    for response, score in results:

        print(
            f"{score:.4f} | {response}"
        )
