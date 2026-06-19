"""
embedder.py
-----------
GloVe Embedder and Persona Centroid Builder for TextMe.

Takes style_features from extractor.py, embeds the user's top words
using GloVe vectors, computes a semantic centroid, measures style
axes (formality, warmth, expressivity), and produces a 104-dim
persona vector representing the user's unique writing style.

Usage:
    from persona_encoder.extractor import extract_style_features
    from persona_encoder.embedder import build_persona_vector, save_persona_profile

    features = extract_style_features(samples)
    persona_vec = build_persona_vector(features, glove_path)
    save_persona_profile(persona_vec, features, "persona_profile.json")
"""

import os
import sys
import json
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import config


def load_glove_vectors(glove_path):
    """
    Load GloVe vectors into a word -> vector dictionary.
    Reuses the same loading pattern as mood_classifier/model.py.

    Args:
        glove_path (str): Path to glove.6B.100d.txt

    Returns:
        dict: word -> np.ndarray vector (100-dim)
    """
    print(f"Loading GloVe vectors from {glove_path} ...")

    glove = {}
    with open(glove_path, "r", encoding="utf-8") as f:
        for line in f:
            values = line.split()
            word   = values[0]
            vector = np.array(values[1:], dtype="float32")
            glove[word] = vector

    print(f"GloVe loaded: {len(glove)} vectors")
    return glove


def compute_centroid(top_words, glove):
    """
    Compute the centroid (mean vector) of a user's top words.
    This is the semantic fingerprint of their vocabulary.

    Args:
        top_words (list[str]): User's most common words
        glove     (dict)     : word -> vector mapping

    Returns:
        np.ndarray: 100-dim centroid vector
    """
    embed_dim = config.EMBED_DIM  # 100

    vectors = [glove[w] for w in top_words if w in glove]

    if len(vectors) == 0:
        print("Warning: no top_words found in GloVe vocabulary. "
              "Returning zero vector.")
        return np.zeros(embed_dim, dtype="float32")

    found_ratio = len(vectors) / len(top_words)
    print(f"Top words found in GloVe: {len(vectors)}/{len(top_words)} "
          f"({found_ratio:.0%})")

    centroid = np.mean(vectors, axis=0)
    return centroid


def cosine_similarity(vec_a, vec_b):
    """
    Compute cosine similarity between two vectors.

    Args:
        vec_a (np.ndarray): First vector
        vec_b (np.ndarray): Second vector

    Returns:
        float: Cosine similarity, range [-1, 1]
    """
    norm_a = np.linalg.norm(vec_a)
    norm_b = np.linalg.norm(vec_b)

    if norm_a == 0 or norm_b == 0:
        return 0.0

    return float(np.dot(vec_a, vec_b) / (norm_a * norm_b))


def compute_style_axes(centroid, glove):
    """
    Measure the user's writing style along three semantic axes
    by comparing their vocabulary centroid to GloVe anchor words.

    Args:
        centroid (np.ndarray): User's vocabulary centroid (100-dim)
        glove    (dict)      : word -> vector mapping

    Returns:
        tuple: (formality, warmth, expressivity) each in range [-1, 1]
    """
    # Anchor words representing each style axis
    anchor_words = {
        "formality":   "formal",
        "warmth":      "warm",
        "expressivity": "expressive",
    }

    scores = {}
    for axis, anchor in anchor_words.items():
        if anchor in glove:
            scores[axis] = cosine_similarity(centroid, glove[anchor])
        else:
            print(f"Warning: anchor word '{anchor}' not in GloVe. "
                  f"Defaulting {axis} to 0.0")
            scores[axis] = 0.0

    return scores["formality"], scores["warmth"], scores["expressivity"]


def build_persona_vector(style_features, glove_path):
    """
    Main entry point. Builds the full 104-dim persona vector from
    style features extracted by extractor.py.

    Persona vector structure:
        [0:100]  - GloVe centroid of user's top words (semantic fingerprint)
        [100]    - formality score (cosine sim to "formal")
        [101]    - warmth score (cosine sim to "warm")
        [102]    - expressivity score (cosine sim to "expressive")
        [103]    - emoji_freq (from style_features)

    Args:
        style_features (dict): Output of extractor.extract_style_features()
        glove_path     (str) : Path to glove.6B.100d.txt

    Returns:
        np.ndarray: 104-dim persona vector
    """
    top_words = style_features.get("top_words", [])
    if len(top_words) == 0:
        raise ValueError("style_features must contain non-empty top_words")

    # Load GloVe and compute centroid
    glove    = load_glove_vectors(glove_path)
    centroid = compute_centroid(top_words, glove)

    # Compute style axes via cosine similarity
    formality, warmth, expressivity = compute_style_axes(centroid, glove)

    print(f"\nStyle axes:")
    print(f"  formality    : {formality:.4f}")
    print(f"  warmth       : {warmth:.4f}")
    print(f"  expressivity : {expressivity:.4f}")

    # Final persona vector: centroid(100) + 4 scalars
    scalar_features = np.array([
        formality,
        warmth,
        expressivity,
        style_features.get("emoji_freq", 0.0)
    ], dtype="float32")

    persona_vec = np.concatenate([centroid, scalar_features])

    print(f"\nPersona vector shape: {persona_vec.shape}")
    return persona_vec


def save_persona_profile(persona_vec, style_features, save_path="persona_profile.json"):
    """
    Save the persona vector and style features to a JSON file.

    Args:
        persona_vec    (np.ndarray): 104-dim persona vector
        style_features (dict)      : Output of extractor.extract_style_features()
        save_path      (str)       : Where to save the JSON file

    Returns:
        dict: The saved profile data
    """
    profile = {
        "persona_vec":    persona_vec.tolist(),
        "style_features": style_features,
    }

    with open(save_path, "w") as f:
        json.dump(profile, f, indent=2)

    print(f"Persona profile saved to {save_path}")
    return profile


def load_persona_profile(load_path="persona_profile.json"):
    """
    Load a previously saved persona profile.

    Args:
        load_path (str): Path to persona_profile.json

    Returns:
        dict: {"persona_vec": np.ndarray, "style_features": dict}
    """
    with open(load_path, "r") as f:
        profile = json.load(f)

    return {
        "persona_vec":    np.array(profile["persona_vec"], dtype="float32"),
        "style_features": profile["style_features"],
    }


# ── Test when run directly ──
if __name__ == "__main__":
    print("=== GloVe Embedder and Persona Centroid Test ===\n")

    # Simulate output from extractor.py
    test_style_features = {
        "avg_len":     3.82,
        "upper_ratio": 0.0,
        "punct_rate":  0.18,
        "emoji_freq":  0.15,
        "top_words":   [
            "fr", "lol", "wait", "today", "bro", "nah", "wild", "omw",
            "give", "seriously", "way", "idk", "man", "kinda", "good"
        ],
        "slang_hits":  [
            "fr", "lol", "bro", "nah", "omw", "idk", "kinda"
        ],
    }

    print(f"Input style features:")
    print(f"  top_words: {test_style_features['top_words']}\n")

    # Build persona vector
    persona_vec = build_persona_vector(test_style_features, config.GLOVE_PATH)

    print(f"\nPersona vector preview (first 5 dims): {persona_vec[:5]}")
    print(f"Persona vector preview (last 4 dims) : {persona_vec[-4:]}")

    # Save and reload to verify round-trip works
    test_save_path = "test_persona_profile.json"
    save_persona_profile(persona_vec, test_style_features, test_save_path)

    loaded = load_persona_profile(test_save_path)

    print(f"\nReloaded persona_vec shape: {loaded['persona_vec'].shape}")

    # Assertions
    assert persona_vec.shape == (104,), \
        f"Expected shape (104,) but got {persona_vec.shape}"
    assert loaded["persona_vec"].shape == (104,), \
        "Reloaded persona_vec should also be 104-dim"
    assert np.allclose(persona_vec, loaded["persona_vec"], atol=1e-5), \
        "Reloaded vector should match original"
    assert loaded["style_features"]["top_words"] == test_style_features["top_words"], \
        "Reloaded style_features should match original"

    # Cleanup test file
    os.remove(test_save_path)

    print("\nAll assertions passed.")
    print("GloVe Embedder and Persona Centroid builder is ready.")
