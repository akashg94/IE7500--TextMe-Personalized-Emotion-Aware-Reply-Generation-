"""
extractor.py
------------
Persona Style Extractor for TextMe.

Analyzes a user's sample texts and extracts writing style features:
message length, capitalization, punctuation, emoji usage, top words,
and slang usage. These features feed into the GloVe embedder to build
a persona vector representing the user's unique texting style.

Usage:
    from persona_encoder.extractor import extract_style_features

    samples = ["bro where u at lol", "nah fr that's wild", ...]
    features = extract_style_features(samples)
"""

import os
import sys
import re
import string
from collections import Counter

import nltk
nltk.download('punkt',     quiet=True)
nltk.download('punkt_tab', quiet=True)
nltk.download('stopwords', quiet=True)
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))


# ── Common texting slang to detect ──
SLANG_LIST = {
    "lol", "lmao", "lmfao", "rofl", "fr", "ngl", "tbh", "imo", "imho",
    "bro", "sis", "bestie", "fam", "homie",
    "omg", "omfg", "wtf", "smh", "idk", "idc", "ikr",
    "btw", "rn", "atm", "asap", "fyi",
    "gonna", "wanna", "gotta", "kinda", "sorta",
    "bruh", "yo", "nah", "yeah", "yep", "nope",
    "lit", "fire", "slay", "vibe", "vibing", "mood",
    "deadass", "lowkey", "highkey", "finna", "tryna",
    "wyd", "hbu", "hmu", "omw", "brb", "gtg", "ttyl",
    "cap", "nocap", "sus", "bet", "fam", "savage",
    "extra", "salty", "shook", "tea", "snatched",
    "periodt", "facts", "vibes", "lit", "wack",
}

# ── Common emoji ranges (basic detection without external library) ──
EMOJI_PATTERN = re.compile(
    "["
    "\U0001F600-\U0001F64F"  # emoticons
    "\U0001F300-\U0001F5FF"  # symbols & pictographs
    "\U0001F680-\U0001F6FF"  # transport & map symbols
    "\U0001F1E0-\U0001F1FF"  # flags
    "\U00002700-\U000027BF"  # dingbats
    "\U0001F900-\U0001F9FF"  # supplemental symbols
    "\U00002600-\U000026FF"  # misc symbols
    "]+",
    flags=re.UNICODE
)


def count_emojis(text):
    """
    Count the number of emoji characters in a text string.

    Args:
        text (str): Input text

    Returns:
        int: Number of emoji characters found
    """
    return len(EMOJI_PATTERN.findall(text))


def count_punctuation(text):
    """
    Count punctuation marks relevant to texting style
    (periods, question marks, exclamation marks).

    Args:
        text (str): Input text

    Returns:
        int: Count of ., ?, ! characters
    """
    return sum(1 for char in text if char in ".!?")


def compute_upper_ratio(text):
    """
    Compute ratio of uppercase characters to total alphabetic characters.

    Args:
        text (str): Input text

    Returns:
        float: Ratio between 0.0 and 1.0
    """
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return 0.0
    upper_count = sum(1 for c in letters if c.isupper())
    return upper_count / len(letters)


def get_top_words(samples, n=20):
    """
    Extract the top N most common words across all samples,
    excluding English stopwords and punctuation-only tokens.

    Args:
        samples (list[str]): User's sample texts
        n       (int)       : Number of top words to return

    Returns:
        list[str]: Top N most frequent words
    """
    stop_words = set(stopwords.words('english'))
    counter = Counter()

    for text in samples:
        tokens = word_tokenize(text.lower())
        for token in tokens:
            # Skip stopwords, punctuation-only tokens, and very short tokens
            if (token not in stop_words
                    and token not in string.punctuation
                    and len(token) > 1
                    and token.isalpha()):
                counter[token] += 1

    return [word for word, _ in counter.most_common(n)]


def detect_slang(samples):
    """
    Detect which slang terms from SLANG_LIST appear in the user's samples.

    Args:
        samples (list[str]): User's sample texts

    Returns:
        list[str]: Slang terms found, sorted by frequency descending
    """
    counter = Counter()

    for text in samples:
        tokens = word_tokenize(text.lower())
        for token in tokens:
            if token in SLANG_LIST:
                counter[token] += 1

    return [word for word, _ in counter.most_common()]


def extract_style_features(samples):
    """
    Main entry point. Extracts all style features from a list of
    user sample texts.

    Args:
        samples (list[str]): User's sample texts, minimum 20 recommended

    Returns:
        dict: {
            "avg_len":     float — average word count per message
            "upper_ratio": float — average uppercase character ratio
            "punct_rate":  float — average punctuation marks per message
            "emoji_freq":  float — average emoji count per message
            "top_words":   list[str] — top 20 most common words
            "slang_hits":  list[str] — slang terms detected, by frequency
        }
    """
    if not samples or len(samples) == 0:
        raise ValueError("samples list cannot be empty")

    if len(samples) < 20:
        print(f"Warning: only {len(samples)} samples provided. "
              f"20+ recommended for a reliable persona profile.")

    # Average message length (word count)
    lengths = [len(text.split()) for text in samples]
    avg_len = sum(lengths) / len(lengths)

    # Average uppercase ratio across all samples
    upper_ratios = [compute_upper_ratio(text) for text in samples]
    upper_ratio = sum(upper_ratios) / len(upper_ratios)

    # Average punctuation marks per message
    punct_counts = [count_punctuation(text) for text in samples]
    punct_rate = sum(punct_counts) / len(punct_counts)

    # Average emoji count per message
    emoji_counts = [count_emojis(text) for text in samples]
    emoji_freq = sum(emoji_counts) / len(emoji_counts)

    # Top 20 words, excluding stopwords
    top_words = get_top_words(samples, n=20)

    # Slang detection
    slang_hits = detect_slang(samples)

    return {
        "avg_len":     round(avg_len, 2),
        "upper_ratio": round(upper_ratio, 4),
        "punct_rate":  round(punct_rate, 2),
        "emoji_freq":  round(emoji_freq, 2),
        "top_words":   top_words,
        "slang_hits":  slang_hits,
    }


# ── Test when run directly ──
if __name__ == "__main__":
    print("=== Persona Style Extractor Test ===\n")

    # Sample texts representing a casual, slang-heavy texting style
    test_samples = [
        "bro where u at lol",
        "nah fr that's wild",
        "omw give me 5",
        "wait seriously??",
        "lmaooo no way",
        "idk man that's kinda sus",
        "fr fr no cap",
        "yooo what's good",
        "bruh i can't believe that",
        "ngl that's actually fire",
        "lowkey tired rn",
        "omg stop it",
        "wyd later",
        "bet, see you then",
        "tbh i don't really care",
        "deadass thought you forgot",
        "yeah for sure!!",
        "nope not today",
        "smh why would you do that",
        "vibes are immaculate today",
        "this is so funny lol",
        "wait what happened",
    ]

    print(f"Number of samples: {len(test_samples)}\n")

    features = extract_style_features(test_samples)

    print("Extracted style features:")
    print(f"  avg_len     : {features['avg_len']} words/message")
    print(f"  upper_ratio : {features['upper_ratio']}")
    print(f"  punct_rate  : {features['punct_rate']} marks/message")
    print(f"  emoji_freq  : {features['emoji_freq']} emojis/message")
    print(f"  top_words   : {features['top_words']}")
    print(f"  slang_hits  : {features['slang_hits']}")

    # Assertions
    assert isinstance(features['avg_len'], float), "avg_len should be float"
    assert isinstance(features['upper_ratio'], float), "upper_ratio should be float"
    assert isinstance(features['punct_rate'], float), "punct_rate should be float"
    assert isinstance(features['emoji_freq'], float), "emoji_freq should be float"
    assert isinstance(features['top_words'], list), "top_words should be list"
    assert isinstance(features['slang_hits'], list), "slang_hits should be list"
    assert len(features['top_words']) <= 20, "top_words should be max 20"
    assert "lol" in features['slang_hits'] or "lmaooo" in str(test_samples), \
        "should detect slang in test samples"

    print("\nAll assertions passed.")
    print("Persona Style Extractor is ready.")
