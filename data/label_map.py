"""
label_map.py
------------
Maps GoEmotions 27 emotion labels to TextMe 18 mood classes.

GoEmotions uses 27 fine-grained emotion labels.
TextMe uses 18 mood classes that reflect real texting scenarios.

This file is imported by:
    - mood_classifier/dataset.py  (for training)
    - mood_classifier/evaluate.py (for reporting)
"""

# ── GoEmotions 27 label names (index 0-26) ──
# These are the official label names from the GoEmotions dataset
GOEMOTIONS_LABEL_NAMES = [
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

# ── TextMe 18 Mood Classes ──
TEXTME_MOOD_CLASSES = [
    "casual",
    "emotional",
    "excited",
    "urgent",
    "romantic",
    "flirty",
    "angry",
    "anxious",
    "grateful",
    "apology",
    "question",
    "checking_in",
    "supportive",
    "curious",
    "professional",
    "naughty",
    "funny",
    "family",
]

# ── Mapping: GoEmotions label name → TextMe mood class ──
GOEMOTIONS_TO_TEXTME = {
    "admiration":     "grateful",      # appreciating someone
    "amusement":      "funny",         # something is funny/entertaining
    "anger":          "angry",         # direct anger
    "annoyance":      "angry",         # mild anger, irritation
    "approval":       "supportive",    # agreeing, encouraging
    "caring":         "supportive",    # showing care for someone
    "confusion":      "curious",       # not understanding, seeking clarity
    "curiosity":      "curious",       # wanting to know more
    "desire":         "romantic",      # wanting someone or something
    "disappointment": "emotional",     # feeling let down
    "disapproval":    "angry",         # strong disagreement
    "disgust":        "angry",         # strong negative reaction
    "embarrassment":  "emotional",     # feeling ashamed or awkward
    "excitement":     "excited",       # very hyped, enthusiastic
    "fear":           "anxious",       # scared of something
    "gratitude":      "grateful",      # saying thank you
    "grief":          "emotional",     # deep sadness, loss
    "joy":            "excited",       # happy, positive feeling
    "love":           "romantic",      # loving feeling toward someone
    "nervousness":    "anxious",       # nervous about something upcoming
    "optimism":       "excited",       # positive about what is ahead
    "pride":          "excited",       # proud of an achievement
    "realization":    "curious",       # understanding something new
    "relief":         "casual",        # relaxed, no longer worried
    "remorse":        "apology",       # feeling guilty, sorry
    "sadness":        "emotional",     # feeling sad or down
    "surprise":       "excited",       # unexpected news or event
    "neutral":        "casual",        # no strong emotion, everyday talk
}


def map_label_to_mood(label_indices, multi_label_strategy="majority"):
    """
    Convert a list of GoEmotions label indices to a single TextMe mood class.

    GoEmotions supports multi-label annotation — one sample can have
    multiple emotion labels. This function resolves them to one mood.

    Args:
        label_indices (list[int]): List of GoEmotions label indices
        multi_label_strategy (str): How to handle multiple labels
            "majority" - return the most common mapped mood
            "first"    - return mood of the first label only

    Returns:
        str: A TextMe mood class name

    Examples:
        map_label_to_mood([17])        # joy -> "excited"
        map_label_to_mood([15])        # gratitude -> "grateful"
        map_label_to_mood([2, 3])      # anger + annoyance -> "angry"
        map_label_to_mood([])          # empty -> "casual"
    """
    if not label_indices:
        return "casual"

    # Convert indices to label names
    moods = []
    for idx in label_indices:
        if idx < len(GOEMOTIONS_LABEL_NAMES):
            label_name = GOEMOTIONS_LABEL_NAMES[idx]
            mood = GOEMOTIONS_TO_TEXTME.get(label_name, "casual")
            moods.append(mood)

    if not moods:
        return "casual"

    if multi_label_strategy == "first":
        return moods[0]

    # majority vote
    return max(set(moods), key=moods.count)


def parse_label_string(label_str):
    """
    Parse a GoEmotions label string into a list of integer indices.

    GoEmotions stores labels as comma-separated strings like "2,14" or "27".

    Args:
        label_str (str): Raw label string from the dataset

    Returns:
        list[int]: List of label indices

    Examples:
        parse_label_string("2")      # [2]
        parse_label_string("2,14")   # [2, 14]
        parse_label_string("27")     # [27]
    """
    try:
        return [int(x.strip()) for x in str(label_str).split(",")]
    except ValueError:
        return []


def get_mood_from_row(label_str):
    """
    Full pipeline: raw label string -> TextMe mood class.
    Combines parse_label_string and map_label_to_mood.

    Args:
        label_str (str): Raw label string from GoEmotions CSV/TSV

    Returns:
        str: TextMe mood class

    Examples:
        get_mood_from_row("17")     # "excited"
        get_mood_from_row("2,3")    # "angry"
        get_mood_from_row("15")     # "grateful"
    """
    indices = parse_label_string(label_str)
    return map_label_to_mood(indices)


# ── Test when run directly ──
if __name__ == "__main__":
    print("=== Label Map Tests ===\n")

    print("Sample mappings (GoEmotions -> TextMe):")
    test_cases = [
        ("17",   "excited"),      # joy
        ("15",   "grateful"),     # gratitude
        ("2",    "angry"),        # anger
        ("18",   "romantic"),     # love
        ("7",    "curious"),      # curiosity
        ("14",   "anxious"),      # fear
        ("24",   "apology"),      # remorse
        ("27",   "casual"),       # neutral
        ("2,3",  "angry"),        # anger + annoyance
        ("17,15","excited"),      # joy + gratitude -> majority
    ]

    all_passed = True
    for label_str, expected in test_cases:
        result = get_mood_from_row(label_str)
        status = "PASS" if result == expected else "FAIL"
        if status == "FAIL":
            all_passed = False
        print(f"  [{status}] label={label_str:8s} -> {result:15s} (expected: {expected})")

    print("\nFull GoEmotions -> TextMe mapping:")
    for emotion, mood in GOEMOTIONS_TO_TEXTME.items():
        print(f"  {emotion:20s} -> {mood}")

    print("\nTextMe mood class distribution check:")
    from collections import Counter
    mood_counts = Counter(GOEMOTIONS_TO_TEXTME.values())
    for mood, count in sorted(mood_counts.items()):
        print(f"  {mood:15s} : {count} GoEmotions labels mapped to it")

    print(f"\nAll tests passed: {all_passed}")
