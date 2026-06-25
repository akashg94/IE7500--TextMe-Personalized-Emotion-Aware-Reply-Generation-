"""
rewriter.py
-----------
Persona Rewriter for TextMe Reply Generator.

Takes raw candidate replies from retrieval.py and rewrites them
to match the user's personal texting style and relationship tone.

Pipeline order (most restrictive first):
    1. apply_relationship_tone() — strip/add formality based on relation
    2. apply_length_style()      — match user's avg message length
    3. apply_capitalization()    — match user's caps usage
    4. apply_punctuation()       — match user's punctuation density
    5. inject_slang()            — swap in user's slang where natural

Usage:
    from reply_generator.rewriter import rewrite_reply, generate_variants

    variants = generate_variants(
        text          = "I am not sure, let me check",
        style_features= style_features,
        relation      = "friend"
    )
"""

import os
import sys
import json
import random
import re

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import config
from pipeline.contact_lookup import get_tone


# ── Relations that require formal tone (no slang) ──
FORMAL_RELATIONS = {
    "manager", "professor", "recruiter", "colleague", "professional"
}

# ── Relations that allow very casual tone ──
CASUAL_RELATIONS = {
    "friend", "sibling", "classmate", "girlfriend", "wife"
}

# ── Slang substitution map ──
# Maps common formal words -> casual slang equivalents
SLANG_MAP = {
    "yes":          ["yeah", "yep", "fr", "bet"],
    "no":           ["nah", "nope"],
    "okay":         ["ok", "bet", "aight"],
    "ok":           ["bet", "aight"],
    "i do not know":["idk"],
    "i don't know": ["idk"],
    "let me":       ["lemme"],
    "going to":     ["gonna"],
    "want to":      ["wanna"],
    "got to":       ["gotta"],
    "kind of":      ["kinda"],
    "sort of":      ["sorta"],
    "because":      ["cuz", "bc"],
    "though":       ["tho"],
    "right now":    ["rn"],
    "to be honest": ["tbh"],
    "not gonna lie":["ngl"],
    "by the way":   ["btw"],
    "laughing":     ["lol"],
    "that is":      ["that's"],
    "i am":         ["i'm"],
    "you are":      ["you're"],
    "what is up":   ["wyd", "wassup"],
}


# ── Punctuation intensifiers ──
HIGH_PUNCT_ADDITIONS = ["!", "!!", "?!", " lol", " haha"]
LOW_PUNCT_REMOVALS   = ["!", "?"]


def load_persona_profile(profile_path="persona_profile.json"):
    """
    Load a saved persona profile from disk.

    Args:
        profile_path (str): Path to persona_profile.json

    Returns:
        dict: {"style_features": dict, "persona_vec": list}
    """
    if not os.path.exists(profile_path):
        raise FileNotFoundError(
            f"Persona profile not found at {profile_path}. "
            f"Run persona_encoder/embedder.py first."
        )

    with open(profile_path, "r") as f:
        return json.load(f)


def apply_relationship_tone(text, relation):
    """
    Adjust reply formality based on relationship type.

    For formal relations (manager, professor, recruiter):
        - Capitalize first letter
        - Ensure proper punctuation
        - Remove any slang markers

    For casual relations (friend, sibling, girlfriend):
        - Allow lowercase, relaxed punctuation

    Args:
        text     (str): Raw candidate reply
        relation (str): Contact relationship type

    Returns:
        str: Tone-adjusted reply
    """
    relation = relation.lower()

    if relation in FORMAL_RELATIONS:
        # Capitalize first letter, ensure ends with period
        text = text.strip()
        if text:
            text = text[0].upper() + text[1:]
        if text and text[-1] not in ".!?":
            text = text + "."

        # Remove common slang from formal replies
        slang_to_remove = [
            "lol", "lmao", "omg", "ngl", "tbh", "fr", "bruh",
            "bro", "sis", "nah", "yep", "bet", "idk", "wyd"
        ]
        words = text.split()
        words = [w for w in words if w.lower() not in slang_to_remove]
        text = " ".join(words)

    elif relation in CASUAL_RELATIONS:
        # Casual — strip trailing period for informal feel
        text = text.strip()
        if text.endswith(".") and not text.endswith("..."):
            text = text[:-1]

    return text


def apply_length_style(text, style_features):
    """
    Trim reply to roughly match the user's average message length.

    If the user's avg_len is short (< 5 words), trim the reply.
    If avg_len is long (> 15 words), allow the full reply.

    Args:
        text           (str) : Candidate reply text
        style_features (dict): User's style features from extractor.py

    Returns:
        str: Length-adjusted reply
    """
    avg_len = style_features.get("avg_len", 8)
    words   = text.split()

    if avg_len < 5:
        # Very short texter — trim to first 5 words
        words = words[:5]
    elif avg_len < 10:
        # Medium — trim to first 12 words
        words = words[:12]
    # If avg_len >= 10, keep full reply

    return " ".join(words)


def apply_capitalization(text, style_features):
    """
    Adjust capitalization to match user's style.

    If upper_ratio < 0.05: user barely uses caps -> lowercase everything
    If upper_ratio > 0.30: user uses lots of caps -> keep normal caps
    Otherwise: keep as-is

    Args:
        text           (str) : Candidate reply text
        style_features (dict): User's style features from extractor.py

    Returns:
        str: Capitalization-adjusted reply
    """
    upper_ratio = style_features.get("upper_ratio", 0.1)

    if upper_ratio < 0.05:
        return text.lower()
    elif upper_ratio > 0.30:
        # Keep normal capitalization (sentence case)
        return text
    else:
        return text


def apply_punctuation(text, style_features, intensity=1.0):
    """
    Adjust punctuation density to match user's style.

    If user's punct_rate is high (> 1.5): add emphasis punctuation
    If user's punct_rate is low (< 0.3): strip some punctuation

    Args:
        text           (str)  : Candidate reply text
        style_features (dict) : User's style features from extractor.py
        intensity      (float): Multiplier to vary between variants (0.5-1.5)

    Returns:
        str: Punctuation-adjusted reply
    """
    punct_rate = style_features.get("punct_rate", 0.5) * intensity
    text       = text.strip()

    if punct_rate > 1.5:
        # High punctuation user — add emphasis if not already there
        if text and text[-1] not in "!?":
            addition = random.choice(HIGH_PUNCT_ADDITIONS)
            text = text + addition

    elif punct_rate < 0.3:
        # Low punctuation user — strip trailing punctuation
        if text and text[-1] in ".!?":
            text = text[:-1]

    return text


def inject_slang(text, style_features, rate=0.5):
    """
    Replace common formal phrases with user's slang where natural.

    Only injects slang if the user actually uses slang (slang_hits non-empty).
    Rate controls how aggressively to inject (0.0 = none, 1.0 = always).

    Args:
        text           (str)  : Candidate reply text
        style_features (dict) : User's style features from extractor.py
        rate           (float): Injection probability (0.0 to 1.0)

    Returns:
        str: Slang-injected reply
    """
    slang_hits = style_features.get("slang_hits", [])

    # Only inject if user actually uses slang
    if not slang_hits:
        return text

    text_lower = text.lower()

    for formal, slang_options in SLANG_MAP.items():
        if formal in text_lower and random.random() < rate:
            # Pick slang that user actually uses if possible
            user_slang = [s for s in slang_options if s in slang_hits]
            chosen     = random.choice(user_slang) if user_slang else random.choice(slang_options)

            # Replace case-insensitively
            pattern = re.compile(re.escape(formal), re.IGNORECASE)
            text    = pattern.sub(chosen, text, count=1)

    return text


def rewrite_reply(text, style_features, relation, slang_rate=0.5, punct_intensity=1.0):
    """
    Main rewrite function. Chains all style transformations in order.

    Pipeline:
        1. Relationship tone (most restrictive — strips slang for formal)
        2. Length adjustment
        3. Capitalization
        4. Punctuation
        5. Slang injection (skipped if formal relation)

    Args:
        text             (str)  : Raw candidate reply from retrieval.py
        style_features   (dict) : User's style_features from extractor.py
        relation         (str)  : Contact relationship type
        slang_rate       (float): How aggressively to inject slang (0.0-1.0)
        punct_intensity  (float): Punctuation intensity multiplier

    Returns:
        str: Rewritten reply matching user's persona and relationship tone
    """
    # Step 1 — Relationship tone (most restrictive)
    text = apply_relationship_tone(text, relation)

    # Step 2 — Length
    text = apply_length_style(text, style_features)

    # Step 3 — Capitalization
    text = apply_capitalization(text, style_features)

    # Step 4 — Punctuation
    text = apply_punctuation(text, style_features, intensity=punct_intensity)

    # Step 5 — Slang (only for non-formal relations)
    if relation.lower() not in FORMAL_RELATIONS:
        text = inject_slang(text, style_features, rate=slang_rate)

    return text.strip()


def generate_variants(text, style_features, relation, n=3):
    """
    Generate N reply variants by varying slang and punctuation intensity.

    Variant 1 — Low slang, low punctuation (reserved)
    Variant 2 — Medium slang, medium punctuation (balanced)
    Variant 3 — High slang, high punctuation (expressive)

    Args:
        text           (str)  : Raw candidate reply from retrieval.py
        style_features (dict) : User's style_features from extractor.py
        relation       (str)  : Contact relationship type
        n              (int)  : Number of variants to generate

    Returns:
        list[str]: N rewritten reply variants
    """
    configs = [
        {"slang_rate": 0.2, "punct_intensity": 0.5},  # reserved
        {"slang_rate": 0.5, "punct_intensity": 1.0},  # balanced
        {"slang_rate": 0.8, "punct_intensity": 1.5},  # expressive
    ]

    variants = []
    for i in range(min(n, len(configs))):
        variant = rewrite_reply(
            text,
            style_features,
            relation,
            slang_rate      = configs[i]["slang_rate"],
            punct_intensity = configs[i]["punct_intensity"]
        )
        variants.append(variant)

    return variants


# ── Test when run directly ──
if __name__ == "__main__":
    print("=== Persona Rewriter Test ===\n")

    # Simulated style_features from extractor.py
    casual_style = {
        "avg_len":     4.0,
        "upper_ratio": 0.02,
        "punct_rate":  0.5,
        "emoji_freq":  0.1,
        "top_words":   ["fr", "lol", "bro", "nah", "idk"],
        "slang_hits":  ["fr", "lol", "bro", "nah", "idk", "bet", "ngl"],
    }

    formal_style = {
        "avg_len":     12.0,
        "upper_ratio": 0.08,
        "punct_rate":  0.8,
        "emoji_freq":  0.0,
        "top_words":   ["meeting", "deadline", "please", "regards"],
        "slang_hits":  [],
    }

    test_cases = [
        {
            "text":     "I am not sure, let me check",
            "relation": "friend",
            "style":    casual_style,
        },
        {
            "text":     "I am not sure, let me check",
            "relation": "professor",
            "style":    formal_style,
        },
        {
            "text":     "Yes I think that is a good idea",
            "relation": "girlfriend",
            "style":    casual_style,
        },
        {
            "text":     "I do not know what to say right now",
            "relation": "manager",
            "style":    formal_style,
        },
    ]

    for case in test_cases:
        print(f"Input    : {case['text']}")
        print(f"Relation : {case['relation']}")
        variants = generate_variants(
            case["text"],
            case["style"],
            case["relation"]
        )
        for i, v in enumerate(variants, 1):
            print(f"  Variant {i}: {v}")
        print()

    print("Rewriter test complete.")
