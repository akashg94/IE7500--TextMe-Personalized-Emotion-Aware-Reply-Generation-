"""
inference.py
------------
End-to-End Inference Pipeline for TextMe.

Takes a raw incoming message and sender name, and outputs
3 ready-to-send reply variants personalized to the user's
writing style and relationship with the sender.

Pipeline:
    1. Contact lookup  — get relation and tone for sender
    2. Safety check    — block scammer/spam senders
    3. Mood classifier — detect mood from incoming message
    4. Persona load    — load user's persona_profile.json
    5. Retrieval       — find top-3 candidate replies from DailyDialog
    6. Rewriter        — apply persona style and relationship tone
    7. Return          — 3 final reply variants

Usage:
    python pipeline/inference.py
"""

import os
import sys
import json
import torch
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import config

from pipeline.contact_lookup       import get_contact, is_safe_to_reply
from mood_classifier.model         import MoodClassifier
from mood_classifier.dataset       import tokenize_and_pad
from reply_generator.retrieval     import retrieve_candidates
from reply_generator.rewriter      import generate_variants

import nltk
nltk.download('punkt',     quiet=True)
nltk.download('punkt_tab', quiet=True)


# ── Global model cache (load once, reuse) ──
_MODEL  = None
_VOCAB  = None
_DEVICE = None


def load_mood_classifier(checkpoint_path=None):
    """
    Load the trained MoodClassifier from checkpoint.
    Reuses pattern from mood_classifier/evaluate.py.

    Args:
        checkpoint_path (str): Path to .pt checkpoint file.
                               Defaults to config.CHECKPOINT_DIR/mood_classifier_best.pt

    Returns:
        tuple: (model, vocab, device)
    """
    global _MODEL, _VOCAB, _DEVICE

    if _MODEL is not None:
        return _MODEL, _VOCAB, _DEVICE

    checkpoint_path = checkpoint_path or os.path.join(
        config.CHECKPOINT_DIR, "mood_classifier_best.pt"
    )

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(
            f"Checkpoint not found at {checkpoint_path}\n"
            f"Run mood_classifier/train.py first."
        )

    device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(checkpoint_path, map_location=device)

    vocab        = checkpoint["vocab"]
    embed_matrix = np.zeros((len(vocab), config.EMBED_DIM), dtype="float32")

    model = MoodClassifier(
        vocab_size   = len(vocab),
        embed_matrix = embed_matrix
    ).to(device)

    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    _MODEL  = model
    _VOCAB  = vocab
    _DEVICE = device

    print(f"Mood classifier loaded from epoch {checkpoint['epoch']} "
          f"(Val F1: {checkpoint['val_f1']:.4f})")

    return model, vocab, device


def predict_mood(message, model, vocab, device):
    """
    Run the mood classifier on an incoming message.

    Args:
        message (str)        : Incoming text message
        model               : Loaded MoodClassifier
        vocab   (dict)      : Word -> index mapping
        device  (torch.device): cpu or cuda

    Returns:
        str: Predicted mood class name (one of config.MOOD_CLASSES)
    """
    input_ids = tokenize_and_pad(message, vocab)
    tensor    = torch.tensor([input_ids], dtype=torch.long).to(device)

    with torch.no_grad():
        logits = model(tensor)
        pred   = torch.argmax(logits, dim=1).item()

    return config.MOOD_CLASSES[pred]


def load_persona_profile(profile_path="persona_profile.json"):
    """
    Load the user's persona profile from disk.

    Args:
        profile_path (str): Path to persona_profile.json

    Returns:
        dict: {"style_features": dict, "persona_vec": list}
              Returns default style_features if file not found.
    """
    if not os.path.exists(profile_path):
        print(f"Warning: persona_profile.json not found at {profile_path}. "
              f"Using default style features.")
        # Default neutral style
        return {
            "style_features": {
                "avg_len":     8.0,
                "upper_ratio": 0.05,
                "punct_rate":  0.5,
                "emoji_freq":  0.0,
                "top_words":   [],
                "slang_hits":  [],
            },
            "persona_vec": [0.0] * 104
        }

    with open(profile_path, "r") as f:
        return json.load(f)


def generate_replies(message, sender_name, profile_path="persona_profile.json"):
    """
    Main pipeline function. Takes an incoming message and sender name,
    returns 3 personalized reply variants.

    Args:
        message     (str): Incoming text message
        sender_name (str): Name of the person who sent the message
        profile_path(str): Path to user's persona_profile.json

    Returns:
        list[str]: 3 reply variants, or [] if unsafe to reply
    """

    # ── Step 1: Contact lookup ──
    contact  = get_contact(sender_name)
    relation = contact["relation"]
    tone     = contact["tone"]

    # ── Step 2: Safety check ──
    if not is_safe_to_reply(relation):
        return []

    # ── Step 3: Mood classification ──
    model, vocab, device = load_mood_classifier()
    mood = predict_mood(message, model, vocab, device)

    # ── Step 4: Load persona ──
    profile        = load_persona_profile(profile_path)
    style_features = profile["style_features"]

    # ── Step 5: Retrieve candidates ──
    candidates = retrieve_candidates(message, mood, top_k=3)

    if not candidates:
        # Fallback if no candidates found for this mood
        candidates = retrieve_candidates(message, "casual", top_k=3)

    if not candidates:
        return ["Sorry, I'll get back to you!", "Let me think about that.", "I'll reply soon."]

    # ── Step 6 + 7: Rewrite and return variants ──
    # Use the top candidate and generate 3 style variants
    top_candidate = candidates[0][0]
    variants      = generate_variants(top_candidate, style_features, relation)

    return variants


def run_test():
    """
    Run 10 test messages across different moods and contact types.
    Prints detected mood, relation, and 3 reply variants for each.
    """
    print("=" * 60)
    print("TextMe Pipeline — End-to-End Test")
    print("=" * 60 + "\n")

    test_cases = [
        {
            "message":     "hey are you free this weekend?",
            "sender_name": "Alex",
            "expected_mood": "casual"
        },
        {
            "message":     "I love you so much, you mean everything to me",
            "sender_name": "Jordan",
            "expected_mood": "romantic"
        },
        {
            "message":     "I am so angry at what happened today, this is unacceptable",
            "sender_name": "Sam",
            "expected_mood": "angry"
        },
        {
            "message":     "Thank you so much for everything you have done for me",
            "sender_name": "Taylor",
            "expected_mood": "grateful"
        },
        {
            "message":     "I am so nervous about the interview tomorrow",
            "sender_name": "Casey",
            "expected_mood": "anxious"
        },
        {
            "message":     "Please send the report ASAP, it is urgent",
            "sender_name": "Morgan",
            "expected_mood": "professional"
        },
        {
            "message":     "haha that was so funny i can't stop laughing",
            "sender_name": "Riley",
            "expected_mood": "funny"
        },
        {
            "message":     "I am so sorry for what I said, I didn't mean it",
            "sender_name": "Drew",
            "expected_mood": "apology"
        },
        {
            "message":     "This is a scam, click here to claim your prize now",
            "sender_name": "Unknown Scammer",
            "expected_mood": "casual"
        },
        {
            "message":     "I just wanted to check in, how are you doing?",
            "sender_name": "Blake",
            "expected_mood": "checking_in"
        },
    ]

    for i, case in enumerate(test_cases, 1):
        print(f"── Test {i} ──")
        print(f"Message : {case['message']}")
        print(f"Sender  : {case['sender_name']}")

        # Get contact info
        contact  = get_contact(case["sender_name"])
        relation = contact["relation"]
        print(f"Relation: {relation}")

        # Safety check
        if not is_safe_to_reply(relation):
            print(f"Mood    : N/A")
            print(f"Result  : BLOCKED (scammer/spam — no reply generated)")
            print()
            continue

        # Generate replies
        replies = generate_replies(case["message"], case["sender_name"])

        # Get mood for display
        model, vocab, device = load_mood_classifier()
        mood = predict_mood(case["message"], model, vocab, device)
        print(f"Mood    : {mood}")

        if replies:
            for j, reply in enumerate(replies, 1):
                print(f"Reply {j} : {reply}")
        else:
            print("Result  : No reply generated")

        print()

    print("=" * 60)
    print("Pipeline test complete.")
    print("=" * 60)


if __name__ == "__main__":
    run_test()
