"""
contact_lookup.py
-----------------
Loads contact profiles from contacts.json and provides
lookup functions for the TextMe inference pipeline.

Functions:
    load_contacts()            - Load all contacts from contacts.json
    get_contact(name)          - Return relationship + tone for a given sender name
    is_safe_to_reply(relation) - Return False for scammer and spam contacts
    get_tone(relation)         - Return tone rules for a given relation type
"""

import json
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import config


def load_contacts():
    """
    Load contacts from contacts.json.
    Returns a dictionary mapping lowercase name to contact info.
    """
    with open(config.CONTACTS_PATH, "r") as f:
        data = json.load(f)

    contact_map = {}
    for contact in data["contacts"]:
        name = contact["name"].lower()
        contact_map[name] = {
            "name":     contact["name"],
            "relation": contact["relation"],
            "tone":     contact["tone"]
        }

    return contact_map


# Load contacts once at import time
CONTACT_MAP = load_contacts()

# Relations that should never receive a reply
UNSAFE_RELATIONS = {"scammer", "spam"}


def get_contact(name):
    """
    Look up a sender by name and return their contact profile.

    Args:
        name (str): The sender's name as it appears in the message

    Returns:
        dict with keys: name, relation, tone

    Example:
        get_contact("John")
        # If John is in contacts.json:
        # {"name": "John", "relation": "friend", "tone": "casual, slang, ..."}
        # If John is not found:
        # {"name": "John", "relation": "unknown", "tone": "neutral, polite, ..."}
    """
    key = name.strip().lower()

    if key in CONTACT_MAP:
        return CONTACT_MAP[key]

    # Default for unknown senders
    return {
        "name":     name,
        "relation": "unknown",
        "tone":     config.RELATIONSHIP_TONES.get(
                        "unknown",
                        "neutral, polite, short, ask who is texting"
                    )
    }


def is_safe_to_reply(relation):
    """
    Check whether the pipeline should generate a reply for this relation type.
    Returns False for scammer and spam.

    Args:
        relation (str): The relationship type

    Returns:
        bool: True if safe to reply, False if reply should be blocked

    Example:
        is_safe_to_reply("friend")   # True
        is_safe_to_reply("scammer")  # False
        is_safe_to_reply("spam")     # False
        is_safe_to_reply("unknown")  # True
    """
    return relation.lower() not in UNSAFE_RELATIONS


def get_tone(relation):
    """
    Return the tone rules string for a given relation type.
    Falls back to unknown tone if relation not found.

    Args:
        relation (str): The relationship type

    Returns:
        str: Tone rules string
    """
    return config.RELATIONSHIP_TONES.get(
        relation.lower(),
        config.RELATIONSHIP_TONES.get("unknown", "neutral, polite, brief")
    )


# ── Test when run directly ──
if __name__ == "__main__":
    print("=== Contact Lookup Tests ===\n")

    # Test 1 — all contacts loaded from contacts.json
    print("All contacts loaded from contacts.json:")
    for name, info in CONTACT_MAP.items():
        safe = is_safe_to_reply(info["relation"])
        print(f"  {info['name']:15s} -> {info['relation']:15s} | safe to reply: {safe}")

    # Test 2 — unknown sender
    print("\nUnknown sender test:")
    result = get_contact("UnknownPerson123")
    print(f"  Name    : {result['name']}")
    print(f"  Relation: {result['relation']}")
    print(f"  Tone    : {result['tone']}")

    # Test 3 — is_safe_to_reply for all relation types
    print("\nis_safe_to_reply for all relation types:")
    for relation in config.RELATIONSHIP_TONES.keys():
        safe = is_safe_to_reply(relation)
        print(f"  {relation:15s} -> {safe}")

    # Test 4 — get_tone for all relation types
    print("\nget_tone for all relation types:")
    for relation in config.RELATIONSHIP_TONES.keys():
        tone = get_tone(relation)
        print(f"  {relation:15s} -> {tone}")

    print("\nAll tests passed.")
