# config.py
# Shared constants — everyone imports from this file

# ── Paths ──
GLOVE_PATH      = "data/glove/glove.6B.100d.txt"
CHECKPOINT_DIR  = "results/checkpoints/"
CONTACTS_PATH   = "contacts.json"
DATA_DIR        = "data/"

# ── Model config ──
MAX_SEQ_LEN     = 64
BATCH_SIZE      = 64
EMBED_DIM       = 100
HIDDEN_SIZE     = 128
NUM_LAYERS      = 2
DROPOUT         = 0.5
NUM_CLASSES     = 18
LEARNING_RATE   = 1e-3
NUM_EPOCHS      = 30
PATIENCE        = 5

# ── Mood classes ──
MOOD_CLASSES = [
    "casual",           # everyday small talk
    "emotional",        # sad, vulnerable, venting
    "excited",          # happy news, hyped up
    "urgent",           # needs immediate response
    "romantic",         # loving, affectionate
    "flirty",           # playful, teasing, light
    "angry",            # confrontational, upset
    "anxious",          # nervous, worried
    "grateful",         # thankful, appreciative
    "apology",          # saying sorry
    "question",         # needs an answer
    "checking_in",      # casual welfare check
    "supportive",       # encouraging, uplifting
    "curious",          # interested, asking about something
    "professional",     # work related, formal
    "naughty",          # cheeky, mischievous
    "funny",            # humorous, joking, sarcastic
    "family"            # family related check in
]

# ── Relationship tones ──
RELATIONSHIP_TONES = {
    "girlfriend":   "warm, affectionate, expressive, emoji okay",
    "wife":         "warm, loving, supportive, personal, expressive, emoji okay",
    "friend":       "casual, slang, short, emoji, playful",
    "sibling":      "very casual, teasing, inside jokes, slang, emoji",
    "parent":       "respectful, warm, full sentences, no slang",
    "grandparent":  "very respectful, simple language, warm, full sentences, no emoji, no slang",
    "manager":      "formal, professional, concise, no slang, no emoji",
    "colleague":    "friendly, semi-formal, light humor okay",
    "professor":    "formal, academic, respectful, full sentences, no slang, no emoji",
    "recruiter":    "professional, enthusiastic, concise, no slang",
    "classmate":    "friendly, casual, academic context, light emoji okay",
    "acquaintance": "polite, neutral, not too familiar, brief",
    "unknown":      "neutral, polite, short, ask who is texting",
    "wrong_number": "polite, very brief, clarify wrong number, no personal info",
    "scammer":      "do not engage, flag as suspicious, no reply",
    "spam":         "do not engage, no reply needed"
}

# ── Special tokens ──
PAD_TOKEN = "<PAD>"
SOS_TOKEN = "<SOS>"
EOS_TOKEN = "<EOS>"
UNK_TOKEN = "<UNK>"
