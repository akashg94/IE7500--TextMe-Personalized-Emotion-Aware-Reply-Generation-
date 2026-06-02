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
NUM_CLASSES     = 6
LEARNING_RATE   = 1e-3
NUM_EPOCHS      = 30
PATIENCE        = 5

# ── Mood classes ──
MOOD_CLASSES = [
    "casual",
    "emotional",
    "urgent",
    "romantic",
    "confrontational",
    "family"
]

# ── Relationship tones ──
RELATIONSHIP_TONES = {
    "girlfriend": "warm, affectionate, expressive, emoji okay",
    "wife":       "warm, loving, supportive, personal",
    "friend":     "casual, slang, short, emoji, playful",
    "manager":    "formal, professional, concise, no slang",
    "colleague":  "friendly, semi-formal, light humor okay",
    "parent":     "respectful, warm, full sentences, no slang"
}

# ── Special tokens ──
PAD_TOKEN = "<PAD>"
SOS_TOKEN = "<SOS>"
EOS_TOKEN = "<EOS>"
UNK_TOKEN = "<UNK>"
