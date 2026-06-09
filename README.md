Project Goal
TextMe is an NLP system that generates personalized text message replies on behalf of a user. The system reads an incoming message, detects its emotional tone, identifies the sender's relationship to the user, and produces a reply that matches the user's personal writing style.
The core problem this solves: existing messaging tools generate generic responses that do not reflect how a specific person writes or adapt to the emotional context of a conversation. A reply to a manager asking for a project update should sound nothing like a reply to a close friend venting about their day. TextMe handles both automatically.

What the System Does
A user receives a text message. The system:

Looks up the sender in the contact profile to determine the relationship type and appropriate tone
Classifies the emotional mood of the incoming message into one of 18 categories
Loads the user's writing style profile built from their own sample texts
Generates three reply options conditioned on the detected mood, relationship tone, and user persona
Returns the three variants for the user to select from

No external API is used. All models are trained locally on open datasets.

Mood Classes
The system classifies incoming messages into 18 mood categories:
casual, emotional, excited, urgent, romantic, flirty, angry, anxious, grateful, apology, question, checking in, supportive, curious, professional, naughty, funny, family

Contact Relationship Types
The system supports 16 relationship types, each with its own tone rules:
girlfriend, wife, friend, sibling, parent, grandparent, manager, colleague, professor, recruiter, classmate, acquaintance, unknown, wrong number, scammer, spam
Scammer and spam contacts return no reply. All others generate a response calibrated to the relationship tone.


Technical Architecture
Incoming Message + Sender Name
        |
        v
Contact Lookup
Reads contacts.json to identify relationship and tone rules
        |
        v
Mood Classifier
Bidirectional LSTM trained on GoEmotions (58k samples, 27 labels mapped to 18 classes)
Outputs mood label and confidence scores
        |
        v
Persona Encoder
Loads user sample texts
Computes GloVe centroid from top vocabulary
Extracts style features: message length, slang, punctuation, emoji frequency
Outputs 104-dimensional persona vector
        |
        v
Retrieval Engine
Searches DailyDialog for messages similar to the incoming text
Filters candidates by detected mood class
Returns top candidate responses
        |
        v
Persona Rewriter
Rewrites candidate response to match user writing style
Applies relationship tone rules
        |
        v
Three Reply Variants
Greedy decode, beam search, top-k sampling

Models
Mood Classifier

Architecture: Bidirectional LSTM
Embedding: GloVe 6B 100-dimensional vectors (frozen)
Hidden size: 128, Layers: 2, Dropout: 0.5
Output: 18-class softmax
Optimizer: Adam, lr=0.001
Loss: CrossEntropyLoss
Training data: GoEmotions

Persona Encoder

Extracts writing style features from user sample texts
Builds GloVe centroid from user vocabulary
Outputs 104-dimensional conditioning vector

Reply Generator

Retrieval-based approach using cosine similarity on GloVe vectors
Searches DailyDialog conversation pairs
Filters by mood class
Rewrites output using persona style rules

Project Outcomes
By the end of this project the system will:

Accurately classify the emotional mood of an incoming text message across 18 categories with a target macro F1 score of 0.65 or higher
Generate replies that reflect the user's personal writing style based on sample texts they provide
Adapt reply tone based on the sender's relationship type
Produce three distinct reply variants per message using different decoding strategies
Run entirely locally with no external API dependencies


Evaluation
Quantitative

Macro F1 score across all 18 mood classes
Per-class precision and recall
BLEU score on generated replies
Perplexity of generated text

Qualitative

User study with 8 to 10 participants rating reply appropriateness on a 1 to 5 scale
Blind test where participants determine whether a reply was written by the system or a human

