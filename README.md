A mood-aware NLP system that learns your personal texting style and generates contextually appropriate replies based on the emotional tone of incoming messages.

What It Does

Detects mood of an incoming message — Casual, Emotional, Urgent, Romantic, Confrontational, or Family Check-In
Learns your style — vocabulary, sentence length, slang, emoji patterns
Generates 3 reply variants conditioned on your persona and the detected mood


Tech Stack

Mood Classifier — Bidirectional LSTM trained on GoEmotions (58k samples)
Persona Engine — GloVe 6B.100d word embeddings
Reply Generator — LSTM Seq2Seq with Bahdanau attention trained on DailyDialog
Framework — PyTorch


Datasets
DatasetUseSourceGoEmotionsMood classifier trainingKaggleDailyDialogReply generator trainingKaggle
