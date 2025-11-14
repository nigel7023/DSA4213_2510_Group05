# Overview
Sarcasm is inherently difficult to detect in text because it relies on subtle linguistic cues, sentiment reversal, and implicit context—signals that are often missing in short online messages. This project aims to build a practical, scalable, and interpretable system that can both detect sarcasm and explain it in natural language, bridging an important gap in NLP.

We implement a two-stage pipeline:

Encoder - Sarcasm Detection
Fine-tuned Transformer encoder to classify whether text is sarcastic.

Decoder - Sarcasm Explanation
FLAN-T5-based models generate explanations clarifying why the text is sarcastic.

This setup enables not only accurate classification but also transparent reasoning suitable for real-world applications like sentiment analysis, content moderation, and conversational AI.


