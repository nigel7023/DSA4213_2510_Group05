# Overview
Sarcasm is inherently difficult to detect in text because it relies on subtle linguistic cues, sentiment reversal, and implicit context—signals that are often missing in short online messages. This project aims to build a practical, scalable, and interpretable system that can both detect sarcasm and explain it in natural language, bridging an important gap in NLP.

We implement a two-stage pipeline:

Encoder - Sarcasm Detection
Fine-tuned Transformer encoder to classify whether text is sarcastic.

Decoder - Sarcasm Explanation
FLAN-T5-based models generate explanations clarifying why the text is sarcastic.

This setup enables not only accurate classification but also transparent reasoning suitable for real-world applications like sentiment analysis, content moderation, and conversational AI.


## Dataset
The Encoder portion of the project uses three publicly available sarcasm datasets from Kaggle, covering news headlines, tweets, and Reddit comments. These diverse sources help the model generalize across different writing styles and platforms.

1. News Headlines (Sarcastic / Non-Sarcastic)

Kaggle: [News Headlines Dataset](https://www.kaggle.com/datasets/rmisra/news-headlines-dataset-for-sarcasm-detection)

Description:
A benchmark dataset of short news headlines labelled as sarcastic or not. Useful for structured, formal text understanding.

2. Tweets with Sarcasm and Irony

Kaggle: [Tweets with Sarcasm and Irony](https://www.kaggle.com/datasets/nikhiljohnk/tweets-with-sarcasm-and-irony)

Description:
A dataset of tweets annotated for sarcasm and irony, ideal for modeling informal, conversational social media language.

3. Sarcastic Comments on Reddit

Kaggle: [Sarcastic Comments on Reddit](https://www.kaggle.com/datasets/sherinclaudia/sarcastic-comments-on-reddit)

Description:
Contains sarcastic comments extracted from Reddit threads, providing longer text examples with richer context.

As raw datasets are too large, they are not included in the repository. The cleaned dataset, which samples from the raw datasets, can be found in combine_data_clean.csv under data


To train and evaluate the explanation component of our pipeline, we use two publicly available sarcasm explanation datasets:

1. MuSE (Multimodal Sarcasm Explanations)

Github: [MuSE Dataset](https://github.com/LCS2-IIITD/Multimodal-Sarcasm-Explanation-MuSE)

Description:
A multimodal dataset containing sarcastic text paired with human-written explanations. Useful for training models to generate interpretable explanations for sarcasm detection.

2. Kaggle Sarcasm Explanation Dataset (by prayag007)

Kaggle: [Sarcasm Explanation Dataset](https://www.kaggle.com/datasets/prayag007/sarcasm-explanation)

Description: 
Contains sarcastic text and human-written explanations for what they actually mean

# Experiments
## Encoder for Sarcasm Detection
We fine-tuned multiple pretrained models:
- DistilBERT
- BERT
- RoBERTa

Experiments included:
- Full fine-tuning vs LoRA parameter-efficient tuning
- Domain generalisation 
- Sentiment masking ablation to test reliance on polarity cues

Hyperparameter search was conducted with Optuna using Bayesian optimisation.

## Decoder for explanation generation
We evaluated several generative models:
- FLAN-T5 (base & small)
- GPT-2
- BART
FLAN-T5-base provided the best explanations on:
- MuSE (Multimodal Sarcasm Explanation) text subset
- Kaggle Sarcasm Explanation Dataset
To address limited explanation data, we:
- Augmented training labels with synthetic explanations from Claude
- Performed knowledge distillation to smaller T5 models for faster inference

