# From Word2Vec to DistilBERT

### Experimental Journey in Amazon Review Sentiment Analysis

> **An experimental NLP project exploring the evolution of text representations—from Word2Vec and recurrent neural networks to attention mechanisms and Transformer-based transfer learning—while investigating the practical challenges of class imbalance, sequence modeling, and model evaluation.**

---

## Why this repository?

Most sentiment analysis projects focus on building a single model and reporting an accuracy score.

This repository takes a different approach.

Instead of asking **"Which model gives the highest accuracy?"**, it investigates:

* How modern NLP architectures evolved
* Why some architectures outperform others
* How class imbalance affects learning
* Why evaluation metrics matter as much as model design
* What practical challenges arise while implementing real NLP systems

Rather than presenting only successful experiments, this repository documents the complete learning process—including failed ideas, unexpected results, implementation challenges, and the reasoning behind each successive model.

---

## Project Roadmap

```text
Amazon Reviews
      │
      ▼
Data Cleaning
      │
      ▼
Word2Vec Embeddings
      │
      ▼
Feed Forward Network
      │
      ▼
LSTM
      │
      ▼
BiLSTM
      │
      ▼
Linear Attention
      │
      ▼
DistilBERT
      │
      ▼
Experimental Analysis
```

Each stage was introduced to solve limitations observed in the previous one rather than simply increasing model complexity.

---

## Repository Structure

```text
amazon-review-sentiment-analysis/

├── data/
├── images/
├── models/
├── notebooks/
│   ├── 01_word2vec/
│   ├── 02_lstm_bilstm/
│   ├── 03_attention/
│   └── 04_distilbert/
│
├── results/
├── requirements.txt
└── README.md
```

Each notebook has its own README explaining the motivation, methodology, implementation details, and observations for that stage.

---

## Experiments

| Stage | Model                       | Objective                                                              |
| ----- | --------------------------- | ---------------------------------------------------------------------- |
| 01    | Word2Vec                    | Learn distributed word embeddings                                      |
| 02    | Feed Forward, LSTM & BiLSTM | Explore sequence modelling using recurrent neural networks             |
| 03    | BiLSTM + Linear Attention   | Learn contextual importance of words through attention                 |
| 04    | DistilBERT                  | Apply Transformer-based transfer learning for sentiment classification |

---

## Dataset

* **Dataset:** Amazon Sports & Outdoors Reviews
* **Total Reviews:** ~296,000
* **Original Labels:** 1–5 Star Ratings
* **Task:** Review Rating Prediction → Sentiment Classification

Original class distribution:

| Rating | Reviews |
| -----: | ------: |
|  ⭐⭐⭐⭐⭐ | 188,208 |
|   ⭐⭐⭐⭐ |  64,809 |
|    ⭐⭐⭐ |  24,071 |
|     ⭐⭐ |  10,204 |
|      ⭐ |   9,045 |

The heavy dominance of positive reviews became one of the central experimental challenges throughout the project.

---

## What this project explores

This repository is not only about model development.

It also explores practical NLP engineering topics including:

* Text preprocessing
* Vocabulary construction
* Word embeddings
* Variable-length sequence handling
* Padding and masking
* Packed sequences in PyTorch
* Mean and Max Pooling
* Attention mechanisms
* Transfer learning
* Gradual layer unfreezing
* Weighted loss functions
* Learning-rate scheduling
* Model evaluation on imbalanced datasets

---

## Key Challenges

During the experiments, several implementation and modeling challenges emerged.

* Correctly handling variable-length sequences using packed sequences.
* Preventing padded tokens from influencing pooling and attention computations through masking.
* Understanding when padding should participate in computation—and when it must be ignored.
* Preventing pretrained Transformer models from catastrophic forgetting during fine-tuning.
* Designing strategies to mitigate severe class imbalance.
* Interpreting seemingly good accuracy scores that masked poor minority-class performance.

Many of these issues required multiple iterations before reaching stable implementations.

---

## A Lesson More Important Than Accuracy

One of the biggest takeaways from this project was that **higher accuracy does not necessarily imply a better classifier.**

Several experiments produced encouraging accuracy values simply because the model became increasingly biased toward the dominant class.

Only after comparing:

* Precision
* Recall
* Macro F1-score
* Weighted F1-score
* Confusion Matrices

did the actual behaviour of the models become clear.

This project therefore emphasizes **careful evaluation** over reporting a single headline metric.

---

## Experimental Highlights

* Learned custom Word2Vec embeddings using Gensim.
* Compared frozen and trainable embeddings.
* Implemented masked mean and max pooling.
* Built LSTM and Bidirectional LSTM classifiers from scratch.
* Designed a custom Linear Attention mechanism.
* Experimented with label reformulation for sentiment prediction.
* Investigated balanced sampling and weighted loss functions.
* Fine-tuned DistilBERT using gradual layer unfreezing.
* Applied discriminative learning rates and custom optimization strategies.
* Compared multiple evaluation metrics to understand model behaviour.

---

## Repository Philosophy

This repository intentionally preserves:

* intermediate experiments,
* commented implementations,
* alternative architectural choices,
* partially executed exploratory sections,
* and unsuccessful ideas.

The notebooks are meant to reflect the actual experimental process rather than present only polished final code.

Negative results were often as informative as successful ones and influenced the direction of subsequent experiments.

---

## Next Steps

Detailed explanations for every stage can be found in the individual notebook READMEs:

* **01 — Word2Vec**
* **02 — LSTM & BiLSTM**
* **03 — Linear Attention**
* **04 — DistilBERT**

Each notebook discusses the motivation, implementation, observations, and lessons learned before moving to the next stage.
