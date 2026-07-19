# 01 — Learning Semantic Word Representations with Word2Vec

## Overview

Before training sequence models such as LSTMs or Transformers, it is useful to learn how words relate to one another in a continuous vector space.

This notebook explores **Word2Vec** using the **Gensim** library to learn distributed word representations from Amazon Sports & Outdoors reviews. Rather than treating words as independent symbols, Word2Vec learns dense vector embeddings where semantically related words occupy nearby locations in the embedding space.

These embeddings are later reused in the BiLSTM experiments as pretrained input representations.

---

## Objective

The primary goals of this experiment were to:

* Learn semantic word embeddings from customer reviews.
* Explore relationships between similar words.
* Verify that the learned embeddings capture meaningful semantic information.
* Generate pretrained embeddings for downstream deep learning models.

---

## Dataset

The model was trained on the cleaned **Amazon Sports & Outdoors Reviews** dataset used throughout this repository.

After preprocessing, each review was tokenized into individual words before training the embedding model.

---

## Methodology

The Word2Vec model was trained using **Gensim** in two stages:

1. Build the vocabulary from all tokenized reviews.
2. Train word embeddings over multiple epochs.

Training summary:

| Property            |                            Value |
| ------------------- | -------------------------------: |
| Training Library    |                           Gensim |
| Epochs              |                               15 |
| Corpus              | Amazon Sports & Outdoors Reviews |
| Vocabulary Size     |             ~31,600 unique words |
| Embedding Dimension |                              100 |

The resulting embedding matrix was later exported and used to initialize the embedding layer of the BiLSTM model.

---

## Sample Results

The learned embedding space successfully grouped semantically related words.

### Similar words to **"awful"**

| Word         | Similarity |
| ------------ | ---------: |
| terrible     |      0.709 |
| horrible     |      0.687 |
| horrendous   |      0.554 |
| overwhelming |      0.530 |
| amazing      |      0.523 |
| overpowering |      0.487 |
| aweful       |      0.486 |
| poor         |      0.479 |
| bad          |      0.475 |
| enormous     |      0.474 |

Although most neighbours are intuitively related, some unexpected associations also appear. This reflects the fact that Word2Vec captures **distributional similarity** (words appearing in similar contexts) rather than purely human notions of semantic similarity.

---

### Word Similarity Examples

| Word Pair     | Cosine Similarity |
| ------------- | ----------------: |
| good ↔ great  |         **0.773** |
| slow ↔ steady |         **0.238** |

The high similarity between *good* and *great* indicates that the model successfully learned positive semantic relationships, whereas *slow* and *steady* appear considerably farther apart, suggesting weaker contextual overlap within this review corpus.

---

## Observations

Some key observations from this experiment include:

* Word embeddings captured meaningful semantic relationships directly from customer reviews.
* Frequently co-occurring words were naturally grouped together in the embedding space.
* Word2Vec measures contextual similarity rather than dictionary definitions, which explains occasional unexpected neighbours.
* The learned embeddings provided a strong initialization for later recurrent neural networks.

---

## Limitations

While Word2Vec provides compact and efficient word representations, it also has important limitations:

* Each word has a **single fixed embedding**, regardless of context.
* Word order is ignored during training.
* Polysemous words (words with multiple meanings) cannot be distinguished.
* Long-range contextual information is not modeled.

These limitations motivated the transition toward recurrent neural networks capable of modelling sequential information.

---

## Why Move Beyond Word2Vec?

Although Word2Vec learns high-quality word representations, it cannot model **sentences**.

Consider the reviews:

> "This product is good."

and

> "This product is not good."

The word **good** receives essentially the same embedding in both sentences, despite expressing opposite sentiment.

To capture sequential context, word order, and long-range dependencies, the next stage introduces **LSTM and Bidirectional LSTM** models.

