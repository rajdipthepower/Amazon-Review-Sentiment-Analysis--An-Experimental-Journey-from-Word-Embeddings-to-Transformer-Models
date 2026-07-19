This directory contains the complete experimental workflow of the project, beginning with learning domain-specific word embeddings and progressing toward Transformer-based transfer learning.

The notebooks are intended to be read sequentially, as each experiment builds upon the observations and limitations of the previous one.

---

## Repository Structure

### 01 — Word2Vec Embeddings

Train domain-specific Word2Vec embeddings using the Amazon Sports & Outdoors review corpus.

**Highlights**

* Text preprocessing
* Word2Vec training with Gensim
* Embedding quality evaluation
* Semantic similarity analysis

---

### 02 — LSTM & BiLSTM

Build recurrent neural network models using the pretrained Word2Vec embeddings.

**Highlights**

* Average and max pooling
* Sequence padding
* Packed sequences
* Padding masks
* BiLSTM architecture
* Comparative evaluation

---

### 03 — BiLSTM with Linear Attention

Extend the BiLSTM by introducing a learnable attention mechanism.

**Highlights**

* Linear attention
* Attention masking
* WeightedRandomSampler
* Handling class imbalance
* Analysis of why accuracy can be misleading on imbalanced datasets

---

### 04 — DistilBERT Transfer Learning

Fine-tune a pretrained DistilBERT model for sentiment classification using modern transfer learning techniques.

**Highlights**

* Hugging Face Transformers
* Dynamic padding
* Balanced training
* Weighted loss
* Gradual layer unfreezing
* Discriminative learning rates
* Learning rate warmup
* Macro F1-score evaluation
* Confusion matrix analysis

---

## Recommended Reading Order

```text
01 Word2Vec
      │
      ▼
02 LSTM & BiLSTM
      │
      ▼
03 BiLSTM + Linear Attention
      │
      ▼
04 DistilBERT Transfer Learning
```

Each notebook is self-contained and includes its own documentation, implementation details, training procedure, experimental observations, and discussion of results.

Together, the notebooks illustrate the progression from traditional neural architectures to Transformer-based transfer learning while emphasizing the practical challenges of sentiment analysis on highly imbalanced real-world datasets.

