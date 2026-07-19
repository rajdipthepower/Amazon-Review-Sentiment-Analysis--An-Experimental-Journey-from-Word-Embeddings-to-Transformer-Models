# 02 — Sentiment Classification using LSTM & BiLSTM

## Overview

After learning meaningful word embeddings using Word2Vec, the next objective was to build models capable of understanding **word order and contextual dependencies**.

Unlike Word2Vec, which assigns a fixed vector to each word, Long Short-Term Memory (LSTM) networks process text sequentially, allowing earlier words to influence later predictions. This notebook progressively explores three architectures:

1. Feed Forward Network using pooled embeddings (baseline)
2. LSTM with masked pooling
3. Bidirectional LSTM initialized with pretrained Word2Vec embeddings

Each model was introduced to overcome limitations observed in the previous one.

---

## Objectives

This stage investigates:

* Sequence modelling with recurrent neural networks
* Efficient handling of variable-length review sequences
* Mean and Max Pooling over LSTM outputs
* Padding and masking strategies
* The impact of pretrained Word2Vec embeddings
* Whether bidirectional context improves sentiment classification

---

## Dataset

The same Amazon Sports & Outdoors Reviews dataset was used throughout the experiments.

### Preprocessing Pipeline

```text
Raw Reviews
      │
      ▼
Cleaning
      │
      ▼
Tokenization
      │
      ▼
Word → Integer Mapping
      │
      ▼
Padding (Length = 200)
      │
      ▼
Length Tensor
      │
      ▼
DataLoader
      │
      ▼
Neural Network
```

### Dataset Statistics

| Property                |   Value |
| ----------------------- | ------: |
| Total Reviews           | 296,337 |
| Maximum Sequence Length |     200 |
| Batch Size              |      60 |

Each token was converted into its corresponding vocabulary index before padding every review to a fixed length of **200 tokens**.

Unknown words were mapped to a dedicated **UNK** token to maintain robustness during inference.

---

# Model Evolution

## 1. Feed Forward Baseline

Before introducing recurrent networks, a simple baseline was created.

Each review was represented by combining

* Mean Pooling
* Max Pooling

over the learned word embeddings.

```text
Word Embeddings
      │
      ├── Mean Pooling
      │
      └── Max Pooling
              │
              ▼
        Concatenation
              │
              ▼
          Dropout
              │
              ▼
     Fully Connected Layer
              │
              ▼
      Rating Prediction
```

This baseline ignores word order entirely but provides a useful reference for comparing sequence models.

---

## 2. LSTM

The next experiment introduced a unidirectional LSTM.

Instead of immediately collapsing the sentence into a single vector, every token was processed sequentially, allowing information to flow through hidden states.

To improve computational efficiency, variable-length reviews were packed before entering the LSTM using `pack_padded_sequence()` and restored afterwards using `pad_packed_sequence()`.

---

## Padding & Masking

One of the most challenging implementation aspects of this project was correctly handling padded sequences.

Padding is necessary because reviews have different lengths, but padded tokens contain no semantic information. If treated like normal words, they distort sentence representations.

To prevent this:

* Packed sequences avoided unnecessary computation.
* Mean pooling ignored padded tokens using sequence-length masks.
* Max pooling replaced padded positions with **−∞**, ensuring they could never become the maximum value.
* Custom masks were created for every mini-batch since sequence lengths differed after shuffling.

This masking strategy became an essential component of every subsequent recurrent model.

---

## Pooling Strategy

Rather than using only the final hidden state, two complementary sentence representations were extracted.

```text
LSTM Output
      │
      ├── Masked Mean Pooling
      │
      └── Masked Max Pooling
              │
              ▼
        Concatenation
              │
              ▼
          Dropout
              │
              ▼
      Fully Connected Layer
```

Combining both pooling operations produced richer sentence representations by capturing both global context and highly activated features.

---

# 3. Bidirectional LSTM

Although the standard LSTM captures information from past tokens, it cannot utilize future context.

The Bidirectional LSTM addresses this limitation by processing every review in both forward and backward directions simultaneously.

This enables each word representation to incorporate information from both preceding and succeeding words.

In addition, the embedding layer was initialized using the pretrained **Word2Vec embeddings** generated in the previous notebook.

After experimentation, freezing these pretrained embeddings produced slightly better validation performance than allowing them to update during training.

---

## Architecture

```text
Pretrained Word2Vec
        │
        ▼
Embedding Layer (Frozen)
        │
        ▼
Bidirectional LSTM
        │
        ├── Masked Mean Pooling
        │
        └── Masked Max Pooling
                │
                ▼
         Concatenation
                │
                ▼
            Dropout
                │
                ▼
       Fully Connected Layer
                │
                ▼
        Rating Prediction
```

---

## Embedding Verification

Before training downstream models, cosine similarity was computed between selected word pairs to verify the quality of the learned embeddings.

| Word Pair           | Cosine Similarity |
| ------------------- | ----------------: |
| Great ↔ Excellent   |         **0.518** |
| Terrible ↔ Horrible |         **0.353** |
| Great ↔ Terrible    |        **−0.357** |

The positive similarity between synonymous words and negative similarity between opposite sentiments indicated that the learned embedding space captured meaningful semantic relationships.

---

# Training Summary

### Feed Forward Baseline

| Metric              | Final Value |
| ------------------- | ----------: |
| Training Accuracy   |      ~69.1% |
| Validation Accuracy |      ~66.7% |

---

### LSTM

| Metric              | Final Value |
| ------------------- | ----------: |
| Training Accuracy   |      ~78.6% |
| Validation Accuracy |      ~65.3% |

The LSTM achieved substantially higher training accuracy but showed limited improvement on the validation set, suggesting the onset of overfitting.

---

### BiLSTM + Pretrained Word2Vec

| Metric              | Final Value |
| ------------------- | ----------: |
| Training Accuracy   |      ~70.5% |
| Validation Accuracy |  **~68.5%** |

Although the BiLSTM achieved lower training accuracy than the standard LSTM, it generalized better and produced the highest validation accuracy among the recurrent architectures explored in this notebook.

---

# Key Observations

* Pretrained Word2Vec embeddings accelerated learning and improved generalization.
* Freezing pretrained embeddings performed slightly better than allowing them to update.
* Correct masking proved essential for reliable pooling over padded sequences.
* Combining mean and max pooling produced richer sentence representations than relying solely on the final hidden state.
* Bidirectional context consistently improved validation performance compared to the standard LSTM.
* Higher training accuracy did not necessarily translate into better generalization.

---

# Limitations

Despite improved contextual modelling, several limitations remained.

* The models still compressed entire reviews into a single pooled representation.
* Every word contributed equally during pooling, regardless of its importance.
* Long reviews often contained only a few sentiment-bearing words, yet all tokens influenced the final representation.
* Performance plateaued despite increasing model complexity.

These observations motivated the next stage of the project: introducing an **attention mechanism** capable of learning which words should contribute most strongly to the final sentence representation.

---

## Next Stage

The next notebook extends the Bidirectional LSTM by incorporating a **Linear Attention** mechanism.

Instead of treating every token equally during pooling, the model learns to assign different importance weights to different words, allowing the network to focus on the most informative parts of each review.

