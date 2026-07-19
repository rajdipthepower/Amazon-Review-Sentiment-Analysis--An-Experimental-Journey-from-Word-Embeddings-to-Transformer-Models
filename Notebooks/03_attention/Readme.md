# 03 — BiLSTM with Linear Attention

## Overview

The previous notebook demonstrated that Bidirectional LSTMs improved contextual understanding compared to standard LSTMs. However, every word in a review still contributed almost equally when constructing the final sentence representation through pooling.

This experiment introduces a **Linear Attention** mechanism, allowing the model to learn which words deserve greater importance during sentiment prediction instead of treating every token equally.

More importantly, this notebook became a turning point in the project by highlighting an important lesson:

> **High accuracy alone does not necessarily indicate a good classifier.**

---

## Objectives

This experiment investigates:

* Adding a learnable attention mechanism on top of a BiLSTM.
* Learning context-dependent importance weights for each token.
* Improving sentence representations beyond mean/max pooling.
* Handling severe class imbalance during training.
* Understanding why evaluation metrics must go beyond accuracy.

---

## Model Architecture

The architecture builds directly upon the BiLSTM developed in the previous notebook.

```text
Pretrained Word2Vec
        │
        ▼
Embedding Layer (Frozen)
        │
        ▼
Bidirectional LSTM
        │
        ▼
Linear Attention
        │
        ▼
Weighted Sentence Representation
        │
        ▼
Dropout
        │
        ▼
Fully Connected Layer
        │
        ▼
Sentiment Prediction
```

Unlike pooling, attention allows the network to assign different importance to different words within the same review.

---

## Attention Mechanism

For each hidden state produced by the BiLSTM,

1. A linear transformation is applied.
2. A **tanh** activation introduces non-linearity.
3. A second linear layer computes an attention score.
4. Padding positions are masked with **−∞**.
5. Softmax converts scores into attention probabilities.
6. A weighted sum of hidden states forms the final sentence representation.

Because padding positions are masked before the softmax operation, they receive zero probability and never contribute to the final representation.

---

## Sentiment Reformulation

The original five-star rating prediction problem was reformulated into **three sentiment classes**.

| Original Rating | Sentiment Class |
| --------------- | --------------: |
| ★–★★            |         Class 1 |
| ★★★             |         Class 2 |
| ★★★★–★★★★★      |         Class 3 |

This simplification was introduced to reduce ambiguity between neighbouring ratings and transform the task into sentiment classification rather than exact rating prediction.

Even after label reformulation, the dataset remained highly imbalanced.

| Sentiment | Samples |
| --------- | ------: |
| Class 1   |  19,249 |
| Class 2   |  24,071 |
| Class 3   | 253,017 |

---

## Handling Class Imbalance

Instead of naively oversampling minority classes, this notebook employed a **WeightedRandomSampler** during training.

Each training sample received a probability inversely proportional to its class frequency, allowing minority classes to appear more often without permanently modifying the dataset.

Only the **training set** used weighted sampling.

Validation and test sets retained their natural distribution to provide a realistic estimate of model performance.

---

## Training Summary

| Metric              | Final Value |
| ------------------- | ----------: |
| Training Accuracy   |  **91.47%** |
| Validation Accuracy |  **89.01%** |

At first glance, these numbers appear to represent a dramatic improvement over previous models.

However, further analysis revealed that the story was considerably more complicated.

---

## Experimental Observation

Although the model achieved nearly **90% validation accuracy**, this result did **not** necessarily indicate superior sentiment understanding.

The dominant positive class represented the overwhelming majority of the dataset. Consequently, predicting this class correctly contributed disproportionately to the overall accuracy.

Despite weighted sampling, repeated exposure to minority samples (sampling with replacement) could not fully eliminate this imbalance.

This experiment became the first indication that **accuracy alone was an unreliable performance metric** for this dataset.

---

## Key Lessons Learned

Several important observations emerged from this experiment:

* Attention produced richer sentence representations than simple pooling.
* Correct masking remained essential to prevent padded tokens from receiving attention.
* Weighted sampling improved minority class exposure during training but was not a complete solution to class imbalance.
* Oversampling with replacement can repeatedly present identical minority examples, increasing the risk of overfitting.
* A model may achieve very high accuracy while still performing poorly on underrepresented classes.

Perhaps the most important takeaway was realizing that evaluation should not stop at a single accuracy value.

---

## Limitations

Despite improved contextual representations, several challenges remained.

* Severe class imbalance continued to bias predictions toward the dominant class.
* Accuracy alone failed to capture minority-class behaviour.
* Weighted sampling alleviated but did not solve the imbalance problem.
* The attention mechanism operated on BiLSTM representations, which still relied on static Word2Vec embeddings.

These limitations motivated the final stage of the project: **Transformer-based transfer learning using DistilBERT**, combined with weighted loss functions, gradual layer unfreezing, discriminative learning rates, and evaluation based on Macro F1-score rather than accuracy alone.

---

## Next Stage

The next notebook introduces **DistilBERT**, marking the transition from static word embeddings to contextual Transformer representations.

Rather than relying solely on architectural improvements, the focus shifts toward:

* transfer learning,
* optimization strategies,
* handling class imbalance,
* and selecting evaluation metrics that more faithfully reflect real model performance.

