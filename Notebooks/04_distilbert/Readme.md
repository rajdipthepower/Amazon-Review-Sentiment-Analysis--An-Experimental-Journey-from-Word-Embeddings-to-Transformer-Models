# 04 — Transfer Learning with DistilBERT

## Overview

The previous experiments relied on **static Word2Vec embeddings** combined with recurrent neural networks. While these models successfully captured sequential information, they still represented each word using a fixed embedding regardless of context.

This notebook marks the transition to **Transformer-based transfer learning** using **DistilBERT**, where word representations become context-dependent and are learned from large-scale pretraining.

More importantly, this stage focuses not only on building a stronger model, but also on **how to fine-tune pretrained Transformers effectively** on a highly imbalanced real-world dataset.

---

## Objectives

This experiment investigates:

* Fine-tuning DistilBERT for sentiment classification.
* Leveraging transfer learning instead of training from scratch.
* Handling severe class imbalance.
* Comparing balanced sampling with weighted loss functions.
* Applying gradual layer unfreezing.
* Using discriminative learning rates.
* Evaluating models using **Macro F1-score** instead of relying solely on accuracy.

---

## Dataset

The same Amazon Sports & Outdoors Reviews dataset was used throughout the project.

### Text Preprocessing

The preprocessing pipeline remained intentionally simple.

* Remove HTML tags
* Remove URLs
* Normalize whitespace
* Preserve natural sentence structure

Unlike earlier models, no manual vocabulary construction was required since DistilBERT uses its own pretrained tokenizer.

---

## Label Reformulation

Rather than predicting the original five-star ratings, the problem was reformulated into a three-class sentiment task.

| Original Rating | Sentiment |
| --------------- | --------: |
| ★, ★★, ★★★      |         0 |
| ★★★★            |         1 |
| ★★★★★           |         2 |

Final class distribution:

| Class | Samples |
| ----: | ------: |
|     0 |  19,249 |
|     1 |  24,071 |
|     2 | 253,017 |

The dataset therefore remained heavily imbalanced, making overall accuracy an unreliable evaluation metric.

---

# Model

The project uses **DistilBERT for Sequence Classification** with three output classes.

| Property                |      Value |
| ----------------------- | ---------: |
| Backbone                | DistilBERT |
| Trainable Parameters    | 66,955,779 |
| Maximum Sequence Length |        512 |
| Mixed Precision         |       FP16 |

Dynamic padding was handled automatically through `DataCollatorWithPadding`, reducing unnecessary computation by padding only to the longest sequence within each mini-batch.

---

# Experiment I — Standard Fine-Tuning

The initial experiment followed the conventional Hugging Face fine-tuning pipeline.

Key features included:

* balanced training dataset
* AdamW optimizer
* FP16 mixed precision
* gradient accumulation
* dynamic padding
* grouped batches of similar sequence lengths
* evaluation every 500 steps

The best model was selected using **Macro F1-score** rather than validation loss.

### Training Progress

| Step | Accuracy | Macro F1 |
| ---: | -------: | -------: |
|  500 |    0.727 |    0.679 |
| 1000 |    0.750 |    0.702 |
| 2000 |    0.764 |    0.711 |
| 3000 |    0.747 |    0.713 |
| 5000 |    0.751 |    0.714 |
| 5500 |    0.737 |    0.708 |

Although the model converged successfully, there remained room for improving minority-class performance.

---

# Experiment II — Advanced Fine-Tuning

The second experiment introduced several optimization strategies inspired by modern transfer learning practices.

Rather than treating every parameter equally, different parts of the network were optimized differently.

---

## Gradual Layer Unfreezing

Instead of updating all Transformer layers from the beginning, the network was progressively unfrozen.

| Training Stage | Trainable Layers              | Purpose                                        |
| -------------- | ----------------------------- | ---------------------------------------------- |
| 0–500          | Classification Head           | Adapt the prediction layer to the new task     |
| 501–1500       | Head + Transformer Blocks 4–5 | Learn high-level task-specific representations |
| 1501+          | Entire Model                  | Fine-tune all Transformer layers               |

This strategy helps preserve useful pretrained representations while gradually adapting the model to the downstream task.

---

## Learning Rate Warmup

One important observation during experimentation was that newly unfrozen layers initially have **no optimizer momentum**.

Updating them immediately with the full learning rate can produce large parameter changes that damage pretrained knowledge.

To avoid this, training employed:

* Linear warmup for the first 500 steps.
* Linear learning rate decay afterwards.

This allowed newly unfrozen layers to adapt more smoothly.

---

## Discriminative Learning Rates

Different parts of the model learned at different speeds.

| Component           | Learning Rate |
| ------------------- | ------------: |
| Classification Head |      1 × 10⁻⁵ |
| DistilBERT Encoder  |      2 × 10⁻⁶ |

The classification head begins with randomly initialized parameters and therefore requires larger updates.

The encoder already contains rich linguistic knowledge from pretraining and only requires small adjustments to avoid catastrophic forgetting.

---

## Weight Decay Strategy

Weight decay was applied selectively.

Regular weights received L2 regularization, while

* bias parameters
* LayerNorm weights

were excluded from weight decay since regularizing these parameters often degrades Transformer performance.

---

## Results

### Training Progress

| Step |  Accuracy |  Macro F1 |
| ---: | --------: | --------: |
|  500 |     0.635 |     0.259 |
| 1000 |     0.720 |     0.557 |
| 2000 |     0.757 |     0.637 |
| 3000 |     0.770 |     0.660 |
| 4000 |     0.780 |     0.690 |
| 5500 | **0.782** | **0.691** |

Although the model started more conservatively due to gradual unfreezing, it steadily improved throughout training and ultimately achieved better overall performance.

---

## Final Evaluation

The final model was evaluated using:

* Accuracy
* Precision
* Recall
* Macro F1-score
* Weighted F1-score
* Confusion Matrix

| Metric          |    Score |
| --------------- | -------: |
| Accuracy        |  **78%** |
| Macro Precision | **0.75** |
| Macro Recall    | **0.64** |
| Macro F1-score  | **0.68** |

The confusion matrix and detailed classification report showed that the dominant positive class was classified reliably, while the intermediate sentiment class remained considerably more challenging.

This reinforced the importance of evaluating individual class performance rather than relying solely on aggregate accuracy.

---

# Lessons Learned

This notebook represents the culmination of the project and highlights several practical lessons:

* Transfer learning substantially outperformed training models from scratch.
* Gradual layer unfreezing provided more stable fine-tuning than updating all layers immediately.
* Warmup scheduling reduced abrupt parameter updates after unfreezing.
* Discriminative learning rates helped preserve pretrained knowledge while adapting the classification head.
* Weighted loss functions improved optimization on imbalanced data but did not completely eliminate minority-class challenges.
* Macro F1-score proved to be a far more informative metric than overall accuracy.

Perhaps the most valuable takeaway was that **building a stronger model is only part of the problem—choosing appropriate optimization strategies and evaluation metrics is equally important.**

---

## Conclusion

Across this repository, the progression from **Word2Vec → LSTM → BiLSTM → Attention → DistilBERT** reflects not only an increase in architectural complexity, but also a deeper understanding of practical NLP.

The project ultimately demonstrated that successful sentiment classification depends on far more than selecting a powerful model. Careful preprocessing, sequence handling, optimization strategies, class imbalance mitigation, and rigorous evaluation all play equally important roles in building reliable NLP systems.

