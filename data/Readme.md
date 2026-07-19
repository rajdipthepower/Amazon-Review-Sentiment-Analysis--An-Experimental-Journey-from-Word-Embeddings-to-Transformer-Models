# Dataset

This directory stores the datasets used throughout the sentiment analysis experiments.

The notebooks in this repository were developed using the **Amazon Sports & Outdoors Reviews** dataset from the **Amazon Review Data (2018)** collection by Julian McAuley and collaborators.

## Dataset Used

* **Dataset:** Sports_and_Outdoors_5.json
* **Reviews:** 296,337
* **Task:** Multi-class sentiment classification

The frequency of each of the five sentiment classes are :

| Original Rating | Sentiment Label |
| --------------- | --------------: |
| 1 ★             |               9045  |
| 2 ★             |               10204 |
| 3 ★             |               24071 |
| 4 ★             |               64809 |
| 5 ★             |              188208 |

The frequency of each label clearly shows a biased dataset towards high frequency ratings and harder prediction task .

## Download

The dataset is **not included** in this repository because of its size and licensing considerations.

Download the dataset from the official Amazon Review Data repository:

**Amazon Review Data (2018)**
https://www.kaggle.com/datasets/aarishasifkhan/sports-and-outdoor-review-dataset

After downloading, place the dataset in this directory:

```text
data/
└── Sports_and_Outdoors_5.json
```

No additional preprocessing files are required—the notebooks perform all preprocessing, cleaning, tokenization, and label mapping automatically.

