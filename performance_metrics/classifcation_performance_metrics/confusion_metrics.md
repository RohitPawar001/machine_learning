# Confusion Metrics

This repository provides tools and examples for calculating and visualizing confusion metrics for classification models. It includes functions for computing accuracy, precision, recall, F1 score, and more.


## Table of Contents

- Installation
- Usage
- Explaination


## Installation

To install the necessary dependencies, run:

```bash
pip install -r requirements.txt
```

## Usage

```python
from confusion_metrics import calculate_confusion_matrix

# Example data
y_true = [0, 1, 0, 1, 0, 1]
y_pred = [0, 0, 0, 1, 0, 1]

conf_matrix = calculate_confusion_matrix(y_true, y_pred)
print(conf_matrix)
```
## Applications

A confusion matrix or error matrix is a table that shows the number of correct and incorrect predictions made by the model compared with the actual classifications in the test set or what type of errors are being made.

This matrix describes the performance of a classification model on test data for which true values are known. It is a n*n matrix, where n is the number of classes. This matrix can be generated after making predictions on the test data.

<img src ="https://miro.medium.com/v2/resize:fit:828/format:webp/0*l30v6Id3wZrw8FAO">

Here, columns represent the count of actual classifications in the test data while rows represent the count of predicted classifications made by the model.

Let’s take an example of a classification problem where we are predicting whether a person is having diabetes or not. Let’s give a label to our target variable:

1: A person is having diabetes | 0: A person is not having diabetes

Four possible outcomes could occur while performing classification predictions:

**True Positives (TP):** Number of outcomes that are actually positive and are predicted positive.
For example: In this case, a person is actually having diabetes(1) and the model predicted that the person has diabetes(1).

**True Negatives (TN):** Number of outcomes that are actually negative and are predicted negative.
For example: In this case, a person actually doesn’t have diabetes(0) and the model predicted that the person doesn’t have diabetes(0).

**False Positives (FP):** Number of outcomes that are actually negative but predicted positive. These errors are also called Type 1 Errors.
For example: In this case, a person actually doesn’t have diabetes(0) but the model predicted that the person has diabetes(1).

**False Negatives (FN):** Number of outcomes that are actually positive but predicted negative. These errors are also called Type 2 Errors.
For example: In this case, a person actually has diabetes(1) but the model predicted that the person doesn’t have diabetes(0).

Positive and Negatives refers to the prediction itself. True and False refers to the correctness of the prediction.




