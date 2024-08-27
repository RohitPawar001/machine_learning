# Accuracy Metrics

This repository provides tools and examples for calculating and visualizing accuracy metrics for classification models. It includes functions for computing accuracy and other related metrics.

## Table of Contents

- Installation
- Usage
- Description

## Installation

To install the necessary dependencies, run:

```bash
pip install -r requirements.txt
```

## usage

```python 
from accuracy_metrics import calculate_accuracy

# Example data
y_true = [0, 1, 0, 1, 0, 1]
y_pred = [0, 0, 0, 1, 0, 1]

accuracy = calculate_accuracy(y_true, y_pred)
print(f"Accuracy: {accuracy:.2f}")
```

## Description

It can also be calculated in terms of positives and negatives for binary classification:

  Accuracy = no of correct predictions/ total no of predictions

It doesn’t grant us much information regarding the distribution of false positives and false negatives.
