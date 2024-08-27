# Recall Metrics

This repository provides tools and examples for calculating and visualizing recall metrics for classification models. It includes functions for computing recall and other related metrics.

## Table of Contents

- Installation
- Usage
- Description

## Installation

To install the necessary dependencies, run:

```bash
pip install -r requirements.txt
```

## Usage
```python
from recall_metrics import calculate_recall

# Example data
y_true = [0, 1, 0, 1, 0, 1]
y_pred = [0, 0, 0, 1, 0, 1]

recall = calculate_recall(y_true, y_pred)
print(f"Recall: {recall:.2f}")
```

## Description

Recall can be calculated as:

Recall = True Positives / (True Positives + False Negatives)

Recall provides insight into the proportion of actual positives that were correctly identified, making it particularly useful in scenarios where the cost of false negatives is high.