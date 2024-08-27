# F1 Score Metrics

This repository provides tools and examples for calculating and visualizing F1 score metrics for classification models. It includes functions for computing the F1 score and other related metrics.

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
from f1_metrics import calculate_f1_score

# Example data
y_true = [0, 1, 0, 1, 0, 1]
y_pred = [0, 0, 0, 1, 0, 1]

f1_score = calculate_f1_score(y_true, y_pred)
print(f"F1 Score: {f1_score:.2f}")
```

## Description

The F1 score can be calculated as:

F1 Score = 2 * (Precision * Recall) / (Precision + Recall)

The F1 score is the harmonic mean of precision and recall, providing a balance between the two. It is particularly useful in scenarios where both false positives and false negatives are important.