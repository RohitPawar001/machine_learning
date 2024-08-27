# AUC Metrics

This repository provides tools and examples for calculating and visualizing AUC metrics for classification models. It includes functions for computing the AUC and other related metrics.

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
from auc_metrics import calculate_auc

# Example data
y_true = [0, 1, 0, 1, 0, 1]
y_scores = [0.1, 0.4, 0.35, 0.8, 0.2, 0.9]

auc = calculate_auc(y_true, y_scores)
print(f"AUC: {auc:.2f}")
```

## Description

An AUC (Area Under the Curve) or Area Under the ROC Curve, thus the term is short for roc_auc.

AUC is a metric used to summarize a graph by using a single number. It is used for binary classification problems.

The AUC (Area Under the Curve) can be calculated as:

AUC = Area under the ROC curve

The AUC provides insight into the model’s ability to distinguish between classes. A higher AUC indicates a better performing model, making it particularly useful for evaluating binary classification models.