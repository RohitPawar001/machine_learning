# ROC Curve Metrics

This repository provides tools and examples for calculating and visualizing ROC curve metrics for classification models. It includes functions for computing the ROC curve and other related metrics.

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
from roc_metrics import calculate_roc_curve

# Example data
y_true = [0, 1, 0, 1, 0, 1]
y_scores = [0.1, 0.4, 0.35, 0.8, 0.2, 0.9]

fpr, tpr, thresholds = calculate_roc_curve(y_true, y_scores)
print(f"False Positive Rates: {fpr}")
print(f"True Positive Rates: {tpr}")
print(f"Thresholds: {thresholds}")
```

## Description

A ROC curve (Receiver Operating Characteristic curve) is a graph showing the performance of a classification model. It is a way to visualize the tradeoff between the True Positive Rate (TPR) and False Positive Rate(FPR) using different decision thresholds (the threshold for deciding whether a prediction is labeled “true” or “false”) for our predictive model.

This threshold is used to control the tradeoff between TPR and FPR. Increasing the threshold will generally increase the precision, but a decrease in recall.

**First, let’s see TPR and FPR:-**

**True Positive Rate (TPR / Sensitivity / Recall):** True Positive Rate corresponds to the proportion of positive data points that are correctly considered as positive, for all positive data points.

**False Positive Rate (FPR):** False Positive Rate corresponds to the proportion of negative data points that are mistakenly considered as positive, for all negative data points.

They both have values in the range of [0,1] which are computed at varying threshold values.

The perfect classifier will have high value of true positive rate and low value of false positive rate.

Below is the ROC curve represents a more precise model:

<img src="https://miro.medium.com/v2/resize:fit:640/format:webp/0*CR1L_wNQbbzEU1Ej.png">

Any model with a ROC curve above the random guessing classifier line can be considered as a better model.
Any model with a ROC curve below the random guessing classifier line can outrightly be rejected.
This curve plots TPR and FPR at different classification thresholds but this is inefficient because we have to evaluate our model at various thresholds. There’s an efficient, sorting-based algorithm that can provide us this information which is AUC.