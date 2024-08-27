# Machine Learning Performance Metrics

This README provides an overview of common performance metrics used in machine learning to evaluate model performance.

## Table of Contents

1. [Regression Metrics](#regression-metrics)
2. [Classification Metrics](#classification-metrics)
3. [Other Metrics](#other-metrics)

## Regression Metrics

### Mean Squared Error (MSE)
- Measures the average squared difference between predicted and actual values
- Sensitive to outliers
- Always non-negative; lower values indicate better performance

### Root Mean Squared Error (RMSE)
- Square root of MSE
- Expressed in the same units as the target variable
- Provides a scale-dependent measure of model accuracy

### Mean Absolute Error (MAE)
- Measures the average absolute difference between predicted and actual values
- Less sensitive to outliers compared to MSE
- Easier to interpret but may be less suitable for optimization

### R-squared (R²)
- Represents the proportion of variance in the dependent variable explained by the model
- Ranges from 0 to 1, with 1 indicating perfect fit
- Can be misleading for non-linear relationships or in the presence of outliers

### Adjusted R-squared
- Modified version of R² that accounts for the number of predictors in the model
- Penalizes the addition of unnecessary variables
- Useful for comparing models with different numbers of features

## Classification Metrics

### Accuracy
- Proportion of correct predictions (both true positives and true negatives) among the total number of cases examined
- Simple to understand but can be misleading for imbalanced datasets

### Precision
- Proportion of true positive predictions among all positive predictions
- Focuses on the accuracy of positive predictions

### Recall (Sensitivity)
- Proportion of true positive predictions among all actual positive cases
- Measures the model's ability to find all positive instances

### F1 Score
- Harmonic mean of precision and recall
- Provides a single score that balances both precision and recall

### Area Under the ROC Curve (AUC-ROC)
- Measures the model's ability to distinguish between classes
- Ranges from 0 to 1, with 0.5 representing random guessing
- Useful for imbalanced datasets and when ranking predictions is important

## Other Metrics

### Confusion Matrix
- Table layout that allows visualization of algorithm performance
- Basis for calculating many classification metrics

### Log Loss
- Measures the performance of a classification model where the prediction is a probability value between 0 and 1
- Heavily penalizes confident misclassifications

### Silhouette Score
- Used for clustering problems
- Measures how similar an object is to its own cluster compared to other clusters
- Ranges from -1 to 1, with higher values indicating better-defined clusters

## Choosing the Right Metric

The choice of metric depends on:
- The type of problem (regression, classification, clustering)
- The specific goals of your model
- The nature of your data (e.g., balanced vs imbalanced classes)
- The requirements of your stakeholders

Always consider multiple metrics for a comprehensive evaluation of your model's performance.