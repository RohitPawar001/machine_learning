# K-Fold Cross-Validation

## Overview

K-fold cross-validation is a statistical method used to estimate the skill of machine learning models. It is commonly used in applied machine learning to compare and select a model for a given predictive modeling problem.

The procedure has a single parameter called `k` that refers to the number of groups that a given data sample is to be split into. As such, the procedure is often called k-fold cross-validation.

## How It Works

1. The dataset is divided into k subsets (or folds) of equal size.
2. The model is trained and tested k times, where each time, it is trained on k-1 folds and tested on the remaining fold.
3. The final performance metric is the average of the k test results.

## Visual Representation

![K-Fold Cross-Validation](k-fold-cross-validation.svg)

## Advantages

- Provides a more accurate estimate of model performance
- Makes efficient use of limited data
- Helps to avoid overfitting
- Reduces bias in model evaluation

## Disadvantages

- Can be computationally expensive for large datasets
- May produce high variance estimates for small datasets

## When to Use K-Fold Cross-Validation

- When you have a limited dataset
- When you want to estimate how well your model generalizes to unseen data
- When you need to tune hyperparameters
- When comparing different machine learning algorithms

## Code Example

Here's a simple Python example using scikit-learn:

```python
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_iris

# Load the iris dataset
X, y = load_iris(return_X_y=True)

# Create a logistic regression model
model = LogisticRegression()

# Perform 5-fold cross-validation
scores = cross_val_score(model, X, y, cv=5)

print(f"Cross-validation scores: {scores}")
print(f"Average score: {scores.mean():.2f}")
```

## Further Reading

- [Scikit-learn Documentation on Cross-validation](https://scikit-learn.org/stable/modules/cross_validation.html)
- [An Introduction to Statistical Learning](http://www-bcf.usc.edu/~gareth/ISL/)
- [Machine Learning Mastery: k-fold Cross-Validation](https://machinelearningmastery.com/k-fold-cross-validation/)

