# Decision Tree with Prepruning

This project implements a Decision Tree algorithm with prepruning techniques, offering a powerful and efficient approach to classification and regression tasks while preventing overfitting.

## Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Prepruning Parameters](#prepruning-parameters)

## Overview

The Decision Tree with Prepruning is an enhanced version of the traditional decision tree algorithm. It incorporates prepruning techniques to stop the tree's growth before it becomes too complex, thus reducing overfitting and improving generalization.

## Features

- Implementation of Decision Tree for both classification and regression tasks
- Built-in prepruning techniques to prevent overfitting
- Support for various splitting criteria (Gini, Entropy, MSE)
- Handles both numerical and categorical features
- Customizable prepruning parameters
- Cross-validation and hyperparameter tuning utilities
- Tree visualization and feature importance analysis

## Installation

To install the Decision Tree with Prepruning, use pip:

```bash
pip install decision-tree-prepruning
```

Or clone this repository and install the required dependencies:



## Usage

Here's a basic example of how to use the Decision Tree with Prepruning:

```python
from decision_tree_prepruning import DecisionTreePrepruning
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

# Generate sample data
X, y = make_classification(n_samples=1000, n_features=20, n_classes=2, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the model
model = DecisionTreePrepruning(
    max_depth=10, 
    min_samples_split=5, 
    min_samples_leaf=2,
    max_features=0.8,
    min_impurity_decrease=0.01
)
model.fit(X_train, y_train)

# Make predictions
predictions = model.predict(X_test)

# Evaluate the model
accuracy = model.score(X_test, y_test)
print(f"Accuracy: {accuracy:.4f}")
```

## Prepruning Parameters

The Decision Tree with Prepruning offers several parameters to control the tree's growth:

- `max_depth`: The maximum depth of the tree.
- `min_samples_split`: The minimum number of samples required to split an internal node.
- `min_samples_leaf`: The minimum number of samples required to be at a leaf node.
- `max_features`: The number of features to consider when looking for the best split.
- `min_impurity_decrease`: A node will be split if this split induces a decrease of the impurity greater than or equal to this value.



