# Decision Tree with Postpruning

This project implements a Decision Tree algorithm with postpruning techniques, offering an effective approach to classification and regression tasks while mitigating overfitting.

## Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Postpruning Methods](#postpruning-methods)


## Overview

The Decision Tree with Postpruning is an advanced implementation of the decision tree algorithm. It first grows a full tree and then applies postpruning techniques to simplify the tree structure, reducing complexity and improving generalization performance.

## Features

- Implementation of Decision Tree for both classification and regression tasks
- Multiple postpruning techniques: Reduced Error Pruning, Cost Complexity Pruning
- Support for various splitting criteria (Gini, Entropy, MSE)
- Handles both numerical and categorical features
- Customizable postpruning parameters
- Cross-validation and hyperparameter tuning utilities
- Tree visualization before and after pruning
- Feature importance analysis

## Installation

To install the Decision Tree with Postpruning, use pip:

```bash
pip install decision-tree-postpruning
```

Or clone this repository and install the required dependencies:



## Usage

Here's a basic example of how to use the Decision Tree with Postpruning:

```python
from decision_tree_postpruning import DecisionTreePostpruning
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

# Generate sample data
X, y = make_classification(n_samples=1000, n_features=20, n_classes=2, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the model
model = DecisionTreePostpruning(
    pruning_method='cost_complexity',
    ccp_alpha=0.01
)
model.fit(X_train, y_train)

# Make predictions
predictions = model.predict(X_test)

# Evaluate the model
accuracy = model.score(X_test, y_test)
print(f"Accuracy: {accuracy:.4f}")

# Get the number of nodes before and after pruning
print(f"Number of nodes before pruning: {model.node_count_before_pruning}")
print(f"Number of nodes after pruning: {model.node_count_after_pruning}")
```

## Postpruning Methods

The Decision Tree with Postpruning offers two main pruning methods:

1. **Reduced Error Pruning (REP)**
   - Iteratively removes subtrees that don't improve validation accuracy
   - Simple and fast, but may overprune in some cases
   - Usage: `pruning_method='reduced_error'`

2. **Cost Complexity Pruning (CCP)**
   - Builds a series of trees with increasing pruning strength
   - Balances accuracy against tree complexity
   - Usage: `pruning_method='cost_complexity', ccp_alpha=0.01`

Parameters:
- `pruning_method`: The method of postpruning to use ('reduced_error' or 'cost_complexity')
- `ccp_alpha`: The complexity parameter for cost complexity pruning
- `validation_fraction`: The fraction of training data to use as a validation set for reduced error pruning

