# AdaBoost Classifier

## Overview

This project implements an AdaBoost (Adaptive Boosting) Classifier, a powerful ensemble learning method used for classification tasks. AdaBoost combines multiple "weak learners" (typically decision trees) into a single strong classifier, adaptively adjusting the weight of instances to focus on the most challenging cases.

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [How It Works](#how-it-works)
- [Features](#features)

## Installation

To use this AdaBoost Classifier, you need Python 3.7 or later. Clone this repository and install the required dependencies:



## Usage

Here's a quick example of how to use the AdaBoost Classifier:

```python
from adaboost import AdaBoostClassifier

# Create and train the model
ada_classifier = AdaBoostClassifier(n_estimators=50, learning_rate=1.0)
ada_classifier.fit(X_train, y_train)

# Make predictions
predictions = ada_classifier.predict(X_test)

# Evaluate the model
accuracy = ada_classifier.evaluate(X_test, y_test)
print(f"Model accuracy: {accuracy}")
```

For more detailed usage instructions, please refer to the [documentation](docs/usage.md).

## How It Works

AdaBoost Classifier works by iteratively improving the model's performance. The key steps are:

1. **Initialize Weights**: Assign equal weights to all training instances.
2. **Train Weak Learner**: Train a weak classifier (usually a decision stump) on the weighted data.
3. **Calculate Error**: Compute the weighted error of the weak learner.
4. **Update Weights**: Increase weights for misclassified instances and decrease for correctly classified ones.
5. **Combine Weak Learners**: Create a strong classifier by combining weak learners, giving more importance to better-performing ones.

This process is repeated for a specified number of iterations, resulting in a powerful ensemble classifier.

## Features

- Customizable number of estimators (weak learners)
- Adjustable learning rate
- Support for various weak learners (default: decision stumps)
- Feature importance calculation
- Support for both binary and multi-class classification
- Cross-validation support
- Parallel processing for faster training and prediction



For more information on AdaBoost and its applications, check out the original paper by [Freund and Schapire (1997)](https://www.sciencedirect.com/science/article/pii/S002200009791504X) or the [scikit-learn documentation](https://scikit-learn.org/stable/modules/ensemble.html#adaboost) on AdaBoost.