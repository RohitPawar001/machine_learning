# Pre-Pruning Decision Tree Classifier

## Overview
This project implements a Decision Tree Classifier with pre-pruning capabilities. The decision tree classifier is a popular machine learning algorithm used for both classification and regression tasks. It works by creating a model that predicts the value of a target variable by learning simple decision rules inferred from the data features. Pre-pruning is applied during the tree growth process to prevent overfitting and improve generalization.

## Features
* Implements a basic decision tree algorithm
* Provides pre-pruning options to prevent overfitting
* Includes visualization tools for the resulting pruned decision tree structure

## Requirements
* Python 3.7+
* NumPy
* Pandas
* Scikit-learn
* Matplotlib (for visualization)



## Usage
Here's a basic example of how to use the Pre-Pruning Decision Tree Classifier:

```python
from pre_pruning_decision_tree import PrePruningDecisionTreeClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

# Load the iris dataset
iris = load_iris()
X, y = iris.data, iris.target

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the classifier with pre-pruning parameters
clf = PrePruningDecisionTreeClassifier(
    max_depth=5,
    min_samples_split=5,
    min_samples_leaf=2,
    max_features=3,
    min_impurity_decrease=0.01
)
clf.fit(X_train, y_train)

# Evaluate the model
accuracy = clf.score(X_test, y_test)
print(f"Accuracy: {accuracy:.2f}")

# Visualize the pre-pruned tree
clf.visualize_tree()

# Make predictions
predictions = clf.predict(X_test)
```

## Pre-Pruning Process
The pre-pruning process in this implementation applies constraints during the tree growth to prevent overfitting:

1. Limit the maximum depth of the tree (`max_depth`).
2. Require a minimum number of samples to split an internal node (`min_samples_split`).
3. Require a minimum number of samples in leaf nodes (`min_samples_leaf`).
4. Limit the number of features considered for splitting at each node (`max_features`).
5. Require a minimum decrease in impurity for a split to be considered (`min_impurity_decrease`).

These constraints are applied at each step of the tree growth, effectively "pruning" branches before they are fully grown.

## Configuration
You can configure the pre-pruning decision tree classifier with the following parameters:
* `max_depth`: Maximum depth of the tree (default: None)
* `min_samples_split`: Minimum number of samples required to split an internal node (default: 2)
* `min_samples_leaf`: Minimum number of samples required to be at a leaf node (default: 1)
* `max_features`: Number of features to consider when looking for the best split (default: None)
* `min_impurity_decrease`: Minimum decrease in impurity required for a split (default: 0.0)
* `criterion`: The function to measure the quality of a split. Supported criteria are "gini" for the Gini impurity and "entropy" for the information gain. (default: "gini")

## Advantages of Pre-Pruning
1. Faster training time compared to post-pruning methods.
2. Prevents the model from learning very specific rules that might only apply to the training data.
3. Reduces the complexity of the final model, making it more interpretable.
4. Can be easily integrated into the tree-building process without requiring a separate validation set.

