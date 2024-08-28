# Post-Pruning Decision Tree Classifier

## Overview
This project implements a Decision Tree Classifier with post-pruning capabilities. The decision tree classifier is a popular machine learning algorithm used for both classification and regression tasks. It works by creating a model that predicts the value of a target variable by learning simple decision rules inferred from the data features. Post-pruning is applied after the tree is fully grown to reduce overfitting and improve generalization.

## Features
* Implements a basic decision tree algorithm
* Provides post-pruning options to prevent overfitting
* Includes visualization tools for the decision tree structure before and after pruning

## Requirements
* Python 3.7+
* NumPy
* Pandas
* Scikit-learn
* Matplotlib (for visualization)



## Usage
Here's a basic example of how to use the Post-Pruning Decision Tree Classifier:

```python
from post_pruning_decision_tree import PostPruningDecisionTreeClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

# Load the iris dataset
iris = load_iris()
X, y = iris.data, iris.target

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the classifier
clf = PostPruningDecisionTreeClassifier(max_depth=10)
clf.fit(X_train, y_train)

# Evaluate the model before pruning
accuracy_before = clf.score(X_test, y_test)
print(f"Accuracy before pruning: {accuracy_before:.2f}")

# Visualize the tree before pruning
clf.visualize_tree(filename="tree_before_pruning.png")

# Perform post-pruning
clf.prune(X_test, y_test)

# Evaluate the model after pruning
accuracy_after = clf.score(X_test, y_test)
print(f"Accuracy after pruning: {accuracy_after:.2f}")

# Visualize the tree after pruning
clf.visualize_tree(filename="tree_after_pruning.png")

# Make predictions with the pruned tree
predictions = clf.predict(X_test)
```

## Post-Pruning Process
The post-pruning process in this implementation follows these steps:

1. Grow a full decision tree to maximum depth or until other stopping criteria are met.
2. Evaluate the performance of the tree on a separate validation set.
3. For each non-leaf subtree, evaluate the change in performance if that subtree is replaced with the best possible leaf node.
4. If the performance improves or stays the same, replace the subtree with the leaf node.
5. Repeat steps 3-4 until no further improvements can be made or a stopping criterion is met.

## Configuration
You can configure the post-pruning decision tree classifier with the following parameters:
* `max_depth`: Maximum depth of the tree before pruning (default: None)
* `min_samples_split`: Minimum number of samples required to split an internal node (default: 2)
* `min_samples_leaf`: Minimum number of samples required to be at a leaf node (default: 1)
* `max_features`: Number of features to consider when looking for the best split (default: None)
* `prune_threshold`: The minimum improvement in accuracy required to make a pruning decision (default: 0.0001)

