# Decision Tree Classifier

## Overview
This project implements a Decision Tree Classifier, a popular machine learning algorithm used for both classification and regression tasks. The decision tree classifier works by creating a model that predicts the value of a target variable by learning simple decision rules inferred from the data features.

## Features

Implements a basic decision tree algorithm
Provides options for tree pruning to prevent overfitting
Includes visualization tools for the decision tree structure

## Requirements

Python 3.7+
NumPy
Pandas
Scikit-learn
Matplotlib (for visualization)


## Usage
Here's a basic example of how to use the Decision Tree Classifier:
```python
from decision_tree_classifier import DecisionTreeClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split


# Load the iris dataset

iris = load_iris()
X, y = iris.data, iris.target

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the classifier
clf = DecisionTreeClassifier(max_depth=3)
clf.fit(X_train, y_train)

# Make predictions
predictions = clf.predict(X_test)

# Evaluate the model
accuracy = clf.score(X_test, y_test)
print(f"Accuracy: {accuracy:.2f}")

# Visualize the tree
clf.visualize_tree()
```