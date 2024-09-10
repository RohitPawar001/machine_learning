# Ensemble Classifier Techniques

This repository provides an overview and implementation of various ensemble classifier techniques. Ensemble methods combine multiple machine learning models to produce better predictive performance than could be obtained from any of the constituent models alone.

## Table of Contents

1. [Introduction](#introduction)
2. [Ensemble Techniques](#ensemble-techniques)
   - [Bagging](#bagging)
   - [Boosting](#boosting)
   - [Stacking](#stacking)
3. [Implementation](#implementation)
4. [Usage](#usage)


## Introduction

Ensemble learning is a machine learning paradigm where multiple models (often called "weak learners") are trained to solve the same problem and combined to get better results. The main hypothesis is that when weak models are correctly combined we can obtain more accurate and/or robust models.

## Ensemble Techniques

### Bagging (Bootstrap Aggregating)

Bagging involves having each model in the ensemble vote with equal weight. In order to promote model variance, bagging trains each model in the ensemble using a randomly drawn subset of the training set.

Example: Random Forest

```python
from sklearn.ensemble import RandomForestClassifier

rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
rf_classifier.fit(X_train, y_train)
```

### Boosting

Boosting involves incrementally building an ensemble by training each new model instance to emphasize the training instances that previous models mis-classified.

Examples: AdaBoost, Gradient Boosting

```python
from sklearn.ensemble import AdaBoostClassifier

adaboost_classifier = AdaBoostClassifier(n_estimators=50, random_state=42)
adaboost_classifier.fit(X_train, y_train)
```

### Stacking

Stacking involves training a learning algorithm to combine the predictions of several other learning algorithms. First, all of the other algorithms are trained using the available data, then a combiner algorithm is trained to make a final prediction using all the predictions of the other algorithms as additional inputs.

```python
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict

# Base models
rf = RandomForestClassifier(n_estimators=100, random_state=42)
gb = GradientBoostingClassifier(n_estimators=100, random_state=42)

# Meta model
meta_model = LogisticRegression()

# Generate base model predictions
rf_predictions = cross_val_predict(rf, X_train, y_train, cv=5)
gb_predictions = cross_val_predict(gb, X_train, y_train, cv=5)

# Train meta model
meta_features = np.column_stack((rf_predictions, gb_predictions))
meta_model.fit(meta_features, y_train)
```

