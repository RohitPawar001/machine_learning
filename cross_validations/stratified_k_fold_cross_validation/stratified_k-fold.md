# Stratified K-Fold Cross-Validation

## Overview

Stratified k-fold cross-validation is a variation of k-fold cross-validation that ensures each fold is representative of the whole dataset in terms of class distribution. This method is particularly useful for dealing with imbalanced datasets or when the target variable is categorical.

## How It Works

1. The dataset is divided into k subsets (or folds).
2. The class distribution in each fold is made to match the overall distribution of the dataset.
3. The model is trained and tested k times, where each time:
   - It is trained on k-1 folds.
   - It is tested on the remaining fold.
4. The final performance metric is the average of the k test results.

## Visual Representation

<img src="https://github.com/user-attachments/assets/4b3123c4-e4de-4cfb-a4df-80ff6236b12c">


## Advantages

- Reduces bias in the evaluation process
- Provides a more reliable estimate of model performance, especially for imbalanced datasets
- Helps in maintaining consistent class distribution across all folds
- Ensures that each class is properly represented in both training and validation sets

## When to Use Stratified K-Fold Cross-Validation

- When dealing with imbalanced datasets
- When the target variable is categorical
- When you want to ensure that the proportion of samples for each class is roughly the same in each fold
- In small datasets where class imbalance can significantly affect model performance

## Comparison with Regular K-Fold Cross-Validation

| Aspect | Regular K-Fold | Stratified K-Fold |
|--------|----------------|-------------------|
| Fold Creation | Randomly splits data | Maintains class distribution |
| Bias | Can be biased for imbalanced data | Reduces bias |
| Variance | May have high variance in class distribution | Lower variance in class distribution |
| Use Case | General purpose | Imbalanced or categorical data |

## Code Example

Here's a Python example using scikit-learn to perform stratified k-fold cross-validation:

```python
from sklearn.model_selection import StratifiedKFold
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import numpy as np

# Load the iris dataset
X, y = load_iris(return_X_y=True)

# Initialize the model
model = LogisticRegression(random_state=42)

# Initialize StratifiedKFold
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Perform stratified k-fold cross-validation
scores = []
for fold, (train_index, val_index) in enumerate(skf.split(X, y), 1):
    X_train, X_val = X[train_index], X[val_index]
    y_train, y_val = y[train_index], y[val_index]
    
    model.fit(X_train, y_train)
    predictions = model.predict(X_val)
    score = accuracy_score(y_val, predictions)
    scores.append(score)
    
    print(f"Fold {fold} - Accuracy: {score:.4f}")

print(f"\nAverage Accuracy: {np.mean(scores):.4f}")
```

## Best Practices

1. Always use stratified k-fold when dealing with classification problems, especially with imbalanced datasets.
2. Choose an appropriate number of folds (k) based on your dataset size. Common choices are 5 or 10.
3. Use shuffling to ensure randomness in fold creation.
4. Set a random state for reproducibility.

## Limitations

- May not be suitable for very small datasets where stratification is difficult.
- Can be computationally expensive for large datasets.
- Not applicable to regression problems (use regular k-fold instead).

## Further Reading

- [Scikit-learn Documentation on Cross-validation](https://scikit-learn.org/stable/modules/cross_validation.html)
- [An Introduction to Statistical Learning](http://www-bcf.usc.edu/~gareth/ISL/)
- [Applied Predictive Modeling](http://appliedpredictivemodeling.com/)

