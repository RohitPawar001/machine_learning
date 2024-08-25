# Linear Regression

Linear regression is a supervised machine learning algorithm used to predict continuous output features, such as house prices or weather conditions. It works with labeled datasets where target values are known.

## Table of Contents
1. [Working Principle](#working-principle)
2. [Types of Linear Regression](#types-of-linear-regression)
3. [Evaluation Metrics](#evaluation-metrics)

## Working Principle

Linear regression algorithms aim to fit the best line to the independent and dependent features.

<img src="https://media.geeksforgeeks.org/wp-content/uploads/20231129130431/11111111.png" width="300" height="300">

### Simple Linear Regression

y = β₀ + β₁X

Where:
- y: dependent feature
- X: independent feature
- β₀: y-axis intercept
- β₁: slope

### Multiple Linear Regression

y = β₀ + β₁X₁ + β₂X₂ + ... + βₙXₙ

Where:
- y: dependent feature
- X₁...Xₙ: independent features
- β₀: y-axis intercept
- β₁...βₙ: slopes

### Goal


The primary objective is to find the best-fit line equation that can predict values based on independent features, minimizing the distance between actual and predicted points (residual error).

### Hypothesis


Assuming a linear relationship between X (experience) and Y (salary):

Ŷᵢ = θ₁ + θ₂xᵢ

Where:
- Yᵢ: (1,2,3...n) dependent features
- Xᵢ: (1,2,3,...n) independent features

### Parameter Updating
To achieve the best-fit regression line, the model updates θ₁ and θ₂ values to minimize the error between predicted and true Y values.

## Types of Linear Regression

### Based on Independent Features
1. **Simple Linear Regression**: One independent and one dependent feature
2. **Multiple Linear Regression**: Multiple independent features and one dependent feature

### Based on Dependent Features
1. **Univariate Linear Regression**: One dependent variable
2. **Multivariate Linear Regression**: Multiple dependent variables

### Cost Function
The cost function (or loss function) measures the error between predicted (Ŷ) and true (Y) values:

Where:
- Yᵢ: (1,2,3...n) dependent features
- Xᵢ: (1,2,3,...n) independent features

### Parameter Updating
To achieve the best-fit regression line, the model updates θ₁ and θ₂ values to minimize the error between predicted and true Y values.

## Types of Linear Regression

### Based on Independent Features
1. **Simple Linear Regression**: One independent and one dependent feature
2. **Multiple Linear Regression**: Multiple independent features and one dependent feature

### Based on Dependent Features
1. **Univariate Linear Regression**: One dependent variable
2. **Multivariate Linear Regression**: Multiple dependent variables

### Cost Function
The cost function (or loss function) measures the error between predicted (Ŷ) and true (Y) values:

Cost function (J) = (1/n) Σᵢ(Ŷᵢ - Yᵢ)²

### Gradient Descent
Gradient descent is an optimization algorithm used to train the model by iteratively modifying parameters to reduce the mean squared error (MSE).

<img src="https://media.geeksforgeeks.org/wp-content/uploads/20230424151248/Gradient-Descent-for-ML-Linear-Regression-(1).webp" width="300" height="300"

## Evaluation Metrics

1. **Mean Squared Error (MSE)**
   - Calculates the average of squared differences between actual and predicted values

2. **Mean Absolute Error (MAE)**

MAE = (1/n) Σᵢ₌₁ⁿ |Yᵢ - Ŷᵢ|

3. **Root Mean Squared Error (RMSE)**
- Square root of MSE, describes the model's absolute fit to the data

4. **R-squared (R²)**
   
R² = 1 - (RSS / TSS)

Where:
- RSS: Residual Sum of Squares
- TSS: Total Sum of Squares

## Conclusion

Linear regression is a powerful tool for predictive modeling, widely used in various fields. Understanding its principles, types, and evaluation metrics is crucial for effective implementation and interpretation of results.


