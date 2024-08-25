# Simple Linear Regression

## Table of Contents
1. [Introduction](#introduction)
2. [Concept and Working](#concept-and-working)
3. [Mathematical Foundation](#mathematical-foundation)
4. [Cost Function](#cost-function)
5. [Gradient Descent](#gradient-descent)
6. [Implementation](#implementation)
7. [Conclusion](#conclusion)

## Introduction

Simple Linear Regression is a fundamental statistical method used to model the relationship between a dependent variable and a single independent variable. This technique assumes a linear relationship between the variables and is widely used in predictive modeling and data analysis.

Key characteristics:
- Works with continuous features
- Assumes normal distribution of features
- Involves one independent feature and one output feature

## Concept and Working

### Basic Principle

The core idea is to find the best-fitting straight line through a set of points. This line can then be used to make predictions.

### Example

Consider the following dataset:

| Experience (years) | Salary ($) |
|--------------------|------------|
| 2                  | 45,000     |
| 4                  | 60,000     |
| 6                  | 80,000     |
| 8                  | 100,000    |

When plotted, we can draw a line that best fits these points. This line allows us to predict salaries for different experience levels.
<img src="https://github.com/user-attachments/assets/20c1a8d6-9a07-4127-bd1f-b719d6c1e45f">

For instance, we could predict the salary for an employee with 5 years of experience:
1. Draw a vertical line at x = 5 years
2. Find where it intersects the best fit line
3. Draw a horizontal line to the y-axis to read the predicted salary

In this example, the predicted salary for 5 years of experience would be approximately $70,000.

## Mathematical Foundation

### Equation of the Best Fit Line

The general equation for a straight line is:

y = mx + c

Where:
- y = Dependent variable (predicted value)
- m = Slope of the line
- x = Independent variable
- c = y-intercept

In the context of simple linear regression, we often use:

h₀(x) = β₀ + β₁x

Where:
- h₀(x) = Hypothesis function (predicted y value)
- β₀ = y-intercept
- β₁ = Slope

## Cost Function

The cost function measures the accuracy of our predictions. It quantifies the difference between predicted and actual values.
<img src="https://github.com/user-attachments/assets/fa817a91-f484-4ce7-b035-0d61df244b57">

### Formula

J(θ) = (1/2m) * Σ(h₀(xᵢ) - yᵢ)²

Where:
- J(θ) = Cost function
- m = Number of training examples
- h₀(xᵢ) = Predicted value for the ith example
- yᵢ = Actual value for the ith example

Our goal is to minimize this cost function to improve the model's accuracy.

## Gradient Descent

Gradient Descent is an optimization algorithm used to minimize the cost function.
<img src="https://github.com/user-attachments/assets/1637cb64-c02c-4bb3-82c9-9752d3fbca94">

### Update Rule

θⱼ := θⱼ - α * (1/m) * Σ(hθ(xᵢ) - yᵢ) * xᵢⱼ
<img src="https://github.com/user-attachments/assets/3e6a658b-f9be-4fe2-b552-81dd257ecf92">

Where:
- θⱼ = Parameter to be updated
- α = Learning rate
- m = Number of training examples
- hθ(xᵢ) = Predicted value for the ith example
- yᵢ = Actual value for the ith example
- xᵢⱼ = Value of feature j in the ith training example

## Implementation

To implement simple linear regression:

1. Collect and prepare your data
2. Choose initial values for β₀ and β₁
3. Calculate predictions using h₀(x) = β₀ + β₁x
4. Compute the cost using the cost function
5. Use gradient descent to update β₀ and β₁
6. Repeat steps 3-5 until convergence or a set number of iterations

## Conclusion

Simple Linear Regression is a powerful tool for understanding relationships between variables and making predictions. While it has limitations, such as assuming a linear relationship, it forms the foundation for more complex regression techniques and is widely used in various fields including economics, finance, and social sciences.

For more advanced scenarios, consider exploring multiple linear regression, polynomial regression, or machine learning techniques.
