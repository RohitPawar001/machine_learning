# Simple Linear Regression

## Introduction

The simple linear regression aims to predict the dependent feature based on independent features. It works with continuous features and assumes that the distribution of the feature is normally distributed. It's called simple linear regression because it involves one independent feature and one output feature.

## Working

Example data:

| Experience (years) | Salary ($) |
|--------------------|------------|
| 2                  | 45000      |
| 4                  | 60000      |
| 6                  | 80000      |
| 8                  | 100000     |

All the points are fitted on the graph.

Suppose we want to predict the salary of an employee based on 5 years of experience:
When we draw a straight line parallel to the y-axis, we reach some point on the best fit line. From that point, draw a parallel line to the x-axis to get the predicted salary. This is the simple working of simple linear regression.

So the expected salary would be 70000.

## Constructing the Best Fit Line

Aim: Find the best fit line which has minimum error/residual error.

Equation of line: y = mx + c
Where:
y = predicted points on best fit line
m = slope
x = independent features
c = intercept

Line equation taking hypothesis in mind:
h₀(x) = β₀ + β₁x

Example:
Suppose β₀ = 0, β₁ = 0.5

h₀(x) = (0 + 0.5 * 2) + (0 + 0.5 * 4) + (0 + 0.5 * 6) + (0 + 0.5 * 8)
h₀(x) = 1 + 2 + 3 + 4 + 5 + ...

## Cost Function of Simple Linear Regression

The cost function is the difference between the actual points and the predicted points, i.e., the residual error. It's also known as the loss function. As the cost function is minimized, the model becomes more accurate. We can increase the accuracy of the model by using gradient descent.

Cost function: (h₀(x) - y)

Where:
h₀(x) = predicted points
y = actual points

To avoid negative values, we square it:
(h₀(x) - y)²

For multiple data points, we take the summation:
(m Σ i=1) (h₀(x) - y)²

To find the average:
1/2m (m Σ i=1) (h₀(x) - y)²

This is the cost function of simple linear regression. This cost function gives the gradient descent.

By minimizing the cost function, we can optimize the model's parameters to make better predictions. This process is often done using optimization algorithms like gradient descent.

To find the global minima, we can use the repeat convergence theorem:

θⱼ := θⱼ - α * (1/m) * Σ(hθ(xᵢ) - yᵢ) * xᵢⱼ

Where:
θⱼ: Weights of the hypothesis
hθ(xᵢ): Predicted y value for ith input
i: Feature index number (can be 0, 1, 2, ..., n)
α: Learning Rate of Gradient Descent
