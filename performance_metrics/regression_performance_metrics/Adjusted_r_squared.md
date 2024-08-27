# Adjusted R-squared in Machine Learning

## Introduction

Adjusted R-squared is a statistical measure used to evaluate the quality of a regression model, similar to R-squared. However, it adjusts for the number of predictors in the model, providing a more accurate measure of model performance. This README provides an overview of Adjusted R-squared, its importance, and how to implement it in Python.

## Table of Contents

- Introduction
- What is Adjusted R-squared?
- Why Use Adjusted R-squared?
- Mathematical Formula
- Conclusion


## What is Adjusted R-squared?

Adjusted R-squared is a modified version of R-squared that adjusts for the number of predictors in a regression model. It provides a more accurate measure of model performance, especially when comparing models with different numbers of predictors.

It penalizes the model for adding more independent variables that don’t necessarily fit the model. Adjusted r-square only increases if the independent variables help in improving the model performance.

## Why Use Adjusted R-squared?

Adjusted R-squared is preferred because:
- It penalizes the addition of irrelevant predictors.
- It provides a more accurate measure of model performance.
- It helps in comparing models with different numbers of predictors.

## Mathematical Formula

The formula for Adjusted R-squared is:

$$
\text{Adjusted } R^2 = 1 - \left( \frac{(1 - R^2) \cdot (n - 1)}{n - k - 1} \right)
$$

Where:
- \( R^2 \) is the R-squared value.
- \( n \) is the number of observations.
- \( k \) is the number of predictors.

## Conclusion

Adjusted R-squared is a crucial metric for evaluating regression models, especially when comparing models with different numbers of predictors. By understanding and implementing Adjusted R-squared, you can better assess the performance of your machine learning models and make necessary improvements.