# R-squared (Coefficient of Determination) in Machine Learning

## Introduction

R-squared, also known as the coefficient of determination, is a statistical measure used to evaluate the quality of a regression model. It represents the proportion of the variance for a dependent variable that's explained by one or more independent variables in a regression model. 



## Table of Contents

- Introduction
- What is R-squared?
- Why Use R-squared?
- Mathematical Formula
- Conclusion

## What is R-squared?

R-squared quantifies how well the regression predictions approximate the real data points. An R-squared of 1 indicates that the regression predictions perfectly fit the data. An R-squared of 0 indicates that the model does not explain any of the variability of the response data around its mean.

it is the ratio of the current model with the baseline model. Here, we consider the baseline model as the one which predicts the mean value of the target variable. It compares our current model with the baseline model and tells us how much better or worse it is performing. R-square will always be less than 1.

If the R-Square value is 0, then our current model is no better than our baseline model and if it is 1, then our current model is predicting the actual values of the target variables. The latter situation is impossible to occur. A negative r-square value will suggest that the current model is worse than the baseline model. Usually, a higher r-square value indicates that our regression model is a good fit for our target observations.

The main disadvantage of this metric is that you cannot estimate that the predictions are biased or not. This can be assessed by using residual plots. Also, it grows with the number of predictor variables, hence it is biased towards more complex models.

## Why Use R-squared?

R-squared is preferred because:
- It provides a measure of how well the model fits the data.
- It helps in comparing different models.
- It is easy to compute and interpret.

## Mathematical Formula

The formula for R-squared is:

$$
R^2 = 1 - \frac{SS_{res}}{SS_{tot}}
$$

Where:
- \( SS_{res} \) is the sum of squares of residuals.
- \( SS_{tot} \) is the total sum of squares.

Alternatively, it can be calculated as:

$$
R^2 = \frac{SS_{reg}}{SS_{tot}}
$$

Where:
- \( SS_{reg} \) is the sum of squares due to regression.

## Conclusion

R-squared is a crucial metric for evaluating regression models. By understanding and implementing R-squared, you can better assess the performance of your machine learning models and make necessary improvements.