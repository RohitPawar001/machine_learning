# Root Mean Square Error (RMSE) in Machine Learning

## Introduction

Root Mean Square Error (RMSE) is a fundamental metric used to evaluate the performance of regression models in machine learning. It measures the average magnitude of the errors between predicted values and actual values. 


## Table of Contents

- Introduction
- What is RMSE?
- Why Use RMSE?
- Mathematical Formula
- Conclusion

## What is RMSE?

Root Mean Square Error (RMSE) quantifies the average magnitude of the errors between predicted values and actual values. It is widely used in regression tasks to measure the accuracy of a model.

RMSE takes care of some of the advantages of MSE. The square root in the RMSE results in the units being the same as the original units of the target values. Also, it removes the inflating effect of the MSE as it is the square root of the same. Like RMSE, MSE also penalizes models with large errors. Hence, it is more useful in scenarios where larger errors are more undesirable than smaller errors. But because we are squaring the differences and then taking the square root, RMSE is also sensitive to outliers.


## Why Use RMSE?

RMSE is preferred because:
- It provides a clear measure of model accuracy.
- It is easy to compute and interpret.
- It penalizes larger errors more than smaller ones, making it sensitive to outliers.

## Mathematical Formula

The formula for RMSE is:

$$
\text{RMSE} = \sqrt{\frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2}
$$

Where:
- \( n \) is the number of data points.
- \( y_i \) is the actual value.
- \( \hat{y}_i \) is the predicted value.

## Conclusion

RMSE is a crucial metric for evaluating regression models. By understanding and implementing RMSE, you can better assess the performance of your machine learning models and make necessary improvements.
