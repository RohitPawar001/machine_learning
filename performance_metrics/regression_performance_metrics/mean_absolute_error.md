# Mean Absolute Error (MAE) in Machine Learning

## Introduction

Mean Absolute Error (MAE) is a fundamental metric used to evaluate the performance of regression models in machine learning. It measures the average absolute difference between the predicted values and the actual values.


## Table of Contents

- Introduction
- What is MAE?
- Why Use MAE?
- Mathematical Formula
- Conclusion


## What is MAE?

Mean Absolute Error (MAE) quantifies the average absolute difference between the predicted values and the actual values. It is widely used in regression tasks to measure the accuracy of a model.

We can see from the below formulation that MAE takes the absolute difference between actual and predicted value hence the error would always be positive. Also, as there is no squaring the units will be the same as the original units of the target value. MAE does not give more or less weight to different types of errors, it will penalize the larger and smaller errors equally. Hence, it is more robust to outliers and increases linearly. In any scenario, if you want to pay much attention to outliers then MAE might not be a suitable choice.

## Why Use MAE?

MAE is preferred because:
- It provides a clear measure of model accuracy.
- It is easy to compute and interpret.
- It is less sensitive to outliers compared to Mean Squared Error (MSE).

## Mathematical Formula

The formula for MAE is:

$$
\text{MAE} = \frac{1}{n} \sum_{i=1}^{n} |y_i - \hat{y}_i|
$$

Where:
- \( n \) is the number of data points.
- \( y_i \) is the actual value.
- \( \hat{y}_i \) is the predicted value.

## Conclusion

MAE is a crucial metric for evaluating regression models. By understanding and implementing MAE, you can better assess the performance of your machine learning models and make necessary improvements.

