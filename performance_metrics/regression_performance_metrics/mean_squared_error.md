# Mean Squared Error (MSE) in Machine Learning

## Introduction

Mean Squared Error (MSE) is a fundamental metric used to evaluate the performance of regression models in machine learning. It measures the average squared difference between the predicted values and the actual values. 


## Table of Contents

- Introduction
- What is MSE?
- Why Use MSE?
- Mathematical Formula
- Conclusion


## What is MSE?

Mean Squared Error (MSE) quantifies the average squared difference between the predicted values and the actual values. It is widely used in regression tasks to measure the accuracy of a model.

The main advantage of MSE is that it is completely differentiable so, it can be easily used as a loss function. But there are few disadvantages as well. Firstly, due to the squaring effect in MSE, the output units you get will always be squared units.

If you have outliers in your data MSE can fluctuate a lot and it will become harder to interpret. Also, if you are using MSE as your metric then it is advised to normalize or standardize your data, otherwise, the metric will inflate a lot.

## Why Use MSE?

MSE is preferred because:
- It penalizes larger errors more than smaller ones, making it sensitive to outliers.
- It provides a clear metric for model optimization.
- It is easy to compute and interpret.

## Mathematical Formula

The formula for MSE is:

$$
\text{MSE} = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2
$$

Where:
- \( n \) is the number of data points.
- \( y_i \) is the actual value.
- \( \hat{y}_i \) is the predicted value.

 
## Conclusion

MSE is a crucial metric for evaluating regression models. By understanding and implementing MSE, you can better assess the performance of your machine learning models and make necessary improvements.

