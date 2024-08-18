# Linear Regression

Linear regression is an supervised machine learning algorithm. which is used to predict the contineous output feature e.g house price prediction, wheather prediction, etc . it is used to work on the datasets whose target values are already known i.e the labeled data.

# Content
<br>
1.Working<br>
2.Types of linear regression<br>
3.Evaluation metrics<br>

# 1.Working
The linear regression algorithms tries to fit the best fit line on to the independent and dependent features
<br>

<img src="https://media.geeksforgeeks.org/wp-content/uploads/20231129130431/11111111.png" style="width: 500px; height: 300px;">

we can obtain the best fit line through <br>
for simple linear regression<br>
**y= β0+β1X**
<br>
where,<br>
y = dependent feature<br>
x = independent feature <br>
β0 = the intercept on y axis<br>
β1 = it is the slope<br>

<br> for multiple linear regression <br>

**y=β0+β1X1+β2X2+………βnXn**
<br> where<br>
y = dependent feature<br>
x1..Xn = independent featurs <br>
β0 = the intercept on y axis<br>
β1..βn = are the slopes<br>


**The goal of this algorithm is to find the best fit line equation that can predict the values based on the independent features.**

**Best fit line**
our aim is to find the best fit line that have the minimun distance between the actual points and the predicated points i.e also known as the resudual error.<br>
We utilize the cost function to compute the best values in order to get the best fit line since different values for weights or the coefficient of lines result in different regression lines.

**Hypothesis of linear regression**
As we have assumed earlier that our independent feature is the experience i.e X and the respective salary Y is the dependent variable. Let’s assume there is a linear relationship between X and Y then the salary can be predicted using

**Y^i = θ1+θ2xi**
where,
Yi = (1,2,3...n) dependent features<br>
Xi = (1,2,3,...n) independent features<br>

**updatation of  θ1 and θ2 values to get the best-fit line​**<br>
To achieve the best-fit regression line, the model aims to predict the target value 𝑌^. Y^ such that the error difference between the predicted value 𝑌^ and the true value Y is minimum. So, it is very important to update the θ1 and θ2 values, to reach the best value that minimizes the error between the predicted y value (pred) and the true y value (y). 

# 2.Types of linear regression
**Based on independent features**
1. Simple linear regression <br>
simple linear regression having only one independent and dependent feature
<br>
2.multiple linear regression<br>
multiple linear regression has more then one independent and one dependent feature

**based on dependent features**<br>
1.Univariate linear regresssion<br>
is having only on dependent variable<br
2.multivariate linear regression<br>
it has more than one dependent variable
<br>
**Cost function of linear regression**<br>
The cost function or the loss function is nothing but the error or difference between the predicted value Y^ and the true value Y.
the linear regression uses the mean squared error (mse) as an cost function which calculates the average of the squared errors between the predicted values and the actual values.

<br>
**Cost function(J)= 1/n n∑i(Y^i−Yi)2**

**Gradient Discent**
A linear regression model can be trained using the optimization algorithm gradient descent by iteratively modifying the model’s parameters to reduce the mean squared error (MSE) of the model on a training dataset. To update θ1 and θ2 values in order to reduce the Cost function (minimizing RMSE value) and achieve the best-fit line the model uses Gradient Descent. The idea is to start with random θ1 and θ2 values and then iteratively update the values, reaching minimum cost. 

<img src="https://media.geeksforgeeks.org/wp-content/uploads/20230424151248/Gradient-Descent-for-ML-Linear-Regression-(1).webp"  style="width: 500px; height: 300px;">


# 3.Evaluation metrics
<br>1.mean squared error(mse)<br>2.mean absolute error<br>3.root mean squared error<br>4.r2_score<br>
1. mse <br>
Mean Squared Error (MSE) is an evaluation metric that calculates the average of the squared differences between the actual and predicted values for all the data points. The difference is squared to ensure that negative and positive differences don’t cancel each other out.

<br>
2.Mae <br>
Mean Absolute Error is an evaluation metric used to calculate the accuracy of a regression model. MAE measures the average absolute difference between the predicted values and actual values.
Mathematically, MAE is expressed as:<br>
**MAE= 1/n n∑i=1∣Yi–Y^i∣**

<br>
3. RMSE<br>
The square root of the residuals’ variance is the Root Mean Squared Error. It describes how well the observed data points match the expected values, or the model’s absolute fit to the data.
<br> **sqrt(mse)**

<br>
4. r2_score<br>
Coefficient of Determination (R-squared)<br>
R-Squared is a statistic that indicates how much variation the developed model can explain or capture. It is always in the range of 0 to 1. In general, the better the model matches the data, the greater the R-squared number.
In mathematical notation, it can be expressed as:<br>
**R2=1−(RSS/TSS)**
<br> RSS = resudual sum of squares <br> TSS = Total Sum of Squares<br>










