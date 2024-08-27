# Support Vector Classifier (SVC)

Support Vector Classifier (SVC) is a specific implementation of the Support Vector Machine (SVM) algorithm designed for classification tasks.

## Overview

SVC is a powerful and flexible supervised learning algorithm that can handle both binary and multiclass classification problems. It aims to find the optimal hyperplane that best separates different classes in the feature space.

## Getting Started

1. **Installation:**
   - Make sure you have Python and the necessary libraries installed (e.g., NumPy, scikit-learn).
   - Install scikit-learn using pip:
     ```
     pip install scikit-learn
     ```

2. **Usage:**
   - Import SVC from scikit-learn:
     ```python
     from sklearn.svm import SVC
     ```

3. **Example:**
   - Load your dataset (e.g., using `make_blobs` from `sklearn.datasets`).
   - Create an SVC model:
     ```python
     svc_model = SVC(kernel='linear', C=1e6)
     svc_model.fit(X, y)  # X: feature matrix, y: target labels
     ```

4. **Hyperparameters:**
   - `kernel`: Choose the kernel type (linear, polynomial, radial basis function, etc.).
   - `C`: Regularization parameter (controls the trade-off between overfitting and underfitting).



## Working

A support vector machine (SVM) is a supervised machine learning algorithm that classifies data by finding an optimal line or hyperplane that maximizes the distance between each class in an N-dimensional space.

SVMs were developed in the 1990s by Vladimir N. Vapnik and his colleagues, and they published this work in a paper titled "Support Vector Method for Function Approximation, Regression Estimation, and Signal Processing"1 in 1995.

<img src="https://www.ibm.com/content/dam/connectedassets-adobe-cms/worldwide-content/creative-assets/s-migr/ul/g/8f/27/3-1_svm_optimal-hyperplane_max-margin_support-vectors-2-1.component.complex-narrative-xl.ts=1723563766425.png/content/adobe-cms/us/en/topics/support-vector-machine/jcr:content/root/table_of_contents/body/content_section_styled/content-section-body/complex_narrative/items/content_group/image">

SVMs are commonly used within classification problems. They distinguish between two classes by finding the optimal hyperplane that maximizes the margin between the closest data points of opposite classes. The number of features in the input data determine if the hyperplane is a line in a 2-D space or a plane in a n-dimensional space. Since multiple hyperplanes can be found to differentiate classes, maximizing the margin between points enables the algorithm to find the best decision boundary between classes. This, in turn, enables it to generalize well to new data and make accurate classification predictions. The lines that are adjacent to the optimal hyperplane are known as support vectors as these vectors run through the data points that determine the maximal margin.

The SVM algorithm is widely used in machine learning as it can handle both linear and nonlinear classification tasks. However, when the data is not linearly separable, kernel functions are used to transform the data higher-dimensional space to enable linear separation. This application of kernel functions can be known as the “kernel trick”, and the choice of kernel function, such as linear kernels, polynomial kernels, radial basis function (RBF) kernels, or sigmoid kernels, depends on data characteristics and the specific use case.

## Evaluation

Assess model performance using metrics like accuracy, precision, recall, and F1-score.
Visualize decision boundaries and margins.