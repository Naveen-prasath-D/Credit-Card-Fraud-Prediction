# Credit Card Fraud Prediction

## Overview

Credit card fraud is a significant concern for both credit card companies and customers. Detecting fraudulent transactions is crucial to prevent customers from being charged for unauthorized purchases. This project aims to predict fraudulent credit card transactions using machine learning techniques.

## Dataset

The dataset contains transactions made by credit cards in September 2013 by European cardholders. It comprises transactions that occurred over two days, with 492 frauds out of 284,807 transactions. The dataset is highly unbalanced, with fraudulent transactions accounting for only 0.172% of all transactions.

Dataset Link: [Credit Card Fraud Dataset](https://www.kaggle.com/mlg-ulb/creditcardfraud)

## Tasks

### 1. Feature Selection

Out of the 30 independent input variables, we aim to identify the variables most useful in predicting the 'Class' of transactions (fraudulent or legitimate).

### 2. Dimensionality Reduction with PCA

Principal Component Analysis (PCA) will be implemented to reduce the number of independent variables. We will evaluate whether PCA improves model performance.

### 3. Classification Models

We will implement the following classification models to predict the 'Class' of transactions:

- Logistic Regression
- Support Vector Classifier (SVC)
- Decision Tree Classifier

## Model Explanations

### Logistic Regression

Logistic regression is a linear classification algorithm used for binary classification tasks. It models the probability that a given input belongs to a particular class using a logistic function. The algorithm estimates the parameters of the logistic function by minimizing a cost function, typically the logistic loss.

### Support Vector Classifier (SVC)

Support Vector Classifier (SVC) is a powerful supervised learning algorithm capable of performing classification tasks. SVC finds the hyperplane that best separates different classes by maximizing the margin between them. It can handle non-linear decision boundaries through the use of kernel functions.

### Decision Tree Classifier

Decision tree classifier is a non-parametric supervised learning algorithm used for classification tasks. It partitions the feature space into regions based on feature values and predicts the class label of a sample by traversing the tree from the root to a leaf node. Decision trees are interpretable and can handle non-linear relationships between features and the target variable.

## Conclusion

This project aims to develop robust models for predicting fraudulent credit card transactions. By leveraging feature selection, dimensionality reduction, and various classification algorithms, we aim to achieve accurate predictions while addressing the challenge of class imbalance in the dataset.
