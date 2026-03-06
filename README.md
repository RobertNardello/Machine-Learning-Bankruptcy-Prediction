# Breast Cancer Prediction Using XGBoost in R

## Overview

This project applies data mining and machine learning techniques in **R** to predict breast cancer classifications using a dataset from the **UCI Machine Learning Repository**. The model used in this analysis is **Extreme Gradient Boosting (XGBoost)**, an ensemble learning method known for strong performance on classification tasks.

The goal of the project is to train a predictive model using breast cancer feature data and evaluate its performance on a held-out test set.

## Objective

The purpose of this project is to:

- build a breast cancer prediction model in R
- apply XGBoost for classification
- evaluate predictive performance on test data
- identify which features contribute most to model predictions

## Dataset

The dataset used in this project was obtained from the **UCI Machine Learning Repository**. It contains predictor variables related to breast cancer observations and a target variable called `class`, which represents the classification outcome.

- **Target variable:** `class`
- **Source:** UCI Machine Learning Repository

The predictor variables include the remaining dataset features used to train the model.

## Tools and Libraries

This project was developed in R using the following packages:

- `xgboost`
- `magrittr`
- `dplyr`
- `Matrix`

## Methodology

### 1. Data Loading and Preparation

The dataset was imported from a CSV file, and the target variable `class` was converted to a factor for classification.

### 2. Train-Test Split

The data was randomly split into:

- 80% training data
- 20% testing data

A random seed was used so the results could be reproduced.

### 3. Feature Matrix Creation

Because XGBoost requires numeric input, the data was transformed using one-hot encoding with `model.matrix()`.

The training and testing datasets were then converted into `xgb.DMatrix` objects for modeling.

### 4. Model Training

The XGBoost model was trained using the following settings:

- **Objective:** `multi:softprob`
- **Evaluation metric:** `mlogloss`
- **Boosting rounds:** 1000
- **Learning rate (`eta`):** 0.001
- **Maximum depth:** 3

The model tracked both training and test loss during training.

### 5. Model Evaluation

Model performance was evaluated using:

- multiclass log loss
- prediction output on the test set
- confusion matrix
- feature importance

A training-versus-test error plot was also generated to visualize learning over time.

## Model Output

This project produces the following outputs:

- trained XGBoost classification model
- training and test multiclass log loss plot
- minimum test log loss
- feature importance rankings
- confusion matrix comparing predicted and actual class labels

## Project Structure

```text
breast-cancer-xgboost/
│
├── data/
│   └── year5.csv
├── scripts/
│   └── xgboost_breast_cancer.R
├── outputs/
│   ├── error_plot.png
│   ├── feature_importance.png
│   └── confusion_matrix.csv
└── README.md
