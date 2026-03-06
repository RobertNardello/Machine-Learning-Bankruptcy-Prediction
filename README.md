# Bankruptcy Prediction Using XGBoost in R

## Overview

This project applies data mining and machine learning techniques in **R** to predict corporate bankruptcy using the **Polish Companies Bankruptcy** dataset from the **UCI Machine Learning Repository**. The model used in this analysis is **Extreme Gradient Boosting (XGBoost)**, an ensemble learning method known for strong performance on structured classification problems.

The goal of the project is to train a predictive model that classifies whether a company is likely to go bankrupt based on financial indicators and to evaluate model performance on a held-out test set.

## Objective

The purpose of this project is to:

- build a bankruptcy prediction model in R
- apply XGBoost for classification
- evaluate predictive performance on test data
- identify which variables contribute most to model predictions

## Dataset

The dataset used in this project was obtained from the **UCI Machine Learning Repository** and is titled **Polish Companies Bankruptcy**.

This dataset contains financial ratios and related variables used to predict whether a company will go bankrupt. The data includes multiple forecasting-period files. Based on the code used in this project, the analysis was performed on the **5thYear** dataset (`year5.csv`), which corresponds to predicting bankruptcy **1 year ahead**.

- **Dataset name:** Polish Companies Bankruptcy
- **Source:** UCI Machine Learning Repository
- **Instances:** 10,503 in the full dataset collection
- **Features:** 65
- **Missing values:** Yes
- **Target variable:** `class`

The predictor variables are financial indicators derived from company financial statements, while the response variable `class` indicates bankruptcy status.

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

A random seed was used to support reproducibility.

### 3. Feature Matrix Creation

Because XGBoost requires numeric input, the data was transformed using one-hot encoding with `model.matrix()`.

The training and testing datasets were then converted into matrix format and stored as `xgb.DMatrix` objects for model training.

### 4. Model Training

The XGBoost model was trained using the following settings:

- **Objective:** `multi:softprob`
- **Evaluation metric:** `mlogloss`
- **Boosting rounds:** 1000
- **Learning rate (`eta`):** 0.001
- **Maximum depth:** 3

The model tracked both training and test loss throughout training.

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
bankruptcy-prediction-xgboost/
│
├── data/
│   └── year5.csv
├── scripts/
│   └── xgboost_bankruptcy_prediction.R
├── outputs/
│   ├── error_plot.png
│   ├── feature_importance.png
│   └── confusion_matrix.csv
└── README.md
