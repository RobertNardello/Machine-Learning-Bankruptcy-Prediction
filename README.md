# Bankruptcy Prediction Using XGBoost and Random Forest in R

## Overview

This project applies data mining and machine learning techniques in **R** to predict corporate bankruptcy using the **Polish Companies Bankruptcy** dataset from the **UCI Machine Learning Repository**. The analysis includes two tree-based ensemble modeling approaches:

- **Extreme Gradient Boosting (XGBoost)**
- **Random Forest**, including a **bagging-style model**

The goal of the project is to classify whether a company is likely to go bankrupt based on financial indicators and to compare the predictive performance of ensemble-based machine learning methods.

## Objective

The purpose of this project is to:

- build bankruptcy prediction models in R
- apply XGBoost and Random Forest for classification
- evaluate model performance on test data
- compare ensemble learning approaches
- identify which financial variables contribute most to predictions

## Dataset

The dataset used in this project was obtained from the **UCI Machine Learning Repository** and is titled **Polish Companies Bankruptcy**. It is a classification dataset in the business domain and contains financial ratios designed to support bankruptcy prediction. :contentReference[oaicite:1]{index=1}

The full dataset collection contains **10,503 instances** and **65 features**, with **missing values present**. The repository separates the data into five forecasting-period files. This project uses the **5thYear** data file (`year5.csv`), which corresponds to predicting bankruptcy **1 year ahead** and contains **5,910 instances**, including bankrupt and non-bankrupt firms. :contentReference[oaicite:2]{index=2}

- **Dataset name:** Polish Companies Bankruptcy
- **Source:** UCI Machine Learning Repository
- **Target variable:** `class`
- **Features:** financial ratios and related accounting indicators
- **Selected file for this project:** `year5.csv`

The response variable `class` indicates bankruptcy status, while the predictor variables are financial measures derived from company financial statements.

## Tools and Libraries

This project was developed in **R** using the following packages:

- `xgboost`
- `randomForest`
- `magrittr`
- `dplyr`
- `Matrix`

## Methodology

### 1. Data Loading and Preparation

The dataset was imported from CSV files into R, and the target variable `class` was converted to a factor for classification.

### 2. Train-Test Split

The data was divided into training and testing sets using random sampling. In the XGBoost workflow, the split was approximately:

- 80% training data
- 20% testing data

A random seed was used to improve reproducibility.

### 3. Feature Processing

For the XGBoost model, predictors were transformed into numeric matrix form using `model.matrix()`, allowing one-hot encoding where needed. The resulting matrices were then converted into `xgb.DMatrix` objects for model training.

For the Random Forest and bagging models, the dataset was modeled directly using the `randomForest` package.

### 4. Models Included

#### XGBoost

The XGBoost model was trained as a classification model using:

- objective: `multi:softprob`
- evaluation metric: `mlogloss`
- boosting rounds: 1000
- learning rate (`eta`): 0.001
- maximum tree depth: 3

Training and test loss were tracked during model fitting, and feature importance was generated after training.

#### Bagging

A bagging-style tree ensemble was created using the `randomForest` package with a high `mtry` value, allowing most or all predictors to be considered at each split.

#### Random Forest

A standard Random Forest model was also trained using the `randomForest` package with a smaller `mtry` value to introduce predictor randomness and improve generalization.

### 5. Model Evaluation

The models were evaluated using outputs such as:

- prediction accuracy
- confusion matrix
- multiclass log loss for XGBoost
- feature importance for XGBoost
- error comparison on the test set

The XGBoost workflow also included a training-versus-test loss plot to visualize model learning over time.

## Model Output

This project produces the following outputs:

- trained XGBoost model
- trained bagging model
- trained Random Forest model
- confusion matrix for classification results
- prediction accuracy for bagging and Random Forest
- multiclass log loss tracking for XGBoost
- feature importance rankings for XGBoost
- training and test loss plot

## Project Structure

```text
bankruptcy-prediction/
│
├── data/
│   ├── year5.csv
│   └── bankruptcy_Train.csv
├── scripts/
│   ├── xgboost_bankruptcy_prediction.R
│   └── random_forest_bankruptcy_prediction.R
├── outputs/
│   ├── error_plot.png
│   ├── feature_importance.png
│   ├── confusion_matrix.csv
│   └── model_metrics.csv
└── README.md
