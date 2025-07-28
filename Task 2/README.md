# 🧠 Telco Customer Churn Prediction

## 📌 Objective
The objective of this task is to build a **machine learning pipeline** that can accurately predict whether a customer will churn or not based on their demographic and service usage data. This solution is aimed to be **production-ready, reusable, and scalable**.

---

## ⚙️ Methodology / Approach

### 🔹 Dataset
- **Source:** [Telco Customer Churn Dataset](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
- **Total Records:** ~7,000
- **Target Variable:** `Churn` (Yes/No)

### 🔹 Preprocessing
- Dropped irrelevant columns (e.g., `customerID`)
- Converted `TotalCharges` to numeric (handling missing or non-numeric values)
- Encoded categorical variables using `OneHotEncoder`
- Scaled numerical features with `StandardScaler`
- Combined steps using `ColumnTransformer` and `Pipeline` for modular processing

### 🔹 Model Training & Tuning
- Models implemented:
  - Logistic Regression
  - Random Forest Classifier
- Used `train_test_split` (80/20) to split the dataset
- Performed **hyperparameter tuning** using `GridSearchCV` with 5-fold cross-validation
- Evaluated using classification metrics (accuracy, precision, recall, F1-score)

### 🔹 Model Export
- Trained model pipelines were saved using `joblib` for future inference:
  - `LogisticRegression_churn_pipeline.joblib`
  - `RandomForest_churn_pipeline.joblib`

---

## 📊 Key Results or Observations

- **Random Forest Classifier** achieved superior F1-score and generalization performance over Logistic Regression.
- Model pipelines are **fully self-contained** — they include preprocessing and are deployable as-is.
- The pipeline ensures **consistency between training and inference**, minimizing data leakage.
- Designed in a modular way to allow easy improvements and model swaps.

---
