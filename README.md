# 🏦 Predictive Modeling and Risk Scoring for Bank Customer Churn

An end-to-end **Machine Learning + MLOps project** that predicts bank customer churn, estimates churn probability, assigns customer risk categories, and prioritizes retention actions based on expected financial impact.

The project combines **supervised machine learning, feature engineering, customer risk scoring, model monitoring, MLflow, FastAPI, Streamlit, Supabase, Docker, and GitHub Actions CI/CD** into a production-style ML workflow.

---

## 🚀 Project Overview

Customer churn is a major challenge for banks because losing a customer can result in significant long-term revenue loss.

This project addresses the problem by building a complete churn analytics and risk-scoring system that answers:

- Which customers are likely to churn?
- What is the probability that a customer will churn?
- Which customers are at critical risk?
- Which customers have the highest business value?
- Which customers should receive retention interventions first?
- What is the expected financial loss from potential churn?
- Is the deployed model performing reliably?
- Is incoming prediction data drifting over time?

The system provides both **machine-learning predictions** and **business-oriented retention prioritization**.

---

## 🎯 Key Objectives

### Machine Learning

- Predict customer churn.
- Estimate churn probability.
- Compare multiple classification models.
- Select a champion model.
- Perform feature engineering.
- Analyze model behavior using ROC and Partial Dependence Analysis.

### Risk Scoring

- Classify customers into risk categories.
- Segment customers according to value.
- Estimate expected loss.
- Estimate expected saved value.
- Calculate retention ROI.
- Prioritize customers for intervention.

### MLOps

- Track experiments using MLflow.
- Manage the champion model.
- Serve predictions through FastAPI.
- Store customer risk data in Supabase.
- Monitor prediction and feature drift.
- Build automated CI testing using GitHub Actions.
- Containerize the application using Docker.

---

## 🧠 Machine Learning Workflow

```text
Raw Customer Data
       │
       ▼
Data Cleaning
       │
       ▼
Exploratory Data Analysis
       │
       ▼
Feature Engineering
       │
       ▼
Model Training
       │
       ├── Logistic Regression
       ├── Decision Tree
       ├── Random Forest
       ├── Gradient Boosting
       └── XGBoost
       │
       ▼
Model Evaluation
       │
       ▼
Champion Model
       │
       ▼
Risk Scoring
       │
       ▼
API / Dashboard
       │
       ▼
Monitoring
```

---

## 📊 Dataset

The project uses a bank customer churn dataset containing customer demographics, financial information, product usage, engagement information, and churn labels.

### Main Features

|   Feature	    |          Description                   |
|-----------------|----------------------------------------|
| CustomerId	    |      Unique customer identifier        |
| Surname	    |        Customer surname                |
| CreditScore	    |       Customer credit score            |
| Geography	    |        Customer country                |
| Gender   	    |         Customer gender                |
| Age	           |          Customer age                  |
| Tenure	    |     Number of years with the bank      |
| Balance	    |         Account balance                |
| NumOfProducts   |       Number of bank products          |
| HasCrCard	    |   Whether customer has a credit card   |
| IsActiveMember  |   Whether customer is an active member |
| EstimatedSalary |       Estimated customer salary        |
| Exited    	    |        Churn target variable           |

---

## 🔧 Feature Engineering

Additional business and behavioral features are generated during preprocessing.

Examples include:

- Balance-to-salary relationship
- Product density
- Engagement-related interactions
- Age-tenure interactions
- Customer value indicators
- Balance presence indicators

Feature engineering helps the models capture relationships that may not be directly represented by the original variables.

---

## 🤖 Models

The project experiments with multiple classification algorithms.

### Logistic Regression

Used as a baseline interpretable classification model.

### Decision Tree

Provides a simple rule-based representation of customer churn behavior.

### Random Forest

An ensemble model consisting of multiple decision trees.

### Gradient Boosting

The primary model used for the production-style prediction workflow.

### XGBoost

An additional gradient-boosting approach used for model comparison where supported by the environment.

---

## 🏆 Champion Model

The project uses a Gradient Boosting classifier as the champion model for the production prediction workflow.

The API can load the model through:
```text
MLflow Model Registry
```
and can fall back to the local model artifact when the MLflow model is unavailable.

Local model artifact:
```text
artifacts/models/gradient_boosting.pkl
```
MLflow model:
```text
Bank-Churn-Gradient-Boosting
```
Champion alias:
```text
champion
```

---

## 📈 Model Evaluation

The models are evaluated using classification metrics including:

- Accuracy
- Precision
- Recall
- F1 Score
- ROC-AUC
- Confusion Matrix
- ROC Curve

ROC curves are available in the Streamlit dashboard.

---

## 📈 ROC Curve & Partial Dependence Analysis

The project includes a dedicated Streamlit page for model interpretation.

The analysis provides:

### ROC Analysis
- Individual ROC curve
- Model comparison
- AUC visualization
- 
### Partial Dependence Analysis

Partial Dependence Plots are used to understand how important numerical features affect predicted churn probability.

Example features include:

- Age
- Credit Score
- Balance
- Estimated Salary
- Number of Products
- Tenure

This provides an additional layer of model interpretability beyond standard feature importance.

---

## 🛡️ Customer Risk Scoring

The prediction system goes beyond simply returning 0 or 1.

Each customer receives:
```text
Churn Probability
Risk Category
Customer Value
Value Category
Expected Loss
Retention Cost
Expected Saved Value
ROI
Priority
```
### Risk Categories

|   Probability     |    Risk   |
|-------------------|-----------|
|    < 0.20	      |    Low    |
|  0.20 – < 0.4     |   Medium  |
|  0.40 – < 0.60    |    High   |
|    >= 0.60	      |  Critical |

