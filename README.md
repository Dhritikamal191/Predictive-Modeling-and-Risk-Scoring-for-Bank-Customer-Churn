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

---

## 💰 Business Risk Scoring

The system estimates the financial importance of each customer.

### Customer Value

Customer value incorporates financial information such as:
```
Balance + Estimated Salary
```
depending on the project's configured scoring implementation.

### Expected Loss

Expected loss estimates the potential financial exposure associated with customer churn.

### Retention Cost

An estimated intervention cost is assigned to the customer.

### Expected Saved Value

The system estimates the value potentially retained through successful intervention.

### ROI

Retention ROI is calculated to help determine whether intervention is economically worthwhile.

---

## 🎯 Retention Priority

Customers can be assigned priorities such as:
```
P1 — Immediate Retention
P2 — High Priority
P3 — Targeted Retention
P4 — Monitor
```

This transforms churn prediction into a business decision-support system.

Instead of asking only:

"Who will churn?"

the system also asks:

"Who should the bank act on first?"

---

🌐 FastAPI

The project provides a production-style REST API for model inference.

API application:
```
api/main.py
```
The API provides:
```
GET /
GET /health
POST /predict
```
Additional endpoints can be added for monitoring and model information.

---

## 🔌 Prediction Endpoint

### Endpoint
```
POST /predict
```
Example request:
```
{
    "CreditScore": 650,
    "Geography": "France",
    "Gender": "Female",
    "Age": 45,
    "Tenure": 5,
    "Balance": 100000,
    "NumOfProducts": 2,
    "HasCrCard": 1,
    "IsActiveMember": 0,
    "EstimatedSalary": 80000
}
```
Example response:
```
{
    "churn_probability": 0.524388,
    "churn_prediction": 1,
    "risk_category": "High"
}
```
## ❤️ Health Endpoint
```
GET /health
```

Example:
```
{
    "status": "healthy",
    "model_loaded": true,
    "model": "Bank-Churn-Gradient-Boosting",
    "alias": "champion",
    "model_source": "MLflow Model Registry"
}
```
The health endpoint is also used by automated tests and deployment validation.

---

## 🗄️ Supabase Integration

Supabase is used as a persistent data layer for customer risk and monitoring information.

The project includes:
```
src/supabase_client.py
```
The application can retrieve customer risk data from Supabase instead of relying entirely on local CSV files.

Example data fields include:
```
CustomerId
Surname
Geography
Gender
Age
CreditScore
Balance
EstimatedSalary
NumOfProducts
IsActiveMember
Exited
Cluster
CustomerValue
ChurnProbability
RiskCategory
ValueCategory
ExpectedLoss
RetentionCost
ExpectedSavedValue
ROI
Priority
```

---

## 🔐 Environment Variables

Sensitive credentials should never be hard-coded into the repository.

Required environment variables:
```
SUPABASE_URL
SUPABASE_SERVICE_KEY
```
For local development, these can be provided through environment variables.

Example:
```
export SUPABASE_URL="your-supabase-url"
export SUPABASE_SERVICE_KEY="your-service-key"
```
On GitHub Actions, configure them as repository secrets.

For deployment platforms such as Render, configure them as environment variables in the service settings.

---

## 📊 Streamlit Dashboard

The project includes an interactive Streamlit dashboard.

Main functionality includes:

- Customer churn prediction
- Risk scoring
- Model comparison
- ROC analysis
- Partial Dependence Analysis
- Monitoring
- Feature drift
- Prediction drift
- Customer segmentation
- Business risk analysis

The dashboard provides a visual interface for both technical and business users.

---

## 📡 MLOps Monitoring

The monitoring dashboard tracks several aspects of model behavior.

### 1. Data Drift

Detects changes in incoming data compared with reference data.

### 2. Feature Drift

Monitors changes in individual feature distributions.

### 3. Feature Distribution

Provides distribution/KDE visualizations for important numerical features.

### 4. Prediction Drift

Tracks changes in model prediction behavior.

### 5. Prediction Monitoring

Records predictions generated by the API.

### 6. Model Performance

Provides model-performance monitoring capabilities.

### 7. Champion Model Validation

Validates the currently deployed champion model.

### 8. Evidently Report

Supports model/data monitoring through Evidently when configured.

## 📁 Project Structure
```
Predictive-Modeling-and-Risk-Scoring-for-Bank-Customer-Churn/
│
├── api/
│   ├── main.py
│   └── schemas.py
│
├── artifacts/
│   ├── models/
│   │   └── gradient_boosting.pkl
│   │
│   ├── metrics/
│   │   └── customer_risk_scoring.csv
│   │
│   ├── monitoring/
│   │   ├── current_data.csv
│   │   └── reference_data.csv
│   │
│   ├── metadata/
│   └── registry/
│
├── data/
│   ├── raw/
│   ├── processed/
│   └── reference/
│
├── pages/
│   ├── 1_*.py
│   ├── 2_*.py
│   ├── 3_📈_ROC_and_PDP.py
│   └── 5_🛡️_Monitoring.py
│
├── src/
│   ├── dashboard/
│   ├── features/
│   ├── monitoring/
│   ├── risk/
│   └── supabase_client.py
│
├── tests/
│   ├── test_api.py
│   └── test_features.py
│
├── app.py
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
├── mlflow.db
├── pytest.ini
└── README.md
```
---

## 🧪 Testing

The project uses pytest for automated testing.

Run:
```
python -m pytest -v
```
Tests cover:

### API
- Root endpoint
- Health endpoint
- Prediction endpoint
- Invalid input handling
  
### Feature Engineering
- Feature creation
- Customer value calculations

Example:
```
6 tests collected
```
The CI pipeline executes these tests automatically.

---

## 🔄 CI/CD

GitHub Actions is used to automatically validate the project.

The CI workflow performs tasks such as:

Checkout Repository
        │
        ▼
Setup Python
        │
        ▼
Install Dependencies
        │
        ▼
Validate Environment
        │
        ▼
Run Pytest
        │
        ▼
Build / Validation

The pipeline uses:
```
Python 3.12
pytest
scikit-learn
joblib
FastAPI
Supabase
MLflow
```
Supabase credentials are supplied through GitHub repository secrets rather than being committed to source control.

---

## 🐳 Docker

The application can be containerized using Docker.

Build the image:
```
docker build -t bank-churn .
```
Run the container:
```
docker run -p 8000:8000 bank-churn
```
The API can then be accessed through:
```
http://localhost:8000
```
Swagger documentation:
```
http://localhost:8000/docs
```

---

## 📚 API Documentation

FastAPI automatically generates interactive documentation.

After starting the API:
```
uvicorn api.main:app --reload
```
Open:
```
/docs
```
or:
```
/redoc
```
---

## 🧰 Technology Stack

### Programming
- Python
  
### Data Science
- Pandas
- NumPy
- Scikit-learn
- SciPy
  
### Machine Learning
- Logistic Regression
- Decision Tree
- Random Forest
- Gradient Boosting
- XGBoost
  
### Visualization
- Plotly
- Streamlit
- 
### MLOps
- MLflow
- Evidently
- GitHub Actions
- Docker
  
### API
- FastAPI
- Uvicorn
- Pydantic
  
### Database
- Supabase
  
### Testing
- Pytest
  
### Version Control
- Git
- GitHub

---

## 🔬 MLOps Architecture

                    ┌───────────────────┐
                    │   Customer Data   │
                    └─────────┬─────────┘
                              │
                              ▼
                    ┌───────────────────┐
                    │ Feature Engineering│
                    └─────────┬─────────┘
                              │
                              ▼
                    ┌───────────────────┐
                    │ Model Training    │
                    │ & Evaluation      │
                    └─────────┬─────────┘
                              │
                              ▼
                    ┌───────────────────┐
                    │      MLflow       │
                    │ Model Registry    │
                    └─────────┬─────────┘
                              │
                         Champion
                              │
                              ▼
                    ┌───────────────────┐
                    │     FastAPI       │
                    │ Prediction API    │
                    └─────────┬─────────┘
                              │
                ┌─────────────┴─────────────┐
                │                           │
                ▼                           ▼
       ┌────────────────┐          ┌────────────────┐
       │   Streamlit    │          │    Supabase    │
       │   Dashboard    │          │   Data Store   │
       └────────────────┘          └────────────────┘
                │                           │
                └─────────────┬─────────────┘
                              ▼
                    ┌───────────────────┐
                    │    Monitoring     │
                    │ Drift / Prediction│
                    │ / Performance     │
                    └───────────────────┘

---

## 📈 Business Impact

The system is designed to help a bank move from:
```
Reactive Customer Retention
```
to:
```
Predictive Customer Retention
```
Instead of treating every customer equally, the system identifies customers based on:
```
Churn Risk
      +
Customer Value
      +
Expected Financial Loss
      +
Retention ROI
      =
Retention Priority
```
This enables retention teams to focus their resources on customers where intervention is expected to have the greatest business impact.

---

## 🔐 Security

The following information should never be committed to GitHub:
```
SUPABASE_SERVICE_KEY
SUPABASE_URL secrets
API keys
Passwords
Private credentials
```
Use environment variables or GitHub/hosting-platform secrets instead.

Do not commit ```.env``` files.

Recommended ```.gitignore``` entries:
```
.env
.venv/
__pycache__/
.pytest_cache/
*.pyc
```

---

## ⚙️ Local Setup
### 1. Clone the repository
```
git clone https://github.com/Dhritikamal191/Predictive-Modeling-and-Risk-Scoring-for-Bank-Customer-Churn.git
```
Enter the project:
```
cd Predictive-Modeling-and-Risk-Scoring-for-Bank-Customer-Churn
```

### 2. Create a virtual environment

Windows:
```
python -m venv bank-churn
```
Activate:
```
bank-churn\Scripts\activate
```

### 3. Install dependencies
```
python -m pip install -r requirements.txt
```

### 4. Configure environment variables

Set:
```
SUPABASE_URL
SUPABASE_SERVICE_KEY
```

### 5. Run tests
```
python -m pytest -v
```

### 6. Run the FastAPI application
```
uvicorn api.main:app --reload
```

Open:
```
http://localhost:8000/docs
```

### 7. Run Streamlit
```
streamlit run app.py
```

---

## ☁️ Deployment

The project can be deployed using platforms supporting Python applications and Docker.

Typical deployment architecture:
```
GitHub
   │
   ▼
CI/CD
   │
   ├──────────────► FastAPI Service
   │
   └──────────────► Streamlit Dashboard
                         │
                         ▼
                      Supabase
```

Environment variables should be configured directly in the deployment platform.

---

## 🧪 Example End-to-End Workflow

A typical prediction workflow is:

User enters customer information
             │
             ▼
       Streamlit UI
             │
             ▼
       FastAPI / Model
             │
             ▼
     Feature Engineering
             │
             ▼
    Gradient Boosting Model
             │
             ▼
      Churn Probability
             │
             ▼
       Risk Category
             │
             ▼
       Business Scoring
             │
             ▼
   Retention Recommendation
             │
             ▼
        Monitoring

---

## 💡 Key Project Highlights

🔹 Predictive Modeling

Built and compared multiple classification algorithms for customer churn prediction.

🔹 Risk-Based Decision Making

Converted churn probability into actionable customer risk categories.

🔹 Business-Oriented Scoring

Combined churn risk and customer value to calculate expected financial impact.

🔹 Production API

Implemented a FastAPI prediction service with validation and health monitoring.

🔹 Model Registry

Integrated MLflow for model tracking and champion model management.

🔹 Cloud Database

Integrated Supabase for persistent customer risk and monitoring data.

🔹 Model Monitoring

Implemented data drift, feature drift, prediction monitoring, and model validation.

🔹 Automated Testing

Integrated pytest with GitHub Actions for continuous validation.

🔹 Containerization

Prepared the application for Docker-based deployment.

---

## 📌 Future Improvements

Potential extensions include:

- Automated model retraining
- Advanced drift-triggered retraining
- SHAP-based explainability
- Automated model promotion
- A/B model testing
- Real-time prediction monitoring
- Advanced retention recommendation engine
- Role-based dashboard access
- Authentication
- Automated alerting
- Model performance degradation alerts
- Automated data-quality validation

---

## 👨‍💻 Author

Dhritikamal Das

MSc MACS
Data Science / Machine Learning / MLOps

---

## ⭐ Project Summary

This project demonstrates a complete machine-learning lifecycle:

Data
 ↓
EDA
 ↓
Feature Engineering
 ↓
Model Development
 ↓
Model Evaluation
 ↓
Risk Scoring
 ↓
MLflow
 ↓
FastAPI
 ↓
Supabase
 ↓
Streamlit
 ↓
Monitoring
 ↓
CI/CD
 ↓
Docker

The goal is not only to predict customer churn, but to transform those predictions into actionable, financially informed retention decisions through an end-to-end MLOps architecture.


### Recommended repository tagline

For the GitHub repository description, use:

> **End-to-end bank customer churn prediction and risk scoring platform with MLflow, FastAPI, Streamlit, Supabase, monitoring, Docker, and CI/CD.**

And your README's strongest opening is the combination of **prediction + risk scoring + MLOps**, rather than presenting it as just another churn-classification project.
