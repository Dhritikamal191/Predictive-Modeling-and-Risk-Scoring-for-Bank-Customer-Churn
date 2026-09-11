import os
import pandas as pd
from supabase import create_client


# =========================================================
# SUPABASE CONNECTION
# =========================================================

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_SERVICE_KEY = os.getenv("SUPABASE_SERVICE_KEY")

if not SUPABASE_URL or not SUPABASE_SERVICE_KEY:
    raise ValueError(
        "SUPABASE_URL or SUPABASE_SERVICE_KEY not found."
    )

supabase = create_client(
    SUPABASE_URL,
    SUPABASE_SERVICE_KEY,
)


# =========================================================
# LOAD CUSTOMER RISK DATA
# =========================================================

def load_customer_risk_data():

    response = (
        supabase
        .table("customer_risk_scoring")
        .select("*")
        .execute()
    )

    if not response.data:
        return pd.DataFrame()

    return pd.DataFrame(response.data)


# =========================================================
# SAVE CUSTOMER PREDICTION
# =========================================================

def save_customer_predictions(
    credit_score,
    age,
    tenure,
    balance,
    products,
    has_card,
    active_member,
    salary,
    gender,
    geography,
    churn_probability,
    prediction,
    risk_category,
    model_name,
):
    """
    Save an individual prediction made through
    the Streamlit Customer Risk Calculator.
    """

    prediction_data = {
        "CreditScore": int(credit_score),
        "Age": int(age),
        "Tenure": int(tenure),
        "Balance": float(balance),
        "NumOfProducts": int(products),
        "HasCrCard": int(has_card),
        "IsActiveMember": int(active_member),
        "EstimatedSalary": float(salary),
        "Gender": gender,
        "Geography": geography,
        "ChurnProbability": float(churn_probability),
        "Prediction": int(prediction),
        "RiskCategory": risk_category,
        "Model": model_name,
    }

    response = (
        supabase
        .table("customer_predictions")
        .insert(prediction_data)
        .execute()
    )

    return response.data

# =========================================================
# PREDICTION MONITORING DATA
# =========================================================

def load_prediction_monitoring_data():
    response = (
        supabase
        .table("prediction_monitoring")
        .select("*")
        .execute()
    )

    if not response.data:
        return pd.DataFrame()

    return pd.DataFrame(response.data)