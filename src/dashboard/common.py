from pathlib import Path

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import plotly.express as px
import plotly.graph_objects as go

from scipy.stats import gaussian_kde
from sklearn.metrics import (
    accuracy_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    confusion_matrix,
)


# =========================================================
# PROJECT PATHS
# =========================================================

ROOT = Path(__file__).resolve().parents[2]

DATA_PATH = ROOT / "Data" /"raw"/ "European_Bank.csv"
MODEL_PATH = ROOT / "all_models_pipeline.pkl"
IMAGE_PATH = ROOT / "Images"


# =========================================================
# PAGE CONFIG / STYLE
# =========================================================

def apply_style():

    st.markdown(
        """
        <style>

        [data-testid="stAppViewContainer"] {
            background-color: #0B132B;
        }

        [data-testid="stSidebar"] {
            background-color: #1C2541;
        }

        h1, h2, h3, h4, h5, h6 {
            color: #EAEAEA !important;
        }

        p, span, div {
            color: #EAEAEA;
        }

        label {
            color: #EAEAEA !important;
        }

        .kpi-card {
            background: linear-gradient(
                135deg,
                #1C2541,
                #3A506B
            );

            padding: 20px;
            border-radius: 15px;
            text-align: center;
            color: white;

            box-shadow:
                0 4px 15px rgba(0,0,0,0.4);
        }

        .kpi-title {
            font-size: 14px;
            opacity: 0.8;
        }

        .kpi-value {
            font-size: 28px;
            font-weight: bold;
            margin-top: 5px;
        }

        .kpi-icon {
            font-size: 30px;
            margin-bottom: 10px;
        }

        input,
        select,
        textarea {
            color: white !important;
            -webkit-text-fill-color: white !important;
        }

        </style>
        """,
        unsafe_allow_html=True,
    )


# =========================================================
# LOAD DATA
# =========================================================

@st.cache_data
def load_data():

    return pd.read_csv(DATA_PATH)


# =========================================================
# LOAD MODELS
# =========================================================

@st.cache_resource
def load_models():

    return joblib.load(MODEL_PATH)


# =========================================================
# KPI CARD
# =========================================================

def kpi_card(title, value, icon):

    return f"""
    <div class="kpi-card">

        <div class="kpi-icon">
            {icon}
        </div>

        <div class="kpi-title">
            {title}
        </div>

        <div class="kpi-value">
            {value}
        </div>

    </div>
    """


# =========================================================
# MODEL INPUTS
# =========================================================

def customer_inputs():

    st.sidebar.subheader(
        "Customer Feature Inputs"
    )

    credit_score = st.sidebar.number_input(
        "Credit Score",
        300,
        900,
        600,
    )

    age = st.sidebar.number_input(
        "Age",
        18,
        100,
        40,
    )

    tenure = st.sidebar.slider(
        "Tenure",
        0,
        10,
        5,
    )

    balance = st.sidebar.slider(
        "Balance",
        0,
        250000,
        50000,
    )

    products = st.sidebar.slider(
        "Number of Products",
        1,
        4,
        2,
    )

    active_member = st.sidebar.selectbox(
        "Active Member",
        [0, 1],
    )

    has_card = st.sidebar.selectbox(
        "Has Credit Card",
        [0, 1],
    )

    salary = st.sidebar.number_input(
        "Estimated Salary",
        min_value=0.0,
        max_value=250000.0,
        value=80000.0,
        step=5000.0,
    )

    gender = st.sidebar.selectbox(
        "Gender",
        ["Male", "Female"],
    )

    geography = st.sidebar.selectbox(
        "Geography",
        ["France", "Germany", "Spain"],
    )

    input_df = pd.DataFrame(
        {
            "CreditScore": [credit_score],
            "Age": [age],
            "Tenure": [tenure],
            "Balance": [balance],
            "NumOfProducts": [products],
            "HasCrCard": [has_card],
            "IsActiveMember": [active_member],
            "EstimatedSalary": [salary],
            "Geography": [geography],
            "Gender": [gender],
        }
    )

    return (
        input_df,
        credit_score,
        age,
        tenure,
        balance,
        products,
        active_member,
        has_card,
        salary,
        gender,
        geography,
    )


# =========================================================
# MODEL SELECTION
# =========================================================

def model_selection(models):

    st.sidebar.subheader(
        "Model Selection"
    )

    model_choice = st.sidebar.radio(
        "Select Model",
        [
            "Logistic Regression",
            "Decision Tree",
            "Random Forest",
            "Gradient Boosting",
            "XGBoost",
        ],
    )

    threshold = st.sidebar.slider(
        "Select Threshold",
        0.0,
        1.0,
        0.50,
        0.01,
    )

    return model_choice, threshold, models[model_choice]