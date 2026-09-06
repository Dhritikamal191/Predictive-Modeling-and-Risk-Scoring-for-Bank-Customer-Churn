import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import numpy as np
import plotly.graph_objects as go
from pathlib import Path
from sklearn.inspection import partial_dependence
import joblib

from sklearn.metrics import (
    roc_curve,
    auc,
    roc_auc_score,
)

from sklearn.inspection import partial_dependence

from src.dashboard.common import (
    apply_style,
    load_data,
    load_models,
    model_selection,
    customer_inputs,
)


st.set_page_config(
    page_title="ROC and PDP",
    page_icon="📈",
    layout="wide",
)

apply_style()

df = load_data()
models = load_models()

model_choice, threshold, model = model_selection(
    models
)

customer_inputs()

X = df.drop(
    ["Exited", "CustomerId", "Surname"],
    axis=1,
    errors="ignore",
)

y = df["Exited"]


st.title("📈 ROC Curve & Partial Dependence")


# =========================================================
# ROC CURVE
# =========================================================

col1, col2 = st.columns(2)


with col1:

    y_prob = model.predict_proba(X)[:, 1]

    fpr, tpr, _ = roc_curve(
        y,
        y_prob,
    )

    roc_auc = auc(
        fpr,
        tpr,
    )

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=fpr,
            y=tpr,
            mode="lines",
            name=f"AUC = {roc_auc:.3f}",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode="lines",
            line=dict(
                dash="dash"
            ),
            name="Random Model",
        )
    )

    fig.update_layout(
        title="ROC Curve",
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate",
        template="plotly_dark",
        height=500,
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
    )


# =========================================================
# MODEL ROC COMPARISON
# =========================================================

with col2:

    fig = go.Figure()

    for name, current_model in models.items():

        probabilities = (
            current_model
            .predict_proba(X)[:, 1]
        )

        fpr, tpr, _ = roc_curve(
            y,
            probabilities,
        )

        roc_auc = auc(
            fpr,
            tpr,
        )

        fig.add_trace(
            go.Scatter(
                x=fpr,
                y=tpr,
                mode="lines",
                name=f"{name} ({roc_auc:.3f})",
            )
        )

    fig.add_trace(
        go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode="lines",
            line=dict(
                dash="dash"
            ),
            name="Random",
        )
    )

    fig.update_layout(
        title="ROC Curve Comparison",
        template="plotly_dark",
        height=500,
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
    )

# =========================================================
# PARTIAL DEPENDENCE — COMBINED
# =========================================================

st.subheader(
    "📈 Partial Dependence Analysis"
)

st.caption(
    "Combined PDP showing how selected features influence "
    "predicted churn probability."
)

ROOT = Path(__file__).resolve().parents[1]

# =========================================================
# PATHS
# =========================================================

MODEL_PATH = (
    ROOT
    / "artifacts"
    / "models"
    / "gradient_boosting.pkl"
)

# Primary artifact containing the complete customer dataset
RISK_DATA_PATH = (
    ROOT
    / "artifacts"
    / "metrics"
    / "customer_risk_scoring.csv"
)

# Fallback artifact
SEGMENT_DATA_PATH = (
    ROOT
    / "artifacts"
    / "metrics"
    / "customer_segments.csv"
)

# =========================================================
# LOAD MODEL
# =========================================================

if not MODEL_PATH.exists():

    st.warning(
        "Gradient Boosting model is unavailable for PDP analysis."
    )

else:

    try:

        model = joblib.load(MODEL_PATH)

        # =================================================
        # LOAD DATA
        # =================================================

        if RISK_DATA_PATH.exists():

            pdp_data = pd.read_csv(
                RISK_DATA_PATH
            )

        elif SEGMENT_DATA_PATH.exists():

            pdp_data = pd.read_csv(
                SEGMENT_DATA_PATH
            )

        else:

            pdp_data = None

        # =================================================
        # CHECK DATA
        # =================================================

        if pdp_data is None:

            st.warning(
                "PDP dataset is unavailable."
            )

        else:

            # =================================================
            # CREATE ENGINEERED FEATURES
            # =================================================

            required_base_columns = [
                "Balance",
                "EstimatedSalary",
                "NumOfProducts",
                "Tenure",
                "IsActiveMember",
                "HasCrCard",
                "Age",
            ]

            missing_base = [
                col
                for col in required_base_columns
                if col not in pdp_data.columns
            ]

            if missing_base:

                st.warning(
                    "Required features are missing for PDP: "
                    + ", ".join(missing_base)
                )

            else:

                # ---------------------------------------------
                # Balance / Salary Ratio
                # ---------------------------------------------

                pdp_data["BalanceSalaryRatio"] = (
                    pdp_data["Balance"]
                    / (
                        pdp_data["EstimatedSalary"]
                        + 1
                    )
                )

                # ---------------------------------------------
                # Customer Value
                # ---------------------------------------------

                pdp_data["CustomerValue"] = (
                    pdp_data["Balance"]
                    * pdp_data["EstimatedSalary"]
                )

                # ---------------------------------------------
                # Product Density
                # ---------------------------------------------

                pdp_data["ProductDensity"] = (
                    pdp_data["NumOfProducts"]
                    / (
                        pdp_data["Tenure"]
                        + 1
                    )
                )

                # ---------------------------------------------
                # Engagement Score
                # ---------------------------------------------

                pdp_data["EngagementScore"] = (
                    pdp_data["IsActiveMember"]
                    + pdp_data["HasCrCard"]
                )

                # ---------------------------------------------
                # Age × Tenure
                # ---------------------------------------------

                pdp_data["AgeTenureInteraction"] = (
                    pdp_data["Age"]
                    * pdp_data["Tenure"]
                )

                # ---------------------------------------------
                # Has Balance
                # ---------------------------------------------

                pdp_data["HasBalance"] = (
                    pdp_data["Balance"] > 0
                ).astype(int)

                # =================================================
                # REMOVE NON-MODEL COLUMNS
                # =================================================

                X_pdp = pdp_data.drop(
                    columns=[
                        "Exited",
                        "CustomerId",
                        "Surname",
                        "Cluster",
                        "ChurnProbability",
                        "RiskCategory",
                        "ValueCategory",
                        "ExpectedLoss",
                        "RetentionCost",
                        "ExpectedSavedValue",
                        "ROI",
                        "Priority",
                    ],
                    errors="ignore",
                )

                # =================================================
                # PDP FEATURES
                # =================================================

                pdp_features = [
                    "Age",
                    "CreditScore",
                    "Balance",
                    "EstimatedSalary",
                    "NumOfProducts",
                    "Tenure",
                ]

                pdp_features = [
                    feature
                    for feature in pdp_features
                    if feature in X_pdp.columns
                ]

                if not pdp_features:

                    st.warning(
                        "No valid PDP features are available."
                    )

                else:

                    # =================================================
                    # GENERATE COMBINED PDP
                    # =================================================

                    fig = go.Figure()

                    for feature in pdp_features:

                        try:

                            # -----------------------------------------
                            # Determine feature range
                            # -----------------------------------------

                            values = (
                                X_pdp[feature]
                                .dropna()
                                .astype(float)
                            )

                            if len(values) < 2:

                                continue

                            feature_min = values.min()
                            feature_max = values.max()

                            if feature_min == feature_max:

                                continue

                            # -----------------------------------------
                            # Create evenly spaced values
                            # -----------------------------------------

                            grid = np.linspace(
                                feature_min,
                                feature_max,
                                50,
                            )

                            pdp_values = []

                            # -----------------------------------------
                            # Partial dependence
                            # -----------------------------------------

                            for value in grid:

                                X_temp = X_pdp.copy()

                                X_temp[feature] = value

                                probabilities = (
                                    model.predict_proba(
                                        X_temp
                                    )[:, 1]
                                )

                                pdp_values.append(
                                    np.mean(
                                        probabilities
                                    )
                                )

                            # -----------------------------------------
                            # Normalize X-axis
                            # -----------------------------------------

                            value_range = (
                                feature_max
                                - feature_min
                            )

                            x_normalized = (
                                (
                                    grid
                                    - feature_min
                                )
                                / value_range
                                * 100
                            )

                            # -----------------------------------------
                            # Add trace
                            # -----------------------------------------

                            fig.add_trace(
                                go.Scatter(
                                    x=x_normalized,
                                    y=pdp_values,
                                    mode="lines",
                                    name=feature,
                                    hovertemplate=(
                                        f"<b>{feature}</b>"
                                        "<br>Relative Feature Value: "
                                        "%{x:.1f}%"
                                        "<br>Predicted Churn Probability: "
                                        "%{y:.3f}"
                                        "<extra></extra>"
                                    ),
                                )
                            )

                        except Exception as feature_error:

                            st.warning(
                                f"Unable to calculate PDP "
                                f"for {feature}: "
                                f"{feature_error}"
                            )

                    # =================================================
                    # LAYOUT
                    # =================================================

                    fig.update_layout(
                        title=(
                            "Combined Partial Dependence Analysis"
                        ),
                        xaxis_title=(
                            "Relative Feature Value "
                            "(0% = Low, 100% = High)"
                        ),
                        yaxis_title=(
                            "Predicted Churn Probability"
                        ),
                        template="plotly_dark",
                        height=550,
                        hovermode="x unified",
                        legend=dict(
                            title="Features",
                            orientation="h",
                            yanchor="bottom",
                            y=1.02,
                            xanchor="center",
                            x=0.5,
                        ),
                    )

                    st.plotly_chart(
                        fig,
                        use_container_width=True,
                    )

    except Exception as e:

        st.error(
            f"Unable to generate PDP analysis: {e}"
<<<<<<< Updated upstream
        )
=======
        )
>>>>>>> Stashed changes
