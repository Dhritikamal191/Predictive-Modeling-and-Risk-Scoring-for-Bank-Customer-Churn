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

DATA_PATH = (
    ROOT
    / "data"
    / "raw"
    / "European_Bank.csv"
)

MODEL_PATH = (
    ROOT
    / "artifacts"
    / "models"
    / "gradient_boosting.pkl"
)

if MODEL_PATH.exists() and DATA_PATH.exists():

    try:

        # -------------------------------------------------
        # Load model
        # -------------------------------------------------

        model = joblib.load(MODEL_PATH)

        # -------------------------------------------------
        # Load data
        # -------------------------------------------------

        pdp_data = pd.read_csv(DATA_PATH)

        # -------------------------------------------------
        # Create the same engineered features used by model
        # -------------------------------------------------

        pdp_data["BalanceSalaryRatio"] = (
            pdp_data["Balance"]
            / (pdp_data["EstimatedSalary"] + 1)
        )

        pdp_data["CustomerValue"] = (
            pdp_data["Balance"]
            * pdp_data["EstimatedSalary"]
        )

        pdp_data["ProductDensity"] = (
            pdp_data["NumOfProducts"]
            / (pdp_data["Tenure"] + 1)
        )

        pdp_data["EngagementScore"] = (
            pdp_data["IsActiveMember"]
            + pdp_data["HasCrCard"]
        )

        pdp_data["AgeTenureInteraction"] = (
            pdp_data["Age"]
            * pdp_data["Tenure"]
        )

        pdp_data["HasBalance"] = (
            pdp_data["Balance"] > 0
        ).astype(int)

        # -------------------------------------------------
        # Remove target / identifier columns
        # -------------------------------------------------

        X_pdp = pdp_data.drop(
            columns=[
                "Exited",
                "CustomerId",
                "Surname",
            ],
            errors="ignore",
        )

        # -------------------------------------------------
        # Features for combined PDP
        # -------------------------------------------------

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

        # -------------------------------------------------
        # Create combined Plotly figure
        # -------------------------------------------------

        fig = go.Figure()

        for feature in pdp_features:

            result = partial_dependence(
                model,
                X_pdp,
                features=[feature],
                kind="average",
                grid_resolution=50,
            )

            values = result["grid_values"][0]
            pdp_values = result["average"][0]

            # Normalize x-axis to percentage of feature range
            x_min = values.min()
            x_max = values.max()

            if x_max != x_min:

                x_normalized = (
                    (values - x_min)
                    / (x_max - x_min)
                    * 100
                )

            else:

                x_normalized = np.zeros(
                    len(values)
                )

            fig.add_trace(
                go.Scatter(
                    x=x_normalized,
                    y=pdp_values,
                    mode="lines",
                    name=feature,
                    hovertemplate=(
                        f"<b>{feature}</b>"
                        "<br>Relative Feature Value: %{x:.1f}%"
                        "<br>Predicted Churn Probability: %{y:.3f}"
                        "<extra></extra>"
                    ),
                )
            )

        fig.update_layout(
            title="Combined Partial Dependence Analysis",
            xaxis_title=(
                "Relative Feature Value "
                "(0% = Low, 100% = High)"
            ),
            yaxis_title="Predicted Churn Probability",
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
        )

else:

    st.info(
        "Model or dataset unavailable for PDP analysis."
    )