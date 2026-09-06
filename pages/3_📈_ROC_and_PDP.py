import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import numpy as np
import plotly.graph_objects as go
from pathlib import Path
from sklearn.inspection import partial_dependence
from sklearn.inspection import PartialDependenceDisplay
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


 # =========================================================
 # LOAD MODEL + DATA
 # =========================================================

 if not MODEL_PATH.exists():

    st.info(
        "Gradient Boosting model unavailable for PDP analysis."
    )

 elif not DATA_PATH.exists():

    st.info(
        "European Bank dataset unavailable for PDP analysis."
    )

 else:

    try:

        # -------------------------------------------------
        # Load model
        # -------------------------------------------------

        model = joblib.load(MODEL_PATH)

        # -------------------------------------------------
        # Load original dataset
        # -------------------------------------------------

        pdp_data = pd.read_csv(DATA_PATH)

        # -------------------------------------------------
        # Verify required raw columns
        # -------------------------------------------------

        required_raw_features = [
            "Age",
            "CreditScore",
            "Balance",
            "EstimatedSalary",
            "NumOfProducts",
            "Tenure",
            "HasCrCard",
        ]

        missing_raw = [
            feature
            for feature in required_raw_features
            if feature not in pdp_data.columns
        ]

        if missing_raw:

            st.error(
                "Required features are missing from the "
                f"dataset: {', '.join(missing_raw)}"
            )

        else:

            # -------------------------------------------------
            # Create engineered features
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
            # Remove target and identifiers
            # -------------------------------------------------

            X_pdp = pdp_data.drop(
                columns=[
                    "Exited",
                    "CustomerId",
                    "Surname",
                    "Year",
                ],
                errors="ignore",
            )

            # -------------------------------------------------
            # Final PDP features
            # -------------------------------------------------

            pdp_features = [
                "Age",
                "CreditScore",
                "Balance",
                "EstimatedSalary",
                "NumOfProducts",
                "Tenure",
                "HasCrCard",
            ]

            # -------------------------------------------------
            # Confirm features are actually present
            # -------------------------------------------------

            missing_features = [
                feature
                for feature in pdp_features
                if feature not in X_pdp.columns
            ]

            if missing_features:

                st.error(
                    "Required features are missing for PDP: "
                    + ", ".join(missing_features)
                )

            else:

                # -------------------------------------------------
                # Create PDP
                # -------------------------------------------------

                from sklearn.inspection import PartialDependenceDisplay

                fig, ax = plt.subplots(
                    figsize=(12, 7)
                )

                PartialDependenceDisplay.from_estimator(
                    model,
                    X_pdp,
                    features=pdp_features,
                    kind="average",
                    grid_resolution=50,
                    ax=ax,
                )

                plt.tight_layout()

                st.pyplot(
                    fig,
                    use_container_width=True,
                )

                plt.close(fig)

     except Exception as e:

        st.error(
            f"Unable to generate PDP analysis: {e}"
        )
