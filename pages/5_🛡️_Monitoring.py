"""
Bank Churn — MLOps Monitoring Dashboard

Displays:
1. Data Drift
2. Feature Drift
3. Feature Distribution — KDE
4. Prediction Drift
5. Prediction Monitoring
6. Model Performance
7. Champion Model Validation
8. Evidently Report
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from scipy.stats import gaussian_kde


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="MLOps Monitoring",
    page_icon="🛡️",
    layout="wide",
)


# =========================================================
# STYLE
# =========================================================

st.markdown(
    """
    <style>

    [data-testid="stAppViewContainer"] {
        background-color: #0B132B;
    }

    [data-testid="stSidebar"] {
        background-color: #1C2541;
    }

    h1, h2, h3 {
        color: white !important;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# =========================================================
# PROJECT PATHS
# =========================================================

ROOT = Path(__file__).resolve().parents[1]

MONITORING_DIR = (
    ROOT
    / "artifacts"
    / "monitoring"
)

METRICS_DIR = (
    ROOT
    / "artifacts"
    / "metrics"
)

METADATA_DIR = (
    ROOT
    / "artifacts"
    / "metadata"
)

REGISTRY_DIR = (
    ROOT
    / "artifacts"
    / "registry"
)


# =========================================================
# LOAD MONITORING POPULATION FOR KDE
# =========================================================
#
# We intentionally DO NOT use:
#
#     data/raw/European_Bank.csv
#
# for this section.
#
# The monitoring population is:
#
#     artifacts/metrics/customer_risk_scoring.csv
#
# This file is already tracked in Git and contains the
# customer-level risk scoring results.
#
# Available numerical fields include:
#
# CreditScore
# Age
# Balance
# NumOfProducts
# EstimatedSalary
# CustomerValue
# ChurnProbability
# ExpectedLoss
# RetentionCost
# ExpectedSavedValue
# ROI
#
# Tenure and HasCrCard are NOT required here.
# =========================================================

reference = None
current = None

RISK_DATA_PATH = (
    METRICS_DIR
    / "customer_risk_scoring.csv"
)


if RISK_DATA_PATH.exists():

    try:

        data = pd.read_csv(
            RISK_DATA_PATH
        )

        # -------------------------------------------------
        # Numerical features actually available in
        # customer_risk_scoring.csv
        # -------------------------------------------------

        numerical_features = [
            "CreditScore",
            "Age",
            "Balance",
            "NumOfProducts",
            "EstimatedSalary",
            "CustomerValue",
            "ChurnProbability",
            "ExpectedLoss",
            "RetentionCost",
            "ExpectedSavedValue",
            "ROI",
        ]

        available_features = [
            feature
            for feature in numerical_features
            if feature in data.columns
        ]

        if available_features:

            # -------------------------------------------------
            # Deterministic 70/30 split
            # -------------------------------------------------

            reference = data.sample(
                frac=0.70,
                random_state=42,
            )

            current = data.drop(
                reference.index,
                errors="ignore",
            )

    except Exception as error:

        st.warning(
            f"Unable to load monitoring population: {error}"
        )


# =========================================================
# JSON LOADER
# =========================================================

def load_json(path):
    """
    Load a JSON file safely.
    """

    if not path.exists():
        return None

    try:

        with open(
            path,
            "r",
            encoding="utf-8",
        ) as file:

            return json.load(file)

    except Exception as error:

        st.warning(
            f"Unable to read {path.name}: {error}"
        )

        return None


# =========================================================
# LOAD MONITORING ARTIFACTS
# =========================================================


# ---------------------------------------------------------
# Overall monitoring status
# ---------------------------------------------------------

monitoring_status = load_json(
    METADATA_DIR
    / "monitoring_status.json"
)


# ---------------------------------------------------------
# Prediction monitoring
# ---------------------------------------------------------

prediction_summary = load_json(
    MONITORING_DIR
    / "prediction_monitoring_summary.json"
)


# ---------------------------------------------------------
# Prediction drift
# ---------------------------------------------------------

prediction_drift = load_json(
    METRICS_DIR
    / "prediction_drift_report.json"
)


# ---------------------------------------------------------
# Performance monitoring
# ---------------------------------------------------------

performance_status = load_json(
    METADATA_DIR
    / "performance_monitoring_status.json"
)

if performance_status is None:

    performance_status = load_json(
        MONITORING_DIR
        / "performance_monitoring_status.json"
    )


performance_alerts = load_json(
    METRICS_DIR
    / "performance_alerts.json"
)

if performance_alerts is None:

    performance_alerts = load_json(
        MONITORING_DIR
        / "performance_alerts.json"
    )


# ---------------------------------------------------------
# Model validation
# ---------------------------------------------------------

model_validation = load_json(
    REGISTRY_DIR
    / "model_validation.json"
)


# =========================================================
# HEADER
# =========================================================

st.title(
    "🛡️ MLOps Monitoring"
)

st.markdown(
    """
    Production monitoring for **data drift, feature drift,
    prediction behavior, model performance, and champion
    model validation**.
    """
)


# =========================================================
# OVERALL MONITORING STATUS
# =========================================================

st.subheader(
    "Monitoring Overview"
)

col1, col2, col3, col4 = st.columns(4)


# ---------------------------------------------------------
# Data Drift Status
# ---------------------------------------------------------

with col1:

    if monitoring_status:

        status = monitoring_status.get(
            "overall_status",
            "UNKNOWN",
        )

        status = str(status).upper()

        st.metric(
            "Data Drift",
            status,
        )

    else:

        st.metric(
            "Data Drift",
            "N/A",
        )


# ---------------------------------------------------------
# Total Predictions
# ---------------------------------------------------------

with col2:

    if prediction_summary:

        total_predictions = (
            prediction_summary.get(
                "total_predictions",
                0,
            )
        )

        st.metric(
            "Predictions",
            total_predictions,
        )

    else:

        st.metric(
            "Predictions",
            "N/A",
        )


# ---------------------------------------------------------
# Average Churn Probability
# ---------------------------------------------------------

with col3:

    if prediction_summary:

        probability = (
            prediction_summary.get(
                "average_churn_probability"
            )
        )

        if probability is not None:

            st.metric(
                "Avg Churn Probability",
                f"{float(probability):.2%}",
            )

        else:

            st.metric(
                "Avg Churn Probability",
                "N/A",
            )

    else:

        st.metric(
            "Avg Churn Probability",
            "N/A",
        )


# ---------------------------------------------------------
# Model Validation
# ---------------------------------------------------------

with col4:

    if model_validation:

        validation_status = (
            model_validation.get(
                "validation_status",
                model_validation.get(
                    "status",
                    "UNKNOWN",
                ),
            )
        )

        st.metric(
            "Model Validation",
            str(validation_status).upper(),
        )

    else:

        st.metric(
            "Model Validation",
            "N/A",
        )


# =========================================================
# DATA DRIFT STATUS
# =========================================================

st.subheader(
    "📊 Data Drift"
)


if monitoring_status:

    overall_status = monitoring_status.get(
        "overall_status",
        "UNKNOWN",
    )

    overall_status = str(
        overall_status
    ).upper()

    if overall_status == "STABLE":

        st.success(
            "✓ Data Drift Status: STABLE"
        )

    elif overall_status == "WARNING":

        st.warning(
            "⚠ Data Drift Status: WARNING"
        )

    elif overall_status == "ALERT":

        st.error(
            "🚨 Data Drift Status: ALERT"
        )

    else:

        st.info(
            f"Data Drift Status: {overall_status}"
        )


    # -----------------------------------------------------
    # Drift counts
    # -----------------------------------------------------

    col1, col2, col3 = st.columns(3)


    with col1:

        st.metric(
            "Significant Feature Drift",
            monitoring_status.get(
                "significant_feature_drift_count",
                0,
            ),
        )


    with col2:

        st.metric(
            "Moderate Feature Drift",
            monitoring_status.get(
                "moderate_feature_drift_count",
                0,
            ),
        )


    with col3:

        prediction_drift_flag = (
            monitoring_status.get(
                "prediction_drift_detected",
                False,
            )
        )

        st.metric(
            "Prediction Drift",
            "YES"
            if prediction_drift_flag
            else "NO",
        )


else:

    st.info(
        "Data drift monitoring status is unavailable."
    )


# =========================================================
# FEATURE DRIFT
# =========================================================

st.subheader(
    "🔍 Feature Drift"
)


feature_drift_file = (
    METRICS_DIR
    / "feature_drift_report.csv"
)


if feature_drift_file.exists():

    try:

        drift_df = pd.read_csv(
            feature_drift_file
        )


        # -------------------------------------------------
        # Summary
        # -------------------------------------------------

        if not drift_df.empty:

            col1, col2, col3 = st.columns(3)


            with col1:

                if "PSIStatus" in drift_df.columns:

                    stable_count = (
                        drift_df["PSIStatus"]
                        .astype(str)
                        .str.lower()
                        .eq("stable")
                        .sum()
                    )

                    st.metric(
                        "Stable Features",
                        int(stable_count),
                    )


            with col2:

                if "PSIStatus" in drift_df.columns:

                    drift_count = (
                        drift_df["PSIStatus"]
                        .astype(str)
                        .str.lower()
                        .ne("stable")
                        .sum()
                    )

                    st.metric(
                        "PSI Drift",
                        int(drift_count),
                    )


            with col3:

                if "KSDrift" in drift_df.columns:

                    ks_drift_count = (
                        drift_df["KSDrift"]
                        .astype(str)
                        .str.contains(
                            "drift",
                            case=False,
                            na=False,
                        )
                        .sum()
                    )

                    st.metric(
                        "KS Drift",
                        int(ks_drift_count),
                    )


        # -------------------------------------------------
        # Feature Drift Table
        # -------------------------------------------------

        st.dataframe(
            drift_df,
            use_container_width=True,
            hide_index=True,
        )


    except Exception as error:

        st.error(
            f"Unable to load feature drift report: {error}"
        )


else:

    st.info(
        "Feature drift report unavailable."
    )

    st.caption(
        "Expected: artifacts/metrics/feature_drift_report.csv"
    )


# =========================================================
# FEATURE DISTRIBUTION — KDE
# =========================================================

st.subheader(
    "📈 Feature Distribution — KDE"
)

st.caption(
    "Reference and current distributions are generated "
    "from the tracked customer risk scoring population."
)


if reference is not None and current is not None:

    numerical_features = [
        "CreditScore",
        "Age",
        "Balance",
        "NumOfProducts",
        "EstimatedSalary",
        "CustomerValue",
        "ChurnProbability",
        "ExpectedLoss",
        "RetentionCost",
        "ExpectedSavedValue",
        "ROI",
    ]


    available_features = [
        feature
        for feature in numerical_features
        if feature in reference.columns
        and feature in current.columns
    ]


    if not available_features:

        st.info(
            "No numerical monitoring features are available."
        )

    else:

        selected_feature = st.selectbox(
            "Select Feature",
            available_features,
        )


        ref_values = (
            pd.to_numeric(
                reference[selected_feature],
                errors="coerce",
            )
            .replace(
                [np.inf, -np.inf],
                np.nan,
            )
            .dropna()
        )


        current_values = (
            pd.to_numeric(
                current[selected_feature],
                errors="coerce",
            )
            .replace(
                [np.inf, -np.inf],
                np.nan,
            )
            .dropna()
        )


        # -------------------------------------------------
        # Validate KDE input
        # -------------------------------------------------

        if (
            len(ref_values) < 2
            or len(current_values) < 2
            or ref_values.nunique() < 2
            or current_values.nunique() < 2
        ):

            st.info(
                f"Insufficient variation in "
                f"{selected_feature} for KDE."
            )

        else:

            try:

                # -------------------------------------------------
                # KDE
                # -------------------------------------------------

                ref_kde = gaussian_kde(
                    ref_values.to_numpy()
                )

                current_kde = gaussian_kde(
                    current_values.to_numpy()
                )


                x_min = min(
                    ref_values.min(),
                    current_values.min(),
                )

                x_max = max(
                    ref_values.max(),
                    current_values.max(),
                )


                # Avoid zero-width x-axis

                if x_min == x_max:

                    x_min -= 1
                    x_max += 1


                x = np.linspace(
                    x_min,
                    x_max,
                    500,
                )


                fig = go.Figure()


                # -------------------------------------------------
                # Reference distribution
                # -------------------------------------------------

                fig.add_trace(
                    go.Scatter(
                        x=x,
                        y=ref_kde(x),
                        mode="lines",
                        name="Reference",
                        fill="tozeroy",
                        opacity=0.35,
                    )
                )


                # -------------------------------------------------
                # Current distribution
                # -------------------------------------------------

                fig.add_trace(
                    go.Scatter(
                        x=x,
                        y=current_kde(x),
                        mode="lines",
                        name="Current",
                        fill="tozeroy",
                        opacity=0.35,
                    )
                )


                fig.update_layout(
                    title=(
                        f"{selected_feature} — "
                        "Reference vs Current KDE"
                    ),
                    xaxis_title=selected_feature,
                    yaxis_title="Density",
                    template="plotly_dark",
                    height=450,
                    hovermode="x unified",
                )


                st.plotly_chart(
                    fig,
                    use_container_width=True,
                )


            except Exception as error:

                st.warning(
                    f"KDE could not be calculated "
                    f"for {selected_feature}: {error}"
                )


else:

    st.info(
        "Reference/current monitoring data is unavailable."
    )

    st.caption(
        "KDE source: artifacts/metrics/customer_risk_scoring.csv"
    )


# =========================================================
# PREDICTION DRIFT
# =========================================================

st.subheader(
    "🎯 Prediction Drift"
)


if prediction_drift:

    # -----------------------------------------------------
    # Extract values
    # -----------------------------------------------------

    prediction_psi = prediction_drift.get(
        "PredictionPSI"
    )

    prediction_ks = prediction_drift.get(
        "PredictionKSPValue"
    )

    prediction_status = prediction_drift.get(
        "PredictionStatus",
        prediction_drift.get(
            "status",
            "UNKNOWN",
        ),
    )


    # -----------------------------------------------------
    # Metrics
    # -----------------------------------------------------

    col1, col2, col3 = st.columns(3)


    with col1:

        if prediction_psi is not None:

            st.metric(
                "Prediction PSI",
                f"{float(prediction_psi):.4f}",
            )

        else:

            st.metric(
                "Prediction PSI",
                "N/A",
            )


    with col2:

        if prediction_ks is not None:

            st.metric(
                "KS p-value",
                f"{float(prediction_ks):.4f}",
            )

        else:

            st.metric(
                "KS p-value",
                "N/A",
            )


    with col3:

        st.metric(
            "Prediction Status",
            str(
                prediction_status
            ).upper(),
        )


    # -----------------------------------------------------
    # Status message
    # -----------------------------------------------------

    if str(prediction_status).upper() == "STABLE":

        st.success(
            "✓ Prediction distribution is stable."
        )

    elif str(prediction_status).upper() in [
        "WARNING",
        "MODERATE",
    ]:

        st.warning(
            "⚠ Prediction drift requires attention."
        )

    elif str(prediction_status).upper() == "ALERT":

        st.error(
            "🚨 Significant prediction drift detected."
        )

    else:

        st.info(
            f"Prediction status: {prediction_status}"
        )


    # -----------------------------------------------------
    # Raw report
    # -----------------------------------------------------

    with st.expander(
        "View Prediction Drift Details"
    ):

        st.json(
            prediction_drift
        )


else:

    st.info(
        "Prediction drift report unavailable."
    )

    st.caption(
        "Expected: artifacts/metrics/prediction_drift_report.json"
    )


# =========================================================
# PREDICTION MONITORING
# =========================================================

st.subheader(
    "📈 Prediction Monitoring"
)


if prediction_summary:

    col1, col2 = st.columns(2)


    # -----------------------------------------------------
    # Predicted Churn Rate
    # -----------------------------------------------------

    with col1:

        churn_rate = (
            prediction_summary.get(
                "predicted_churn_rate"
            )
        )

        if churn_rate is not None:

            st.metric(
                "Predicted Churn Rate",
                f"{float(churn_rate):.2%}",
            )

        else:

            st.metric(
                "Predicted Churn Rate",
                "N/A",
            )


    # -----------------------------------------------------
    # Risk Distribution
    # -----------------------------------------------------

    with col2:

        risk_distribution = (
            prediction_summary.get(
                "risk_distribution",
                {},
            )
        )

        if risk_distribution:

            risk_df = pd.DataFrame(
                {
                    "Risk": list(
                        risk_distribution.keys()
                    ),
                    "Predictions": list(
                        risk_distribution.values()
                    ),
                }
            )

            st.bar_chart(
                risk_df.set_index(
                    "Risk"
                )
            )

        else:

            st.info(
                "Risk distribution unavailable."
            )


else:

    st.info(
        "Prediction monitoring data unavailable."
    )


# =========================================================
# MODEL PERFORMANCE
# =========================================================

st.subheader(
    "📉 Model Performance Monitoring"
)


performance_file = (
    METRICS_DIR
    / "performance_comparison.csv"
)


if performance_file.exists():

    try:

        performance_df = pd.read_csv(
            performance_file
        )


        st.dataframe(
            performance_df,
            use_container_width=True,
            hide_index=True,
        )


    except Exception as error:

        st.error(
            f"Unable to load performance report: {error}"
        )


else:

    st.info(
        "Performance comparison report unavailable."
    )

    st.caption(
        "Expected: artifacts/metrics/performance_comparison.csv"
    )


# =========================================================
# PERFORMANCE STATUS
# =========================================================

if performance_status:

    status = performance_status.get(
        "status",
        performance_status.get(
            "monitoring_status",
            performance_status.get(
                "overall_status",
                "UNKNOWN",
            ),
        ),
    )


    status = str(status).upper()


    if status == "STABLE":

        st.success(
            f"✓ Performance Monitoring: {status}"
        )

    elif status in [
        "WARNING",
        "MODERATE",
    ]:

        st.warning(
            f"⚠ Performance Monitoring: {status}"
        )

    elif status == "ALERT":

        st.error(
            f"🚨 Performance Monitoring: {status}"
        )

    else:

        st.info(
            f"Performance Monitoring: {status}"
        )


# =========================================================
# PERFORMANCE ALERTS
# =========================================================

if performance_alerts:

    with st.expander(
        "View Performance Alerts"
    ):

        st.json(
            performance_alerts
        )


# =========================================================
# CHAMPION MODEL VALIDATION
# =========================================================

st.subheader(
    "🏆 Champion Model Validation"
)


if model_validation:

    validation_status = model_validation.get(
        "validation_status",
        model_validation.get(
            "status",
            "UNKNOWN",
        ),
    )


    if str(validation_status).upper() in [
        "PASSED",
        "PASS",
        "SUCCESS",
    ]:

        st.success(
            "✓ Champion model validation PASSED"
        )

    else:

        st.warning(
            f"Champion model validation: "
            f"{validation_status}"
        )


    with st.expander(
        "View Validation Details"
    ):

        st.json(
            model_validation
        )


else:

    st.warning(
        "Model validation artifact unavailable."
    )

    st.caption(
        "Expected: artifacts/registry/model_validation.json"
    )


# =========================================================
# EVIDENTLY REPORT
# =========================================================

st.subheader(
    "🧪 Evidently Data Drift Report"
)


evidently_html = (
    MONITORING_DIR
    / "data_drift_report.html"
)


evidently_json = (
    MONITORING_DIR
    / "data_drift_report.json"
)


if evidently_html.exists():

    st.success(
        "✓ Evidently HTML report generated"
    )

    st.caption(
        str(evidently_html)
    )


    # -----------------------------------------------------
    # Download button
    # -----------------------------------------------------

    with open(
        evidently_html,
        "rb",
    ) as file:

        st.download_button(
            label="Download Evidently HTML Report",
            data=file,
            file_name="data_drift_report.html",
            mime="text/html",
        )


else:

    st.warning(
        "Evidently HTML report unavailable."
    )


if evidently_json.exists():

    with st.expander(
        "View Evidently JSON Report"
    ):

        evidently_data = load_json(
            evidently_json
        )

        if evidently_data:

            st.json(
                evidently_data
            )


# =========================================================
# MONITORING ARTIFACT LOCATIONS
# =========================================================

with st.expander(
    "📁 Monitoring Artifact Locations"
):

    st.write(
        "**Monitoring:**",
        str(MONITORING_DIR),
    )

    st.write(
        "**Metrics:**",
        str(METRICS_DIR),
    )

    st.write(
        "**Metadata:**",
        str(METADATA_DIR),
    )

    st.write(
        "**Registry:**",
        str(REGISTRY_DIR),
    )

    st.write(
        "**KDE Data:**",
        str(RISK_DATA_PATH),
    )