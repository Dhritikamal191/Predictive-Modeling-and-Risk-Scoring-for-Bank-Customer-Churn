"""
Bank Churn — MLOps Model Monitoring Dashboard

Displays:
- Model performance monitoring
- Feature drift
- Prediction drift
- Monitoring alerts
- Champion model status
"""

from pathlib import Path
import json

import pandas as pd
import streamlit as st


ROOT = Path(__file__).resolve().parents[2]

METRICS_DIR = ROOT / "artifacts" / "metrics"
METADATA_DIR = ROOT / "artifacts" / "metadata"


def load_json(path):
    """Safely load a JSON file."""

    if not path.exists():
        return {}

    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def show_model_monitoring():
    """Render the MLOps monitoring dashboard."""

    st.header("Model Monitoring")

    st.caption(
        "Production-style monitoring of model performance, "
        "data drift and prediction drift."
    )

    # =====================================================
    # LOAD MONITORING ARTIFACTS
    # =====================================================

    performance_file = METRICS_DIR / "performance_comparison.csv"
    alerts_file = METRICS_DIR / "performance_alerts.json"
    performance_status_file = (
        METADATA_DIR / "performance_monitoring_status.json"
    )
    monitoring_status_file = (
        METADATA_DIR / "monitoring_status.json"
    )
    feature_drift_file = (
        METRICS_DIR / "feature_drift_report.csv"
    )
    prediction_drift_file = (
        METRICS_DIR / "prediction_drift_report.json"
    )

    performance_df = pd.DataFrame()

    if performance_file.exists():
        try:
            performance_df = pd.read_csv(performance_file)
        except Exception as e:
            st.warning(f"Unable to read performance report: {e}")

    alerts = load_json(alerts_file)
    performance_status = load_json(
        performance_status_file
    )
    monitoring_status = load_json(
        monitoring_status_file
    )
    prediction_drift = load_json(
        prediction_drift_file
    )

    # =====================================================
    # STATUS
    # =====================================================

    status = (
        performance_status.get("status")
        or monitoring_status.get("status")
        or "UNKNOWN"
    )

    status_upper = str(status).upper()

    if status_upper == "STABLE":
        st.success("MONITORING STATUS: STABLE")

    elif status_upper in ["WARNING", "WARN"]:
        st.warning("MONITORING STATUS: WARNING")

    elif status_upper in ["CRITICAL", "ALERT"]:
        st.error("MONITORING STATUS: CRITICAL")

    else:
        st.info(f"MONITORING STATUS: {status_upper}")

    # =====================================================
    # KPI CARDS
    # =====================================================

    champion = (
        performance_status.get("model")
        or performance_status.get("model_name")
        or "Gradient Boosting"
    )

    alert_count = 0

    if isinstance(alerts, list):
        alert_count = len(alerts)

    elif isinstance(alerts, dict):
        alert_count = len(alerts.get("alerts", []))

    drift_status = (
        monitoring_status.get("status")
        or prediction_drift.get("status")
        or "Unknown"
    )

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Champion Model",
            champion,
        )

    with col2:
        st.metric(
            "Performance Alerts",
            alert_count,
        )

    with col3:
        st.metric(
            "Drift Status",
            str(drift_status).upper(),
        )

    with col4:
        st.metric(
            "Monitoring",
            status_upper,
        )

    st.divider()

    # =====================================================
    # MODEL PERFORMANCE
    # =====================================================

    st.subheader("Model Performance")

    if performance_df.empty:
        st.info(
            "No performance comparison report found."
        )

    else:

        display_columns = [
            "Metric",
            "Reference",
            "Current",
            "AbsoluteChange",
            "PercentageChange",
        ]

        available_columns = [
            column
            for column in display_columns
            if column in performance_df.columns
        ]

        if available_columns:
            st.dataframe(
                performance_df[available_columns],
                use_container_width=True,
                hide_index=True,
            )

        # -------------------------------------------------
        # PERFORMANCE CHART
        # -------------------------------------------------

        if {
            "Metric",
            "Reference",
            "Current",
        }.issubset(performance_df.columns):

            chart_df = performance_df[
                ["Metric", "Reference", "Current"]
            ].copy()

            chart_df = chart_df.set_index("Metric")

            st.subheader(
                "Reference vs Current Performance"
            )

            st.bar_chart(
                chart_df,
                use_container_width=True,
            )

    # =====================================================
    # ALERTS
    # =====================================================

    st.divider()

    st.subheader("Performance Alerts")

    alert_items = []

    if isinstance(alerts, list):
        alert_items = alerts

    elif isinstance(alerts, dict):
        alert_items = alerts.get(
            "alerts",
            []
        )

    if not alert_items:
        st.success(
            "✓ No performance alerts detected."
        )

    else:

        for alert in alert_items:

            if isinstance(alert, dict):

                metric = alert.get(
                    "metric",
                    "Unknown metric"
                )

                message = alert.get(
                    "message",
                    str(alert)
                )

                severity = str(
                    alert.get(
                        "severity",
                        "warning"
                    )
                ).lower()

                if severity == "critical":
                    st.error(
                        f"🔴 {metric}: {message}"
                    )

                else:
                    st.warning(
                        f"🟡 {metric}: {message}"
                    )

            else:
                st.warning(str(alert))

    # =====================================================
    # FEATURE DRIFT
    # =====================================================

    st.divider()

    st.subheader("Feature Drift")

    if feature_drift_file.exists():

        try:

            drift_df = pd.read_csv(
                feature_drift_file
            )

            st.dataframe(
                drift_df,
                use_container_width=True,
                hide_index=True,
            )

        except Exception as e:

            st.warning(
                f"Unable to read feature drift report: {e}"
            )

    else:

        st.info(
            "Feature drift report not available."
        )

    # =====================================================
    # PREDICTION DRIFT
    # =====================================================

    st.divider()

    st.subheader("Prediction Drift")

    if prediction_drift:

        if isinstance(
            prediction_drift,
            dict
        ):

            prediction_data = {
                str(k): str(v)
                for k, v in prediction_drift.items()
            }

            prediction_df = pd.DataFrame(
                prediction_data.items(),
                columns=[
                    "Metric",
                    "Value",
                ],
            )

            st.dataframe(
                prediction_df,
                use_container_width=True,
                hide_index=True,
            )

        else:

            st.write(prediction_drift)

    else:

        st.info(
            "Prediction drift report not available."
        )

    # =====================================================
    # MONITORING DETAILS
    # =====================================================

    st.divider()

    st.subheader("Monitoring Details")

    details = {}

    if isinstance(
        performance_status,
        dict,
    ):
        details.update(
            performance_status
        )

    if isinstance(
        monitoring_status,
        dict,
    ):
        details.update(
            monitoring_status
        )

    if details:

        details_df = pd.DataFrame(
            details.items(),
            columns=[
                "Property",
                "Value",
            ],
        )

        st.dataframe(
            details_df,
            use_container_width=True,
            hide_index=True,
        )

    else:

        st.info(
            "No monitoring metadata available."
        )
