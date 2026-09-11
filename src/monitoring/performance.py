"""
Bank Churn — Model Performance Monitoring

Compares model performance between a reference dataset
and a current dataset.

Metrics:
- Accuracy
- Precision
- Recall
- F1
- ROC-AUC
- PR-AUC
- Log Loss
- Brier Score

Also generates an automated monitoring status.
"""

from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    log_loss,
    brier_score_loss,
)


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "European_Bank.csv"
)

MODEL_PATH = (
    PROJECT_ROOT
    / "artifacts"
    / "models"
    / "gradient_boosting.pkl"
)

METRICS_DIR = (
    PROJECT_ROOT
    / "artifacts"
    / "metrics"
)

METADATA_DIR = (
    PROJECT_ROOT
    / "artifacts"
    / "metadata"
)

METRICS_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

METADATA_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# THRESHOLD
# ============================================================

DEFAULT_THRESHOLD = 0.50


# ============================================================
# FEATURE ENGINEERING
# ============================================================

def create_features(df):

    data = df.copy()

    data["BalanceSalaryRatio"] = np.where(
        data["EstimatedSalary"] > 0,
        data["Balance"] / data["EstimatedSalary"],
        0,
    )

    data["CustomerValue"] = (
        data["Balance"]
        + data["EstimatedSalary"]
    )

    data["ProductDensity"] = (
        data["NumOfProducts"]
        / data["Tenure"].clip(lower=1)
    )

    data["EngagementScore"] = (
        data["IsActiveMember"]
        + data["HasCrCard"]
        + (data["NumOfProducts"] > 1).astype(int)
    )

    data["AgeTenureInteraction"] = (
        data["Age"] * data["Tenure"]
    )

    data["HasBalance"] = (
        data["Balance"] > 0
    ).astype(int)

    return data


# ============================================================
# PREPARE DATA
# ============================================================

def prepare_data(df):

    df = create_features(df)

    X = df.drop(
        columns=[
            "Exited",
            "Year",
            "CustomerId",
            "Surname",
        ]
    )

    y = df["Exited"]

    return X, y


# ============================================================
# PERFORMANCE METRICS
# ============================================================

def calculate_metrics(
    model,
    X,
    y,
    threshold=DEFAULT_THRESHOLD,
):

    probabilities = model.predict_proba(
        X
    )[:, 1]

    predictions = (
        probabilities >= threshold
    ).astype(int)

    metrics = {
        "Accuracy": accuracy_score(
            y,
            predictions,
        ),

        "Precision": precision_score(
            y,
            predictions,
            zero_division=0,
        ),

        "Recall": recall_score(
            y,
            predictions,
            zero_division=0,
        ),

        "F1": f1_score(
            y,
            predictions,
            zero_division=0,
        ),

        "ROC_AUC": roc_auc_score(
            y,
            probabilities,
        ),

        "PR_AUC": average_precision_score(
            y,
            probabilities,
        ),

        "LogLoss": log_loss(
            y,
            probabilities,
        ),

        "BrierScore": brier_score_loss(
            y,
            probabilities,
        ),
    }

    return metrics


# ============================================================
# PERFORMANCE CHANGE
# ============================================================

def calculate_change(
    reference_metrics,
    current_metrics,
):

    rows = []

    for metric in reference_metrics:

        reference_value = (
            reference_metrics[metric]
        )

        current_value = (
            current_metrics[metric]
        )

        absolute_change = (
            current_value
            - reference_value
        )

        if reference_value != 0:

            percentage_change = (
                absolute_change
                / abs(reference_value)
                * 100
            )

        else:

            percentage_change = np.nan

        rows.append(
            {
                "Metric": metric,
                "Reference": reference_value,
                "Current": current_value,
                "AbsoluteChange": absolute_change,
                "PercentageChange": percentage_change,
            }
        )

    return pd.DataFrame(rows)


# ============================================================
# ALERT RULES
# ============================================================

def evaluate_alerts(
    comparison,
):

    alerts = []

    for _, row in comparison.iterrows():

        metric = row["Metric"]
        change = row["AbsoluteChange"]

        # Metrics where lower is worse
        higher_is_better = metric not in [
            "LogLoss",
            "BrierScore",
        ]

        if higher_is_better:

            if change <= -0.10:

                alerts.append(
                    {
                        "Metric": metric,
                        "Severity": "CRITICAL",
                        "Message": (
                            f"{metric} dropped by "
                            f"{abs(change):.4f}"
                        ),
                    }
                )

            elif change <= -0.05:

                alerts.append(
                    {
                        "Metric": metric,
                        "Severity": "WARNING",
                        "Message": (
                            f"{metric} dropped by "
                            f"{abs(change):.4f}"
                        ),
                    }
                )

        else:

            if change >= 0.10:

                alerts.append(
                    {
                        "Metric": metric,
                        "Severity": "CRITICAL",
                        "Message": (
                            f"{metric} increased by "
                            f"{change:.4f}"
                        ),
                    }
                )

            elif change >= 0.05:

                alerts.append(
                    {
                        "Metric": metric,
                        "Severity": "WARNING",
                        "Message": (
                            f"{metric} increased by "
                            f"{change:.4f}"
                        ),
                    }
                )

    if any(
        alert["Severity"] == "CRITICAL"
        for alert in alerts
    ):

        overall_status = "RETRAIN_RECOMMENDED"

    elif len(alerts) > 0:

        overall_status = "WARNING"

    else:

        overall_status = "STABLE"

    return alerts, overall_status


# ============================================================
# MAIN
# ============================================================

def main():

    print("\n" + "=" * 65)
    print("BANK CHURN — MODEL PERFORMANCE MONITORING")
    print("=" * 65)

    # --------------------------------------------------------
    # 1. Load
    # --------------------------------------------------------

    print(
        "\n[1/6] Loading model and dataset..."
    )

    df = pd.read_csv(
        DATA_PATH
    )

    model = joblib.load(
        MODEL_PATH
    )

    print(
        f"Dataset rows: {len(df):,}"
    )

    print(
        "✓ Gradient Boosting model loaded"
    )

    # --------------------------------------------------------
    # 2. Split reference/current
    # --------------------------------------------------------

    print(
        "\n[2/6] Creating reference/current populations..."
    )

    reference = df.sample(
        frac=0.70,
        random_state=42,
    )

    current = df.drop(
        reference.index
    )

    print(
        f"Reference rows: {len(reference):,}"
    )

    print(
        f"Current rows:   {len(current):,}"
    )

    # --------------------------------------------------------
    # 3. Prepare
    # --------------------------------------------------------

    print(
        "\n[3/6] Preparing features..."
    )

    X_reference, y_reference = (
        prepare_data(reference)
    )

    X_current, y_current = (
        prepare_data(current)
    )

    # --------------------------------------------------------
    # 4. Calculate
    # --------------------------------------------------------

    print(
        "\n[4/6] Calculating performance..."
    )

    reference_metrics = calculate_metrics(
        model,
        X_reference,
        y_reference,
    )

    current_metrics = calculate_metrics(
        model,
        X_current,
        y_current,
    )

    comparison = calculate_change(
        reference_metrics,
        current_metrics,
    )

    print("\nPerformance comparison:")

    print(
        comparison.to_string(
            index=False
        )
    )

    # --------------------------------------------------------
    # 5. Alerts
    # --------------------------------------------------------

    print(
        "\n[5/6] Evaluating monitoring alerts..."
    )

    alerts, overall_status = (
        evaluate_alerts(
            comparison
        )
    )

    if alerts:

        for alert in alerts:

            print(
                f"  [{alert['Severity']}] "
                f"{alert['Message']}"
            )

    else:

        print(
            "  ✓ No performance alerts"
        )

    # --------------------------------------------------------
    # 6. Save
    # --------------------------------------------------------

    print(
        "\n[6/6] Saving monitoring artifacts..."
    )

    comparison.to_csv(
        METRICS_DIR
        / "performance_comparison.csv",
        index=False,
    )

    with open(
        METRICS_DIR
        / "performance_alerts.json",
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            alerts,
            f,
            indent=4,
        )

    monitoring_status = {
        "model": "Gradient Boosting",
        "status": overall_status,
        "reference_rows": len(reference),
        "current_rows": len(current),
        "alerts": len(alerts),
        "reference_metrics": reference_metrics,
        "current_metrics": current_metrics,
    }

    with open(
        METADATA_DIR
        / "performance_monitoring_status.json",
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            monitoring_status,
            f,
            indent=4,
        )

    print(
        "\n" + "-" * 65
    )

    print(
        f"MONITORING STATUS: {overall_status}"
    )

    print(
        "-" * 65
    )

    print(
        "\nSaved:"
    )

    print(
        "   ✓ performance_comparison.csv"
    )

    print(
        "   ✓ performance_alerts.json"
    )

    print(
        "   ✓ performance_monitoring_status.json"
    )

    print(
        "\n" + "=" * 65
    )

    print(
        "PERFORMANCE MONITORING COMPLETE"
    )

    print(
        "=" * 65
    )


if __name__ == "__main__":
    main()