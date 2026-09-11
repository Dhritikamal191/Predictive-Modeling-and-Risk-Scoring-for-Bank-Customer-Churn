"""
Probability Calibration Analysis

Evaluates whether churn probabilities are reliable enough
to support downstream risk and financial calculations.
"""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    brier_score_loss,
    log_loss,
)
from sklearn.calibration import calibration_curve


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

MODEL_DIR = PROJECT_ROOT / "artifacts" / "models"
METRICS_DIR = PROJECT_ROOT / "artifacts" / "metrics"
DOCS_DIR = PROJECT_ROOT / "docs"

METRICS_DIR.mkdir(parents=True, exist_ok=True)
DOCS_DIR.mkdir(parents=True, exist_ok=True)


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
# DATA
# ============================================================

def prepare_data():

    df = pd.read_csv(DATA_PATH)

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

    _, X_test, _, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    return X_test, y_test


# ============================================================
# MODELS
# ============================================================

MODEL_FILES = {
    "Logistic Regression":
        "logistic_regression.pkl",

    "Random Forest":
        "random_forest.pkl",

    "Gradient Boosting":
        "gradient_boosting.pkl",

    "XGBoost":
        "xgboost.pkl",
}


# ============================================================
# CALIBRATION
# ============================================================

def evaluate_calibration(X_test, y_test):

    print("\n" + "=" * 65)
    print("BANK CHURN — PROBABILITY CALIBRATION")
    print("=" * 65)

    results = []

    calibration_data = {}

    for name, filename in MODEL_FILES.items():

        print(f"\nEvaluating: {name}")

        model = joblib.load(
            MODEL_DIR / filename
        )

        probabilities = model.predict_proba(
            X_test
        )[:, 1]

        brier = brier_score_loss(
            y_test,
            probabilities,
        )

        loss = log_loss(
            y_test,
            probabilities,
        )

        fraction_positive, mean_predicted = (
            calibration_curve(
                y_test,
                probabilities,
                n_bins=10,
                strategy="quantile",
            )
        )

        calibration_data[name] = (
            mean_predicted,
            fraction_positive,
        )

        results.append(
            {
                "Model": name,
                "BrierScore": brier,
                "LogLoss": loss,
            }
        )

        print(
            f"  Brier Score: {brier:.4f}"
        )

        print(
            f"  Log Loss:    {loss:.4f}"
        )

    results_df = pd.DataFrame(results)

    results_df = results_df.sort_values(
        "BrierScore"
    )

    return results_df, calibration_data


# ============================================================
# SAVE CALIBRATION CURVE
# ============================================================

def save_calibration_plot(calibration_data):

    print("\nCreating calibration plot...")

    plt.figure(figsize=(8, 6))

    for name, values in calibration_data.items():

        mean_predicted, fraction_positive = values

        plt.plot(
            mean_predicted,
            fraction_positive,
            marker="o",
            label=name,
        )

    plt.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        label="Perfect Calibration",
    )

    plt.xlabel(
        "Mean Predicted Probability"
    )

    plt.ylabel(
        "Observed Churn Rate"
    )

    plt.title(
        "Churn Probability Calibration"
    )

    plt.legend()

    plt.tight_layout()

    output_path = (
        DOCS_DIR
        / "calibration_curve.png"
    )

    plt.savefig(
        output_path,
        dpi=200,
    )

    plt.close()

    print(
        f"✓ Saved: {output_path}"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    X_test, y_test = prepare_data()

    results_df, calibration_data = (
        evaluate_calibration(
            X_test,
            y_test,
        )
    )

    print("\n" + "-" * 65)
    print("CALIBRATION COMPARISON")
    print("-" * 65)

    print(
        results_df.to_string(
            index=False
        )
    )

    results_df.to_csv(
        METRICS_DIR
        / "calibration_metrics.csv",
        index=False,
    )

    save_calibration_plot(
        calibration_data
    )

    print("\n" + "=" * 65)
    print("CALIBRATION ANALYSIS COMPLETE")
    print("=" * 65)


if __name__ == "__main__":
    main()