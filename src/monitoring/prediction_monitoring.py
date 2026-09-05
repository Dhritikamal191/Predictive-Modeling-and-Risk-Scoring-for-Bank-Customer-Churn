"""
Bank Churn — Prediction Monitoring

Monitors production prediction outputs from predictions.jsonl.
"""

from pathlib import Path
import json
from datetime import datetime, timezone

import pandas as pd
import matplotlib.pyplot as plt


# ---------------------------------------------------------
# Project paths
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

PREDICTION_LOG = (
    PROJECT_ROOT
    / "artifacts"
    / "monitoring"
    / "predictions.jsonl"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "artifacts"
    / "monitoring"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ---------------------------------------------------------
# Load prediction logs
# ---------------------------------------------------------

def load_predictions():

    if not PREDICTION_LOG.exists():
        raise FileNotFoundError(
            f"Prediction log not found:\n{PREDICTION_LOG}"
        )

    records = []

    with open(
        PREDICTION_LOG,
        "r",
        encoding="utf-8",
    ) as file:

        for line in file:

            line = line.strip()

            if line:
                records.append(
                    json.loads(line)
                )

    if not records:
        raise ValueError(
            "Prediction log is empty."
        )

    df = pd.DataFrame(records)

    df["timestamp"] = pd.to_datetime(
        df["timestamp"],
        utc=True,
    )

    return df


# ---------------------------------------------------------
# Calculate monitoring statistics
# ---------------------------------------------------------

def calculate_metrics(df):

    total_predictions = len(df)

    average_probability = (
        df["churn_probability"].mean()
    )

    predicted_churn_rate = (
        df["churn_prediction"].mean()
    )

    risk_distribution = (
        df["risk_category"]
        .value_counts()
        .to_dict()
    )

    return {
        "total_predictions": total_predictions,
        "average_churn_probability": round(
            float(average_probability),
            6,
        ),
        "predicted_churn_rate": round(
            float(predicted_churn_rate),
            6,
        ),
        "risk_distribution": risk_distribution,
    }


# ---------------------------------------------------------
# Save monitoring summary
# ---------------------------------------------------------

def save_summary(metrics):

    output_file = (
        OUTPUT_DIR
        / "prediction_monitoring_summary.json"
    )

    with open(
        output_file,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            metrics,
            file,
            indent=4,
        )

    return output_file


# ---------------------------------------------------------
# Generate prediction probability chart
# ---------------------------------------------------------

def generate_probability_plot(df):

    output_file = (
        OUTPUT_DIR
        / "prediction_probability_distribution.png"
    )

    plt.figure(figsize=(10, 6))

    plt.hist(
        df["churn_probability"],
        bins=10,
        edgecolor="black",
    )

    plt.xlabel(
        "Predicted Churn Probability"
    )

    plt.ylabel(
        "Number of Predictions"
    )

    plt.title(
        "Bank Churn — Prediction Probability Distribution"
    )

    plt.tight_layout()

    plt.savefig(
        output_file,
        dpi=150,
    )

    plt.close()

    return output_file


# ---------------------------------------------------------
# Generate risk distribution chart
# ---------------------------------------------------------

def generate_risk_plot(df):

    output_file = (
        OUTPUT_DIR
        / "risk_category_distribution.png"
    )

    counts = (
        df["risk_category"]
        .value_counts()
        .reindex(
            [
                "Low",
                "Medium",
                "High",
                "Critical",
            ],
            fill_value=0,
        )
    )

    plt.figure(figsize=(10, 6))

    counts.plot(
        kind="bar",
    )

    plt.xlabel(
        "Risk Category"
    )

    plt.ylabel(
        "Number of Predictions"
    )

    plt.title(
        "Bank Churn — Prediction Risk Distribution"
    )

    plt.xticks(
        rotation=0
    )

    plt.tight_layout()

    plt.savefig(
        output_file,
        dpi=150,
    )

    plt.close()

    return output_file


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------

def main():

    print("=" * 60)
    print("BANK CHURN — PREDICTION MONITORING")
    print("=" * 60)

    print()
    print("Prediction log:")
    print(f"  {PREDICTION_LOG}")

    print()
    print("Loading predictions...")

    df = load_predictions()

    print(
        f"  Predictions loaded: {len(df)}"
    )

    print()
    print("Calculating monitoring metrics...")

    metrics = calculate_metrics(df)

    print()
    print("-" * 60)
    print("PREDICTION MONITORING SUMMARY")
    print("-" * 60)

    print(
        f"Total predictions: "
        f"{metrics['total_predictions']}"
    )

    print(
        f"Average churn probability: "
        f"{metrics['average_churn_probability']:.2%}"
    )

    print(
        f"Predicted churn rate: "
        f"{metrics['predicted_churn_rate']:.2%}"
    )

    print()
    print("Risk distribution:")

    for risk, count in metrics[
        "risk_distribution"
    ].items():

        print(
            f"  {risk}: {count}"
        )

    summary_file = save_summary(
        metrics
    )

    probability_plot = (
        generate_probability_plot(df)
    )

    risk_plot = generate_risk_plot(df)

    print()
    print("=" * 60)
    print("PREDICTION MONITORING COMPLETE")
    print("=" * 60)

    print()
    print("Summary:")
    print(f"  {summary_file}")

    print()
    print("Probability distribution:")
    print(f"  {probability_plot}")

    print()
    print("Risk distribution:")
    print(f"  {risk_plot}")

    print()


if __name__ == "__main__":
    main()