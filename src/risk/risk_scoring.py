"""
Bank Customer Risk Scoring & Financial Prioritization

Combines:
    - Churn probability
    - Customer value
    - Expected loss
    - Retention economics
    - Customer segment

Important:
Model probability is kept separate from business assumptions.
"""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd


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

SEGMENT_PATH = (
    PROJECT_ROOT
    / "artifacts"
    / "metrics"
    / "customer_segments.csv"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "artifacts"
    / "metrics"
    / "customer_risk_scoring.csv"
)

METADATA_PATH = (
    PROJECT_ROOT
    / "artifacts"
    / "metadata"
    / "risk_scoring_metadata.json"
)


# ============================================================
# BUSINESS ASSUMPTIONS
# ============================================================

# Estimated proportion of customer value that can be retained
# if a successful intervention is applied.
TREATMENT_EFFECTIVENESS = 0.30

# Estimated campaign/retention cost as a proportion
# of customer value.
RETENTION_COST_RATE = 0.05


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
        data["Balance"] +
        data["EstimatedSalary"]
    )

    data["ProductDensity"] = (
        data["NumOfProducts"] /
        data["Tenure"].clip(lower=1)
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
# RISK CATEGORY
# ============================================================

def assign_risk_category(probability):

    if probability < 0.20:
        return "Low Risk"

    elif probability < 0.40:
        return "Medium Risk"

    elif probability < 0.60:
        return "High Risk"

    else:
        return "Critical Risk"


# ============================================================
# VALUE CATEGORY
# ============================================================

def assign_value_category(value, q1, q2, q3):

    if value <= q1:
        return "Low Value"

    elif value <= q2:
        return "Medium Value"

    elif value <= q3:
        return "High Value"

    else:
        return "Very High Value"


# ============================================================
# PRIORITY
# ============================================================

def assign_priority(row):

    risk = row["RiskCategory"]
    value = row["ValueCategory"]

    if risk in ["High Risk", "Critical Risk"]:

        if value in ["High Value", "Very High Value"]:
            return "P1 — Immediate Retention"

        elif value == "Medium Value":
            return "P2 — High Priority"

        else:
            return "P3 — Targeted Retention"

    elif risk == "Medium Risk":

        if value in ["High Value", "Very High Value"]:
            return "P2 — High Priority"

        else:
            return "P3 — Targeted Retention"

    else:

        if value == "Very High Value":
            return "P3 — Relationship Management"

        else:
            return "P4 — Monitor"


# ============================================================
# MAIN
# ============================================================

def main():

    print("\n" + "=" * 65)
    print("BANK CHURN — RISK & VALUE SCORING")
    print("=" * 65)

    # --------------------------------------------------------
    # 1. Load customer data
    # --------------------------------------------------------

    print("\n[1/7] Loading customer data...")

    df = pd.read_csv(DATA_PATH)

    print(f"Customers: {len(df):,}")

    # --------------------------------------------------------
    # 2. Load trained model
    # --------------------------------------------------------

    print("\n[2/7] Loading churn model...")

    model = joblib.load(MODEL_PATH)

    print("✓ Gradient Boosting model loaded")

    # --------------------------------------------------------
    # 3. Prepare model features
    # --------------------------------------------------------

    print("\n[3/7] Preparing features...")

    data = create_features(df)

    X = data.drop(
        columns=[
            "Exited",
            "Year",
            "CustomerId",
            "Surname",
        ]
    )

    print(f"Model features: {X.shape[1]}")

    # --------------------------------------------------------
    # 4. Predict churn probability
    # --------------------------------------------------------

    print("\n[4/7] Calculating churn probability...")

    churn_probability = model.predict_proba(X)[:, 1]

    data["ChurnProbability"] = churn_probability

    data["RiskCategory"] = (
        data["ChurnProbability"]
        .apply(assign_risk_category)
    )

    # --------------------------------------------------------
    # 5. Financial calculations
    # --------------------------------------------------------

    print("\n[5/7] Calculating financial risk...")

    # Expected financial loss if the customer churns
    data["ExpectedLoss"] = (
        data["ChurnProbability"]
        * data["CustomerValue"]
    )

    # Estimated retention campaign cost
    data["RetentionCost"] = (
        data["CustomerValue"]
        * RETENTION_COST_RATE
    )

    # Expected amount of customer value saved
    data["ExpectedSavedValue"] = (
        data["ExpectedLoss"]
        * TREATMENT_EFFECTIVENESS
    )

    # ROI of intervention
    data["ROI"] = np.where(
        data["RetentionCost"] > 0,
        (
            data["ExpectedSavedValue"]
            - data["RetentionCost"]
        )
        / data["RetentionCost"],
        0,
    )

    # --------------------------------------------------------
    # 6. Customer value segmentation
    # --------------------------------------------------------

    print("\n[6/7] Building risk × value matrix...")

    q1, q2, q3 = data[
        "CustomerValue"
    ].quantile(
        [0.25, 0.50, 0.75]
    )

    data["ValueCategory"] = data[
        "CustomerValue"
    ].apply(
        lambda x: assign_value_category(
            x,
            q1,
            q2,
            q3,
        )
    )

    data["Priority"] = data.apply(
        assign_priority,
        axis=1,
    )

    # --------------------------------------------------------
    # Add cluster
    # --------------------------------------------------------

    segments = pd.read_csv(
        SEGMENT_PATH
    )[
        ["CustomerId", "Cluster"]
    ]

    data = data.merge(
        segments,
        on="CustomerId",
        how="left",
    )

    # --------------------------------------------------------
    # 7. Save
    # --------------------------------------------------------

    print("\n[7/7] Saving risk scoring artifacts...")

    output_columns = [
        "CustomerId",
        "Surname",
        "Geography",
        "Gender",
        "Age",
        "CreditScore",
        "Balance",
        "EstimatedSalary",
        "NumOfProducts",
        "IsActiveMember",
        "Exited",
        "Cluster",
        "CustomerValue",
        "ChurnProbability",
        "RiskCategory",
        "ValueCategory",
        "ExpectedLoss",
        "RetentionCost",
        "ExpectedSavedValue",
        "ROI",
        "Priority",
    ]

    result = data[
        output_columns
    ].copy()

    result.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    summary = (
        result
        .groupby(
            ["RiskCategory", "ValueCategory"]
        )
        .agg(
            Customers=("CustomerId", "count"),
            AvgChurnProbability=(
                "ChurnProbability",
                "mean",
            ),
            TotalExpectedLoss=(
                "ExpectedLoss",
                "sum",
            ),
            TotalRetentionCost=(
                "RetentionCost",
                "sum",
            ),
            TotalExpectedSavedValue=(
                "ExpectedSavedValue",
                "sum",
            ),
            AvgROI=("ROI", "mean"),
        )
        .reset_index()
    )

    summary.to_csv(
        PROJECT_ROOT
        / "artifacts"
        / "metrics"
        / "risk_value_summary.csv",
        index=False,
    )

    # --------------------------------------------------------
    # Priority summary
    # --------------------------------------------------------

    priority_summary = (
        result
        .groupby("Priority")
        .agg(
            Customers=("CustomerId", "count"),
            TotalExpectedLoss=(
                "ExpectedLoss",
                "sum",
            ),
            TotalExpectedSavedValue=(
                "ExpectedSavedValue",
                "sum",
            ),
            TotalRetentionCost=(
                "RetentionCost",
                "sum",
            ),
            AvgROI=("ROI", "mean"),
        )
        .reset_index()
    )

    priority_summary.to_csv(
        PROJECT_ROOT
        / "artifacts"
        / "metrics"
        / "retention_priority_summary.csv",
        index=False,
    )

    # --------------------------------------------------------
    # Metadata
    # --------------------------------------------------------

    import json

    metadata = {
        "model": "Gradient Boosting",
        "treatment_effectiveness": (
            TREATMENT_EFFECTIVENESS
        ),
        "retention_cost_rate": (
            RETENTION_COST_RATE
        ),
        "customer_value_formula": (
            "Balance + EstimatedSalary"
        ),
        "expected_loss_formula": (
            "ChurnProbability × CustomerValue"
        ),
        "expected_saved_value_formula": (
            "ExpectedLoss × TreatmentEffectiveness"
        ),
        "roi_formula": (
            "(ExpectedSavedValue - RetentionCost)"
            " / RetentionCost"
        ),
    }

    with open(
        METADATA_PATH,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            metadata,
            f,
            indent=4,
        )

    # ========================================================
    # DISPLAY RESULTS
    # ========================================================

    print("\n" + "-" * 65)
    print("RISK DISTRIBUTION")
    print("-" * 65)

    print(
        result["RiskCategory"]
        .value_counts()
    )

    print("\n" + "-" * 65)
    print("PRIORITY DISTRIBUTION")
    print("-" * 65)

    print(
        result["Priority"]
        .value_counts()
    )

    print("\n" + "-" * 65)
    print("TOP 10 CUSTOMERS BY EXPECTED LOSS")
    print("-" * 65)

    top_customers = (
        result
        .sort_values(
            "ExpectedLoss",
            ascending=False,
        )
        [
            [
                "CustomerId",
                "CustomerValue",
                "ChurnProbability",
                "ExpectedLoss",
                "ROI",
                "Priority",
            ]
        ]
        .head(10)
    )

    print(
        top_customers.to_string(
            index=False
        )
    )

    print("\n" + "=" * 65)
    print("RISK & VALUE SCORING COMPLETE")
    print("=" * 65)

    print(
        f"\nSaved: {OUTPUT_PATH}"
    )


if __name__ == "__main__":
    main()