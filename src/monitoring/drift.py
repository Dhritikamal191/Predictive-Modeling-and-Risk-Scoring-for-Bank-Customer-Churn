"""
Bank Churn — Data & Prediction Drift Monitoring

Compares a reference dataset against a current dataset.

Metrics:
- PSI for numerical/categorical distributions
- KS test for numerical variables
- Categorical distribution drift
- Prediction probability drift

Reference:
    Training/reference population

Current:
    New/current population
"""

from pathlib import Path

import json
import joblib
import numpy as np
import pandas as pd

from scipy.stats import ks_2samp


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
# CONFIGURATION
# ============================================================

NUMERICAL_FEATURES = [
    "CreditScore",
    "Age",
    "Tenure",
    "Balance",
    "NumOfProducts",
    "EstimatedSalary",
]

CATEGORICAL_FEATURES = [
    "Geography",
    "Gender",
]

PSI_BINS = 10


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
# PSI
# ============================================================

def calculate_psi(
    reference,
    current,
    bins=10,
):
    """
    Population Stability Index.

    Approximate interpretation:
        PSI < 0.10       → Little/no drift
        0.10–0.25       → Moderate drift
        > 0.25          → Significant drift
    """

    reference = pd.Series(
        reference
    ).replace(
        [np.inf, -np.inf],
        np.nan,
    ).dropna()

    current = pd.Series(
        current
    ).replace(
        [np.inf, -np.inf],
        np.nan,
    ).dropna()

    if len(reference) == 0 or len(current) == 0:
        return np.nan

    # Quantile bins based on reference population
    quantiles = np.linspace(
        0,
        1,
        bins + 1,
    )

    breakpoints = np.unique(
        reference.quantile(
            quantiles
        ).values
    )

    if len(breakpoints) < 3:
        return 0.0

    reference_binned = pd.cut(
        reference,
        bins=breakpoints,
        include_lowest=True,
    )

    current_binned = pd.cut(
        current,
        bins=breakpoints,
        include_lowest=True,
    )

    reference_dist = (
        reference_binned
        .value_counts(
            normalize=True,
            sort=False,
        )
    )

    current_dist = (
        current_binned
        .value_counts(
            normalize=True,
            sort=False,
        )
    )

    # Align bins
    current_dist = current_dist.reindex(
        reference_dist.index,
        fill_value=0,
    )

    # Prevent division/log(0)
    epsilon = 1e-6

    reference_dist = (
        reference_dist + epsilon
    )

    current_dist = (
        current_dist + epsilon
    )

    psi = (
        (
            current_dist
            - reference_dist
        )
        * np.log(
            current_dist
            / reference_dist
        )
    ).sum()

    return float(psi)


# ============================================================
# PSI FOR CATEGORICAL FEATURES
# ============================================================

def calculate_categorical_psi(
    reference,
    current,
):
    reference = pd.Series(
        reference
    ).astype(str)

    current = pd.Series(
        current
    ).astype(str)

    categories = sorted(
        set(reference.unique())
        | set(current.unique())
    )

    reference_dist = (
        reference
        .value_counts(
            normalize=True
        )
        .reindex(
            categories,
            fill_value=0,
        )
    )

    current_dist = (
        current
        .value_counts(
            normalize=True
        )
        .reindex(
            categories,
            fill_value=0,
        )
    )

    epsilon = 1e-6

    reference_dist = (
        reference_dist + epsilon
    )

    current_dist = (
        current_dist + epsilon
    )

    psi = (
        (
            current_dist
            - reference_dist
        )
        * np.log(
            current_dist
            / reference_dist
        )
    ).sum()

    return float(psi)


# ============================================================
# PSI INTERPRETATION
# ============================================================

def interpret_psi(value):

    if pd.isna(value):
        return "Unavailable"

    if value < 0.10:
        return "Stable"

    elif value < 0.25:
        return "Moderate Drift"

    return "Significant Drift"


# ============================================================
# NUMERICAL DRIFT
# ============================================================

def numerical_drift(
    reference,
    current,
):

    results = []

    for feature in NUMERICAL_FEATURES:

        ref = reference[feature].dropna()
        cur = current[feature].dropna()

        psi = calculate_psi(
            ref,
            cur,
        )

        ks_stat, ks_pvalue = ks_2samp(
            ref,
            cur,
        )

        results.append(
            {
                "Feature": feature,
                "Type": "Numerical",
                "PSI": psi,
                "PSIStatus": interpret_psi(
                    psi
                ),
                "KSStatistic": ks_stat,
                "KSPValue": ks_pvalue,
                "KSDrift": (
                    "Drift Detected"
                    if ks_pvalue < 0.05
                    else "Stable"
                ),
            }
        )

    return pd.DataFrame(results)


# ============================================================
# CATEGORICAL DRIFT
# ============================================================

def categorical_drift(
    reference,
    current,
):

    results = []

    for feature in CATEGORICAL_FEATURES:

        psi = calculate_categorical_psi(
            reference[feature],
            current[feature],
        )

        results.append(
            {
                "Feature": feature,
                "Type": "Categorical",
                "PSI": psi,
                "PSIStatus": interpret_psi(
                    psi
                ),
                "KSStatistic": np.nan,
                "KSPValue": np.nan,
                "KSDrift": "Not Applicable",
            }
        )

    return pd.DataFrame(results)


# ============================================================
# PREDICTION DRIFT
# ============================================================

def prediction_drift(
    model,
    reference,
    current,
):

    reference_features = (
        reference.drop(
            columns=[
                "Exited",
                "Year",
                "CustomerId",
                "Surname",
            ],
            errors="ignore",
        )
    )

    current_features = (
        current.drop(
            columns=[
                "Exited",
                "Year",
                "CustomerId",
                "Surname",
            ],
            errors="ignore",
        )
    )

    reference_probability = (
        model.predict_proba(
            reference_features
        )[:, 1]
    )

    current_probability = (
        model.predict_proba(
            current_features
        )[:, 1]
    )

    psi = calculate_psi(
        reference_probability,
        current_probability,
    )

    ks_stat, ks_pvalue = ks_2samp(
        reference_probability,
        current_probability,
    )

    result = {
        "PredictionPSI": psi,
        "PredictionPSIStatus": interpret_psi(
            psi
        ),
        "PredictionKSStatistic": ks_stat,
        "PredictionKSPValue": ks_pvalue,
        "PredictionKSDrift": (
            "Drift Detected"
            if ks_pvalue < 0.05
            else "Stable"
        ),
        "ReferenceMeanProbability": float(
            np.mean(reference_probability)
        ),
        "CurrentMeanProbability": float(
            np.mean(current_probability)
        ),
    }

    return result


# ============================================================
# MAIN
# ============================================================

def main():

    print("\n" + "=" * 65)
    print("BANK CHURN — MLOPS DRIFT MONITORING")
    print("=" * 65)

    # --------------------------------------------------------
    # 1. Load reference/current
    # --------------------------------------------------------

    print(
        "\n[1/5] Loading reference dataset..."
    )

    reference = pd.read_csv(
        DATA_PATH
    )

    # For the initial monitoring baseline,
    # use a deterministic reference/current split.
    #
    # In production, "current" would be a genuinely
    # newly received customer batch.

    reference = reference.sample(
        frac=0.70,
        random_state=42,
    )

    current = pd.read_csv(
        DATA_PATH
    )

    current = current.drop(
        reference.index,
        errors="ignore",
    )

    print(
        f"Reference rows: {len(reference):,}"
    )

    print(
        f"Current rows:   {len(current):,}"
    )

    # --------------------------------------------------------
    # Feature engineering
    # --------------------------------------------------------

    print(
        "\n[2/5] Creating monitoring features..."
    )

    reference = create_features(
        reference
    )

    current = create_features(
        current
    )

    # --------------------------------------------------------
    # Feature drift
    # --------------------------------------------------------

    print(
        "\n[3/5] Calculating feature drift..."
    )

    numerical_results = numerical_drift(
        reference,
        current,
    )

    categorical_results = categorical_drift(
        reference,
        current,
    )

    drift_results = pd.concat(
        [
            numerical_results,
            categorical_results,
        ],
        ignore_index=True,
    )

    drift_results = drift_results.sort_values(
        "PSI",
        ascending=False,
    )

    print("\nFeature drift:")
    print(
        drift_results.to_string(
            index=False
        )
    )

    # --------------------------------------------------------
    # Prediction drift
    # --------------------------------------------------------

    print(
        "\n[4/5] Calculating prediction drift..."
    )

    model = joblib.load(
        MODEL_PATH
    )

    prediction_results = prediction_drift(
        model,
        reference,
        current,
    )

    print(
        f"Prediction PSI: "
        f"{prediction_results['PredictionPSI']:.4f}"
    )

    print(
        f"Prediction KS p-value: "
        f"{prediction_results['PredictionKSPValue']:.6f}"
    )

    print(
        f"Prediction status: "
        f"{prediction_results['PredictionPSIStatus']}"
    )

    # --------------------------------------------------------
    # Overall status
    # --------------------------------------------------------

    significant_feature_drift = (
        drift_results["PSI"] > 0.25
    ).sum()

    moderate_feature_drift = (
        (
            drift_results["PSI"] >= 0.10
        )
        & (
            drift_results["PSI"] <= 0.25
        )
    ).sum()

    prediction_drift_detected = (
        prediction_results[
            "PredictionPSI"
        ] > 0.25
    )

    if (
        significant_feature_drift > 0
        or prediction_drift_detected
    ):
        overall_status = "ALERT"

    elif moderate_feature_drift > 0:
        overall_status = "WARNING"

    else:
        overall_status = "STABLE"

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    print(
        "\n[5/5] Saving monitoring artifacts..."
    )

    drift_results.to_csv(
        METRICS_DIR
        / "feature_drift_report.csv",
        index=False,
    )

    with open(
        METRICS_DIR
        / "prediction_drift_report.json",
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            prediction_results,
            f,
            indent=4,
        )

    monitoring_summary = {
        "overall_status": overall_status,
        "significant_feature_drift_count": int(
            significant_feature_drift
        ),
        "moderate_feature_drift_count": int(
            moderate_feature_drift
        ),
        "prediction_drift_detected": bool(
            prediction_drift_detected
        ),
        "reference_rows": len(reference),
        "current_rows": len(current),
    }

    with open(
        METADATA_DIR
        / "monitoring_status.json",
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            monitoring_summary,
            f,
            indent=4,
        )

    print(
        "\n" + "-" * 65
    )

    print(
        "MONITORING STATUS:"
    )

    print(
        f"   {overall_status}"
    )

    print(
        f"\nSignificant feature drift: "
        f"{significant_feature_drift}"
    )

    print(
        f"Moderate feature drift: "
        f"{moderate_feature_drift}"
    )

    print(
        f"Prediction drift: "
        f"{'YES' if prediction_drift_detected else 'NO'}"
    )

    print(
        "\nSaved:"
    )

    print(
        "   ✓ feature_drift_report.csv"
    )

    print(
        "   ✓ prediction_drift_report.json"
    )

    print(
        "   ✓ monitoring_status.json"
    )

    print(
        "\n" + "=" * 65
    )

    print(
        "DRIFT MONITORING COMPLETE"
    )

    print(
        "=" * 65
    )


if __name__ == "__main__":
    main()