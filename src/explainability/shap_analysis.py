"""
SHAP Explainability for Bank Churn

Provides:
- Global feature importance
- Local customer-level explanations
- Feature impact direction
"""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import shap


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
# LOAD DATA
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

    return df, X


# ============================================================
# EXTRACT TRANSFORMED FEATURES
# ============================================================

def transform_features(model, X):

    preprocessor = model.named_steps[
        "preprocessor"
    ]

    X_transformed = preprocessor.transform(X)

    feature_names = (
        preprocessor
        .get_feature_names_out()
    )

    return X_transformed, feature_names


# ============================================================
# SHAP ANALYSIS
# ============================================================

def main():

    print("\n" + "=" * 65)
    print("BANK CHURN — SHAP EXPLAINABILITY")
    print("=" * 65)

    # --------------------------------------------------------
    # Load
    # --------------------------------------------------------

    print("\n[1/5] Loading model and data...")

    model = joblib.load(
        MODEL_PATH
    )

    df, X = prepare_data()

    print(
        f"Customers: {len(X):,}"
    )

    # --------------------------------------------------------
    # Transform
    # --------------------------------------------------------

    print(
        "\n[2/5] Transforming model features..."
    )

    X_transformed, feature_names = (
        transform_features(
            model,
            X,
        )
    )

    # --------------------------------------------------------
    # Sample for SHAP
    # --------------------------------------------------------

    print(
        "\n[3/5] Calculating SHAP values..."
    )

    # 2,000 observations are enough for
    # a strong global explanation while
    # keeping computation manageable.
    sample_size = min(
        2000,
        X_transformed.shape[0],
    )

    rng = np.random.default_rng(42)

    sample_indices = rng.choice(
        X_transformed.shape[0],
        size=sample_size,
        replace=False,
    )

    X_sample = X_transformed[
        sample_indices
    ]

    # GradientBoosting works well with
    # TreeExplainer.
    explainer = shap.TreeExplainer(
        model.named_steps["model"]
    )

    shap_values = explainer.shap_values(
        X_sample
    )

    # --------------------------------------------------------
    # Handle SHAP output formats
    # --------------------------------------------------------

    if isinstance(
        shap_values,
        list,
    ):

        shap_values = shap_values[0]

    shap_values = np.asarray(
        shap_values
    )

    # Some SHAP versions may return
    # an additional output dimension.
    if shap_values.ndim == 3:

        shap_values = shap_values[:, :, 0]

    # --------------------------------------------------------
    # Global importance
    # --------------------------------------------------------

    importance = np.abs(
        shap_values
    ).mean(axis=0)

    global_importance = pd.DataFrame(
        {
            "Feature": feature_names,
            "MeanAbsoluteSHAP": importance,
        }
    )

    global_importance = (
        global_importance
        .sort_values(
            "MeanAbsoluteSHAP",
            ascending=False,
        )
        .reset_index(drop=True)
    )

    global_importance[
        "Rank"
    ] = np.arange(
        1,
        len(global_importance) + 1,
    )

    # --------------------------------------------------------
    # Direction of influence
    # --------------------------------------------------------

    mean_signed_shap = (
        shap_values.mean(axis=0)
    )

    direction = pd.DataFrame(
        {
            "Feature": feature_names,
            "MeanSHAP": mean_signed_shap,
        }
    )

    direction["ImpactDirection"] = np.where(
        direction["MeanSHAP"] > 0,
        "Increases Churn Risk",
        "Decreases Churn Risk",
    )

    explanation = global_importance.merge(
        direction,
        on="Feature",
    )

    # --------------------------------------------------------
    # Local explanations
    # --------------------------------------------------------

    print(
        "\n[4/5] Creating customer-level explanations..."
    )

    local_rows = []

    # Explain first 500 customers
    local_count = min(
        500,
        len(sample_indices),
    )

    for i in range(local_count):

        original_index = (
            sample_indices[i]
        )

        customer_id = df.iloc[
            original_index
        ]["CustomerId"]

        probability = model.predict_proba(
            X.iloc[
                [original_index]
            ]
        )[0, 1]

        values = shap_values[i]

        top_indices = np.argsort(
            np.abs(values)
        )[::-1][:10]

        for rank, feature_index in enumerate(
            top_indices,
            start=1,
        ):

            local_rows.append(
                {
                    "CustomerId": customer_id,
                    "ChurnProbability": probability,
                    "Feature": feature_names[
                        feature_index
                    ],
                    "SHAPValue": values[
                        feature_index
                    ],
                    "AbsoluteSHAP": abs(
                        values[
                            feature_index
                        ]
                    ),
                    "ImpactDirection": (
                        "Increases Churn Risk"
                        if values[
                            feature_index
                        ] > 0
                        else
                        "Decreases Churn Risk"
                    ),
                    "ImportanceRank": rank,
                }
            )

    local_explanations = pd.DataFrame(
        local_rows
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    print(
        "\n[5/5] Saving explainability artifacts..."
    )

    explanation.to_csv(
        METRICS_DIR
        / "shap_global_importance.csv",
        index=False,
    )

    local_explanations.to_csv(
        METRICS_DIR
        / "shap_local_explanations.csv",
        index=False,
    )

    metadata = {
        "model": "Gradient Boosting",
        "sample_size": sample_size,
        "local_explanation_count": local_count,
        "method": "SHAP TreeExplainer",
    }

    joblib.dump(
        metadata,
        METADATA_DIR
        / "shap_metadata.pkl",
    )

    # --------------------------------------------------------
    # Display
    # --------------------------------------------------------

    print("\n" + "-" * 65)
    print("TOP 15 FEATURES BY SHAP IMPORTANCE")
    print("-" * 65)

    print(
        explanation[
            [
                "Rank",
                "Feature",
                "MeanAbsoluteSHAP",
                "MeanSHAP",
                "ImpactDirection",
            ]
        ]
        .head(15)
        .to_string(index=False)
    )

    print("\n" + "=" * 65)
    print("SHAP ANALYSIS COMPLETE")
    print("=" * 65)

    print(
        "\nSaved:"
    )

    print(
        "   ✓ artifacts/metrics/"
        "shap_global_importance.csv"
    )

    print(
        "   ✓ artifacts/metrics/"
        "shap_local_explanations.csv"
    )

    print(
        "   ✓ artifacts/metadata/"
        "shap_metadata.pkl"
    )


if __name__ == "__main__":
    main()