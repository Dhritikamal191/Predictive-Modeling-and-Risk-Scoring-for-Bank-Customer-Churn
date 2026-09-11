"""
Logistic Regression Explainability

Extracts:
- Model coefficients
- Odds ratios
- Direction of influence
- Ranked feature importance

The Logistic Regression model is retained as the
interpretable statistical baseline.
"""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

MODEL_PATH = (
    PROJECT_ROOT
    / "artifacts"
    / "models"
    / "logistic_regression.pkl"
)

METRICS_DIR = (
    PROJECT_ROOT
    / "artifacts"
    / "metrics"
)

METRICS_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# MAIN
# ============================================================

def main():

    print("\n" + "=" * 65)
    print("BANK CHURN — LOGISTIC REGRESSION EXPLAINABILITY")
    print("=" * 65)

    # --------------------------------------------------------
    # 1. Load model
    # --------------------------------------------------------

    print("\n[1/4] Loading Logistic Regression model...")

    pipeline = joblib.load(
        MODEL_PATH
    )

    print("✓ Model loaded")

    # --------------------------------------------------------
    # 2. Extract preprocessor + model
    # --------------------------------------------------------

    print(
        "\n[2/4] Extracting coefficients..."
    )

    preprocessor = pipeline.named_steps[
        "preprocessor"
    ]

    model = pipeline.named_steps[
        "model"
    ]

    feature_names = (
        preprocessor
        .get_feature_names_out()
    )

    coefficients = model.coef_[0]

    # --------------------------------------------------------
    # 3. Build explanation table
    # --------------------------------------------------------

    print(
        "\n[3/4] Building coefficient analysis..."
    )

    results = pd.DataFrame(
        {
            "Feature": feature_names,
            "Coefficient": coefficients,
        }
    )

    results["OddsRatio"] = np.exp(
        results["Coefficient"]
    )

    results["AbsoluteCoefficient"] = (
        results["Coefficient"].abs()
    )

    results["ImpactDirection"] = np.where(
        results["Coefficient"] > 0,
        "Increases Churn Risk",
        "Decreases Churn Risk",
    )

    results = results.sort_values(
        "AbsoluteCoefficient",
        ascending=False,
    ).reset_index(drop=True)

    results["Rank"] = (
        np.arange(len(results)) + 1
    )

    results = results[
        [
            "Rank",
            "Feature",
            "Coefficient",
            "OddsRatio",
            "ImpactDirection",
            "AbsoluteCoefficient",
        ]
    ]

    # --------------------------------------------------------
    # 4. Save
    # --------------------------------------------------------

    print(
        "\n[4/4] Saving coefficient analysis..."
    )

    output_path = (
        METRICS_DIR
        / "logistic_coefficients.csv"
    )

    results.to_csv(
        output_path,
        index=False,
    )

    print(
        "\n" + "-" * 65
    )

    print(
        "TOP LOGISTIC REGRESSION FEATURES"
    )

    print(
        "-" * 65
    )

    print(
        results
        .head(15)
        .to_string(index=False)
    )

    print(
        "\n" + "=" * 65
    )

    print(
        "LOGISTIC EXPLAINABILITY COMPLETE"
    )

    print(
        "=" * 65
    )

    print(
        f"\nSaved: {output_path}"
    )


if __name__ == "__main__":
    main()