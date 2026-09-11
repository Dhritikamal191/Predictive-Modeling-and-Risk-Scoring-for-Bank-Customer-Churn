"""
Bank Churn — Model Registry Validation

Validates the MLflow champion model before production use.
"""

from pathlib import Path
import json

import mlflow
import pandas as pd

from src.features.engineering import create_features


# ---------------------------------------------------------
# Project paths
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_PATH = (
    PROJECT_ROOT
    / "Data"
    / "raw"
    / "European_Bank.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "artifacts"
    / "registry"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ---------------------------------------------------------
# MLflow configuration
# ---------------------------------------------------------

MLFLOW_DB = PROJECT_ROOT / "mlflow.db"

MODEL_NAME = "Bank-Churn-Gradient-Boosting"
MODEL_ALIAS = "champion"

MODEL_URI = (
    f"models:/{MODEL_NAME}@{MODEL_ALIAS}"
)


# ---------------------------------------------------------
# Main validation
# ---------------------------------------------------------

def main():

    print("=" * 60)
    print("BANK CHURN — MODEL VALIDATION")
    print("=" * 60)

    # -----------------------------------------------------
    # Configure MLflow
    # -----------------------------------------------------

    mlflow.set_tracking_uri(
        f"sqlite:///{MLFLOW_DB}"
    )

    # -----------------------------------------------------
    # Load registered model
    # -----------------------------------------------------

    print()
    print("Loading registered model...")
    print(f"  {MODEL_URI}")

    try:

        model = mlflow.pyfunc.load_model(
            MODEL_URI
        )

        print(
            "  ✓ Champion model loaded"
        )

    except Exception as e:

        print(
            "  ✗ Model loading failed"
        )
        print(f"    {e}")

        raise

    # -----------------------------------------------------
    # Load validation data
    # -----------------------------------------------------

    print()
    print("Loading validation data...")

    df = pd.read_csv(DATA_PATH)

    if df.empty:
        raise RuntimeError(
            "Validation dataset is empty."
        )

    sample = df.head(5).copy()

    print(
        f"  ✓ Dataset loaded: "
        f"{len(df):,} rows"
    )

    # -----------------------------------------------------
    # Feature engineering
    # -----------------------------------------------------

    print()
    print("Creating model features...")

    sample = create_features(
        sample
    )

    print(
        f"  ✓ Features created: "
        f"{sample.shape[1]}"
    )

    # -----------------------------------------------------
    # Remove fields not used by model
    # -----------------------------------------------------

    X = sample.drop(
        columns=[
            "Year",
            "CustomerId",
            "Surname",
            "Exited",
        ],
        errors="ignore",
    )

    print(
        f"  ✓ Model input shape: "
        f"{X.shape}"
    )

    # -----------------------------------------------------
    # Prediction
    # -----------------------------------------------------

    print()
    print("Running prediction...")

    try:

        predictions = model.predict(
            X
        )

    except Exception as e:

        print(
            "  ✗ Prediction failed"
        )
        print(f"    {e}")

        raise

    predictions = list(
        predictions
    )

    # -----------------------------------------------------
    # Validate predictions
    # -----------------------------------------------------

    if len(predictions) != len(X):

        raise RuntimeError(
            "Prediction count does not "
            "match input row count."
        )

    invalid_predictions = [
        value
        for value in predictions
        if int(value) not in [0, 1]
    ]

    if invalid_predictions:

        raise RuntimeError(
            f"Invalid prediction values: "
            f"{invalid_predictions}"
        )

    print(
        "  ✓ Predictions valid"
    )

    print(
        f"  Predictions: {predictions}"
    )

    # -----------------------------------------------------
    # Validation result
    # -----------------------------------------------------

    result = {
        "model_name": MODEL_NAME,
        "model_alias": MODEL_ALIAS,
        "model_uri": MODEL_URI,
        "validation_status": "PASSED",
        "sample_size": len(X),
        "input_features": list(
            X.columns
        ),
        "predictions": [
            int(value)
            for value in predictions
        ],
    }

    # -----------------------------------------------------
    # Save result
    # -----------------------------------------------------

    output_file = (
        OUTPUT_DIR
        / "model_validation.json"
    )

    with open(
        output_file,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            result,
            file,
            indent=4,
        )

    # -----------------------------------------------------
    # Complete
    # -----------------------------------------------------

    print()
    print("=" * 60)
    print("MODEL VALIDATION PASSED")
    print("=" * 60)

    print()
    print("Saved:")
    print(
        f"  ✓ {output_file}"
    )


if __name__ == "__main__":
    main()