import os
import json
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE

from src.data.validation import validate_dataset
from src.features.engineering import create_features
from src.supervised.models import get_models
from src.supervised.evaluate import (
    evaluate_model,
    find_best_threshold,
    metrics_to_dataframe
)


DATA_PATH = "data/raw/European_Bank.csv"

MODEL_DIR = "artifacts/models"
METRICS_DIR = "artifacts/metrics"
METADATA_DIR = "artifacts/metadata"


def build_preprocessor(X):

    categorical_cols = [
        "Geography",
        "Gender"
    ]

    numeric_cols = [
        col for col in X.columns
        if col not in categorical_cols
    ]

    return ColumnTransformer(
        transformers=[
            (
                "num",
                StandardScaler(),
                numeric_cols
            ),
            (
                "cat",
                OneHotEncoder(
                    drop="first",
                    handle_unknown="ignore"
                ),
                categorical_cols
            )
        ]
    )


def main():

    print("\n" + "=" * 60)
    print("BANK CHURN — SUPERVISED ML TRAINING")
    print("=" * 60)

    # -----------------------------------------
    # Create directories
    # -----------------------------------------

    os.makedirs(
        MODEL_DIR,
        exist_ok=True
    )

    os.makedirs(
        METRICS_DIR,
        exist_ok=True
    )

    os.makedirs(
        METADATA_DIR,
        exist_ok=True
    )

    # -----------------------------------------
    # Load data
    # -----------------------------------------

    print("\n[1/6] Loading dataset...")

    df = pd.read_csv(DATA_PATH)

    print(
        f"Dataset shape: {df.shape}"
    )

    # -----------------------------------------
    # Validate
    # -----------------------------------------

    print("\n[2/6] Validating dataset...")

    validation = validate_dataset(df)

    if not validation["valid"]:

        print("\n❌ Dataset validation failed.")

        for error in validation["errors"]:
            print(f"   - {error}")

        raise ValueError(
            "Training stopped because dataset validation failed."
        )

    print("✅ Dataset validation passed.")

    # -----------------------------------------
    # Feature engineering
    # -----------------------------------------

    print("\n[3/6] Creating features...")

    data = create_features(df)

    # -----------------------------------------
    # Target and predictors
    # -----------------------------------------

    y = data["Exited"]

    drop_columns = [
        "Exited",
        "Year",
        "CustomerId",
        "Surname"
    ]

    X = data.drop(
        columns=drop_columns
    )

    print(
        f"Training features: {X.shape[1]}"
    )

    # -----------------------------------------
    # Train/test split
    # -----------------------------------------

    print("\n[4/6] Splitting data...")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    print(
        f"Training rows: {len(X_train)}"
    )

    print(
        f"Testing rows: {len(X_test)}"
    )

    # -----------------------------------------
    # Models
    # -----------------------------------------

    models = get_models()

    print("\nModels to train:")

    for name in models:
        print(f"   ✓ {name}")

    results = {}
    thresholds = {}
    trained_models = {}

    # -----------------------------------------
    # Train
    # -----------------------------------------

    print("\n[5/6] Training models...")
    print("-" * 60)

    for name, model in models.items():

        print(f"\nTraining: {name}")

        preprocessor = build_preprocessor(
            X_train
        )

        pipeline = ImbPipeline(
            steps=[
                (
                    "preprocessor",
                    preprocessor
                ),
                (
                    "smote",
                    SMOTE(
                        random_state=42
                    )
                ),
                (
                    "model",
                    model
                )
            ]
        )

        pipeline.fit(
            X_train,
            y_train
        )

        metrics = evaluate_model(
            pipeline,
            X_test,
            y_test
        )

        threshold = find_best_threshold(
            pipeline,
            X_test,
            y_test
        )

        results[name] = metrics

        thresholds[name] = threshold

        trained_models[name] = pipeline

        print(
            f"Accuracy : {metrics['Accuracy']:.4f}"
        )

        print(
            f"Precision: {metrics['Precision']:.4f}"
        )

        print(
            f"Recall   : {metrics['Recall']:.4f}"
        )

        print(
            f"F1       : {metrics['F1']:.4f}"
        )

        print(
            f"ROC-AUC  : {metrics['ROC-AUC']:.4f}"
        )

        print(
            f"PR-AUC   : {metrics['PR-AUC']:.4f}"
        )

        print(
            f"Best F1 threshold: "
            f"{threshold['threshold']:.2f}"
        )

    # -----------------------------------------
    # Save models
    # -----------------------------------------

    print("\n[6/6] Saving artifacts...")

    for name, pipeline in trained_models.items():

        filename = (
            name.lower()
            .replace(" ", "_")
            .replace("-", "_")
            + ".pkl"
        )

        path = os.path.join(
            MODEL_DIR,
            filename
        )

        joblib.dump(
            pipeline,
            path
        )

    # -----------------------------------------
    # Comparison table
    # -----------------------------------------

    comparison = metrics_to_dataframe(
        results
    )

    comparison.to_csv(
        os.path.join(
            METRICS_DIR,
            "model_comparison.csv"
        )
    )

    # -----------------------------------------
    # Threshold metadata
    # -----------------------------------------

    with open(
        os.path.join(
            METADATA_DIR,
            "thresholds.json"
        ),
        "w"
    ) as file:

        json.dump(
            thresholds,
            file,
            indent=4
        )

    # -----------------------------------------
    # Model metadata
    # -----------------------------------------

    metadata = {
        "dataset": DATA_PATH,
        "rows": len(df),
        "features": X.columns.tolist(),
        "models": list(trained_models.keys()),
        "target": "Exited",
        "test_size": 0.20,
        "random_state": 42,
        "smote": True
    }

    with open(
        os.path.join(
            METADATA_DIR,
            "model_metadata.json"
        ),
        "w"
    ) as file:

        json.dump(
            metadata,
            file,
            indent=4
        )

    # -----------------------------------------
    # Save comparison
    # -----------------------------------------

    print("\n" + "=" * 60)
    print("MODEL COMPARISON")
    print("=" * 60)

    print(
        comparison[
            [
                "Accuracy",
                "Precision",
                "Recall",
                "F1",
                "ROC-AUC",
                "PR-AUC"
            ]
        ].round(4)
    )

    print("\n✅ All five models trained successfully.")

    print("\nSaved models:")

    for name in trained_models:
        print(f"   ✓ {name}")

    print(
        "\nArtifacts saved under: artifacts/"
    )


if __name__ == "__main__":
    main()
