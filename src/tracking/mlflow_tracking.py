"""
MLflow Experiment Tracking

Tracks supervised model experiments for the Bank Churn project.
"""

from pathlib import Path
import json

import pandas as pd
import joblib
import mlflow
import mlflow.sklearn


PROJECT_ROOT = Path(__file__).resolve().parents[2]

ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
MODELS_DIR = ARTIFACTS_DIR / "models"
METRICS_DIR = ARTIFACTS_DIR / "metrics"
METADATA_DIR = ARTIFACTS_DIR / "metadata"

MODEL_COMPARISON_FILE = METRICS_DIR / "model_comparison.csv"


EXPERIMENT_NAME = "Bank Churn — Supervised ML"


MODEL_FILES = {
    "Logistic Regression": "logistic_regression.pkl",
    "Decision Tree": "decision_tree.pkl",
    "Random Forest": "random_forest.pkl",
    "Gradient Boosting": "gradient_boosting.pkl",
    "XGBoost": "xgboost.pkl",
}


def setup_mlflow():
    """Configure MLflow with the local SQLite backend."""

    tracking_uri = f"sqlite:///{PROJECT_ROOT / 'mlflow.db'}"

    mlflow.set_tracking_uri(tracking_uri)

    experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)

    if experiment is None:
        mlflow.create_experiment(EXPERIMENT_NAME)

    mlflow.set_experiment(EXPERIMENT_NAME)


def load_model_metadata():
    """Load existing model metadata."""

    metadata_file = METADATA_DIR / "model_metadata.json"

    if not metadata_file.exists():
        return {}

    try:
        with open(metadata_file, "r", encoding="utf-8") as file:
            return json.load(file)
    except Exception:
        return {}


def find_model_row(model_name, comparison_df):
    """Find the row corresponding to a model."""

    possible_columns = [
        "Model",
        "model",
        "ModelName",
        "model_name",
        "Classifier",
    ]

    model_column = None

    for column in possible_columns:
        if column in comparison_df.columns:
            model_column = column
            break

    if model_column is None:
        return None

    matches = comparison_df[
        comparison_df[model_column]
        .astype(str)
        .str.lower()
        .str.contains(model_name.lower().replace(" ", ".*"), regex=True)
    ]

    if len(matches) == 0:
        return None

    return matches.iloc[0]


def extract_metrics(row):
    """Extract numeric evaluation metrics from a model comparison row."""

    if row is None:
        return {}

    excluded = {
        "Model",
        "model",
        "ModelName",
        "model_name",
        "Classifier",
    }

    metrics = {}

    for column in row.index:

        if column in excluded:
            continue

        value = row[column]

        try:
            if pd.notna(value):
                metrics[str(column)] = float(value)
        except (ValueError, TypeError):
            continue

    return metrics


def get_model_params(model):
    """Extract useful hyperparameters from a trained sklearn model."""

    params = {}

    if hasattr(model, "get_params"):
        try:
            raw_params = model.get_params()

            important_params = [
                "n_estimators",
                "max_depth",
                "learning_rate",
                "min_samples_split",
                "min_samples_leaf",
                "criterion",
                "C",
                "solver",
                "max_features",
                "subsample",
                "random_state",
            ]

            for key in important_params:
                if key in raw_params and raw_params[key] is not None:
                    params[key] = raw_params[key]

        except Exception:
            pass

    return params


def log_model_run(model_name, model_path, metrics):
    """Log one trained model to MLflow."""

    print()
    print("-" * 60)
    print(f"Logging: {model_name}")
    print("-" * 60)

    if not model_path.exists():
        print(f"⚠ Model file not found: {model_path}")
        return

    model = joblib.load(model_path)

    params = get_model_params(model)

    with mlflow.start_run(run_name=model_name) as run:

        # Model identification
        mlflow.set_tag("model_name", model_name)
        mlflow.set_tag("project", "Bank Customer Churn")
        mlflow.set_tag("task", "Binary Classification")
        mlflow.set_tag("framework", "scikit-learn")

        # Parameters
        if params:
            clean_params = {}

            for key, value in params.items():
                clean_params[str(key)] = str(value)

            mlflow.log_params(clean_params)

        # Metrics
        if metrics:
            clean_metrics = {}

            for key, value in metrics.items():

                # MLflow metric names cannot contain certain characters
                clean_key = (
                    str(key)
                    .strip()
                    .replace(" ", "_")
                    .replace("-", "_")
                    .replace("/", "_")
                )

                if clean_key:
                    clean_metrics[clean_key] = float(value)

            if clean_metrics:
                mlflow.log_metrics(clean_metrics)

        # Existing model metadata
        metadata_file = METADATA_DIR / "model_metadata.json"

        if metadata_file.exists():
            mlflow.log_artifact(
                str(metadata_file),
                artifact_path="metadata"
            )

        # Log trained sklearn model
        mlflow.sklearn.log_model(
            model,
            name="model",
            skops_trusted_types=[
         "imblearn.over_sampling._smote.base.SMOTE",
         "imblearn.pipeline.Pipeline",
         "sklearn.compose._column_transformer._RemainderColsList",
         "xgboost.core.Booster",
         "xgboost.sklearn.XGBClassifier"
         ],

        )

        print(f"✓ Run logged successfully")
        print(f"  Run ID: {run.info.run_id}")


def main():
    """Log all existing supervised models to MLflow."""

    print("=" * 60)
    print("BANK CHURN — MLFLOW EXPERIMENT TRACKING")
    print("=" * 60)

    setup_mlflow()

    print()
    print(f"Experiment: {EXPERIMENT_NAME}")
    print(f"Tracking database: {PROJECT_ROOT / 'mlflow.db'}")
    print()

    if not MODEL_COMPARISON_FILE.exists():
        print("❌ model_comparison.csv not found.")
        print()
        print("Run:")
        print("python -m src.supervised.train")
        return

    print("Loading model comparison...")
    comparison_df = pd.read_csv(MODEL_COMPARISON_FILE)

    print()
    print("Models found:")
    print(comparison_df.to_string(index=False))

    print()
    print("=" * 60)
    print("LOGGING MODEL RUNS")
    print("=" * 60)

    logged = 0

    for model_name, filename in MODEL_FILES.items():

        model_path = MODELS_DIR / filename

        row = find_model_row(model_name, comparison_df)

        metrics = extract_metrics(row)

        if row is None:
            print()
            print(f"⚠ No comparison row found for {model_name}")

        log_model_run(
            model_name=model_name,
            model_path=model_path,
            metrics=metrics,
        )

        logged += 1

    print()
    print("=" * 60)
    print("MLFLOW TRACKING COMPLETE")
    print("=" * 60)

    print()
    print(f"✓ Models logged: {logged}")
    print(f"✓ Experiment: {EXPERIMENT_NAME}")
    print()
    print("Start MLflow UI with:")
    print()
    print("mlflow ui --backend-store-uri sqlite:///mlflow.db")
    print()


if __name__ == "__main__":
    main()