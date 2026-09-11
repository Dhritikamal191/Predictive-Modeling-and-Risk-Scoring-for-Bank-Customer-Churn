"""
MLflow Model Registry

Registers the selected production model and assigns
the champion alias.
"""

from pathlib import Path

import mlflow
from mlflow import MlflowClient


PROJECT_ROOT = Path(__file__).resolve().parents[2]

TRACKING_URI = f"sqlite:///{PROJECT_ROOT / 'mlflow.db'}"

EXPERIMENT_NAME = "Bank Churn — Supervised ML"

MODEL_NAME = "Bank-Churn-Gradient-Boosting"


def main():

    print("=" * 60)
    print("BANK CHURN — MLFLOW MODEL REGISTRY")
    print("=" * 60)

    mlflow.set_tracking_uri(TRACKING_URI)

    client = MlflowClient()

    experiment = client.get_experiment_by_name(EXPERIMENT_NAME)

    if experiment is None:
        raise RuntimeError(
            f"Experiment not found: {EXPERIMENT_NAME}"
        )

    print()
    print(f"Experiment: {EXPERIMENT_NAME}")
    print(f"Model: {MODEL_NAME}")
    print()

    # Find the Gradient Boosting run
    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string="attributes.run_name = 'Gradient Boosting'",
        order_by=["attributes.start_time DESC"],
    )

    if not runs:
        raise RuntimeError(
            "No Gradient Boosting MLflow run found."
        )

    run = runs[0]

    run_id = run.info.run_id

    print(f"✓ Gradient Boosting run found")
    print(f"  Run ID: {run_id}")

    model_uri = f"runs:/{run_id}/model"

    print()
    print(f"Model URI:")
    print(f"  {model_uri}")

    print()
    print("Registering model...")

    model_version = mlflow.register_model(
        model_uri=model_uri,
        name=MODEL_NAME,
        tags={
            "project": "bank-churn",
            "model_type": "gradient_boosting",
            "task": "customer_churn_prediction",
        },
    )

    print()
    print("✓ Model registered")
    print(f"  Name: {model_version.name}")
    print(f"  Version: {model_version.version}")

    print()
    print("Assigning champion alias...")

    client.set_registered_model_alias(
        MODEL_NAME,
        "champion",
        model_version.version,
    )

    print()
    print("✓ Champion alias assigned")

    print()
    print("=" * 60)
    print("MODEL REGISTRY COMPLETE")
    print("=" * 60)

    print()
    print("Registered model:")
    print(f"  {MODEL_NAME}")

    print()
    print("Production alias:")
    print("  champion")

    print()
    print("Model URI:")
    print(f"  models:/{MODEL_NAME}@champion")

    print()


if __name__ == "__main__":
    main()