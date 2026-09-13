from pathlib import Path
import traceback
from typing import Any
import joblib
import mlflow
import mlflow.sklearn
import pandas as pd

from fastapi import FastAPI, HTTPException

from api.schemas import CustomerInput, PredictionResponse
from src.features.engineering import create_features
from src.monitoring.prediction_monitoring import record_prediction
from src.supabase_client import load_prediction_monitoring_data

# =========================================================
# PROJECT CONFIGURATION
# =========================================================

ROOT = Path(__file__).resolve().parents[1]

MLFLOW_DB = ROOT / "mlflow.db"

EXPERIMENT_NAME = "Bank Churn — Supervised ML"

MODEL_NAME = "Bank-Churn-Gradient-Boosting"
MODEL_ALIAS = "champion"

LOCAL_MODEL_PATH = (
    ROOT
    / "artifacts"
    / "models"
    / "gradient_boosting.pkl"
)


# =========================================================
# MLFLOW CONFIGURATION
# =========================================================

mlflow.set_tracking_uri(
    f"sqlite:///{MLFLOW_DB}"
)

try:
    mlflow.set_experiment(EXPERIMENT_NAME)
except Exception as e:
    print(f"MLflow experiment configuration warning: {e}")


# =========================================================
# FASTAPI APPLICATION
# =========================================================

app = FastAPI(
    title="Bank Customer Churn Prediction API",
    description=(
        "Production-style API for bank customer churn "
        "prediction and risk classification using a "
        "Gradient Boosting champion model."
    ),
    version="2.0.0",
)


# =========================================================
# MODEL VARIABLES
# =========================================================

model = None

model_source = None

model_load_error = None


# =========================================================
# LOAD MODEL
# =========================================================

MODEL_URI = (
    f"models:/{MODEL_NAME}@{MODEL_ALIAS}"
)


# ---------------------------------------------------------
# 1. TRY MLFLOW MODEL REGISTRY
# ---------------------------------------------------------

try:

    print("=" * 70)
    print("BANK CHURN API — LOADING MODEL")
    print("=" * 70)

    print(f"MLflow URI: {MODEL_URI}")

    model = mlflow.sklearn.load_model(
        MODEL_URI
    )

    model_source = "MLflow Model Registry"

    print("✓ Champion model loaded from MLflow")
    print(f"Model:  {MODEL_NAME}")
    print(f"Alias:  {MODEL_ALIAS}")
    print(f"Type:   {type(model)}")
    print(
        "predict_proba available:",
        hasattr(model, "predict_proba"),
    )

# ---------------------------------------------------------
# 2. FALLBACK TO LOCAL MODEL
# ---------------------------------------------------------

except Exception as mlflow_error:

    print("⚠ MLflow model unavailable")
    print(f"MLflow error: {mlflow_error}")

    try:

        if not LOCAL_MODEL_PATH.exists():

            raise FileNotFoundError(
                f"Local model not found: "
                f"{LOCAL_MODEL_PATH}"
            )

        model = joblib.load(
            LOCAL_MODEL_PATH
        )

        model_source = "Local Gradient Boosting Model"

        print(
            "✓ Gradient Boosting model loaded locally"
        )

        print(
            f"Path: {LOCAL_MODEL_PATH}"
        )

        print(
            f"Type: {type(model)}"
        )

    except Exception as local_error:

        model = None

        model_source = None

        model_load_error = (
            f"MLflow error: {mlflow_error}; "
            f"Local model error: {local_error}"
        )

        print("=" * 70)
        print("✗ BANK CHURN API — MODEL LOAD FAILED")
        print("=" * 70)
        print(model_load_error)

        traceback.print_exc()

        print("=" * 70)


# =========================================================
# HEALTH CHECK
# =========================================================

@app.get("/health")
def health():

    if model is None:

        return {
            "status": "unhealthy",
            "model_loaded": False,
            "model": MODEL_NAME,
            "alias": MODEL_ALIAS,
            "model_source": None,
            "error": model_load_error,
        }

    return {
        "status": "healthy",
        "model_loaded": True,
        "model": MODEL_NAME,
        "alias": MODEL_ALIAS,
        "model_source": model_source,
        "model_uri": MODEL_URI,
    }

# =========================================================
# MODEL INFORMATION ENDPOINT
# =========================================================

@app.get("/model-info")
def model_info():

    if model is None:
        return {
            "model_loaded": False,
            "model": MODEL_NAME,
            "alias": MODEL_ALIAS,
            "model_source": None,
            "error": model_load_error,
        }

    return {
        "model_loaded": True,
        "model": MODEL_NAME,
        "alias": MODEL_ALIAS,
        "model_source": model_source,
        "model_type": type(model).__name__,
        "model_uri": MODEL_URI,
        "predict_proba": hasattr(
            model,
            "predict_proba",
        ),
    }


# =========================================================
# RECENT PREDICTIONS ENDPOINT
# =========================================================

@app.get("/predictions/recent")
def recent_predictions(limit: int = 10):

    if limit < 1 or limit > 100:
        raise HTTPException(
            status_code=400,
            detail="limit must be between 1 and 100",
        )

    try:

        from src.supabase_client import (
            load_customer_risk_data
        )

        df = load_customer_risk_data()

        if df.empty:
            return {
                "count": 0,
                "predictions": [],
            }

        # Return most recent records
        df = df.tail(limit)

        # Convert pandas values to JSON-safe values
        records = df.where(
            pd.notnull(df),
            None
        ).to_dict(orient="records")

        return {
            "count": len(records),
            "predictions": records,
        }

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=f"Unable to load predictions: {str(e)}",
        )


# =========================================================
# RISK DISTRIBUTION ENDPOINT
# =========================================================

@app.get("/risk-distribution")
def risk_distribution():

    try:

        from src.supabase_client import (
            load_customer_risk_data
        )

        df = load_customer_risk_data()

        if df.empty:
            return {
                "total_customers": 0,
                "distribution": {},
            }

        if "RiskCategory" not in df.columns:
            raise RuntimeError(
                "RiskCategory column is missing "
                "from customer risk data."
            )

        distribution = (
            df["RiskCategory"]
            .value_counts()
            .to_dict()
        )

        return {
            "total_customers": int(len(df)),
            "distribution": {
                str(key): int(value)
                for key, value in distribution.items()
            },
        }

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=f"Unable to calculate risk distribution: {str(e)}",
        )


# =========================================================
# MONITORING SUMMARY ENDPOINT
# =========================================================

@app.get("/monitoring/summary")
def monitoring_summary():

    try:

        from src.supabase_client import (
            load_customer_risk_data
        )

        df = load_customer_risk_data()

        if df.empty:
            return {
                "total_customers": 0,
                "average_churn_probability": 0,
                "predicted_churn": 0,
                "predicted_retention": 0,
            }

        if "ChurnProbability" not in df.columns:
            raise RuntimeError(
                "ChurnProbability column is missing."
            )

        if "Exited" in df.columns:

            predicted_churn = int(
                df["Exited"].sum()
            )

        else:

            predicted_churn = 0

        predicted_retention = (
            len(df) - predicted_churn
        )

        return {
            "total_customers": int(len(df)),
            "average_churn_probability": float(
                df["ChurnProbability"].mean()
            ),
            "predicted_churn": predicted_churn,
            "predicted_retention": int(
                predicted_retention
            ),
        }

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=f"Unable to generate monitoring summary: {str(e)}",
        )

# =========================================================
# PREDICTION ENDPOINT
# =========================================================

@app.post(
    "/predict",
    response_model=PredictionResponse,
)
@mlflow.trace(
    name="bank-churn-prediction",
    span_type="CHAIN",
)
def predict(
    customer: CustomerInput,
):

    # -----------------------------------------------------
    # MODEL AVAILABILITY
    # -----------------------------------------------------

    if model is None:

        raise HTTPException(
            status_code=503,
            detail=(
                "Champion model is not available. "
                "Check MLflow registry or local model file."
            ),
        )

    try:

        # -------------------------------------------------
        # TRACE METADATA — INPUT
        # -------------------------------------------------

        mlflow.update_current_trace(
            metadata={
                "model": MODEL_NAME,
                "model_alias": MODEL_ALIAS,
                "model_source": model_source,
                "api_endpoint": "/predict",
                "prediction_type": "binary_churn",
            },
            tags={
                "application": "bank-churn",
                "environment": "local",
                "model_type": "gradient_boosting",
                "model_registry": "mlflow",
            },
        )

        # -------------------------------------------------
        # CONVERT REQUEST TO DATAFRAME
        # -------------------------------------------------

        df = pd.DataFrame(
            [customer.model_dump()]
        )

        # -------------------------------------------------
        # FEATURE ENGINEERING
        # -------------------------------------------------

        df = create_features(df)

        # -------------------------------------------------
        # REMOVE NON-MODEL COLUMNS
        # -------------------------------------------------

        X = df.drop(
            columns=[
                "Year",
                "CustomerId",
                "Surname",
                "Exited",
            ],
            errors="ignore",
        )

        # -------------------------------------------------
        # MODEL PREDICTION
        # -------------------------------------------------

        prediction = int(
            model.predict(X)[0]
        )

        # -------------------------------------------------
        # CHURN PROBABILITY
        # -------------------------------------------------

        if not hasattr(
            model,
            "predict_proba",
        ):

            raise RuntimeError(
                "Loaded model does not support "
                "predict_proba()."
            )

        probability = float(
            model.predict_proba(X)[0][1]
        )

        # -------------------------------------------------
        # RISK CATEGORY
        # -------------------------------------------------

        if probability < 0.20:

            risk = "Low"

        elif probability < 0.40:

            risk = "Medium"

        elif probability < 0.60:

            risk = "High"

        else:

            risk = "Critical"

        # -------------------------------------------------
        # TRACE METADATA — OUTPUT
        # -------------------------------------------------

        mlflow.update_current_trace(
            metadata={
                "churn_probability": round(
                    probability,
                    6,
                ),
                "churn_prediction": prediction,
                "risk_category": risk,
            }
        )

        # -------------------------------------------------
        # RECORD PREDICTION FOR MONITORING
        # -------------------------------------------------

        record_prediction(
            churn_probability=probability,
            churn_prediction=prediction,
            risk_category=risk,
        )

        # -------------------------------------------------
        # API RESPONSE
        # -------------------------------------------------

        return PredictionResponse(
            churn_probability=round(
                probability,
                6,
            ),
            churn_prediction=prediction,
            risk_category=risk,
        )

    except HTTPException:
        raise

    except Exception as e:

        print()
        print("=" * 70)
        print("PREDICTION ERROR")
        print("=" * 70)
        print(f"Error: {e}")

        traceback.print_exc()

        print("=" * 70)
        print()

        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}",
        )

# =========================================================
# BATCH PREDICTION ENDPOINT
# =========================================================

@app.post("/predict/batch")
def predict_batch(
    customers: list[CustomerInput],
):

    if model is None:

        raise HTTPException(
            status_code=503,
            detail=(
                "Champion model is not available. "
                "Check MLflow registry or local model file."
            ),
        )

    if not customers:

        raise HTTPException(
            status_code=400,
            detail="At least one customer is required.",
        )

    try:

        results = []

        for customer in customers:

            df = pd.DataFrame(
                [customer.model_dump()]
            )

            df = create_features(df)

            X = df.drop(
                columns=[
                    "Year",
                    "CustomerId",
                    "Surname",
                    "Exited",
                ],
                errors="ignore",
            )

            prediction = int(
                model.predict(X)[0]
            )

            probability = float(
                model.predict_proba(X)[0][1]
            )

            if probability < 0.20:
                risk = "Low"

            elif probability < 0.40:
                risk = "Medium"

            elif probability < 0.60:
                risk = "High"

            else:
                risk = "Critical"

            results.append(
                {
                    "customer_id": getattr(
                        customer,
                        "CustomerId",
                        None,
                    ),
                    "churn_probability": round(
                        probability,
                        6,
                    ),
                    "churn_prediction": prediction,
                    "risk_category": risk,
                }
            )

        return {
            "count": len(results),
            "predictions": results,
        }

    except Exception as e:

        traceback.print_exc()

        raise HTTPException(
            status_code=500,
            detail=f"Batch prediction failed: {str(e)}",
        )

# =========================================================
# MONITORING STATISTICS ENDPOINT
# =========================================================

@app.get("/monitoring/stats")
def monitoring_stats():

    try:

        df = load_prediction_monitoring_data()

        if df.empty:

            return {
                "count": 0,
                "average_churn_probability": 0,
                "predicted_churn_rate": 0,
                "risk_distribution": {},
            }

        probability_column = "churn_probability"
        prediction_column = "churn_prediction"
        risk_column = "risk_category"

        result = {
            "count": len(df),
            "average_churn_probability": round(
                float(
                    df[probability_column].mean()
                ),
                6,
            ),
            "predicted_churn_rate": round(
                float(
                    df[prediction_column].mean()
                ),
                6,
            ),
            "risk_distribution": (
                df[risk_column]
                .value_counts()
                .to_dict()
            ),
        }

        return result

    except Exception as e:

        traceback.print_exc()

        raise HTTPException(
            status_code=500,
            detail=(
                f"Unable to load monitoring "
                f"statistics: {str(e)}"
            ),
        )

# =========================================================
# ROOT ENDPOINT
# =========================================================

@app.get("/")
def root():

    return {
        "message": (
            "Bank Customer Churn Prediction API"
        ),
        "version": "2.0.0",
        "model": MODEL_NAME,
        "alias": MODEL_ALIAS,
        "model_source": model_source,
        "docs": "/docs",
        "health": "/health",
        "prediction": "/predict",
    }