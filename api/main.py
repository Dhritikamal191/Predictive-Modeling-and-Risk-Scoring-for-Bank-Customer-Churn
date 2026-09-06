from pathlib import Path
import traceback
import mlflow
import mlflow.sklearn
import pandas as pd
from fastapi import FastAPI, HTTPException
from api.schemas import CustomerInput, PredictionResponse
from src.features.engineering import create_features
from src.monitoring.prediction_monitoring import record_prediction

# =========================================================
# PROJECT CONFIGURATION
# =========================================================

ROOT = Path(__file__).resolve().parents[1]

MLFLOW_DB = ROOT / "mlflow.db"

EXPERIMENT_NAME = "Bank Churn — Supervised ML"

MODEL_NAME = "Bank-Churn-Gradient-Boosting"
MODEL_ALIAS = "champion"


# =========================================================
# MLFLOW CONFIGURATION
# =========================================================

mlflow.set_tracking_uri(
    f"sqlite:///{MLFLOW_DB}"
)

mlflow.set_experiment(EXPERIMENT_NAME)


# =========================================================
# FASTAPI APPLICATION
# =========================================================

app = FastAPI(
    title="Bank Customer Churn Prediction API",
    description=(
        "Production-style API for bank customer churn "
        "prediction and risk classification using an "
        "MLflow Model Registry champion model."
    ),
    version="2.0.0",
)

# =========================================================
# LOAD CHAMPION MODEL
# =========================================================

MODEL_URI = f"models:/{MODEL_NAME}@{MODEL_ALIAS}"

LOCAL_MODEL_PATH = (
    ROOT
    / "artifacts"
    / "models"
    / "gradient_boosting.pkl"
)

model = None
model_load_error = None
model_source = None


# ---------------------------------------------------------
# PRIMARY: LOAD FROM MLFLOW MODEL REGISTRY
# ---------------------------------------------------------

try:

    model = mlflow.sklearn.load_model(
        MODEL_URI
    )

    model_source = "MLflow Model Registry"

    print("=" * 70)
    print("BANK CHURN API — MODEL LOADED")
    print("=" * 70)
    print(f"Model:  {MODEL_NAME}")
    print(f"Alias:  {MODEL_ALIAS}")
    print(f"Source: {model_source}")
    print(f"URI:    {MODEL_URI}")
    print(f"Type:   {type(model)}")
    print(
        f"predict_proba available: "
        f"{hasattr(model, 'predict_proba')}"
    )
    print("=" * 70)


# ---------------------------------------------------------
# FALLBACK: LOAD LOCAL GRADIENT BOOSTING MODEL
# ---------------------------------------------------------

except Exception as mlflow_error:

    print("=" * 70)
    print("MLFLOW MODEL LOAD FAILED")
    print("=" * 70)
    print(f"MLflow error: {mlflow_error}")
    print("=" * 70)

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

        print("=" * 70)
        print("BANK CHURN API — LOCAL MODEL LOADED")
        print("=" * 70)
        print(f"Source: {model_source}")
        print(f"Path:   {LOCAL_MODEL_PATH}")
        print(f"Type:   {type(model)}")
        print("=" * 70)

    except Exception as local_error:

        model = None

        model_load_error = (
            f"MLflow error: {mlflow_error}; "
            f"Local model error: {local_error}"
        )

        print("=" * 70)
        print("BANK CHURN API — MODEL LOAD FAILED")
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
            "error": model_load_error,
        }

    return {
        "status": "healthy",
        "model_loaded": True,
        "model": MODEL_NAME,
        "alias": MODEL_ALIAS,
        "model_uri": MODEL_URI,
    }


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
def predict(customer: CustomerInput):

    if model is None:

        raise HTTPException(
            status_code=503,
            detail="Champion model is not available.",
        )

    try:

        # -------------------------------------------------
        # TRACE METADATA — INPUT
        # -------------------------------------------------

        mlflow.update_current_trace(
            metadata={
                "model": MODEL_NAME,
                "model_alias": MODEL_ALIAS,
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

        if not hasattr(model, "predict_proba"):

            raise RuntimeError(
                "Loaded champion model does not support "
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

        print("\n")
        print("=" * 70)
        print("PREDICTION ERROR")
        print("=" * 70)
        print(f"Error: {e}")
        traceback.print_exc()
        print("=" * 70)
        print("\n")

        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}",
        )


# =========================================================
# ROOT ENDPOINT
# =========================================================

@app.get("/")
def root():

    return {
        "message": "Bank Customer Churn Prediction API",
        "version": "2.0.0",
        "model": MODEL_NAME,
        "alias": MODEL_ALIAS,
        "docs": "/docs",
        "health": "/health",
        "prediction": "/predict",
    }