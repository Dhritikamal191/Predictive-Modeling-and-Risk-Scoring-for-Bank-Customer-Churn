from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException

from api.schemas import CustomerInput, PredictionResponse
from src.features.engineering import create_features


ROOT = Path(__file__).resolve().parents[1]

MODEL_PATH = (
    ROOT
    / "artifacts"
    / "models"
    / "gradient_boosting.pkl"
)


app = FastAPI(
    title="Bank Customer Churn Prediction API",
    description=(
        "Production-style API for bank customer churn "
        "prediction and risk classification."
    ),
    version="1.0.0",
)


# ---------------------------------------------------------
# Load model once when API starts
# ---------------------------------------------------------

try:
    model = joblib.load(MODEL_PATH)

except Exception as e:
    model = None
    model_load_error = str(e)


# ---------------------------------------------------------
# Health check
# ---------------------------------------------------------

@app.get("/health")
def health():

    if model is None:
        return {
            "status": "unhealthy",
            "model_loaded": False,
            "error": model_load_error,
        }

    return {
        "status": "healthy",
        "model_loaded": True,
        "model": "Gradient Boosting",
    }


# ---------------------------------------------------------
# Prediction
# ---------------------------------------------------------

@app.post(
    "/predict",
    response_model=PredictionResponse
)
def predict(customer: CustomerInput):

    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model is not available."
        )

    try:

        # Convert request to DataFrame
        df = pd.DataFrame([customer.model_dump()])

        # Apply EXACT same feature engineering
        df = create_features(df)

        # Remove fields not used by the model
        X = df.drop(
            columns=[
                "Year",
                "CustomerId",
                "Surname",
                "Exited",
            ],
            errors="ignore",
        )

        # Probability
        probability = float(
            model.predict_proba(X)[0][1]
        )

        # Default classification threshold
        prediction = int(
            probability >= 0.50
        )

        # Risk category
        if probability < 0.20:
            risk = "Low"

        elif probability < 0.40:
            risk = "Medium"

        elif probability < 0.60:
            risk = "High"

        else:
            risk = "Critical"

        return PredictionResponse(
            churn_probability=round(
                probability,
                6
            ),
            churn_prediction=prediction,
            risk_category=risk,
        )

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        )


# ---------------------------------------------------------
# Root endpoint
# ---------------------------------------------------------

@app.get("/")
def root():

    return {
        "message": "Bank Customer Churn Prediction API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health",
        "prediction": "/predict",
    }