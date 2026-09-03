from fastapi.testclient import TestClient

from api.main import app


client = TestClient(app)


def test_root():
    response = client.get("/")

    assert response.status_code == 200

    data = response.json()

    assert "message" in data
    assert "docs" in data
    assert "prediction" in data


def test_health():
    response = client.get("/health")

    assert response.status_code == 200

    data = response.json()

    assert data["status"] == "healthy"
    assert data["model_loaded"] is True


def test_prediction():

    payload = {
        "CreditScore": 650,
        "Geography": "France",
        "Gender": "Female",
        "Age": 45,
        "Tenure": 5,
        "Balance": 100000,
        "NumOfProducts": 2,
        "HasCrCard": 1,
        "IsActiveMember": 0,
        "EstimatedSalary": 80000,
    }

    response = client.post(
        "/predict",
        json=payload
    )

    assert response.status_code == 200

    data = response.json()

    assert "churn_probability" in data
    assert "churn_prediction" in data
    assert "risk_category" in data

    assert 0 <= data["churn_probability"] <= 1
    assert data["churn_prediction"] in [0, 1]

    assert data["risk_category"] in [
        "Low",
        "Medium",
        "High",
        "Critical",
    ]


def test_invalid_credit_score():

    payload = {
        "CreditScore": 1000,
        "Geography": "France",
        "Gender": "Female",
        "Age": 45,
        "Tenure": 5,
        "Balance": 100000,
        "NumOfProducts": 2,
        "HasCrCard": 1,
        "IsActiveMember": 0,
        "EstimatedSalary": 80000,
    }

    response = client.post(
        "/predict",
        json=payload
    )

    assert response.status_code == 422