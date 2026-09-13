import pandas as pd

from src.features.engineering import create_features


def test_feature_engineering():

    df = pd.DataFrame({
        "CreditScore": [650],
        "Geography": ["France"],
        "Gender": ["Female"],
        "Age": [45],
        "Tenure": [5],
        "Balance": [100000.0],
        "NumOfProducts": [2],
        "HasCrCard": [1],
        "IsActiveMember": [0],
        "EstimatedSalary": [80000.0],
    })

    result = create_features(df)

    expected_features = [
        "BalanceSalaryRatio",
        "CustomerValue",
        "ProductDensity",
        "EngagementScore",
        "AgeTenureInteraction",
        "HasBalance",
    ]

    for feature in expected_features:
        assert feature in result.columns


def test_customer_value():

    df = pd.DataFrame({
        "CreditScore": [650],
        "Geography": ["France"],
        "Gender": ["Female"],
        "Age": [45],
        "Tenure": [5],
        "Balance": [100000.0],
        "NumOfProducts": [2],
        "HasCrCard": [1],
        "IsActiveMember": [0],
        "EstimatedSalary": [80000.0],
    })

    result = create_features(df)

    assert result["CustomerValue"].iloc[0] == 180000.0