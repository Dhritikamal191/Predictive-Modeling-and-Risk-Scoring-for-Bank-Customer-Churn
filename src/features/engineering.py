import numpy as np
import pandas as pd


def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create reusable business and ML features.

    The function does not modify the original DataFrame.
    """

    data = df.copy()

    # ---------------------------------
    # Balance-to-Salary Ratio
    # ---------------------------------
    data["BalanceSalaryRatio"] = (
        data["Balance"]
        / data["EstimatedSalary"].replace(0, np.nan)
    )

    data["BalanceSalaryRatio"] = (
        data["BalanceSalaryRatio"]
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0)
    )

    # ---------------------------------
    # Customer Value
    # ---------------------------------
    data["CustomerValue"] = (
        data["Balance"] +
        data["EstimatedSalary"]
    )

    # ---------------------------------
    # Product Density
    # ---------------------------------
    data["ProductDensity"] = (
        data["NumOfProducts"]
        / data["Tenure"].replace(0, 1)
    )

    # ---------------------------------
    # Engagement Score
    # ---------------------------------
    data["EngagementScore"] = (
        data["IsActiveMember"]
        + data["HasCrCard"]
        + (data["NumOfProducts"] > 1).astype(int)
    )

    # ---------------------------------
    # Age × Tenure interaction
    # ---------------------------------
    data["AgeTenureInteraction"] = (
        data["Age"] * data["Tenure"]
    )

    # ---------------------------------
    # Has Balance
    # ---------------------------------
    data["HasBalance"] = (
        data["Balance"] > 0
    ).astype(int)

    return data


def get_model_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create features and remove identifiers/target
    that should not be used directly as predictors.
    """

    data = create_features(df)

    drop_columns = [
        "Year",
        "CustomerId",
        "Surname",
        "Exited",
    ]

    return data.drop(
        columns=[
            col for col in drop_columns
            if col in data.columns
        ]
    )


if __name__ == "__main__":

    df = pd.read_csv(
        "data/raw/European_Bank.csv"
    )

    features = create_features(df)

    print("\nFEATURE ENGINEERING REPORT")
    print("=" * 50)

    print(f"Original columns: {len(df.columns)}")
    print(f"New columns: {len(features.columns)}")

    print("\nNew features:")

    original_columns = set(df.columns)

    for column in features.columns:
        if column not in original_columns:
            print(f"  ✓ {column}")

    print("\nFeature preview:")
    print(features.head())
