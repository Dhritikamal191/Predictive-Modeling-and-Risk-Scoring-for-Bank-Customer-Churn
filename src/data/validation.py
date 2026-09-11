import pandas as pd


REQUIRED_COLUMNS = [
    "Year",
    "CustomerId",
    "Surname",
    "CreditScore",
    "Geography",
    "Gender",
    "Age",
    "Tenure",
    "Balance",
    "NumOfProducts",
    "HasCrCard",
    "IsActiveMember",
    "EstimatedSalary",
    "Exited",
]


def validate_dataset(df: pd.DataFrame) -> dict:
    """
    Validate the bank customer dataset before model training.
    """

    errors = []
    warnings = []

    # -----------------------------
    # Required columns
    # -----------------------------
    missing_columns = [
        col for col in REQUIRED_COLUMNS
        if col not in df.columns
    ]

    if missing_columns:
        errors.append(
            f"Missing required columns: {missing_columns}"
        )

    # Stop further validation if schema is broken
    if errors:
        return {
            "valid": False,
            "errors": errors,
            "warnings": warnings,
        }

    # -----------------------------
    # Missing values
    # -----------------------------
    missing_values = df[REQUIRED_COLUMNS].isnull().sum()

    missing_columns_found = (
        missing_values[missing_values > 0]
        .to_dict()
    )

    if missing_columns_found:
        errors.append(
            f"Missing values found: {missing_columns_found}"
        )

    # -----------------------------
    # Duplicate customers
    # -----------------------------
    duplicate_ids = df["CustomerId"].duplicated().sum()

    if duplicate_ids > 0:
        warnings.append(
            f"{duplicate_ids} duplicate CustomerId values found."
        )

    # -----------------------------
    # Target validation
    # -----------------------------
    invalid_target = ~df["Exited"].isin([0, 1])

    if invalid_target.any():
        errors.append(
            "Exited must contain only 0 and 1."
        )

    # -----------------------------
    # Numeric range checks
    # -----------------------------
    if (df["CreditScore"] < 0).any():
        errors.append("CreditScore contains negative values.")

    if (df["Age"] < 18).any():
        warnings.append(
            "Customers below age 18 were detected."
        )

    if (df["Tenure"] < 0).any():
        errors.append("Tenure contains negative values.")

    if (df["Balance"] < 0).any():
        errors.append("Balance contains negative values.")

    if (df["NumOfProducts"] < 1).any():
        errors.append(
            "NumOfProducts contains values below 1."
        )

    if (df["EstimatedSalary"] < 0).any():
        errors.append(
            "EstimatedSalary contains negative values."
        )

    # -----------------------------
    # Binary columns
    # -----------------------------
    binary_columns = [
        "HasCrCard",
        "IsActiveMember",
        "Exited",
    ]

    for column in binary_columns:
        invalid = ~df[column].isin([0, 1])

        if invalid.any():
            errors.append(
                f"{column} must contain only 0 and 1."
            )

    # -----------------------------
    # Categorical validation
    # -----------------------------
    if df["Geography"].isnull().any():
        errors.append("Geography contains null values.")

    if df["Gender"].isnull().any():
        errors.append("Gender contains null values.")

    # -----------------------------
    # Return results
    # -----------------------------
    return {
        "valid": len(errors) == 0,
        "errors": errors,
        "warnings": warnings,
    }


def validate_file(path: str) -> dict:
    """
    Load and validate a CSV file.
    """

    df = pd.read_csv(path)

    result = validate_dataset(df)

    result["rows"] = len(df)
    result["columns"] = len(df.columns)

    return result


if __name__ == "__main__":

    path = "data/raw/European_Bank.csv"

    result = validate_file(path)

    print("\nDATA VALIDATION REPORT")
    print("=" * 50)

    print(f"Rows: {result['rows']}")
    print(f"Columns: {result['columns']}")
    print(f"Valid: {result['valid']}")

    if result["errors"]:
        print("\nERRORS:")

        for error in result["errors"]:
            print(f"❌ {error}")

    if result["warnings"]:
        print("\nWARNINGS:")

        for warning in result["warnings"]:
            print(f"⚠️ {warning}")

    if result["valid"]:
        print("\n✅ Dataset validation passed.")
    else:
        print("\n❌ Dataset validation failed.")
