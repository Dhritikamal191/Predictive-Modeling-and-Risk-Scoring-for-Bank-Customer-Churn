"""
Customer Segment Profiling and PCA Visualization
"""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_PATH = PROJECT_ROOT / "data" / "raw" / "European_Bank.csv"

SEGMENTS_PATH = (
    PROJECT_ROOT
    / "artifacts"
    / "metrics"
    / "customer_segments.csv"
)

METRICS_DIR = PROJECT_ROOT / "artifacts" / "metrics"
METADATA_DIR = PROJECT_ROOT / "artifacts" / "metadata"

METRICS_DIR.mkdir(parents=True, exist_ok=True)
METADATA_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# FEATURES
# ============================================================

PROFILE_FEATURES = [
    "CreditScore",
    "Age",
    "Tenure",
    "Balance",
    "NumOfProducts",
    "IsActiveMember",
    "EstimatedSalary",
    "BalanceSalaryRatio",
    "CustomerValue",
    "ProductDensity",
    "EngagementScore",
    "AgeTenureInteraction",
    "HasBalance",
]


# ============================================================
# FEATURE ENGINEERING
# ============================================================

def create_features(df):

    data = df.copy()

    data["BalanceSalaryRatio"] = np.where(
        data["EstimatedSalary"] > 0,
        data["Balance"] / data["EstimatedSalary"],
        0,
    )

    data["CustomerValue"] = (
        data["Balance"] + data["EstimatedSalary"]
    )

    data["ProductDensity"] = (
        data["NumOfProducts"]
        / data["Tenure"].clip(lower=1)
    )

    data["EngagementScore"] = (
        data["IsActiveMember"]
        + data["HasCrCard"]
        + (data["NumOfProducts"] > 1).astype(int)
    )

    data["AgeTenureInteraction"] = (
        data["Age"] * data["Tenure"]
    )

    data["HasBalance"] = (
        data["Balance"] > 0
    ).astype(int)

    return data


# ============================================================
# LOAD
# ============================================================

def load_data():

    print("\n[1/5] Loading customer data...")

    df = pd.read_csv(DATA_PATH)

    segments = pd.read_csv(SEGMENTS_PATH)

    print(f"Customers: {len(df):,}")

    return df, segments


# ============================================================
# BUILD PROFILE
# ============================================================

def build_profile(df, segments):

    print("\n[2/5] Building cluster profiles...")

    data = create_features(df)

    # Use CustomerId to safely map cluster assignments
    cluster_map = segments[
        ["CustomerId", "Cluster"]
    ]

    data = data.merge(
        cluster_map,
        on="CustomerId",
        how="inner",
    )

    # Aggregate important business metrics
    profile = (
        data
        .groupby("Cluster")
        .agg(
            CustomerCount=("CustomerId", "count"),
            ChurnRate=("Exited", "mean"),
            AvgAge=("Age", "mean"),
            AvgCreditScore=("CreditScore", "mean"),
            AvgBalance=("Balance", "mean"),
            AvgSalary=("EstimatedSalary", "mean"),
            AvgCustomerValue=("CustomerValue", "mean"),
            AvgProducts=("NumOfProducts", "mean"),
            ActiveMemberRate=("IsActiveMember", "mean"),
            AvgEngagement=("EngagementScore", "mean"),
            AvgTenure=("Tenure", "mean"),
        )
        .reset_index()
    )

    profile["CustomerPercentage"] = (
        profile["CustomerCount"]
        / profile["CustomerCount"].sum()
        * 100
    )

    profile["ChurnRate"] *= 100
    profile["ActiveMemberRate"] *= 100

    numeric_cols = profile.columns[
        profile.columns != "Cluster"
    ]

    profile[numeric_cols] = profile[numeric_cols].round(2)

    profile = profile.sort_values("Cluster")

    return data, profile


# ============================================================
# COMPARE CLUSTERS
# ============================================================

def create_comparison(profile):

    print("\n[3/5] Creating cluster comparison...")

    comparison = profile.copy()

    metrics = [
        "ChurnRate",
        "AvgAge",
        "AvgCreditScore",
        "AvgBalance",
        "AvgSalary",
        "AvgCustomerValue",
        "AvgProducts",
        "ActiveMemberRate",
        "AvgEngagement",
        "AvgTenure",
    ]

    rows = []

    for metric in metrics:

        values = comparison.set_index(
            "Cluster"
        )[metric]

        if len(values) >= 2:

            cluster_0 = values.iloc[0]
            cluster_1 = values.iloc[1]

            difference = cluster_1 - cluster_0

            if cluster_0 != 0:
                percentage_difference = (
                    difference
                    / abs(cluster_0)
                    * 100
                )
            else:
                percentage_difference = np.nan

            rows.append(
                {
                    "Metric": metric,
                    "Cluster_0": cluster_0,
                    "Cluster_1": cluster_1,
                    "Difference": difference,
                    "PercentageDifference": (
                        percentage_difference
                    ),
                }
            )

    comparison_df = pd.DataFrame(rows)

    return comparison_df


# ============================================================
# PCA
# ============================================================

def create_pca(data):

    print("\n[4/5] Creating PCA representation...")

    X = data[PROFILE_FEATURES].copy()

    X = X.replace(
        [np.inf, -np.inf],
        np.nan,
    )

    X = X.fillna(0)

    scaler = StandardScaler()

    X_scaled = scaler.fit_transform(X)

    pca = PCA(
        n_components=2,
        random_state=42,
    )

    components = pca.fit_transform(
        X_scaled
    )

    pca_df = pd.DataFrame(
        {
            "CustomerId": data["CustomerId"],
            "Cluster": data["Cluster"],
            "PCA1": components[:, 0],
            "PCA2": components[:, 1],
        }
    )

    explained_variance = (
        pca.explained_variance_ratio_
        * 100
    )

    print(
        f"PCA1 explained variance: "
        f"{explained_variance[0]:.2f}%"
    )

    print(
        f"PCA2 explained variance: "
        f"{explained_variance[1]:.2f}%"
    )

    print(
        f"Total explained variance: "
        f"{explained_variance.sum():.2f}%"
    )

    return pca_df, explained_variance


# ============================================================
# MAIN
# ============================================================

def main():

    print("\n" + "=" * 60)
    print("BANK CHURN — SEGMENT PROFILING")
    print("=" * 60)

    # 1
    df, segments = load_data()

    # 2
    data, profile = build_profile(
        df,
        segments,
    )

    print("\nCluster profile:")
    print(profile.to_string(index=False))

    # 3
    comparison = create_comparison(
        profile
    )

    # 4
    pca_df, explained_variance = create_pca(
        data
    )

    # ========================================================
    # SAVE
    # ========================================================

    print("\n[5/5] Saving profiling artifacts...")

    profile.to_csv(
        METRICS_DIR
        / "cluster_business_profile.csv",
        index=False,
    )

    comparison.to_csv(
        METRICS_DIR
        / "cluster_comparison.csv",
        index=False,
    )

    pca_df.to_csv(
        METRICS_DIR
        / "pca_customer_segments.csv",
        index=False,
    )

    metadata = {
        "pca_components": 2,
        "pca_explained_variance": (
            explained_variance.tolist()
        ),
        "total_explained_variance": float(
            explained_variance.sum()
        ),
        "features": PROFILE_FEATURES,
    }

    joblib.dump(
        metadata,
        METADATA_DIR
        / "pca_metadata.pkl",
    )

    print("\nSaved artifacts:")

    print(
        "   ✓ cluster_business_profile.csv"
    )

    print(
        "   ✓ cluster_comparison.csv"
    )

    print(
        "   ✓ pca_customer_segments.csv"
    )

    print(
        "   ✓ pca_metadata.pkl"
    )

    print("\n" + "=" * 60)
    print("PROFILING COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()