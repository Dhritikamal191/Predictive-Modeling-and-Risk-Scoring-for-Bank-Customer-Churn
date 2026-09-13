"""
Customer Segmentation using K-Means Clustering

Purpose:
- Build customer segments using behavioral/value features
- Evaluate candidate K values
- Select optimal K using clustering metrics
- Save model, scaler, assignments, metrics and cluster profiles
"""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.cluster import KMeans
from sklearn.metrics import (
    silhouette_score,
    calinski_harabasz_score,
    davies_bouldin_score,
)
from sklearn.preprocessing import StandardScaler


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_PATH = PROJECT_ROOT / "data" / "raw" / "European_Bank.csv"

MODEL_DIR = PROJECT_ROOT / "artifacts" / "models"
METRICS_DIR = PROJECT_ROOT / "artifacts" / "metrics"
METADATA_DIR = PROJECT_ROOT / "artifacts" / "metadata"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
METRICS_DIR.mkdir(parents=True, exist_ok=True)
METADATA_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# CONFIGURATION
# ============================================================

RANDOM_STATE = 42

K_RANGE = range(2, 9)

SEGMENT_FEATURES = [
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

def create_segmentation_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create customer-level behavioral and value features.
    """

    data = df.copy()

    # Avoid division by zero
    data["BalanceSalaryRatio"] = np.where(
        data["EstimatedSalary"] > 0,
        data["Balance"] / data["EstimatedSalary"],
        0,
    )

    data["CustomerValue"] = (
        data["Balance"] + data["EstimatedSalary"]
    )

    data["ProductDensity"] = (
        data["NumOfProducts"] /
        data["Tenure"].clip(lower=1)
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
# LOAD DATA
# ============================================================

def load_data() -> pd.DataFrame:

    print("\n[1/6] Loading dataset...")

    df = pd.read_csv(DATA_PATH)

    print(f"Dataset shape: {df.shape}")

    return df


# ============================================================
# PREPARE FEATURES
# ============================================================

def prepare_features(df: pd.DataFrame):

    print("\n[2/6] Creating segmentation features...")

    data = create_segmentation_features(df)

    X = data[SEGMENT_FEATURES].copy()

    # Replace problematic values
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(0)

    print(f"Segmentation features: {X.shape[1]}")

    print("\nFeatures used:")

    for feature in SEGMENT_FEATURES:
        print(f"   ✓ {feature}")

    return data, X


# ============================================================
# SCALE FEATURES
# ============================================================

def scale_features(X: pd.DataFrame):

    print("\n[3/6] Scaling features...")

    scaler = StandardScaler()

    X_scaled = scaler.fit_transform(X)

    return scaler, X_scaled


# ============================================================
# EVALUATE K VALUES
# ============================================================

def evaluate_k_values(X_scaled):

    print("\n[4/6] Evaluating K-Means configurations...")
    print("-" * 60)

    results = []

    for k in K_RANGE:

        model = KMeans(
            n_clusters=k,
            random_state=RANDOM_STATE,
            n_init=20,
            max_iter=500,
        )

        labels = model.fit_predict(X_scaled)

        inertia = model.inertia_

        silhouette = silhouette_score(
            X_scaled,
            labels,
        )

        calinski = calinski_harabasz_score(
            X_scaled,
            labels,
        )

        davies = davies_bouldin_score(
            X_scaled,
            labels,
        )

        results.append(
            {
                "K": k,
                "Inertia": inertia,
                "Silhouette": silhouette,
                "Calinski_Harabasz": calinski,
                "Davies_Bouldin": davies,
            }
        )

        print(
            f"K={k} | "
            f"Inertia={inertia:.2f} | "
            f"Silhouette={silhouette:.4f} | "
            f"CH={calinski:.2f} | "
            f"DB={davies:.4f}"
        )

    metrics_df = pd.DataFrame(results)

    return metrics_df


# ============================================================
# SELECT BEST K
# ============================================================

def select_best_k(metrics_df: pd.DataFrame) -> int:
    """
    Select K using the strongest overall silhouette score.

    Silhouette:
        Higher = better separated clusters.

    We also retain the other metrics for analysis.
    """

    best_row = metrics_df.loc[
        metrics_df["Silhouette"].idxmax()
    ]

    best_k = int(best_row["K"])

    print("\nSelected K:")
    print(f"   ✓ K = {best_k}")
    print(
        f"   Silhouette = {best_row['Silhouette']:.4f}"
    )

    return best_k


# ============================================================
# TRAIN FINAL MODEL
# ============================================================

def train_final_model(X_scaled, best_k):

    print("\n[5/6] Training final K-Means model...")

    model = KMeans(
        n_clusters=best_k,
        random_state=RANDOM_STATE,
        n_init=20,
        max_iter=500,
    )

    labels = model.fit_predict(X_scaled)

    print(f"Final clusters: {best_k}")

    return model, labels


# ============================================================
# CLUSTER PROFILING
# ============================================================

def create_cluster_profile(
    data: pd.DataFrame,
    labels,
):

    data = data.copy()

    data["Cluster"] = labels

    profile_features = [
        "Age",
        "CreditScore",
        "Tenure",
        "Balance",
        "EstimatedSalary",
        "NumOfProducts",
        "IsActiveMember",
        "CustomerValue",
        "EngagementScore",
        "Exited",
    ]

    profile = (
        data
        .groupby("Cluster")[profile_features]
        .agg(["mean", "median"])
        .round(2)
    )

    cluster_size = (
        data["Cluster"]
        .value_counts()
        .sort_index()
        .rename("CustomerCount")
    )

    cluster_percentage = (
        data["Cluster"]
        .value_counts(normalize=True)
        .sort_index()
        .mul(100)
        .round(2)
        .rename("CustomerPercentage")
    )

    churn_rate = (
        data
        .groupby("Cluster")["Exited"]
        .mean()
        .mul(100)
        .round(2)
        .rename("ChurnRate")
    )

    summary = pd.concat(
        [
            cluster_size,
            cluster_percentage,
            churn_rate,
        ],
        axis=1,
    )

    return data, profile, summary


# ============================================================
# MAIN
# ============================================================

def main():

    print("\n" + "=" * 60)
    print("BANK CHURN — CUSTOMER SEGMENTATION")
    print("=" * 60)

    # 1
    df = load_data()

    # 2
    data, X = prepare_features(df)

    # 3
    scaler, X_scaled = scale_features(X)

    # 4
    metrics_df = evaluate_k_values(X_scaled)

    # 5
    best_k = select_best_k(metrics_df)

    model, labels = train_final_model(
        X_scaled,
        best_k,
    )

    # Cluster profiles
    clustered_data, profile, summary = (
        create_cluster_profile(
            data,
            labels,
        )
    )

    print("\nCluster summary:")
    print(summary)

    # ========================================================
    # SAVE ARTIFACTS
    # ========================================================

    print("\n[6/6] Saving segmentation artifacts...")

    # Model
    joblib.dump(
        model,
        MODEL_DIR / "kmeans_customer_segmentation.pkl",
    )

    # Scaler
    joblib.dump(
        scaler,
        MODEL_DIR / "segmentation_scaler.pkl",
    )

    # Customer assignments
    customer_columns = [
        "CustomerId",
        "Surname",
        "Geography",
        "Gender",
        "Age",
        "CreditScore",
        "Balance",
        "EstimatedSalary",
        "NumOfProducts",
        "IsActiveMember",
        "Exited",
        "Cluster",
    ]

    clustered_data[
        customer_columns
    ].to_csv(
        METRICS_DIR / "customer_segments.csv",
        index=False,
    )

    # K evaluation
    metrics_df.to_csv(
        METRICS_DIR / "clustering_metrics.csv",
        index=False,
    )

    # Cluster profile
    profile.to_csv(
        METRICS_DIR / "cluster_profiles.csv",
    )

    # Cluster summary
    summary.to_csv(
        METRICS_DIR / "cluster_summary.csv",
    )

    # Metadata
    metadata = {
        "best_k": best_k,
        "random_state": RANDOM_STATE,
        "k_range": list(K_RANGE),
        "features": SEGMENT_FEATURES,
        "model": "KMeans",
    }

    joblib.dump(
        metadata,
        METADATA_DIR / "segmentation_metadata.pkl",
    )

    print("\n" + "=" * 60)
    print("SEGMENTATION COMPLETE")
    print("=" * 60)

    print("\nSaved artifacts:")

    print(
        "   ✓ artifacts/models/"
        "kmeans_customer_segmentation.pkl"
    )

    print(
        "   ✓ artifacts/models/"
        "segmentation_scaler.pkl"
    )

    print(
        "   ✓ artifacts/metrics/"
        "customer_segments.csv"
    )

    print(
        "   ✓ artifacts/metrics/"
        "clustering_metrics.csv"
    )

    print(
        "   ✓ artifacts/metrics/"
        "cluster_profiles.csv"
    )

    print(
        "   ✓ artifacts/metrics/"
        "cluster_summary.csv"
    )

    print(
        "\nBest K: "
        f"{best_k}"
    )


if __name__ == "__main__":
    main()