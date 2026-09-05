"""
Bank Churn MLOps Pipeline

Runs the complete machine learning workflow:
1. Data validation
2. Feature engineering
3. Supervised model training
4. Customer segmentation
5. Risk & value scoring
6. Drift monitoring
7. Performance monitoring
"""

import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent


STEPS = [
    (
        "DATA VALIDATION",
        "src.data.validation",
    ),
    (
        "FEATURE ENGINEERING",
        "src.features.engineering",
    ),
    (
        "SUPERVISED MODEL TRAINING",
        "src.supervised.train",
    ),
    (
        "CUSTOMER SEGMENTATION",
        "src.unsupervised.clustering",
    ),
    (
        "RISK & VALUE SCORING",
        "src.risk.risk_scoring",
    ),
    (
        "DRIFT MONITORING",
        "src.monitoring.drift",
    ),
    (
        "PERFORMANCE MONITORING",
        "src.monitoring.performance",
    ),
]


def run_step(name, module):
    print("\n")
    print("=" * 70)
    print(f"RUNNING: {name}")
    print("=" * 70)

    result = subprocess.run(
        [sys.executable, "-m", module],
        cwd=PROJECT_ROOT,
    )

    if result.returncode != 0:
        print("\n")
        print("=" * 70)
        print(f"❌ FAILED: {name}")
        print("=" * 70)
        sys.exit(result.returncode)

    print("\n")
    print(f"✅ COMPLETED: {name}")


def main():
    print("\n")
    print("=" * 70)
    print("BANK CHURN — COMPLETE MLOPS PIPELINE")
    print("=" * 70)

    for name, module in STEPS:
        run_step(name, module)

    print("\n")
    print("=" * 70)
    print("🎉 COMPLETE MLOPS PIPELINE FINISHED")
    print("=" * 70)
    print("\nAll major pipeline stages completed successfully.")
    print("Artifacts are available under:")
    print("   artifacts/")
    print("\n")


if __name__ == "__main__":
    main()