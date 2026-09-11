import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    log_loss,
    confusion_matrix,
    brier_score_loss
)


def evaluate_model(model, X_test, y_test):
    """
    Evaluate a binary classification model.
    """

    y_prob = model.predict_proba(X_test)[:, 1]

    # Default classification threshold
    y_pred = (y_prob >= 0.5).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_test,
        y_pred
    ).ravel()

    metrics = {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(
            y_test,
            y_pred,
            zero_division=0
        ),
        "Recall": recall_score(
            y_test,
            y_pred,
            zero_division=0
        ),
        "F1": f1_score(
            y_test,
            y_pred,
            zero_division=0
        ),
        "ROC-AUC": roc_auc_score(
            y_test,
            y_prob
        ),
        "PR-AUC": average_precision_score(
            y_test,
            y_prob
        ),
        "Log Loss": log_loss(
            y_test,
            y_prob
        ),
        "Brier Score": brier_score_loss(
            y_test,
            y_prob
        ),
        "True Negative": tn,
        "False Positive": fp,
        "False Negative": fn,
        "True Positive": tp
    }

    return metrics


def find_best_threshold(model, X_test, y_test):
    """
    Find the classification threshold that maximizes F1.
    """

    probabilities = model.predict_proba(X_test)[:, 1]

    thresholds = np.arange(
        0.10,
        0.91,
        0.01
    )

    best_threshold = 0.5
    best_f1 = -1

    for threshold in thresholds:

        predictions = (
            probabilities >= threshold
        ).astype(int)

        score = f1_score(
            y_test,
            predictions,
            zero_division=0
        )

        if score > best_f1:
            best_f1 = score
            best_threshold = threshold

    return {
        "threshold": float(best_threshold),
        "f1": float(best_f1)
    }


def metrics_to_dataframe(results):
    """
    Convert model evaluation results into
    a comparison DataFrame.
    """

    return pd.DataFrame(results).T.sort_values(
        "ROC-AUC",
        ascending=False
    )
