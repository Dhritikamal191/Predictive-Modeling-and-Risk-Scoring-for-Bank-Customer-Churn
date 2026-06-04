import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

from sklearn.metrics import accuracy_score, roc_auc_score

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier
)

from xgboost import XGBClassifier

from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE


# ==========================
# LOAD DATA
# ==========================

df = pd.read_csv("Bank.csv")

df = df.drop(
    columns=["CustomerId", "Surname"],
    errors="ignore"
)

X = df.drop("Exited", axis=1)
y = df["Exited"]


# ==========================
# FEATURE GROUPS
# ==========================

categorical_cols = [
    "Geography",
    "Gender"
]

numeric_cols = [
    col for col in X.columns
    if col not in categorical_cols
]


# ==========================
# PREPROCESSOR
# ==========================

preprocessor = ColumnTransformer(
    transformers=[
        (
            "num",
            StandardScaler(),
            numeric_cols
        ),
        (
            "cat",
            OneHotEncoder(
                drop="first",
                handle_unknown="ignore"
            ),
            categorical_cols
        )
    ]
)


# ==========================
# TRAIN TEST SPLIT
# ==========================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


# ==========================
# MODELS
# ==========================

models = {

    "Logistic Regression": LogisticRegression(
        max_iter=3000,
        C=0.5,
        solver="lbfgs", 
        random_state=42
    ),

    "Decision Tree": DecisionTreeClassifier(
        max_depth=6,
        random_state=42
    ),

    "Random Forest": RandomForestClassifier(
        n_estimators=100,
        max_depth=8,
        random_state=42
    ),

    "Gradient Boosting": GradientBoostingClassifier(
        n_estimators=200,
        random_state=42
    ),

    "XGBoost": XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss",
        random_state=42
    )
}


# ==========================
# TRAIN MODELS
# ==========================

pipelines = {}

print("\nMODEL PERFORMANCE\n")

for name, model in models.items():

    pipeline = ImbPipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("smote", SMOTE(random_state=42)),
            ("model", model)
        ]
    )

    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)

    y_prob = pipeline.predict_proba(
        X_test
    )[:, 1]

    acc = accuracy_score(
        y_test,
        y_pred
    )

    auc = roc_auc_score(
        y_test,
        y_prob
    )

    print(
        f"{name} | "
        f"Accuracy={acc:.4f} | "
        f"ROC-AUC={auc:.4f}"
    )

    pipelines[name] = pipeline


# ==========================
# VERIFY LOGISTIC REGRESSION
# ==========================

lr_model = pipelines[
    "Logistic Regression"
].named_steps["model"]

print("\nLOGISTIC REGRESSION CHECK")
print("Logistic Regression Model:")
print(type(lr_model))
print(lr_model)

# ==========================
# SAVE MODELS
# ==========================

joblib.dump(
    pipelines,
    "all_models_pipeline.pkl"
)

print(
    "\nSUCCESS: all_models_pipeline.pkl saved"
)
