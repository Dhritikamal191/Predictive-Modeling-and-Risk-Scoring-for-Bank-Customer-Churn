import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, roc_auc_score

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier

from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE

Load data

df = pd.read_csv("European_Bank.csv")
df = df.drop(["Year","CustomerId", "Surname"], axis=1)

Split features

X = df.drop("Exited", axis=1)
y = df["Exited"]

Identify columns

categorical_cols = ["Geography", "Gender"]
numeric_cols = [col for col in X.columns if col not in categorical_cols]

Preprocessing

from sklearn.preprocessing import OneHotEncoder

preprocessor = ColumnTransformer([
("num", StandardScaler(), numeric_cols),
("cat", OneHotEncoder(drop="first"), categorical_cols)
])

Train-test split

X_train, X_test, y_train, y_test = train_test_split(
X, y, test_size=0.2, random_state=42
)

Models

models = {
"Logistic Regression": LogisticRegression(max_iter=2000, C=0.5),
"Decision Tree": DecisionTreeClassifier(max_depth=6),
"Random Forest": RandomForestClassifier(n_estimators=100, max_depth=8),
"Gradient Boosting": GradientBoostingClassifier(n_estimators=200),
"XGBoost": XGBClassifier(n_estimators=200, eval_metric='logloss')
}

Create pipelines

pipelines = {}

print("\nModel Performance:\n")

for name, model in models.items():

pipeline = ImbPipeline(steps=[  
    ("preprocessor", preprocessor),  
    ("smote", SMOTE(random_state=42)),  
    ("model", model)  
])  

pipeline.fit(X_train, y_train)  

y_pred = pipeline.predict(X_test)  
y_prob = pipeline.predict_proba(X_test)[:, 1]  

acc = accuracy_score(y_test, y_pred)  
auc = roc_auc_score(y_test, y_prob)  

print(f"{name}: Accuracy={acc:.4f}, ROC-AUC={auc:.4f}")  

pipelines[name] = pipeline

✅ Save ONE PKL

joblib.dump(pipelines, "all_models_pipeline.pkl")

print("✅ All models saved in ONE pipeline file")
