import os
import joblib
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, roc_auc_score

# Load dataset
df = pd.read_csv("Bank.csv")

# Drop unnecessary columns
df = df.drop(["CustomerId", "Surname"], axis=1)

# Convert categorical variables to numeric
df=pd.get_dummies(df,columns=["Geography","Gender"])

# Features and target
X = df.drop("Exited", axis=1)
y = df["Exited"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
feature_names=X.columns

scaler= StandardScaler()
X_scaled=scaler.fit_transform(X)
X_train= scaler.fit_transform(X_train)
X_test= scaler.transform(X_test)
joblib.dump(scaler, "scaler.pkl")

# Train model
dt_model= DecisionTreeClassifier(max_depth=5, random_state=42)
dt_model.fit(X_train, y_train)

rf_model = RandomForestClassifier(n_estimators=200, random_state=42)
rf_model.fit(X_train, y_train)

gb_model= GradientBoostingClassifier(n_estimators=200, learning_rate=0.1, random_state=42)
gb_model.fit(X_train, y_train)

xgb_model=XGBClassifier(n_estimators=200,learning_rate=0.1, max_depth=5, use_label_encoder=False, eval_metric='logloss')
xgb_model.fit(X_train, y_train)

models={"Decision Tree": dt_model, "Random Forest": rf_model, "Gradient Boosting": gb_model, "XGBoost": xgb_model}

print("\nModel Performance:\n")
for feature_names, model in models.items():
    y_pred= model.predict(X_test)
    y_prob= model.predict_proba(X_test)[:,1]

    acc= accuracy_score(y_test, y_pred)
    auc= roc_auc_score(y_test, y_prob)

    print(f"{feature_names}:")
    print(f" Accuracy: {acc:.4f}")
    print(f" ROC-AUC: {auc:.4f}\n")

model_path=os.path.abspath("model.pkl")
joblib.dump((dt_model,feature_names),"decision_tree.pkl")
joblib.dump((rf_model,feature_names),"random_forest.pkl")
joblib.dump((gb_model,feature_names),"gradient_boosting.pkl")
joblib.dump((xgb_model,feature_names),"xgboost.pkl")
joblib.dump(scaler,"scaler.pkl")
joblib.dump(X.columns, "columns.pkl")
joblib.dump(X_scaled, "X_scaled.pkl")
print("All models saved successfully")