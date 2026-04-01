import streamlit as st
from PIL import Image
import pandas as pd
import numpy as np
import pickle
import joblib
import shap
import plotly.express as px
import matplotlib.pyplot as plt
from streamlit_autorefresh import st_autorefresh
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go

st.set_page_config(page_title="Predictive Modeling and Risk Scoring for Bank Customer Churn",layout="wide")

col1,col2=st.columns([0.5,6])

with col1:
     st.image("office.png",width=60)
with col2:
     st.title("Predictive Modeling and Risk Scoring for Bank Customer Churn")

# --------------------------------------------------
# Load Model
# --------------------------------------------------
dt_model,feature_names=joblib.load("decision_tree.pkl")
rf_model,feature_names=joblib.load("random_forest.pkl")
gb_model,feature_names=joblib.load("gradient_boosting.pkl")
xgb_model,feature_names=joblib.load("xgboost.pkl")
scaler=joblib.load("scaler.pkl")
columns=joblib.load("columns.pkl")
X_scaled=joblib.load("X_scaled.pkl")

# Load dataset for visualization

df=pd.read_csv("Bank.csv")

st.sidebar.image("mentor.png",width=150)
st.sidebar.header("Customer Feature Inputs")

credit_score = st.sidebar.number_input("Credit Score", 300, 900, 600)
age = st.sidebar.number_input("Age", 18, 100, 40)
tenure = st.sidebar.slider("Tenure", 0, 10, 5)
balance = st.sidebar.number_input("Balance", 0.0, 250000.0, 50000.0)
products = st.sidebar.slider("Number of Products", 1, 4, 2)
active_member = st.sidebar.selectbox("Active Member", [0,1])
has_card = st.sidebar.selectbox("Has Credit Card", [0,1])
salary = st.sidebar.number_input("Estimated Salary", 0.0, 200000.0, 50000.0)
gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
geography = st.sidebar.selectbox("Geography", ["France", "Germany", "Spain"])

# --------------------------------------------------
# Prediction Input
# --------------------------------------------------

input_df =pd.DataFrame({
    "CreditScore":[credit_score],
    "Age":[age],
    "Tenure":[tenure],
    "Balance":[balance],
    "NumOfProducts":[products],
    "HasCrCard":[has_card],
    "IsActiveMember":[active_member],
    "EstimatedSalary":[salary],
    "Geography":[geography],
    "Gender":[gender]
})
input_encoded = pd.get_dummies(input_df)
input_encoded= input_encoded.reindex(columns=columns,fill_value=0)
input_scaled= scaler.transform(input_encoded)
model_choice=st.radio("Select Model",["Decision Tree", "Random Forest", "Gradient Boosting", "XGBoost"], key="model_selector")

model= rf_model

if model_choice=="Decision Tree": 
   model=dt_model
elif model_choice=="Gradient Boosting":
     model=gb_model
elif model_choice=="XGBoost":
     model=xgb_model

if st.button("Predict"):
                        prediction= model.predict(input_encoded)[0]

probability = model.predict_proba(input_encoded)[0][1]
risk_score = probability * 100

X = df.drop(["Exited","CustomerId","Surname"],axis=1,errors="ignore")
input_df = input_df.reindex(columns=columns,fill_value=0)
df=df.drop(["RowNumber","CustomerId","Surname"],axis=1,errors="ignore")
X=pd.get_dummies(X,columns=["Geography","Gender"],drop_first=True)
X=df.copy()
print("Model features:",columns)
print("input features:",columns)
X=X.drop(columns=["Exited","probability"],errors="ignore")
X=X.reindex(columns=columns,fill_value=0)

input_scaled= scaler.transform(input_df)

# --------------------------------------------------
# Churn Prediction
# --------------------------------------------------
pred=model.predict(input_scaled)[0]
df["probability"]=model.predict_proba(input_encoded)[0][1]


prob = model.predict_proba(input_scaled)[0][1]

if pred==1:
           st.error("Customer is likely to CHURN")
else:
     st.success("Customer is NOT likely to churn")
risk_score = prob * 100

# --------------------------------------------------
# Risk Category
# --------------------------------------------------
if risk_score < 30:
    risk = "Low Risk"
elif risk_score < 70:
    risk = "Medium Risk"
else:
    risk = "High Risk"

# --------------------------------------------------
# Display Risk Calculator
# --------------------------------------------------
col1, col2, col3 = st.columns(3)

with col1:  
     icon,metric=st.columns([1,3])
     with icon:
          st.image("churn.jpg",width=70)   
     with metric:
          st.metric("Churn Probability",round(prob,3))
with col2:
     icon,metric=st.columns([1,3])
     with icon:
          st.image("risk.png",width=70)   
     with metric:
          st.metric("Risk Score", f"{risk_score:.0f}/100")
with col3:
     icon,metric=st.columns([1,3])
     with icon:
          st.image("category.png",width=70)   
     with metric:
          st.metric("Risk Category", risk)

probs= model.predict_proba(X_scaled)[:,1]
col1,col2=st.columns(2)

with col1:
     st.metric("Average Churn Probability",round(probs.mean(),3))

with col2:
     st.metric("Max Risk Score",f"{round(probs.max()*100,1)}%")

st.divider()

scenario_df = input_df.copy()
scenario_encoded=scenario_df.reindex(columns=columns, fill_value=0)
scenario_df["NumOfProducts"] = products
scenario_df["IsActiveMember"] = active_member
scenario_df["HasCrCard"] = has_card

new_probability = model.predict_proba(input_encoded)[0][1]
new_risk = new_probability * 100

st.subheader("Customer Churn Risk Calculator")
col1,col2=st.columns(2)

with col1:
     
     fig1 = go.Figure(go.Indicator(mode="gauge+number+delta",
value=new_risk,
    
     title={'text': "Customer Churn Risk (%)", 'font': {'size': 20}},
    
     delta={'reference': risk_score, 'increasing': {'color': "red"}},
    
     gauge={'axis': {'range': [0, 100], 'tickwidth': 1},'bar': {'color': "darkblue", 'thickness': 0.25},'bgcolor': "white",'borderwidth': 2,'bordercolor': "gray",'steps': [{'range': [0, 40], 'color': "#2ecc71"},{'range': [40, 70],'color': "#f1c40f"},{'range': [70, 100], 'color': "#e74c3c"}],'threshold': {'line': {'color': "black", 'width': 4},'thickness': 0.75,'value': new_risk}}))

     fig1.update_layout(height=400,margin=dict(l=20, r=20, t=50, b=20))

     st.plotly_chart(fig1)

with col2:
     compare_df=pd.DataFrame({"Type":["Original","Adjusted"],"Risk":[risk_score,new_risk]})
     fig2=px.bar(compare_df,x="Type",y="Risk",color="Type",text="Risk",title="Customer Churn Risk Comparison")
     fig2.update_traces(texttemplate='%{text:.2f}',textposition='outside')
     fig2.update_layout(yaxis_title="Risk Score (%)",xaxis_title="Scenario",title_x=0.3,height=400,template="plotly_white")
     st.plotly_chart(fig2)
# --------------------------------------------------
# Probability Distribution Visualization
# --------------------------------------------------

st.subheader("Probability Distribution Visualization")

fig = px.histogram(df, probs, nbins=30, color="Exited", title="Distribution of Customer Churn Probability")

st.plotly_chart(fig, use_container_width=True)

# --------------------------------------------------
# Feature Importance Dashboard
# --------------------------------------------------
st.subheader("Feature Importance Dashboard")

importance = model.feature_importances_
features =df.drop("Exited", axis=1).columns

importance_df = pd.DataFrame({"Feature": columns,"Importance": importance}).sort_values(by="Importance", ascending=False)

fig2 = px.bar(importance_df,x="Importance",y="Feature",orientation="h",
title="Feature Importance")

st.plotly_chart(fig2, use_container_width=True)

explainer=shap.Explainer(model)
shap_values=explainer(input_scaled)

values=np.array(shap_values.values).reshape(-1)
features=list(input_encoded.columns)
min_len=min(len(values), len(features))
values=values[:min_len]
features=features[:min_len]
base_value=shap_values.base_values[0]

shap_df=pd.DataFrame({"Feature": features, "SHAP Value": values})
shap_df=shap_df.sort_values(by="SHAP Value", key=abs, ascending=True)

import plotly.graph_objects as go

fig=go.Figure(go.Waterfall(name="SHAP", orientation="h", y=shap_df["Feature"],x=shap_df["SHAP Value"], text=shap_df["SHAP Value"].round(3), measure=["relative"]*len(shap_df)))

fig.update_layout(title="Feature Contribution to Prediction (SHAP Waterfall)", xaxis_title="Impact on Prediction", yaxis_title="Features")
st.plotly_chart(fig)


colors = ["red" if v > 0 else "green" for v in shap_df["SHAP Value"]]

fig = go.Figure(go.Scatter(x=shap_df["SHAP Value"],y=shap_df["Feature"],mode="markers",marker=dict(size=12,color=colors,
line=dict(width=1)),text=[f"{v:.3f}" for v in shap_df["SHAP Value"]],hovertemplate="<b>%{y}</b><br>Impact: %{x}<extra></extra>"))

fig.update_layout(title="Feature Impact on Churn Prediction",
xaxis_title="Impact on Prediction (SHAP Value)", yaxis_title="Features",template="plotly_dark",height=500)

st.plotly_chart(fig)

# --------------------------------------------------
# Customer Feature Visualization
# --------------------------------------------------
st.subheader("Customer Data Exploration")

col1,col2=st.columns(2)

with col1:
     fig3 = px.violin(df, x="Exited", y="Age",color="Exited", points="outliers",box=True,title="Age Distribution by Churn")
     
     fig.update_layout( template="plotly_dark",xaxis_title="Churn (0=No, 1=Yes)", yaxis_title="Age", height=500)

     st.plotly_chart(fig3)

with col2:
     fig4 = px.violin(df, x="Exited", y="Balance",color="Exited",points="outliers",box=True,title="Balance Distribution by Churn")
     
     fig.update_layout( template="plotly_dark",xaxis_title="Churn (0=No, 1=Yes)", yaxis_title="Age", height=500)
     st.plotly_chart(fig4)

from sklearn.metrics import roc_curve, auc
import plotly.graph_objects as go

from sklearn.metrics import roc_curve, auc

🔹 Force y into correct format
y_true = np.array(y)

# If it's 2D → make it 1D
if len(y_true.shape) > 1:
    y_true = y_true.ravel()

# 🔹 Convert to numeric (handles Yes/No, True/False)
try:
    y_true = y_true.astype(int)
except:
    # fallback mapping
    y_true = np.where((y_true == "Yes") | (y_true == True), 1, 0)

# 🔹 Get probabilities
y_prob = model.predict_proba(X_scaled)[:, 1]

# Ensure 1D
y_prob = np.array(y_prob).ravel()

# 🔹 ROC
fpr, tpr, _ = roc_curve(y_true, y_prob)
roc_auc = auc(fpr, tpr)

fig = go.Figure()

fig.add_trace(go.Scatter(
    x=fpr,
    y=tpr,
    mode="lines",
    name=f"AUC = {roc_auc:.3f}"
))

# Diagonal line (random model)
fig.add_trace(go.Scatter(
    x=[0, 1],
    y=[0, 1],
    mode="lines",
    line=dict(dash="dash"),
    name="Random Model"
))

fig.update_layout(
    title="ROC Curve",
    xaxis_title="False Positive Rate",
    yaxis_title="True Positive Rate",
    template="plotly_dark"
)

st.plotly_chart(fig)

models = {
    "Random Forest": rf_model,
    "Gradient Boosting": gb_model,
    "XGBoost": xgb_model
}

fig = go.Figure()

for name, m in models.items():
    y_prob = m.predict_proba(X_scaled)[:, 1]
    fpr, tpr, _ = roc_curve(y, y_prob)
    roc_auc = auc(fpr, tpr)

    fig.add_trace(go.Scatter(
        x=fpr,
        y=tpr,
        mode="lines",
        name=f"{name} (AUC={roc_auc:.3f})"
    ))

fig.add_trace(go.Scatter(
    x=[0, 1], y=[0, 1],
    mode="lines",
    line=dict(dash="dash"),
    name="Random"
))

st.plotly_chart(fig)