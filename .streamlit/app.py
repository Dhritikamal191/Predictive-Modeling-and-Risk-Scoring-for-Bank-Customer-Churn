import streamlit as st
from PIL import Image
import pandas as pd
import numpy as np
import pickle
import joblib
import shap
import plotly.express as px
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
from sklearn.metrics import confusion_matrix
import plotly.figure_factory as ff
from scipy.stats import gaussian_kde

st.markdown("""
<style>

/*  Main background */
[data-testid="stAppViewContainer"] {
    background-color: #0B132B;
}

/*  Sidebar */
[data-testid="stSidebar"] {
    background-color: #1C2541;
}

/*  Headers (VERY IMPORTANT FIX) */
h1, h2, h3, h4, h5, h6 {
    color: #EAEAEA !important;
}

/* General text */
p, span, div {
    color: #EAEAEA;
}

/*  Tabs (FIX INVISIBLE TAB NAMES) */
button[data-baseweb="tab"] {
    color: #A9BCD0;
    font-weight: 600;
}

/*  Active tab */
button[data-baseweb="tab"][aria-selected="true"] {
    color: #FFFFFF;
    border-bottom: 2px solid #00D4FF;
}


/*  KPI cards */
[data-testid="stMetric"] {
    background-color: #1C2541;
    padding: 15px;
    border-radius: 12px;
    text-align: center;
    border: 1px solid #3A506B;
}

/*  Buttons */
.stButton > button {
    background-color: #2563EB;
    color: blue;
    border-radius: 8px;
}

/*  Labels (inputs, sliders) */
label {
    color: #EAEAEA !important;
}

/*  Fix dataframe text */
[data-testid="stDataFrame"] {
    color: black;  /* keep table readable */
}

/*  TOP HEADER FIX */
header {
    background-color: #0B132B !important;
}
</style>
""", unsafe_allow_html=True)


st.markdown("""
<style>

.kpi-card {
    background: linear-gradient(135deg, #1C2541, #3A506B);
    padding: 20px;
    border-radius: 15px;
    text-align: center;
    color: white;
    box-shadow: 0 4px 15px rgba(0,0,0,0.4);
    transition: 0.3s;
}

.kpi-card:hover {
    transform: translateY(-5px);
    box-shadow: 0 8px 25px rgba(0,0,0,0.6);
}

.kpi-title {
    font-size: 14px;
    opacity: 0.8;
}

.kpi-value {
    font-size: 28px;
    font-weight: bold;
    margin-top: 5px;
}

.kpi-icon {
    font-size: 30px;
    margin-bottom: 10px;
}

/* 1. Fix the background of the dropdown menu popup */
div[data-baseweb="popover"] ul {
    background-color: #1a1c24 !important; /* Dark background for the list */
    border: 1px solid rgba(255, 255, 255, 0.1);
}

/* 2. Fix the individual items inside the dropdown */
div[data-baseweb="popover"] li {
    background-color: transparent !important;
    color: white !important; /* Makes '0' and '1' visible */
}

/* 3. Style the item when you hover over it */
div[data-baseweb="popover"] li:hover {
    background-color: rgba(255, 255, 255, 0.1) !important;
}

/* 4. Fix selected item color */
div[data-baseweb="popover"] li[aria-selected="true"] {
    background-color: #00D4FF !important; /* Highlight color for selected */
    color: black !important;
}

</style>
""", unsafe_allow_html=True)

st.markdown("""
    <style>
    /* 1. Remove the white background from the container and the input area */
    div[data-baseweb="input"], 
    div[data-baseweb="base-input"], 
    div[data-baseweb="select"] > div {
        background-color: transparent !important;
        background: transparent !important;
        border-color: rgba(255, 255, 255, 0.2) !important;
    }

    /* 2. Make the text visible (White) */
    input, select, textarea {
        color: white !important;
        -webkit-text-fill-color: white !important;
    }

    /* 3. Specifically target the inner div that Streamlit uses for Selectboxes */
    div[data-testid="stSelectbox"] > div:nth-child(1) > div {
        background-color: transparent !important;
        color: white !important;
    }

    /* 4. Fix Number Input buttons and spacing */
    div[data-testid="stNumberInput"] button {
        background-color: transparent !important;
        color: white !important;
        border: none !important;
    }

    /* 5. Ensure labels and placeholder text are visible */
    label p {
        color: white !important;
    }
    
    ::placeholder {
        color: rgba(255, 255, 255, 0.5) !important;
    }
    </style>
    """, unsafe_allow_html=True)


st.markdown("""
<style>

/*  Block containers */
[data-testid="stVerticalBlock"] {
    background-color: transparent;
}
</style>
""", unsafe_allow_html=True)

def styled_table(df):

    html = """
    <style>
    .custom-table {
        width: 100%;
        border-collapse: collapse;
        background-color: #1C2541;
        color: white;
        border-radius: 10px;
        overflow: hidden;
    }

    .custom-table th {
        background-color: #3A506B;
        padding: 10px;
        text-align: center;
    }

    .custom-table td {
        padding: 10px;
        text-align: center;
        border-top: 1px solid #2D3748;
    }

    .custom-table tr:hover {
        background-color: #2A3A5A;
    }

    </style>

    <table class="custom-table">
    <tr>
    """

    # Header
    for col in df.columns:
        html += f"<th>{col}</th>"
    html += "</tr>"

    # Rows
    for _, row in df.iterrows():
        html += "<tr>"
        for val in row:
            html += f"<td>{val}</td>"
        html += "</tr>"

    html += "</table>"

    return html

def kpi_card(title, value, icon):
    return f"""
    <div class="kpi-card">
        <div class="kpi-icon">{icon}</div>
        <div class="kpi-title">{title}</div>
        <div class="kpi-value">{value}</div>
    </div>
    """

st.set_page_config(page_title="Predictive Modeling and Risk Scoring for Bank Customer Churn",layout="wide")

col1,col2=st.columns([0.5,6])

with col1:
     st.image("Images/office.png",width=60)
with col2:
     st.title("Predictive Modeling and Risk Scoring for Bank Customer Churn")

st.divider()

# --------------------------------------------------
# Load Model
# --------------------------------------------------
lr_model,feature_names= joblib.load("models/logistic_regression.pkl")
dt_model,feature_names= joblib.load("models/decision_tree.pkl")
rf_model,feature_names= joblib.load("models/random_forest.pkl")
gb_model,feature_names= joblib.load("models/gradient_boosting.pkl")
xgb_model,feature_names= joblib.load("models/xgboost.pkl")
scaler= joblib.load("models/scaler.pkl")
columns= joblib.load("models/columns.pkl")
X_test_scaled=joblib.load("models/X_test_scaled.pkl")
y_test=joblib.load("models/y_test.pkl")
# Load dataset for visualization

df=pd.read_csv("Data/European_Bank.csv")

st.sidebar.image("Images/mentor.png",width=150)

model_choice=st.sidebar.radio("Select Model",["Logistic Regression","Decision Tree", "Random Forest", "Gradient Boosting", "XGBoost"], key="model_selector")

threshold=st.sidebar.slider("Select Threshold",0.0, 1.0, 0.50, 0.01)

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

model= rf_model

if model_choice=="Logisitc Regression":
   model=lr_model
elif model_choice=="Decision Tree": 
     model=dt_model
elif model_choice=="Gradient Boosting":
     model=gb_model
elif model_choice=="XGBoost":
     model=xgb_model

if st.button("Predict"):
   prediction= model.predict(input_encoded)[0]

   y_prob=model.predict_proba(X_test_scaled)[:,1]
   y_pred=(y_prob>threshold).astype(int)
   
   st.write("Prediction:", y_pred[0])
   st.write("Max Probability:", y_prob.max())
   st.write("Min Probability:", y_prob.min())
   st.write("Churn Probability:", y_prob[0])

   st.success(f"Prediction:{y_pred[0]}")

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

probs= model.predict_proba(X_test_scaled)[:,1]

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

col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    st.markdown(kpi_card("Churn Probability", f"{prob:.3f}", "📉"), unsafe_allow_html=True)

with col2:
    st.markdown(kpi_card("Risk Score", f"{risk_score:.0f}/100", "⚠️"), unsafe_allow_html=True)

with col3:
    st.markdown(kpi_card("Risk Category", risk, "🚦"), unsafe_allow_html=True)

with col4:
    st.markdown(kpi_card("Avg. Churn Probability", round(probs.mean(),3), "📊"), unsafe_allow_html=True)

with col5:
     st.markdown(kpi_card("Max Risk Score", f"{round(probs.max()*100,1)}%","📈"), unsafe_allow_html=True)

st.divider()

scenario_df = input_df.copy()
scenario_encoded=scenario_df.reindex(columns=columns, fill_value=0)
scenario_df["NumOfProducts"] = products
scenario_df["IsActiveMember"] = active_member
scenario_df["HasCrCard"] = has_card

new_probability = model.predict_proba(input_encoded)[0][1]
new_risk = new_probability * 100

tab1, tab2, tab3, tab4= st.tabs(["Customer Risk Calculator","Feature Importance","ROC and PDP","Model Comparison"])

with tab1:
     st.subheader("Customer Churn Risk Calculator")
     col1,col2=st.columns(2)

     with col1:
     
          fig1 = go.Figure(go.Indicator(mode="gauge+number+delta", value=new_risk, title={'text': "Customer Churn Risk (%)",'font': {'size': 20}}, delta={'reference': risk_score, 'increasing': {'color': "red"}},gauge={'axis': {'range': [0, 100], 'tickwidth': 1},'bar': {'color':"#2563eb", 'thickness': 0.25},'bgcolor':"#111827",'borderwidth':2,'bordercolor':"#374151",'steps': [{'range': [0, 40], 'color': "#16a34a"},{'range': [40, 70],'color':"#ca8a04"},{'range': [70, 100], 'color':"#dc2626"}],'threshold': {'line': {'color':"black", 'width':4},'thickness':0.75,'value':new_risk}}))

          fig1.update_layout(height=400,margin=dict(l=20, r=20, t=50, b=20),template="plotly_dark",paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")

          st.plotly_chart(fig1)

     with col2:
          compare_df=pd.DataFrame({"Type":["Original","Adjusted"],"Risk":[risk_score,new_risk]})
          fig2=px.bar(compare_df,x="Type",y="Risk",color="Type",text="Risk",title="Customer Churn Risk Comparison",color_discrete_sequence=["#6366f1","#f43f5e"])
          fig2.update_traces(texttemplate='%{text:.2f}',textposition='outside')
          fig2.update_layout(yaxis_title="Risk Score (%)",xaxis_title="Scenario",title_x=0.3,height=400,template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)")
          st.plotly_chart(fig2)
         
     st.subheader("Confusion Matrix")
     y_prob=model.predict_proba(X_test_scaled)[:,1]
     y_pred=(y_prob> threshold).astype(int)

     cm =confusion_matrix(y_test, y_pred)
     cm=cm[::-1]

     labels = ["Churn", "No Churn"]
     fig = ff.create_annotated_heatmap(z=cm, x=labels, y=labels, colorscale="Reds")

     fig.update_layout(xaxis_title="Predicted", yaxis_title="Actual",template="plotly_dark",paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)")

     st.plotly_chart(fig)

     # --------------------------------------------------
     # Probability Distribution Visualization
     # --------------------------------------------------

     st.subheader("Probability Distribution Visualization")
     plot_results=pd.DataFrame({"Probability":y_prob, "Actual_Status": y_test})
     plot_results["Actual_Status"]=plot_results["Actual_Status"].map({1:"Churned", 0: "Stayed"})
     fig = px.histogram(plot_results, x="Probability", nbins=30, color="Actual_Status", color_discrete_sequence=["#6366f1","#f43f5e"], barmode="overlay", opacity=0.5)
     fig.update_traces(marker_line_width=0)
     fig.update_layout(bargap=0.1, legend_title_text="Customer Status",template="plotly_dark",paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)")
     st.plotly_chart(fig, use_container_width=True)

     st.subheader("Churn Probability Density Distribution")  
     def render_comparison_kde(model, X_test_scaled, y_test):
    
         kde_probs = model.predict_proba(X_test_scaled)[:, 1]
         kde_mean = np.mean(kde_probs)

         data_0 =kde_probs[y_test == 0]
         data_1 =kde_probs[y_test == 1]
         
         fig=go.Figure()
          
         for data, label, color in zip([data_0, data_1], ["Stayed", "Churned"],["rgba(0,200,150,0.4)","rgba(200,0,200,0.4)"]):
             if len(data) > 1:
                kde=gaussian_kde(data)
                x_range = np.linspace(0, 1, 500)
                y_range = kde(x_range)
         
                fig.add_trace(go.Scatter(x=x_range, y=y_range, mode='lines', name=label, fill='tozeroy',  line=dict(color=color, width=2), fillcolor=color, opacity=0.5))
         m_val = np.mean(kde_probs)
         fig.add_vline(x=m_val, line_dash="dash", line_color="red")
         fig.add_annotation(x=m_val, text=f"Mean: {m_val:.2f}", showarrow=False, yshift=10)

         fig.update_layout(title="Probability Distribution (Churned vs. Stayed)", xaxis_title="Probability", yaxis_title="Density", template="plotly_dark",paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", xaxis=dict(range=[0, 1]), legend=dict(yanchor="top", y=0.99, xanchor="right", x=0.99))

         return fig

     st.plotly_chart(render_comparison_kde(model, X_test_scaled, y_test), use_container_width=True)


with tab2:
     # --------------------------------------------------
     # Feature Importance Dashboard
     # --------------------------------------------------
     st.subheader("Feature Importance Dashboard")

     importance = model.feature_importances_
     features =df.drop("Exited", axis=1).columns

     importance_df = pd.DataFrame({"Feature": columns,"Importance": importance}).sort_values(by="Importance", ascending=False)
     max_val=importance_df["Importance"].max()
     fig2 = px.bar(importance_df,x="Importance",y="Feature",orientation="h", color="Importance", color_discrete_sequence=px.colors.qualitative.Set3)
     fig2.update_traces(text=importance_df["Importance"],textposition="outside")
     fig2.update_layout(template="plotly_dark", yaxis=dict(autorange="reversed"), height=500,paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)")
     st.plotly_chart(fig2, use_container_width=True)
    
     st.subheader("Feature Contribution to Prediction (SHAP Waterfall)")
     explainer=shap.Explainer(model)
     shap_values=explainer(X_test_scaled)

     values=np.array(shap_values.values).reshape(-1)
     features=list(input_encoded.columns)
     min_len=min(len(values), len(features))
     values=values[:min_len]
     features=features[:min_len]
     base_value=shap_values.base_values[0]

     shap_df=pd.DataFrame({"Feature": features, "SHAP Value": values})
     shap_df=shap_df.sort_values(by="SHAP Value", key=abs, ascending=True)

     fig=go.Figure(go.Waterfall(name="SHAP", orientation="h", y=shap_df["Feature"],x=shap_df["SHAP Value"], text=shap_df["SHAP Value"].round(3), measure=["relative"]*len(shap_df), increasing=dict(marker=dict(color="#22c55e")), decreasing=dict(marker=dict(color="#ef4444")), totals=dict(marker=dict(color="#3b82f6"))))

     fig.update_layout(xaxis_title="Impact on Prediction", yaxis_title="Features", template="plotly_dark", height=500,paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)")
     st.plotly_chart(fig)

     st.subheader("Feature Impact on Churn Prediction")
     colors = shap_df["SHAP Value"]

     fig = go.Figure(go.Scatter(x=shap_df["SHAP Value"], y=shap_df["Feature"], mode="markers", marker=dict(size=20,color=colors, colorscale ="Plasma", showscale=True,colorbar=dict(title="Impact")), text=[f"{v:.3f}" for v in shap_df["SHAP Value"]], hovertemplate="<b>%{y}</b><br>" +"Impact: %{x:.3f}<br>" +"<extra></extra>"))

     fig.update_layout(xaxis_title="Impact on Prediction (SHAP Value)", yaxis_title="Features",template="plotly_dark",paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",height=550)

     fig.update_yaxes(autorange="reversed")

     st.plotly_chart(fig, use_container_width=True)
     # --------------------------------------------------
     # Customer Feature Visualization
     # --------------------------------------------------
     st.subheader("Customer Data Exploration")

     st.subheader("Age Distribution by Churn")
    
     df0 = df[df["Exited"] == 0]
     df1 = df[df["Exited"] == 1]

     fig = go.Figure()

     fig.add_trace(go.Violin(y=df0["Age"],name="Not Churned",box_visible=True,meanline_visible=True,line_color="#00D4FF",fillcolor="rgba(0,212,255,0.4)",opacity=0.7))

     fig.add_trace(go.Violin(y=df1["Age"],name="Churned",box_visible=True,meanline_visible=True,line_color="#FF6B6B",fillcolor="rgba(255,107,107,0.4)",opacity=0.7))

     fig.add_scatter(x=["Not Churned"],y=[np.median(df0["Age"])],mode="markers",marker=dict(color="black", size=8),name="Median (0)")

     fig.add_scatter(x=["Churned"],y=[np.median(df1["Age"])],mode="markers",marker=dict(color="white", size=8),name="Median (1)")

     fig.update_layout(template="plotly_dark",paper_bgcolor="#0B132B",plot_bgcolor="#0B132B",yaxis_title="Age",showlegend=True)

     st.plotly_chart(fig, use_container_width=True)

     st.subheader("Balance Distribution by Churn")

     fig = go.Figure()

     fig.add_trace(go.Violin(y=df0["Balance"],name="Not Churned",box_visible=True,meanline_visible=True,line_color="#00D4FF",fillcolor="rgba(0,212,255,0.4)",opacity=0.7))

     fig.add_trace(go.Violin(y=df1["Balance"],name="Churned",box_visible=True,meanline_visible=True,line_color="#FF6B6B",fillcolor="rgba(255,107,107,0.4)",opacity=0.7))

     fig.add_scatter(x=["Not Churned"],y=[np.median(df0["Balance"])],mode="markers",marker=dict(color="black", size=8),name="Median (0)")

     fig.add_scatter(x=["Churned"],y=[np.median(df1["Balance"])],mode="markers",marker=dict(color="white", size=8),name="Median (1)")

     fig.update_layout(template="plotly_dark",paper_bgcolor="#0B132B",plot_bgcolor="#0B132B",yaxis_title="Balance")

     st.plotly_chart(fig, use_container_width=True)

with tab3:
     st.subheader("ROC Curve")
     from sklearn.metrics import roc_curve, auc

     y= df["Exited"]
     y_true = df["Exited"].to_numpy().ravel()
     y_prob = model.predict_proba(X_test_scaled)[:, 1]

     fpr, tpr, _ = roc_curve(y_test, y_prob)
     roc_auc = auc(fpr, tpr)   

     fig = go.Figure()  
     fig.add_trace(go.Scatter(x=fpr,y=tpr, mode="lines", name=f"AUC = {roc_auc:.3f}"))

     fig.add_trace(go.Scatter(x=[0, 1],y=[0, 1],mode="lines",line=dict(color="grey",dash="dash"),name="Random Model"))

     fig.update_layout(xaxis_title="False Positive Rate", yaxis_title="True Positive Rate", template="plotly_dark",paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)", height=550, legend=dict(orientation="h",yanchor="bottom", y=1.02, xanchor="center", x=0.5), hovermode="x unified")
     fig.update_xaxes(showgrid=True, gridcolor="rgba(255, 255,255,0.1)")
     fig.update_yaxes(showgrid=True, gridcolor="rgba(255,255,255,0.1)")
     
     st.plotly_chart(fig, use_container_width=True)

     st.subheader("ROC with respect to Random Line")
     fig = go.Figure()
     models = {"Logistic Regression":lr_model,"Decision Tree": dt_model,"Random Forest": rf_model,"Gradient Boosting": gb_model,"XGBoost": xgb_model}
     for name, m in models.items():
         y_prob = m.predict_proba(X_test_scaled)[:, 1]
         fpr, tpr, _ = roc_curve(y_test, y_prob)
         roc_auc = auc(fpr, tpr)

     fig.add_trace(go.Scatter(x=fpr,y=tpr, mode="lines",name=f"{name} (AUC={roc_auc:.3f})"))

     fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1],mode="lines",line=dict(dash="dash"),name="Random"))
     fig.update_layout(template="plotly_dark",paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)")
     st.plotly_chart(fig)

     from sklearn.inspection import partial_dependence

     features = ["Age", "Balance", "NumOfProducts", "EstimatedSalary"]

     fig = go.Figure()

     colors=["#00D4FF","#FF6B6B","#FFD93D","#6BCB77"]

     st.subheader("Partial Dependence Plots (Key Features)")
     for i, feature in enumerate(features):
         feature_index=X.columns.get_loc(feature)

         pdp=partial_dependence(model,X_test_scaled,features=[feature_index])
         x_vals=pdp["grid_values"][0]
         y_vals=pdp["average"][0].flatten()
         fig.add_trace(go.Scatter(x=x_vals,y=y_vals,mode="lines+markers",name=feature,line=dict(width=3,color=colors[i]), marker=dict(size=5),hovertemplate=f"<b>{feature}</b><br>Value:%{{x}}<br>Churn Prob: %{{y:.3f}}<extra></extra>"))
         fig.update_layout(xaxis_title="Feature Value", yaxis_title="Churn Probability", template="plotly_dark",paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)", height=550, margin=dict(l=40, r=40, t=60, b=40),legend=dict(title="Feature", orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5), hovermode="x unified")
         fig.update_xaxes(showgrid=True, gridcolor="rgba(255,255,255,0.1)")
         fig.update_yaxes(showgrid=True, gridcolor="rgba(255,255,255,0.1)")
         st.plotly_chart(fig, use_container_width=True)
with tab4:
     
     from sklearn.metrics import accuracy_score, recall_score, f1_score

     def get_metrics(model, X, y, threshold):
         y_prob=model.predict_proba(X)[:,1]
         y_pred=(y_prob>=threshold).astype(int)

         acc= accuracy_score(y, y_pred)
         rec=recall_score(y, y_pred)
         f1= f1_score(y, y_pred)
         return acc, rec, f1
    
     models={"Logistic Regression": lr_model,"Decision Tree": dt_model,"Random Forest": rf_model,"Gradient Boosting": gb_model,"XGBoost": xgb_model}
     metrics_data=[]
     for name, model in models.items():
         acc, rec, f1= get_metrics(model, X_test_scaled, y_test, threshold)
         metrics_data.append({"Model": name,"Accuracy":acc,"Recall":rec,"F1 Score":f1})
     
     df_metrics=pd.DataFrame(metrics_data)

     best_row=df_metrics.sort_values(by="F1 Score", ascending=False).iloc[0]
     best_model_name=best_row["Model"]
     best_accuracy=best_row["Accuracy"]
     best_recall=best_row["Recall"]
     best_f1=best_row["F1 Score"]

     col1, col2, col3, col4=st.columns(4)
     col1.metric("🏆 Best Model", best_model_name)
     col2.metric("Accuracy",f"{best_accuracy:.2f}")
     col3.metric("recall",f"{best_recall:.2f}")
     col4.metric("F1 Score",f"{best_f1:.2f}")

     st.subheader("Model Comaprison Table")
     
     df_metrics = df_metrics.sort_values(by="F1 Score", ascending=False).reset_index(drop=True)
    
     df_metrics.insert(0, "Rank", range(1, len(df_metrics) + 1))
     
     df_display = df_metrics.copy()
     for col in ["Accuracy", "Recall", "F1 Score"]:
         df_display[col] = df_display[col].apply(lambda x: f"{x:.2%}")
     styled_df=df_display.style \
     .highlight_max(subset=["Accuracy"], color="#00D4FF")\
     .highlight_max(subset=["Recall"], color="#FF6B6B")\
     .highlight_max(subset=["F1 Score"], color="#FFD93D")\
     .set_properties(**{"text-align":"center","font-weight":"bold"})

     html_table=styled_table(df_metrics)
     st.markdown(html_table, unsafe_allow_html=True)

     st.subheader("Model Comparison Table (Percentage Based)")

     html_table=styled_df.to_html()
     st.markdown(html_table, unsafe_allow_html=True)
    
     st.subheader ("Model Comparison Graph")
     df_melted = df_metrics.melt(id_vars="Model", var_name="Metric", value_name="Score")

     color_map = {"Accuracy": "#00D4FF", "Recall": "#FF6B6B", "F1 Score": "#FFD93D"}

     fig = px.bar(df_melted,x="Model",y="Score",color="Metric",barmode="group",color_discrete_map=color_map,text="Score",template="plotly_dark")

     fig.update_traces(texttemplate='%{text:.2f}',textposition='outside',marker_line_width=1.5)

     best_model = df_metrics.sort_values(by="F1 Score", ascending=False).iloc[0]["Model"]

     fig.add_vrect(x0=best_model,x1=best_model,fillcolor="rgba(255, 255, 255, 0.1)",line_width=0,layer="below")

     fig.update_layout(title="Model Performance Comparison",xaxis_title="Model",yaxis_title="Score",height=550,margin=dict(l=40, r=40, t=60, b=40),legend=dict(orientation="h",yanchor="bottom",y=1.02,xanchor="center",x=0.5,title=""),template="plotly_dark",paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)", hovermode="x unified")

     fig.update_xaxes(showgrid=False)
    
     fig.update_yaxes(showgrid=True, gridcolor="rgba(255,255,255,0.1)")
    
     fig.update_traces(marker=dict(line=dict(width=1)))
    
     st.plotly_chart(fig, use_container_width=True)
