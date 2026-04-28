import streamlit as st
from PIL import Image
import pandas as pd
import numpy as np
import pickle
import joblib
import shap
import os
import plotly.express as px
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
from sklearn.metrics import confusion_matrix
import plotly.figure_factory as ff
from scipy.stats import gaussian_kde
from sklearn.metrics import (accuracy_score, recall_score, f1_score, roc_auc_score, roc_curve, confusion_matrix)
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

/*  Fix the background of the dropdown menu popup */
div[data-baseweb="popover"] ul {
    background-color: #1a1c24 !important; /* Dark background for the list */
    border: 1px solid rgba(255, 255, 255, 0.1);
}

/*  Fix the individual items inside the dropdown */
div[data-baseweb="popover"] li {
    background-color: transparent !important;
    color: white !important; /* Makes '0' and '1' visible */
}

/*  Style the item when you hover over it */
div[data-baseweb="popover"] li:hover {
    background-color: rgba(255, 255, 255, 0.1) !important;
}

/*  Fix selected item color */
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
models=joblib.load("models/all_models_pipeline.pkl")

# Load dataset for visualization

df=pd.read_csv("Data/European_Bank.csv")
X=df.drop(["Exited","CustomerId","Surname"], axis=1, errors="ignore")
y_test=df["Exited"]
st.sidebar.image("Images/mentor.png",width=150)

st.sidebar.subheader("Model Selection")
model_choice=st.sidebar.radio("Select Model",["Logistic Regression","Decision Tree", "Random Forest", "Gradient Boosting", "XGBoost"], key="model_selector")
threshold=st.sidebar.slider("Select Threshold",0.0, 1.0, 0.50, 0.01)
    
model=models[model_choice]
     
st.sidebar.subheader("Customer Feature Inputs")

credit_score = st.sidebar.number_input("Credit Score", 300, 900, 600)
age = st.sidebar.number_input("Age", 18, 100, 40)
tenure = st.sidebar.slider("Tenure", 0, 10, 5)
balance = st.sidebar.slider("Balance", 0, 250000, 50000)
products = st.sidebar.slider("Number of Products", 1, 4, 2)
active_member = st.sidebar.selectbox("Active Member", [0,1])
has_card = st.sidebar.selectbox("Has Credit Card", [0,1])
salary = st.sidebar.slider("Estimated Salary", 0,1000, 200000, 50000)
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
   
X=df.copy()
# --------------------------------------------------
# Churn Prediction
# --------------------------------------------------

if st.button("Predict"):
   try:
       pred = model.predict(input_df)[0]
       prob = model.predict_proba(input_df)[0][1]
       risk_score = prob * 100
       if pred == 1:
          st.error(f"Customer likely to CHURN")
       else:
            st.success(f"Customer NOT likely to churn")

   except Exception as e:
          st.error(f"Prediction Error: {e}")

pred = model.predict(input_df)[0]
prob = model.predict_proba(input_df)[0][1]
risk_score = prob * 100
probs= model.predict_proba(X)[:,1]
if credit_score <= 0 or age <= 0 or balance <= 0 or salary <= 0 :
      st.error(" Invalid input values. Please check inputs.")
      st.stop()

if risk_score < 30:
   risk = "Low Risk"
elif risk_score < 70:
     risk = "Medium Risk"
else:
     risk = "High Risk"
       
if pred is None:
   st.warning("Please click Predict first")
 
elif pred not in [0, 1]:
     st.error("Invalid prediction output")

elif prob < 0 or prob > 1:
     st.error("Probability out of range")

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
scenario_df["Balance"]=balance + 20000
scenario_df["NumOfProducts"] = products
scenario_df["IsActiveMember"] = active_member
scenario_df["HasCrCard"] = has_card
new_probability = model.predict_proba(X)[0][1]
new_risk = new_probability * 100

tab1, tab2, tab3, tab4, tab5= st.tabs(["Customer Risk Calculator","Feature Importance","ROC and PDP","Model Comparison","Monitoring"])

with tab1:
     col1,col2=st.columns(2)

     with col1:
          fig1 = go.Figure(go.Indicator(mode="gauge+number+delta", value=new_risk, title={'text': "Customer Churn Risk (%)",'font': {'size': 20}}, delta={'reference': risk_score,'relative':False,'valueformat':"+.2f",'increasing': {'color': "red"},'decreasing':{'color':"green"}},gauge={'axis': {'range': [0, 100], 'tickwidth': 1},'bar': {'color':"#2563eb", 'thickness': 0.25},'bgcolor':"#111827",'borderwidth':2,'bordercolor':"#374151",'steps': [{'range': [0, 40], 'color': "#16a34a"},{'range': [40, 70],'color':"#ca8a04"},{'range': [70, 100], 'color':"#dc2626"}],'threshold': {'line': {'color':"black", 'width':4},'thickness':0.75,'value':new_risk}}))

          fig1.update_layout(title="Customer Churn Risk (%)",x=0.5, xanchor="center",font=dict(size=20, color="white"),height=400,margin=dict(l=20, r=20, t=50, b=20),template="plotly_dark",paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")

          st.plotly_chart(fig1)

     with col2:
          compare_df = pd.DataFrame({"Type": ["Original", "Adjusted"],"Risk": [risk_score, new_risk]})

          fig2 = px.pie(compare_df,names="Type",values="Risk",hole=0.5,color="Type",color_discrete_sequence=["#6366f1", "#f43f5e"])

          fig2.update_traces(textinfo="label+percent",hovertemplate="<b>%{label}</b><br>Risk: %{value:.2f}%<extra></extra>")

          fig2.update_layout(title="Customer Churn Risk Comparison",x=0.5, xanchor="center",font=dict(size=20, color="white"),title_x=0.3,font=dict(color="white"),legend=dict(font=dict(color="white")),height=400,template="plotly_dark",paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)")

          st.plotly_chart(fig2)
         
     st.subheader("Confusion Matrix")
     y_prob=model.predict_proba(X)[:,1]
     y_pred=(y_prob>=threshold).astype(int)
     cm =confusion_matrix(y_test, y_pred)
     cm=cm[::-1]
     labels = ["Churn", "No Churn"]
     fig = ff.create_annotated_heatmap(z=cm, x=labels, y=labels, colorscale="Reds")
     fig.update_layout(xaxis_title="Predicted", yaxis_title="Actual",template="plotly_dark",paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)")
     st.plotly_chart(fig)

     st.subheader("🔍 Model Explainability (SHAP)")

     try:
         preprocessor = model.named_steps["preprocessor"]
         actual_model = model.named_steps["model"]

         X_transformed = preprocessor.transform(input_df)
        
         if hasattr(actual_model, "feature_importances_"):
            explainer = shap.TreeExplainer(actual_model)
            shap_values = explainer(X_transformed)
            values = shap_values.values

            if len(values.shape) == 3:
               values = values[0, :, 1]
            elif len(values.shape) == 2:
                 values = values[0]
            feature_names = preprocessor.get_feature_names_out()
            min_len = min(len(feature_names), len(values))
            feature_names = feature_names[:min_len]
            values = values[:min_len]
            shap_df = pd.DataFrame({"Feature": feature_names,"SHAP Value": values}).sort_values(by="SHAP Value", key=np.abs, ascending=False)
            top_n=10
            shap_df_top=shap_df.head(top_n) 
            html_table=styled_table(shap_df_top)
            st.markdown(html_table, unsafe_allow_html=True)
            fig = px.bar(shap_df_top,x="SHAP Value",y="Feature",orientation="h",color="SHAP Value",color_continuous_scale="RdBu",title="Feature Impact on Prediction")
            fig.update_layout(template="plotly_dark",xaxis_title="Impact on Prediction",yaxis_title="Features",yaxis=dict(autorange="reversed"),coloraxis_colorbar=dict(title="SHAP Value"),margin=dict(l=50,r=50,t=50,b=50),height=400,font=dict(color="white"), legend=dict(font=dict(color="white")),paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)")
            fig.update_traces(text=shap_df_top["SHAP Value"].round(3),textposition="outside")
            st.plotly_chart(fig, use_container_width=True)
     
         elif hasattr(actual_model, "coef_"):
              coef = actual_model.coef_[0]
              feature_names = preprocessor.get_feature_names_out()
              coef_df = pd.DataFrame({"Feature": feature_names,"Impact": coef}).sort_values(by="Impact", key=np.abs, ascending=False)
              st.info("Showing feature impact using Logistic regression coefficients")
              fig = px.bar(coef_df.head(10),x="Impact",y="Feature",orientation="h",color="Impact")
              fig.update_layout(font=dict(color="white"), legend=dict(font=dict(color="white")),paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",template="plotly_dark")
              fig.update_yaxes(autorange="reversed")
              st.plotly_chart(fig)
         else:
              st.warning("Explainability not available for this model")
     except Exception as e:
             st.error(f"SHAP Error: {e}")
     
     # --------------------------------------------------
     # Probability Distribution Visualization
     # --------------------------------------------------

     st.subheader("Probability Distribution Visualization")
     plot_results=pd.DataFrame({"Probability":y_prob, "Actual_Status": y_test})
     plot_results["Actual_Status"]=plot_results["Actual_Status"].map({1:"Churned", 0: "Stayed"})
     fig = px.histogram(plot_results, x="Probability", nbins=30, color="Actual_Status", color_discrete_sequence=["#6366f1","#f43f5e"], barmode="overlay", opacity=0.5)
     fig.update_traces(marker_line_width=0)
     fig.update_layout(bargap=0.1, legend_title_text="Customer Status",template="plotly_dark",font=dict(color="white"), legend=dict(font=dict(color="white")),paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)")
     fig.update_xaxes(showgrid=False)
     fig.update_yaxes(showgrid=False)
     st.plotly_chart(fig, use_container_width=True)

     st.subheader("Churn Probability Density Distribution")  
     def render_comparison_kde(model, X, y_test):
    
         kde_probs = model.predict_proba(X)[:, 1]
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
         fig.update_layout(title="Probability Distribution (Churned vs. Stayed)", xaxis_title="Probability", yaxis_title="Density",font=dict(color="white"), template="plotly_dark",paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", xaxis=dict(range=[0, 1]), legend=dict(yanchor="top", y=0.99, xanchor="right", x=0.99,font=dict(color="white")))
         fig.update_xaxes(showgrid=False)
         fig.update_yaxes(showgrid=False)
         return fig

     st.plotly_chart(render_comparison_kde(model, X, y_test), use_container_width=True)

with tab2:
     # --------------------------------------------------
     # Feature Importance Dashboard
     # --------------------------------------------------
     st.subheader("Feature Importance Dashboard")
   
     actual_model = list(model.named_steps.values())[-1]

     if hasattr(actual_model, "feature_importances_"):
        importance = actual_model.feature_importances_

     elif hasattr(actual_model, "coef_"):
          importance = np.abs(actual_model.coef_[0])

     else:
          st.warning("Feature importance not available for this model")
          importance = None

     feature_names = model.named_steps["preprocessor"].get_feature_names_out()

     if importance is not None:

        feature_names = model.named_steps["preprocessor"].get_feature_names_out()

        min_len = min(len(feature_names), len(importance))

        importance_df = pd.DataFrame({"Feature": feature_names[:min_len],"Importance": importance[:min_len]}).sort_values(by="Importance", ascending=False)

        max_val = importance_df["Importance"].max()

        fig2 = px.bar(importance_df,x="Importance",y="Feature",orientation="h", color="Importance", color_discrete_sequence=px.colors.qualitative.Set3)
        fig2.update_traces(text=importance_df["Importance"],textposition="outside")
        fig2.update_layout(template="plotly_dark", yaxis=dict(autorange="reversed"), height=500,font=dict(color="white"), legend=dict(font=dict(color="white")), paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig2, use_container_width=True)

     st.subheader("Feature Contribution and Feature Impact on Churn Prediction")

     col1, col2 = st.columns(2)
     
     X_sample = X.sample(100, random_state=42)
     X_transformed = preprocessor.transform(X_sample)

     explainer = shap.Explainer(actual_model.predict_proba, X_transformed)

     shap_values = explainer(X_transformed)

     values = shap_values.values

     if len(values.shape) == 3:
        values = values[:, :, 1]

     mean_shap = np.abs(values).mean(axis=0)

     feature_names = preprocessor.get_feature_names_out()

     min_len = min(len(feature_names), len(mean_shap))

     shap_df = pd.DataFrame({"Feature": feature_names[:min_len],"SHAP Value": mean_shap[:min_len]}).sort_values(by="SHAP Value", ascending=False)

     with col1:
          fig = go.Figure(go.Bar(x=shap_df["SHAP Value"],y=shap_df["Feature"],orientation="h",marker=dict(color=shap_df["SHAP Value"], colorscale="RdBu")))

          fig.update_layout(title="Global Feature Contribution",xaxis_title="Mean Impact on Prediction",yaxis_title="Features",template="plotly_dark",height=500,paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)")
          fig.update_xaxes(showgrid=False)
          fig.update_yaxes(showgrid=False)
          st.plotly_chart(fig, use_container_width=True)

    
     with col2:
          fig = go.Figure(go.Scatter(x=shap_df["SHAP Value"],y=shap_df["Feature"],mode="markers",marker=dict(size=12,color=shap_df["SHAP Value"],colorscale="Viridis",showscale=True,colorbar=dict(title="Impact")),text=[f"{v:.3f}" for v in shap_df["SHAP Value"]],hovertemplate="<b>%{y}</b><br>Impact: %{x:.3f}<extra></extra>"))

          fig.update_layout(title="Feature Impact Distribution",xaxis_title="SHAP Value",yaxis_title="Features",template="plotly_dark",height=550,paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)")
          fig.update_xaxes(showgrid=False)
          fig.update_yaxes(showgrid=False,autorange="reversed")
          st.plotly_chart(fig, use_container_width=True)

     
     # --------------------------------------------------
     # Customer Feature Visualization
     # --------------------------------------------------
     st.subheader("Customer Data Exploration")
     col1,col2=st.columns(2)
     with col1:
          st.subheader("Age Distribution by Churn")
    
          df0 = df[df["Exited"] == 0]
          df1 = df[df["Exited"] == 1]

          fig = go.Figure()

          fig.add_trace(go.Violin(y=df0["Age"],name="Not Churned",box_visible=True,meanline_visible=True,line_color="#00D4FF",fillcolor="rgba(0,212,255,0.4)",opacity=0.7))
          fig.add_trace(go.Violin(y=df1["Age"],name="Churned",box_visible=True,meanline_visible=True,line_color="#FF6B6B",fillcolor="rgba(255,107,107,0.4)",opacity=0.7))
          fig.add_scatter(x=["Not Churned"],y=[np.median(df0["Age"])],mode="markers",marker=dict(color="grey", size=8),name="Median (0)")
          fig.add_scatter(x=["Churned"],y=[np.median(df1["Age"])],mode="markers",marker=dict(color="black", size=8),name="Median (1)")
          fig.update_layout(font=dict(color="white"), legend=dict(font=dict(color="white")),template="plotly_dark",paper_bgcolor="#0B132B",plot_bgcolor="#0B132B",yaxis_title="Age",showlegend=True)
          fig.update_xaxes(showgrid=False)
          fig.update_yaxes(showgrid=False)
          st.plotly_chart(fig, use_container_width=True)
     with col2:
          st.subheader("Balance Distribution by Churn")

          fig = go.Figure()

          fig.add_trace(go.Violin(y=df0["Balance"],name="Not Churned",box_visible=True,meanline_visible=True,line_color="#00D4FF",fillcolor="rgba(0,212,255,0.4)",opacity=0.7))
          fig.add_trace(go.Violin(y=df1["Balance"],name="Churned",box_visible=True,meanline_visible=True,line_color="#FF6B6B",fillcolor="rgba(255,107,107,0.4)",opacity=0.7))
          fig.add_scatter(x=["Not Churned"],y=[np.median(df0["Balance"])],mode="markers",marker=dict(color="black", size=8),name="Median (0)")
          fig.add_scatter(x=["Churned"],y=[np.median(df1["Balance"])],mode="markers",marker=dict(color="grey", size=8),name="Median (1)")
          fig.update_layout(font=dict(color="white"), legend=dict(font=dict(color="white")),template="plotly_dark",paper_bgcolor="#0B132B",plot_bgcolor="#0B132B",yaxis_title="Balance")
          fig.update_xaxes(showgrid=False)
          fig.update_yaxes(showgrid=False)
          st.plotly_chart(fig, use_container_width=True)

with tab3:
     col1, col2=st.columns(2)
     with col1:
          st.subheader("ROC Curve")
          from sklearn.metrics import roc_curve, auc
          y= df["Exited"]
          y_true = df["Exited"].to_numpy().ravel()
          y_prob = model.predict_proba(X)[:, 1]
          fpr, tpr, _ = roc_curve(y_test, y_prob)
          roc_auc = auc(fpr, tpr)   
          fig = go.Figure()  
          fig.add_trace(go.Scatter(x=fpr,y=tpr, mode="lines", name=f"AUC = {roc_auc:.3f}"))
          fig.add_trace(go.Scatter(x=[0, 1],y=[0, 1],mode="lines",line=dict(color="grey",dash="dash"),name="Random Model"))
          fig.update_layout(xaxis_title="False Positive Rate", yaxis_title="True Positive Rate",font=dict(color="white"), template="plotly_dark",paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)", height=550, legend=dict(orientation="h",yanchor="bottom", y=1.02, xanchor="center", x=0.5,font=dict(color="white")), hovermode="x unified")
          fig.update_xaxes(showgrid=True, gridcolor="rgba(255, 255,255,0.1)")
          fig.update_yaxes(showgrid=True, gridcolor="rgba(255,255,255,0.1)")
          st.plotly_chart(fig, use_container_width=True)
     with col2:
          st.subheader("ROC with respect to Random Line")
          fig = go.Figure()
          for name, m in models.items():
              y_prob = m.predict_proba(X)[:, 1]
              fpr, tpr, _ = roc_curve(y_test, y_prob)
              roc_auc = auc(fpr, tpr)

          fig.add_trace(go.Scatter(x=fpr,y=tpr, mode="lines",name=f"{name} (AUC={roc_auc:.3f})"))
          fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1],mode="lines",line=dict(dash="dash"),name="Random"))
          fig.update_layout(font=dict(color="white"), legend=dict(font=dict(color="white")),template="plotly_dark",paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)")
          fig.update_xaxes(showgrid=False)
          fig.update_yaxes(showgrid=False)
          st.plotly_chart(fig)

     results = []

     for name, model in models.items():
         y_pred=model.predict(X)
         y_prob=model.predict_proba(X)[:,1]
         accuracy = accuracy_score(y_test, y_pred)
         recall = recall_score(y_test, y_pred)
         f1 = f1_score(y_test, y_pred)
         roc_auc = roc_auc_score(y_test, y_prob)
         results.append({"Model": name,"Accuracy": accuracy,"Recall": recall,"F1 Score": f1,"ROC-AUC": roc_auc})
         
     st.subheader("Model Comparison (with ROC-AUC)") 
     df_results=pd.DataFrame(results)
     html_table=styled_table(df_results)
     st.markdown(html_table, unsafe_allow_html=True)

     for name, model in models.items():
         y_pred=model.predict(X)
         y_prob=model.predict_proba(X)[:,1]
         fpr, tpr, _ = roc_curve(y_test, y_prob)

         fig.add_trace(go.Scatter(x=fpr, y=tpr,mode='lines',name=name))

     fig.add_shape(type='line',line=dict(dash='dash'),x0=0, x1=1, y0=0, y1=1)

     fig.update_layout(title="ROC Curve Comparison",xaxis_title="False Positive Rate",yaxis_title="True Positive Rate",template="plotly_dark")

     st.plotly_chart(fig, use_container_width=True)

     from sklearn.inspection import partial_dependence

     features = ["Age", "Balance", "NumOfProducts", "EstimatedSalary"]

     fig = go.Figure()

     colors=["#00D4FF","#FF6B6B","#FFD93D","#6BCB77"]

     st.subheader("Partial Dependence Plots (Key Features)")
     for i, feature in enumerate(features):
         feature_index=X.columns.get_loc(feature)

         pdp=partial_dependence(model,X,features=[feature_index])
         x_vals=pdp["grid_values"][0]
         y_vals=pdp["average"][0].flatten()
         fig.add_trace(go.Scatter(x=x_vals,y=y_vals,mode="lines+markers",name=feature,line=dict(width=3,color=colors[i]), marker=dict(size=5),hovertemplate=f"<b>{feature}</b><br>Value:%{{x}}<br>Churn Prob: %{{y:.3f}}<extra></extra>"))
         fig.update_layout(font=dict(color="white"),xaxis_title="Feature Value", yaxis_title="Churn Probability", template="plotly_dark",paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)", height=550, margin=dict(l=40, r=40, t=60, b=40),legend=dict(title="Feature", orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5,font=dict(color="white")), hovermode="x unified")
         fig.update_xaxes(showgrid=True, gridcolor="rgba(255,255,255,0.1)")
         fig.update_yaxes(showgrid=True, gridcolor="rgba(255,255,255,0.1)")
         st.plotly_chart(fig, use_container_width=True)
with tab4:

     def get_metrics(model, X, y, threshold):
         y_prob=model.predict_proba(X)[:,1]
         y_pred=(y_prob>=threshold).astype(int)

         acc= accuracy_score(y, y_pred)
         rec=recall_score(y, y_pred)
         f1= f1_score(y, y_pred)
         return acc, rec, f1
    
     metrics_data=[]
     for name, model in models.items():
         acc, rec, f1= get_metrics(model, X, y_test, threshold)
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

     fig.update_layout(title="Model Performance Comparison",xaxis_title="Model",yaxis_title="Score",font=dict(color="white"),height=550,margin=dict(l=40, r=40, t=60, b=40),legend=dict(orientation="h",yanchor="bottom",y=1.02,xanchor="center",x=0.5,title="",font=dict(color="white")),template="plotly_dark",paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)", hovermode="x unified")

     fig.update_xaxes(showgrid=False)
    
     fig.update_yaxes(showgrid=True, gridcolor="rgba(255,255,255,0.1)")
    
     fig.update_traces(marker=dict(line=dict(width=1)))
    
     st.plotly_chart(fig, use_container_width=True)
    
with tab5:
     churn_rate=y_pred.mean()
     st.metric("Predicted Churn Rate", f"{churn_rate:.2%}")
     st.subheader("Prediction Distribution")
     fig=px.histogram(x=y_prob, nbins=30, title="Churn Probability Distribution")
     fig.update_layout(template="plotly_dark",font=dict(color="white"), legend=dict(font=dict(color="white")),paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)")
     fig.update_xaxes(showgrid=False)
     fig.update_yaxes(showgrid=False)
     st.plotly_chart(fig, use_container_width=True)

     st.subheader("Feature Drift Check")
     import datetime
     log_file="drift_log.csv"
     if not os.path.exists(log_file):
        pd.DataFrame(columns=["time","feature","drift"]).to_csv(log_file, index=False)
     current_data=df.drop("Exited", axis=1)
     current_mean=current_data.mean(numeric_only=True)
     train_mean=current_data.mean(numeric_only=True)
     recent_data=df.sample(200, random_state=42)
     current_mean=recent_data.mean(numeric_only=True)
     train_mean=df.mean(numeric_only=True)
     drift_df=pd.DataFrame({"Feature": current_mean.index,"Current": current_mean.values,"Training": train_mean[current_mean.index].values})
     drift_df["Drift"]=abs(drift_df["Current"]-drift_df["Training"])
     html_table=styled_table(drift_df)
     st.markdown(html_table, unsafe_allow_html=True)
     
     roc_auc_val=roc_auc_score(y_test,y_prob)
     if roc_auc_val < 0.7:
        st.error("Model performance dropped! Consider retraining.")
        if churn_rate > 0.5:
           st.warning("High churn rate detected!")

     import datetime
     log_file="drift_log.csv"
     if not os.path.exists(log_file):
        pd.DataFrame(columns=["time","feature","drift"]).to_csv(log_file, index=False)

     train_data=df.drop("Exited", axis=1, errors="ignore")
     recent_data=df.sample(200, random_state=42)
     train_mean=train_data.mean(numeric_only=True)
     recent_mean=recent_data.mean(numeric_only=True)
     drift_values=abs(train_mean-recent_mean)
     rows=[]
    
     now =datetime.datetime.now()
     for f in drift_values.index:
         rows.append({"time":now,"feature":f,"drift":drift_values[f]})

     pd.DataFrame(rows).to_csv(log_file,mode="a", header=False, index=False)
     drift_history=pd.read_csv(log_file)
     drift_history["time"]=pd.to_datetime(drift_history["time"])

     st.subheader("Feature Drift Trend Over Time")
     selected_feature=st.selectbox("Select Feature", drift_history["feature"].unique())
     feature_df=drift_history[drift_history["feature"]==selected_feature]
     fig=px.line(feature_df, x="time",y="drift",title=f"Drift Trend for {selected_feature}", markers=True)
     fig.update_layout(template="plotly_dark",xaxis_title="Time", yaxis_title="Drift",font=dict(color="white"), legend=dict(font=dict(color="white")),paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)")
     fig.update_xaxes(showgrid=False)
     fig.update_yaxes(showgrid=False)
     st.plotly_chart(fig, use_container_width=True)
