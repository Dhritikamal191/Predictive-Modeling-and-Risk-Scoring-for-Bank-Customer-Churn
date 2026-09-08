import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from sklearn.metrics import confusion_matrix
from src.supabase_client import save_customer_prediction
import seaborn as sns
from src.dashboard.common import (
    apply_style,
    load_data,
    load_models,
    model_selection,
    customer_inputs,
    kpi_card,
)


st.set_page_config(
    page_title="Customer Risk Calculator",
    page_icon="📊",
    layout="wide",
)

apply_style()

df = load_data()
models = load_models()

model_choice, threshold, model = model_selection(
    models
)

(
    input_df,
    credit_score,
    age,
    tenure,
    balance,
    products,
    active_member,
    has_card,
    salary,
    gender,
    geography,
) = customer_inputs()


st.title("📊 Customer Risk Calculator")

st.markdown(
    "Predict individual customer churn probability "
    "and evaluate retention risk."
)


# =========================================================
# PREDICTION
# =========================================================

try:

    pred = model.predict(input_df)[0]
    prob = model.predict_proba(input_df)[0][1]

except Exception as e:

    st.error(
        f"Prediction Error: {e}"
    )
    st.stop()


risk_score = prob * 100


if risk_score >= 70:
    risk = "High Risk"

elif risk_score >= 40:
    risk = "Medium Risk"

else:
    risk = "Low Risk"

# =========================================================
# SAVE PREDICTION TO SUPABASE
# =========================================================

try:

    save_customer_prediction(
        {
            "CreditScore": credit_score,
            "Age": age,
            "Tenure": tenure,
            "Balance": balance,
            "NumOfProducts": products,
            "HasCrCard": has_card,
            "IsActiveMember": active_member,
            "EstimatedSalary": salary,
            "Geography": geography,
            "Gender": gender,
            "Model": model_choice,
            "ChurnProbability": float(prob),
            "RiskScore": float(risk_score),
            "RiskCategory": risk,
            "Prediction": int(pred),
        }
    )

except Exception as e:

    st.warning(
        f"Prediction generated, but could not be saved: {e}"
    )
    
if pred == 1:

    st.error(
        "Customer likely to CHURN"
    )

else:

    st.success(
        "Customer NOT likely to churn"
    )


# =========================================================
# KPIs
# =========================================================

X = df.drop(
    ["Exited", "CustomerId", "Surname"],
    axis=1,
    errors="ignore",
)

y = df["Exited"]

probs = model.predict_proba(X)[:, 1]

col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    st.markdown(
        kpi_card(
            "Churn Probability",
            f"{prob:.3f}",
            "📉",
        ),
        unsafe_allow_html=True,
    )

with col2:
    st.markdown(
        kpi_card(
            "Risk Score",
            f"{risk_score:.0f}/100",
            "⚠️",
        ),
        unsafe_allow_html=True,
    )

with col3:
    st.markdown(
        kpi_card(
            "Risk Category",
            risk,
            "🚦",
        ),
        unsafe_allow_html=True,
    )

with col4:
    st.markdown(
        kpi_card(
            "Avg. Churn Probability",
            f"{probs.mean():.3f}",
            "📊",
        ),
        unsafe_allow_html=True,
    )

with col5:
    st.markdown(
        kpi_card(
            "Max Risk Score",
            f"{probs.max() * 100:.1f}%",
            "📈",
        ),
        unsafe_allow_html=True,
    )


st.divider()


# =========================================================
# RISK VISUALIZATION
# =========================================================

col1, col2 = st.columns(2)

with col1:

    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=risk_score,
            gauge={
                "axis": {
                    "range": [0, 100]
                },
                "steps": [
                    {
                        "range": [0, 40],
                        "color": "#16a34a",
                    },
                    {
                        "range": [40, 70],
                        "color": "#ca8a04",
                    },
                    {
                        "range": [70, 100],
                        "color": "#dc2626",
                    },
                ],
            },
        )
    )

    fig.update_layout(
        title="Customer Churn Risk (%)",
        template="plotly_dark",
        height=400,
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
    )


with col2:

    scenario_df = pd.DataFrame(
        {
            "Type": [
                "Original",
                "Adjusted",
            ],
            "Risk": [
                risk_score,
                risk_score,
            ],
        }
    )

    fig = px.pie(
        scenario_df,
        names="Type",
        values="Risk",
        hole=0.5,
        color="Type",
    )

    fig.update_layout(
        title="Customer Churn Risk Comparison",
        template="plotly_dark",
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
    )


# =========================================================
# RECOMMENDATIONS
# =========================================================

st.subheader("Recommendations")

recommendations = []

if age > 50:
    recommendations.append(
        "Provide personalized relationship management "
        "for senior customers."
    )

if balance > 100000:
    recommendations.append(
        "Offer premium banking benefits and wealth "
        "management services."
    )

if active_member == 0:
    recommendations.append(
        "Increase engagement through loyalty programs "
        "and personalized offers."
    )

if products <= 1:
    recommendations.append(
        "Recommend additional banking products "
        "to improve retention."
    )

if credit_score < 500:
    recommendations.append(
        "Provide financial wellness support and "
        "flexible credit solutions."
    )

if geography == "Germany":
    recommendations.append(
        "Customers from Germany show relatively "
        "higher churn tendency."
    )

if salary > 120000:
    recommendations.append(
        "Offer premium investment and savings plans."
    )

for recommendation in recommendations:
    st.markdown(
        f"• {recommendation}"
    )


# =========================================================
# CONFUSION MATRIX
# =========================================================

st.subheader("Prediction Confusion Matrix")

y_prob = model.predict_proba(X)[:, 1]

y_pred = (
    y_prob >= threshold
).astype(int)

cm = confusion_matrix(
    y,
    y_pred,
)

# ---------------------------------------------------------
# CONFUSION MATRIX
# ---------------------------------------------------------

fig = go.Figure(
    data=go.Heatmap(
        z=cm,
        x=["No Churn", "Churn"],
        y=["No Churn", "Churn"],
        colorscale="Reds",
        text=cm,
        texttemplate="%{text}",
        textfont=dict(size=16),
        hovertemplate=(
            "Actual: %{y}<br>"
            "Predicted: %{x}<br>"
            "Count: %{z}"
            "<extra></extra>"
        ),
        showscale=True,
    )
)

fig.update_layout(
    template="plotly_dark",
    xaxis_title="Predicted",
    yaxis_title="Actual",
)

st.plotly_chart(
    fig,
    use_container_width=True,
)

# --------------------------------------------------
# Probability Distribution Visualization
# --------------------------------------------------

st.subheader("Customer Distribution Plot")
plot_results=pd.DataFrame({"Probability":y_prob, "Actual_Status": y})
plot_results["Actual_Status"]=plot_results["Actual_Status"].map({1:"Churned", 0: "Stayed"})
fig = px.histogram(plot_results, x="Probability", nbins=30, color="Actual_Status", color_discrete_sequence=["#6366f1","#f43f5e"], barmode="overlay", opacity=0.5)
fig.update_traces(marker_line_width=0)
fig.update_layout(title=dict(text="Probability Distribution Visualization",x=0.5, xanchor="center",font=dict(size=17, color="white")),
bargap=0.1, legend_title_text="Customer Status",template="plotly_dark", 
legend=dict(font=dict(color="white")),paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)")
fig.update_xaxes(showgrid=False)
fig.update_yaxes(showgrid=False)
st.plotly_chart(fig, use_container_width=True)
