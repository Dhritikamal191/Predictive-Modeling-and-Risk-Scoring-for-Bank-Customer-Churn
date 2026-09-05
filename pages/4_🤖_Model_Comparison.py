import streamlit as st
import pandas as pd
import plotly.express as px

from sklearn.metrics import (
    accuracy_score,
    recall_score,
    f1_score,
)

from src.dashboard.common import (
    apply_style,
    load_data,
    load_models,
    model_selection,
    customer_inputs,
)


st.set_page_config(
    page_title="Model Comparison",
    page_icon="🤖",
    layout="wide",
)

apply_style()

df = load_data()
models = load_models()

model_selection(models)
customer_inputs()

X = df.drop(
    ["Exited", "CustomerId", "Surname"],
    axis=1,
    errors="ignore",
)

y = df["Exited"]


st.title("🤖 Model Comparison")


# =========================================================
# CALCULATE METRICS
# =========================================================

results = []

for name, current_model in models.items():

    probabilities = (
        current_model
        .predict_proba(X)[:, 1]
    )

    predictions = (
        probabilities >= 0.50
    ).astype(int)

    results.append(
        {
            "Model": name,
            "Accuracy": accuracy_score(
                y,
                predictions,
            ),
            "Recall": recall_score(
                y,
                predictions,
            ),
            "F1 Score": f1_score(
                y,
                predictions,
            ),
        }
    )


metrics_df = pd.DataFrame(
    results
)


# =========================================================
# BEST MODEL
# =========================================================

best_row = (
    metrics_df
    .sort_values(
        "F1 Score",
        ascending=False,
    )
    .iloc[0]
)


col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "🏆 Best Model",
    best_row["Model"],
)

col2.metric(
    "Accuracy",
    f"{best_row['Accuracy']:.2%}",
)

col3.metric(
    "Recall",
    f"{best_row['Recall']:.2%}",
)

col4.metric(
    "F1 Score",
    f"{best_row['F1 Score']:.2%}",
)


# =========================================================
# TABLE
# =========================================================

st.subheader(
    "Model Comparison Table"
)

display_df = metrics_df.copy()

for column in [
    "Accuracy",
    "Recall",
    "F1 Score",
]:

    display_df[column] = (
        display_df[column]
        .map(lambda x: f"{x:.2%}")
    )


st.dataframe(
    display_df,
    use_container_width=True,
    hide_index=True,
)


# =========================================================
# CHART
# =========================================================

melted = metrics_df.melt(
    id_vars="Model",
    var_name="Metric",
    value_name="Score",
)

fig = px.bar(
    melted,
    x="Model",
    y="Score",
    color="Metric",
    barmode="group",
    text="Score",
)

fig.update_traces(
    texttemplate="%{text:.2f}",
    textposition="outside",
)

fig.update_layout(
    title="Model Performance Comparison",
    template="plotly_dark",
    height=550,
)

st.plotly_chart(
    fig,
    use_container_width=True,
)