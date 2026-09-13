import streamlit as st
import pandas as pd
import numpy as np
import shap
import plotly.express as px
import plotly.graph_objects as go

from src.dashboard.common import (
    apply_style,
    load_data,
    load_models,
    model_selection,
    customer_inputs,
)


st.set_page_config(
    page_title="Feature Importance",
    page_icon="🔍",
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
    *_,
) = customer_inputs()


st.title("🔍 Feature Importance & Explainability")


# =========================================================
# MODEL FEATURE IMPORTANCE
# =========================================================

st.subheader(
    "Feature Importance Dashboard"
)

actual_model = list(
    model.named_steps.values()
)[-1]

preprocessor = model.named_steps[
    "preprocessor"
]

feature_names = (
    preprocessor
    .get_feature_names_out()
)


if hasattr(
    actual_model,
    "feature_importances_",
):

    importance = (
        actual_model
        .feature_importances_
    )

elif hasattr(
    actual_model,
    "coef_",
):

    importance = np.abs(
        actual_model.coef_[0]
    )

else:

    st.warning(
        "Feature importance not available."
    )

    importance = None


if importance is not None:

    min_len = min(
        len(feature_names),
        len(importance),
    )

    importance_df = pd.DataFrame(
        {
            "Feature":
                feature_names[:min_len],

            "Importance":
                importance[:min_len],
        }
    ).sort_values(
        "Importance",
        ascending=False,
    )

    fig = px.bar(
        importance_df.head(15),
        x="Importance",
        y="Feature",
        orientation="h",
        color="Importance",
    )

    fig.update_layout(
        template="plotly_dark",
        yaxis=dict(
            autorange="reversed"
        ),
        height=550,
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
    )


# =========================================================
# SHAP
# =========================================================

st.subheader(
    "Global SHAP Feature Contribution"
)

X = df.drop(
    ["Exited", "CustomerId", "Surname"],
    axis=1,
    errors="ignore",
)

try:

    X_sample = X.sample(
        min(100, len(X)),
        random_state=42,
    )

    X_transformed = (
        preprocessor.transform(
            X_sample
        )
    )

    if hasattr(
        actual_model,
        "feature_importances_",
    ):

        explainer = shap.TreeExplainer(
            actual_model
        )

    else:

        explainer = shap.Explainer(
            actual_model.predict_proba,
            X_transformed,
        )

    shap_values = explainer(
        X_transformed
    )

    values = shap_values.values

    if len(values.shape) == 3:
        values = values[:, :, 1]

    mean_shap = np.abs(
        values
    ).mean(axis=0)

    min_len = min(
        len(feature_names),
        len(mean_shap),
    )

    shap_df = pd.DataFrame(
        {
            "Feature":
                feature_names[:min_len],

            "SHAP Value":
                mean_shap[:min_len],
        }
    ).sort_values(
        "SHAP Value",
        ascending=False,
    )

    col1, col2 = st.columns(2)

    with col1:

        fig = px.bar(
            shap_df.head(15),
            x="SHAP Value",
            y="Feature",
            orientation="h",
            color="SHAP Value",
        )

        fig.update_layout(
            title="Global Feature Contribution",
            template="plotly_dark",
            height=550,
        )

        st.plotly_chart(
            fig,
            use_container_width=True,
        )

    with col2:

        fig = go.Figure(
            go.Scatter(
                x=shap_df.head(15)[
                    "SHAP Value"
                ],
                y=shap_df.head(15)[
                    "Feature"
                ],
                mode="markers",
                marker=dict(
                    size=12,
                    color=shap_df.head(15)[
                        "SHAP Value"
                    ],
                    colorscale="Viridis",
                    showscale=True,
                ),
            )
        )

        fig.update_layout(
            title="Feature Impact Distribution",
            template="plotly_dark",
            height=550,
            yaxis=dict(
                autorange="reversed"
            ),
        )

        st.plotly_chart(
            fig,
            use_container_width=True,
        )

except Exception as e:

    st.error(
        f"SHAP Error: {e}"
    )