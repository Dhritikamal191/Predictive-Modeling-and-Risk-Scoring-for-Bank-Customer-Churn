import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px


from src.dashboard.common import (
    apply_style,
    load_data,
    load_models,
    model_selection,
    customer_inputs,
)


st.set_page_config(
    page_title="Quantitative Modeling",
    page_icon="💰",
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


st.title(
    "💰 Quantitative Modeling"
)


# =========================================================
# RETENTION EFFECTIVENESS
# =========================================================

treatment_effectiveness = st.sidebar.slider(
    "Retention Effectiveness",
    0.0,
    0.5,
    0.20,
    0.01,
)


# =========================================================
# ONLY LOGISTIC REGRESSION
# =========================================================

if model_choice == "Logistic Regression":

    st.subheader(
        "Logistic Regression Probability Model"
    )

    st.latex(
        r"""
        z = \beta_0 +
        \beta_1 X_1 +
        \beta_2 X_2 +
        \cdots +
        \beta_n X_n
        """
    )

    st.latex(
        r"""
        P(Churn)=\frac{1}{1+e^{-z}}
        """
    )

    st.latex(
        r"""
        \mathbb{E}[Loss]
        =
        P(Churn)\times Customer\ Value
        """
    )

    st.latex(
        r"""
        ROI=
        \frac{Retention\ Savings-Retention\ Cost}
        {Retention\ Cost}
        """
    )


    # -----------------------------------------------------
    # MODEL
    # -----------------------------------------------------

    lr_pipeline = models[
        "Logistic Regression"
    ]

    preprocessor = (
        lr_pipeline
        .named_steps["preprocessor"]
    )

    lr_model = (
        lr_pipeline
        .named_steps["model"]
    )


    # -----------------------------------------------------
    # TRANSFORM INPUT
    # -----------------------------------------------------

    input_scaled = (
        preprocessor.transform(
            input_df
        )
    )

    coefficients = (
        lr_model.coef_[0]
    )

    intercept = (
        lr_model.intercept_[0]
    )


    # -----------------------------------------------------
    # CALCULATE Z
    # -----------------------------------------------------

    z = (
        intercept
        +
        np.dot(
            input_scaled[0],
            coefficients,
        )
    )

    probability_formula = (
        1 /
        (
            1 +
            np.exp(-z)
        )
    )


    st.subheader(
        "Quantitative Probability Calculation"
    )

    st.write(
        f"Intercept (β₀): {intercept:.4f}"
    )

    st.write(
        f"Linear Combination (z): {z:.4f}"
    )

    st.write(
        f"Calculated Probability: "
        f"{probability_formula:.4f}"
    )


    # -----------------------------------------------------
    # MODEL PROBABILITY
    # -----------------------------------------------------

    model_prob = (
        lr_pipeline
        .predict_proba(
            input_df
        )[0][1]
    )


    # -----------------------------------------------------
    # BUSINESS VALUES
    # -----------------------------------------------------

    customer_value = (
        balance + salary
    )

    expected_loss = (
        probability_formula
        * customer_value
    )

    retention_cost = (
        customer_value
        * treatment_effectiveness
    )

    expected_saved_value = (
        expected_loss
        * treatment_effectiveness
    )

    roi = (
        expected_saved_value
        - retention_cost
    ) / retention_cost


    potential_loss = (
        customer_value
        * (1 + model_prob)
    )


    treatment_probability = max(
        0,
        probability_formula
        - treatment_effectiveness,
    )

    treatment_risk = (
        treatment_probability * 100
    )


    # -----------------------------------------------------
    # BUSINESS KPIs
    # -----------------------------------------------------

    col1, col2, col3, col4, col5 = (
        st.columns(5)
    )

    with col1:

        st.metric(
            "Expected Loss",
            f"₹{expected_loss:,.0f}",
        )

    with col2:

        st.metric(
            "Retention ROI",
            f"{roi:.2f}x",
        )

    with col3:

        st.metric(
            "Customer Value",
            f"₹{customer_value:,.0f}",
        )

    with col4:

        st.metric(
            "Potential Churn Loss",
            f"₹{potential_loss:,.0f}",
        )

    with col5:

        st.metric(
            "Retention Cost",
            f"₹{retention_cost:,.0f}",
        )


    # -----------------------------------------------------
    # COEFFICIENT CONTRIBUTIONS
    # -----------------------------------------------------

    feature_names = (
        preprocessor
        .get_feature_names_out()
    )

    min_len = min(
        len(feature_names),
        len(coefficients),
    )

    coef_df = pd.DataFrame(
        {
            "Feature":
                feature_names[:min_len],

            "Coefficient":
                coefficients[:min_len],

            "Contribution":
                (
                    input_scaled[0][:min_len]
                    *
                    coefficients[:min_len]
                ),
        }
    )

    st.subheader(
        "Feature Contributions"
    )

    st.dataframe(
        coef_df,
        use_container_width=True,
        hide_index=True,
    )


    # =====================================================
    # A/B TESTING
    # =====================================================

    st.subheader(
        "Retention Strategy A/B Testing"
    )

    uplift = (
        model_prob
        - treatment_probability
    )

    saved_revenue = (
        uplift * customer_value
    )

    ab_roi = (
        saved_revenue
        - retention_cost
    ) / retention_cost


    col1, col2, col3, col4 = (
        st.columns(4)
    )

    with col1:

        st.metric(
            "Control Churn",
            f"{probability_formula:.2%}",
        )

    with col2:

        st.metric(
            "Treatment Churn",
            f"{treatment_probability:.2%}",
        )

    with col3:

        st.metric(
            "Uplift",
            f"{uplift:.2%}",
        )

    with col4:

        st.metric(
            "Experimental ROI",
            f"{ab_roi:.2f}x",
        )


    ab_df = pd.DataFrame(
        {
            "Type": [
                "Control",
                "Treatment",
            ],
            "Risk": [
                probability_formula * 100,
                treatment_risk,
            ],
        }
    )

    fig = px.pie(
        ab_df,
        names="Type",
        values="Risk",
        hole=0.5,
    )

    fig.update_layout(
        template="plotly_dark"
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
    )


else:

    st.info(
        "Select Logistic Regression from "
        "the sidebar to use the quantitative "
        "probability model."
    )