import streamlit as st

from src.dashboard.common import apply_style


st.set_page_config(
    page_title="Bank Customer Churn",
    page_icon="🏦",
    layout="wide",
)

apply_style()


# =========================================================
# HEADER
# =========================================================

col1, col2 = st.columns([0.5, 6])

with col1:

    st.image(
        "Images/office.png",
        width=60,
    )

with col2:

    st.title(
        "Predictive Modeling and Risk Scoring "
        "for Bank Customer Churn"
    )


st.divider()


# =========================================================
# WELCOME
# =========================================================

st.header(
    "Bank Customer Churn Analytics Platform"
)

st.markdown(
    """
    Explore the complete customer churn prediction
    and risk-scoring platform using the navigation
    menu on the left.
    """
)


# =========================================================
# PLATFORM OVERVIEW
# =========================================================

col1, col2, col3 = st.columns(3)

with col1:

    st.info(
        """
        ### 📊 Customer Analytics

        Explore customer behavior, churn patterns,
        feature importance and model predictions.
        """
    )

with col2:

    st.info(
        """
        ### 🤖 Machine Learning

        Compare multiple classification models
        and analyze their predictive performance.
        """
    )

with col3:

    st.info(
        """
        ### 🛡️ MLOps

        Monitor data drift, prediction drift,
        model performance and model validation.
        """
    )


st.divider()

st.markdown(
    """
    ### Available Modules

    **📊 Customer Risk Calculator**  
    Individual customer churn probability and risk scoring.

    **🔍 Feature Importance**  
    Feature importance and SHAP-based model explainability.

    **📈 ROC and PDP**  
    ROC curves and partial dependence analysis.

    **🤖 Model Comparison**  
    Compare Logistic Regression, Decision Tree,
    Random Forest, Gradient Boosting and XGBoost.

    **🛡️ Monitoring**  
    Data drift, prediction monitoring and model performance.

    **💰 Quantitative Modeling**  
    Expected loss, retention ROI and A/B retention analysis.
    """
)