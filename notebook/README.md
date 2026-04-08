# Predictive Modeling and Risk Scoring for Bank Customer Churn
# Project Overview
This project focuses on predicting customer churn using multiple machine learning models and providing an interactive dashboard for business insights. The system not only predicts whether a customer is likely to churn but also calculates a churn risk score, identifies key drivers, and enables scenario-based analysis.

# Objectives
# Primary Objectives
- Predict customer churn with high accuracy
- Generate churn probability scores
- Identify key factors influencing churn
# Secondary Objectives
- Reduce false positives in churn prediciton
- Improve interpretability of ML models
- Enable scenario-based churn risk analysis
# Dataset Description
The dataset contains customer-level information typically used in banking/telecom churn analysis. Key features include:
- Demographics: Age, Gender, Credit Score, Tenure
- Customer Behavior: Number of Products, Activity Status
- Target Variable: Exited (1=Churn, 0=Retained)
# Machine Learning Models Used
- Logistic Regression
- Decision Tree
- Random Forest
- Gradient Boosting
- XGBoost
Each model was evaluated using performance metrics such as Accuracy, Precision, Recall, and F1 Score

# Methodology
# Data Preprocessing
- Handling missing values
- Encoding categorical variables using one-hot encoding
- Feature scaling using StandardScaler
- Train-test split
# Model Training
- Models trained on preprocessed dataset
- Hyperparameter tuning applied
- Models saved using joblib
# Feature Engineering
- Created consistent feature schema for deployment
- Stored training columns to ensure alignment in prediction
# Exploratory Data Analysis (EDA)
- Distribution plots for numerical features
- Churn comparison across categorical variables
- Violin plots for feature relationships
-  Correlation analysis
# Model Evaluation Metrics
- Accuracy: Overall correctness
- Precision: Correct churn customers
- F1 Score: Balance between precison and recall
- 
# Model Explainability
# SHAP Analysis
- Identifies feature contribution to individual predictions
- Visualized using SHAP summary and waterfall plots
# Feature Importance 
- Highlights most influencial variables
- Used for business insights
# Partial Dependence Plots (PDP)
- Shows relationship between features and churn probability

Dashboard Features (Streamlit)
- Churn Prediction with probability score
- Risk Score Gauge Visualization
- Feature Importance Graph
- SHAP Explainability (Waterfall Plot)
- Distribution & Violin Plots
- Interactive Sliders for Scenario Analysis
- Model Comparison
# Tech Stack
- Python
- Pandas, NumPy
- Scikit-learn
- XGBoost
- SHAP
- Plotly
- Streamlit

# Key Challenges & Solutions
|          Challange                    |                Solution                       |
|---------------------------------------|-----------------------------------------------|
|XGboost not responding to categorical  | Fixed by proper encoding and column alignment |
|inputs                                 |                                               |
|Feature mismatch during deployment     | Stored training columns and reindexed input   |
|Visualization inconsistencies          | Applied custom Plotly styling                 |
|Model interpretability                 | Integrated SHAP and PDP                       |

# Key Insights 
- Customers with higher balance and age are more likely to churn
- Active members are less likely to churn
- Certain geographic regions show higher churn patterns
- Feature importancve varies across models, with XGBoost focusing on strong predictors

  ![Feature Importance](Images/feature_importance.png)

# Future Work
- Improve recall for churn detection
- Add deep learning models
- deploy using cloud pltforms (Streamlit Cloud)
- Real-time prediction API
- Advanced customer segmentation

# Recommendations
- Focus on high-risk customers identified by model
- Improve engagement for inactive users
- Offer targeted retention strategies
- Monitor key churn indicators continuously
